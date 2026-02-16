"""
Market Research Assistant
=========================
A Streamlit-based RAG application that generates industry reports
from Wikipedia sources. Built for the MSIN0231 Machine Learning
for Business individual assignment.

Architecture:
    Sidebar  -> LLM configuration (model selection + API key)
    Step 1   -> Industry input + LLM-based validation
    Step 2   -> Wikipedia retrieval (5 most relevant pages)
    Step 3   -> Industry report generation (<500 words)

Design Principles:
    - Modular: each pipeline stage is an isolated function
    - Separation of concerns: retrieval logic =/= generation logic
    - Fail-safe: every external call has error handling
    - KISS: minimal complexity, maximum clarity
"""

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
APP_TITLE = "Market Research Assistant"
APP_ICON = "📊"
LLM_OPTIONS = ["Gemini 2.0 Flash"]
LLM_MODEL_MAP = {"Gemini 2.0 Flash": "gemini-2.0-flash"}
DEFAULT_TEMPERATURE = 0.2          # Low temperature for factual output
MAX_WIKI_RESULTS = 8               # Retrieve more, then filter to best 5
FINAL_SOURCE_COUNT = 5             # Exactly 5 URLs returned to user
MAX_REPORT_WORDS = 480             # Target word count (buffer under 500)


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — modular pipeline stages
# ──────────────────────────────────────────────────────────────

def initialise_llm(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
    """Create a LangChain LLM instance with the given configuration.

    Uses a low temperature to favour factual, deterministic outputs
    suitable for market research analysis.
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_MAP[model_name],
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )


def validate_industry(llm: ChatGoogleGenerativeAI, user_input: str) -> dict:
    """Use the LLM to determine whether user_input names a real industry.

    Returns:
        dict with keys:
            - is_valid (bool): True if the input is a recognised industry
            - reason  (str) : Explanation for the decision
            - normalised (str): Cleaned industry name if valid, else empty

    Design choice: LLM-based validation catches typos, slang, and
    non-industry inputs (e.g. 'pizza' vs 'pizza industry') that
    simple string checks would miss.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a validation assistant. Your ONLY job is to decide "
         "whether the user's input refers to a legitimate industry or "
         "economic sector. Respond in EXACTLY this format:\n"
         "VALID: yes or no\n"
         "NORMALISED: <the cleaned, standard industry name if valid, "
         "otherwise empty>\n"
         "REASON: <one-sentence explanation>\n\n"
         "Examples of VALID industries: semiconductor manufacturing, "
         "renewable energy, pharmaceutical, fast fashion, fintech.\n"
         "Examples of INVALID inputs: empty text, random words, "
         "individual company names, people's names, jokes."),
        ("human", "Is this a valid industry? Input: '{input}'"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": user_input})

    # Parse structured response
    lines = response.strip().split("\n")
    result = {"is_valid": False, "reason": "", "normalised": ""}

    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("valid:"):
            result["is_valid"] = "yes" in lower
        elif lower.startswith("normalised:") or lower.startswith("normalized:"):
            result["normalised"] = line.split(":", 1)[1].strip()
        elif lower.startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


def retrieve_wikipedia_pages(industry: str) -> list[dict]:
    """Retrieve relevant Wikipedia pages for the given industry.

    Strategy:
        1. Fetch up to MAX_WIKI_RESULTS pages via WikipediaRetriever
        2. Each result includes title, content summary, and source URL
        3. The caller (Step 2) then uses the LLM to rank and select
           the FINAL_SOURCE_COUNT most relevant pages

    This two-stage approach (broad retrieval -> LLM-based filtering)
    ensures high relevance and avoids returning tangential pages.
    """
    retriever = WikipediaRetriever(
        top_k_results=MAX_WIKI_RESULTS,
        doc_content_chars_max=4000,   # Enough context without overwhelming
    )

    docs = retriever.invoke(industry)

    pages = []
    seen_titles = set()
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        # Deduplicate by title
        if title in seen_titles:
            continue
        seen_titles.add(title)
        source = doc.metadata.get("source", "")
        # Build URL if not provided directly
        if not source:
            source = (
                "https://en.wikipedia.org/wiki/"
                + title.replace(" ", "_")
            )
        pages.append({
            "title": title,
            "url": source,
            "content": doc.page_content,
        })

    return pages


def select_top_pages(
    llm: ChatGoogleGenerativeAI,
    industry: str,
    pages: list[dict],
) -> list[dict]:
    """Use the LLM to rank retrieved pages and select the 5 most relevant.

    This filtering step is critical for retrieval quality. Without it,
    WikipediaRetriever may return tangentially related pages (e.g.
    a biography of an industry founder instead of the industry itself).

    The LLM sees each page's title and a content snippet, then returns
    the indices of the 5 most relevant pages ranked by usefulness for
    an industry market research report.
    """
    if len(pages) <= FINAL_SOURCE_COUNT:
        return pages

    # Build a numbered list of candidates for the LLM to evaluate
    candidate_descriptions = ""
    for i, page in enumerate(pages):
        snippet = page["content"][:500]
        candidate_descriptions += (
            f"[{i}] Title: {page['title']}\n"
            f"    Snippet: {snippet}...\n\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research librarian selecting the most relevant "
         "Wikipedia sources for a market research report.\n\n"
         "TASK: From the numbered candidate pages below, select EXACTLY "
         "5 that are MOST relevant to writing an industry report about "
         "'{industry}'. Prioritise pages that cover:\n"
         "- The industry overview and structure\n"
         "- Market size, trends, and growth drivers\n"
         "- Key companies and competitive landscape\n"
         "- Regulation, risks, and challenges\n"
         "- Technology and innovation in the sector\n\n"
         "Respond with ONLY the 5 index numbers, one per line, "
         "in order of relevance (most relevant first).\n"
         "Example response:\n3\n0\n7\n1\n5"),
        ("human",
         "Industry: {industry}\n\nCandidate pages:\n{candidates}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "industry": industry,
        "candidates": candidate_descriptions,
    })

    # Parse indices from response
    selected = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("[]").strip()
        try:
            idx = int(line)
            if 0 <= idx < len(pages) and idx not in [s["_idx"] for s in selected if "_idx" in s]:
                page = pages[idx].copy()
                page["_idx"] = idx
                selected.append(page)
        except ValueError:
            continue

    # Fallback: if parsing fails, return the first 5
    if len(selected) < FINAL_SOURCE_COUNT:
        for page in pages:
            if page not in selected:
                selected.append(page)
            if len(selected) == FINAL_SOURCE_COUNT:
                break

    # Remove internal index key
    for page in selected:
        page.pop("_idx", None)

    return selected[:FINAL_SOURCE_COUNT]


def generate_report(
    llm: ChatGoogleGenerativeAI,
    industry: str,
    pages: list[dict],
) -> str:
    """Generate a structured industry report (<500 words) from Wikipedia sources.

    The prompt is carefully engineered to produce analyst-grade output:
    - Structured sections (overview, drivers, risks, outlook)
    - Evidence-based language ('According to...', 'Data suggests...')
    - Synthesis across sources (not copy-paste from one page)
    - Hard word limit enforced in the prompt

    Design choice: we pass the full content of all 5 pages to give
    the LLM maximum context for synthesis. The temperature is kept
    low (0.2) to minimise hallucination beyond source material.
    """
    # Combine source material with clear attribution
    source_material = ""
    for i, page in enumerate(pages, 1):
        source_material += (
            f"--- Source {i}: {page['title']} ---\n"
            f"URL: {page['url']}\n"
            f"{page['content']}\n\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior market research analyst at a large corporation. "
         "Write a concise industry report based ONLY on the provided "
         "Wikipedia sources. Do NOT invent facts beyond what the sources "
         "contain.\n\n"
         "REPORT REQUIREMENTS:\n"
         "1. STRICTLY under {max_words} words (this is a hard limit)\n"
         "2. Use this structure:\n"
         "   **Industry Overview** - definition, scope, and market structure\n"
         "   **Key Drivers & Trends** - growth factors, technology shifts\n"
         "   **Competitive Landscape** - major players, market concentration\n"
         "   **Risks & Challenges** - regulatory, economic, operational\n"
         "   **Outlook** - future direction based on available evidence\n"
         "3. Use evidence language: 'According to [Source]...', "
         "'Data from [Source] suggests...'\n"
         "4. Synthesise across multiple sources — do not summarise "
         "each source separately\n"
         "5. Write in a professional, analytical tone suitable for "
         "a business audience\n"
         "6. End with exactly one line: 'Word count: [N]' where N is "
         "the actual word count of the report above"),
        ("human",
         "Write an industry report on: **{industry}**\n\n"
         "Sources:\n{sources}"),
    ])

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "industry": industry,
        "sources": source_material,
        "max_words": MAX_REPORT_WORDS,
    })

    return report


# ──────────────────────────────────────────────────────────────
# SESSION STATE — manages the multi-step flow
# ──────────────────────────────────────────────────────────────

def init_session_state():
    """Initialise session state variables on first run.

    Using session state ensures the app maintains context across
    Streamlit reruns (which happen on every user interaction).
    """
    defaults = {
        "current_step": 1,
        "industry_input": "",
        "validated_industry": "",
        "wiki_pages": [],
        "report": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline():
    """Clear all pipeline state to start fresh.

    Called when the user changes their industry input, ensuring
    stale results from a previous query are not displayed.
    """
    st.session_state.current_step = 1
    st.session_state.validated_industry = ""
    st.session_state.wiki_pages = []
    st.session_state.report = ""


# ──────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str]:
    """Render the sidebar with LLM configuration controls.

    Returns:
        (selected_model, api_key) tuple

    Security: API key is entered as a password field and stored
    only in session state. It is never written to disk or logged.
    """
    with st.sidebar:
        st.header("🔧 Configuration")

        selected_model = st.selectbox(
            "Select LLM",
            options=LLM_OPTIONS,
            index=0,
            help="The LLM used for validation and report generation.",
        )

        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your Google AI API key",
            help="Your key is not stored and is only used for this session.",
        )

        st.divider()
        st.caption(
            "💡 Get a free API key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)"
        )

        # Display pipeline status
        st.divider()
        st.subheader("Pipeline Status")
        step = st.session_state.get("current_step", 1)
        steps = [
            ("1️⃣", "Industry Input", step >= 2),
            ("2️⃣", "Source Retrieval", step >= 3),
            ("3️⃣", "Report Generation", step >= 4),
        ]
        for icon, label, done in steps:
            if done:
                st.markdown(f"{icon} ~~{label}~~ ✅")
            elif step == steps.index((icon, label, done)) + 1:
                st.markdown(f"{icon} **{label}** ⏳")
            else:
                st.markdown(f"{icon} {label}")

    return selected_model, api_key


def render_step_1(llm: ChatGoogleGenerativeAI):
    """Step 1: Industry input and validation."""
    st.header("Step 1: Enter an Industry")
    st.markdown(
        "Provide the name of an industry or economic sector to research. "
        "The assistant will validate your input before proceeding."
    )

    # Text input with callback to reset if changed
    industry = st.text_input(
        "Industry name",
        placeholder="e.g. Renewable Energy, Semiconductor Manufacturing, Fintech",
        key="industry_text_input",
    )

    if st.button("Validate Industry", type="primary", disabled=not industry):
        # Reset downstream results when input changes
        reset_pipeline()
        st.session_state.industry_input = industry

        with st.spinner("Validating your input..."):
            try:
                result = validate_industry(llm, industry)
            except Exception as e:
                st.error(f"Validation failed: {e}")
                return

        if result["is_valid"]:
            normalised = result["normalised"] or industry
            st.session_state.validated_industry = normalised
            st.session_state.current_step = 2
            st.success(
                f"✅ Recognised industry: **{normalised}**"
            )
            if result["reason"]:
                st.caption(f"ℹ️ {result['reason']}")
            st.rerun()
        else:
            st.warning(
                "⚠️ That doesn't appear to be a recognised industry. "
                "Please try again with a specific industry name."
            )
            if result["reason"]:
                st.caption(f"ℹ️ {result['reason']}")

    elif not industry:
        st.info("👆 Please enter an industry name above to get started.")


def render_step_2(llm: ChatGoogleGenerativeAI):
    """Step 2: Retrieve and display 5 most relevant Wikipedia sources."""
    industry = st.session_state.validated_industry

    st.header("Step 2: Relevant Wikipedia Sources")
    st.markdown(f"Retrieving sources for: **{industry}**")

    if not st.session_state.wiki_pages:
        with st.spinner("Searching Wikipedia and ranking by relevance..."):
            try:
                # Stage A: Broad retrieval
                raw_pages = retrieve_wikipedia_pages(industry)

                if not raw_pages:
                    st.error(
                        "No Wikipedia pages found for this industry. "
                        "Try a broader or more common industry name."
                    )
                    return

                # Stage B: LLM-based relevance filtering
                top_pages = select_top_pages(llm, industry, raw_pages)
                st.session_state.wiki_pages = top_pages

            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

    # Display the 5 selected sources
    pages = st.session_state.wiki_pages
    st.markdown(f"**Top {len(pages)} sources selected by relevance:**")

    for i, page in enumerate(pages, 1):
        st.markdown(f"{i}. [{page['title']}]({page['url']})")

    st.divider()

    if st.button("Generate Industry Report →", type="primary"):
        st.session_state.current_step = 3
        st.rerun()


def render_step_3(llm: ChatGoogleGenerativeAI):
    """Step 3: Generate and display the industry report."""
    industry = st.session_state.validated_industry
    pages = st.session_state.wiki_pages

    st.header("Step 3: Industry Report")

    if not st.session_state.report:
        with st.spinner("Generating your industry report... (this may take a moment)"):
            try:
                report = generate_report(llm, industry, pages)
                st.session_state.report = report
                st.session_state.current_step = 4
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                return

    # Display the report
    st.markdown(st.session_state.report)

    # Show the sources used
    st.divider()
    st.subheader("Sources")
    for i, page in enumerate(st.session_state.wiki_pages, 1):
        st.markdown(f"{i}. [{page['title']}]({page['url']})")

    # Option to start over
    st.divider()
    if st.button("🔄 Research a different industry"):
        reset_pipeline()
        st.rerun()


# ──────────────────────────────────────────────────────────────
# MAIN — application entry point
# ──────────────────────────────────────────────────────────────

def main():
    """Main application entry point.

    Orchestrates the three-step pipeline:
        1. Industry validation
        2. Wikipedia retrieval
        3. Report generation

    Each step only renders when its prerequisites are met,
    creating a clear, guided user experience.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )

    init_session_state()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("AI-powered industry research from Wikipedia sources")

    # Sidebar — LLM configuration
    selected_model, api_key = render_sidebar()

    # Gate: require API key before proceeding
    if not api_key:
        st.info(
            "👈 Please enter your API key in the sidebar to begin. "
            "You can get a free key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)."
        )
        return

    # Initialise the LLM
    llm = initialise_llm(selected_model, api_key)

    # Render the current step
    step = st.session_state.current_step

    if step == 1:
        render_step_1(llm)
    elif step == 2:
        render_step_1(llm)     # Keep input visible
        st.divider()
        render_step_2(llm)
    elif step >= 3:
        render_step_1(llm)     # Keep input visible
        st.divider()
        render_step_2(llm)
        st.divider()
        render_step_3(llm)


if __name__ == "__main__":
    main()
