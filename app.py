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
APP_ICON = ":material/query_stats:"
LLM_OPTIONS = ["Gemini 2.5 Flash"]
LLM_MODEL_MAP = {"Gemini 2.5 Flash": "gemini-2.5-flash"}
DEFAULT_TEMPERATURE = 0.2          # Low temperature for factual output
MAX_WIKI_RESULTS = 6               # Results per search query
FINAL_SOURCE_COUNT = 5             # Exactly 5 URLs returned to user
MAX_REPORT_WORDS = 480             # Target word count (buffer under 500)
HARD_WORD_LIMIT = 500              # Absolute maximum enforced programmatically
WIKI_CONTENT_CHARS = 8000          # Characters per Wikipedia page (more context)


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — modular pipeline stages
# ──────────────────────────────────────────────────────────────

def handle_api_error(e: Exception, context: str = "Operation") -> None:
    """Display a user-friendly error message for common API failures.

    Centralised error handling ensures consistent messaging and avoids
    exposing raw API error details to the end user.
    """
    error_msg = str(e).lower()
    if "api key" in error_msg or "api_key" in error_msg:
        st.error(
            "**Invalid API key.** Please check your Google AI API key "
            "in the sidebar. Get a free key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)."
        )
    elif "resource_exhausted" in error_msg or "429" in error_msg or "quota" in error_msg:
        st.warning(
            "**Rate limit reached.** Please wait 1-2 minutes and try again."
        )
    else:
        st.error(f"{context} failed: {e}")


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


def generate_search_queries(llm: ChatGoogleGenerativeAI, industry: str) -> list[str]:
    """Use the LLM to generate multiple diverse search queries for Wikipedia.

    Design choice: A single search query (e.g. 'renewable energy') often
    misses important sub-topics like market size, regulation, or key
    companies. By generating 3 targeted queries, we cast a wider net
    and retrieve more diverse, relevant pages for the final report.

    This is a key retrieval improvement over naive single-query approaches.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research strategist. Given an industry, generate exactly "
         "3 distinct Wikipedia search queries that together would provide "
         "comprehensive coverage for a market research report.\n\n"
         "The queries should target different aspects:\n"
         "1. The industry itself (overview, definition)\n"
         "2. The market or economics of the industry\n"
         "3. Key technology, regulation, or major companies in the industry\n\n"
         "Respond with ONLY the 3 queries, one per line. No numbering, "
         "no explanation."),
        ("human", "Industry: {industry}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"industry": industry})

    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    # Always include the original industry as a query too
    if industry not in queries:
        queries.insert(0, industry)
    return queries[:4]  # Cap at 4 queries to control API usage


def retrieve_wikipedia_pages(industry: str, queries: list[str]) -> list[dict]:
    """Retrieve relevant Wikipedia pages using multiple search queries.

    Strategy:
        1. Run WikipediaRetriever for each generated query
        2. Deduplicate results by page title
        3. Collect content, title, and URL for each unique page
        4. The caller (Step 2) then uses the LLM to rank and select
           the FINAL_SOURCE_COUNT most relevant pages

    Using multiple queries (broad retrieval -> LLM-based filtering)
    ensures high coverage and avoids missing important sub-topics.
    """
    retriever = WikipediaRetriever(
        top_k_results=MAX_WIKI_RESULTS,
        doc_content_chars_max=WIKI_CONTENT_CHARS,
    )

    pages = []
    seen_titles = set()

    for query in queries:
        try:
            docs = retriever.invoke(query)
        except Exception:
            continue  # Skip failed queries, try remaining ones

        for doc in docs:
            title = doc.metadata.get("title", "Unknown")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            source = doc.metadata.get("source", "")
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
        snippet = page["content"][:600]
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
         "AVOID selecting pages about:\n"
         "- Individual people or biographies\n"
         "- Unrelated tangential topics\n"
         "- Overly narrow sub-topics that don't inform the big picture\n\n"
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
    seen_indices = set()
    for line in response.strip().split("\n"):
        line = line.strip().strip("[]").strip()
        try:
            idx = int(line)
            if 0 <= idx < len(pages) and idx not in seen_indices:
                seen_indices.add(idx)
                selected.append(pages[idx].copy())
        except ValueError:
            continue

    # Fallback: if parsing fails, return the first 5
    if len(selected) < FINAL_SOURCE_COUNT:
        for i, page in enumerate(pages):
            if i not in seen_indices:
                selected.append(page)
                seen_indices.add(i)
            if len(selected) == FINAL_SOURCE_COUNT:
                break

    return selected[:FINAL_SOURCE_COUNT]


def enforce_word_limit(text: str, limit: int = HARD_WORD_LIMIT) -> str:
    """Programmatically enforce the word limit on generated reports.

    Design choice: LLMs are unreliable at counting words. Rather than
    trusting the model, we truncate at the sentence boundary nearest
    to the limit. This guarantees compliance with the <500 word
    requirement regardless of LLM behaviour.
    """
    words = text.split()
    if len(words) <= limit:
        return text

    # Truncate to limit, then find the last sentence boundary
    truncated = " ".join(words[:limit])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.5:  # Only cut at sentence if reasonable
        truncated = truncated[:last_period + 1]

    return truncated


def count_words(text: str) -> int:
    """Count words in text, excluding markdown formatting symbols."""
    clean = text.replace("**", "").replace("*", "").replace("#", "")
    return len(clean.split())


def generate_report(
    llm: ChatGoogleGenerativeAI,
    industry: str,
    pages: list[dict],
) -> str:
    """Generate a structured, executive-ready industry report (<500 words).

    The prompt is engineered to produce consulting-grade output:
    - Executive summary with strategic implications
    - Market structure and competitive dynamics analysis
    - Key data table summarising quantitative findings
    - Strategic interpretation (insight over description)
    - Inline citations referencing specific Wikipedia source titles
    - Hard word limit enforced both in prompt and programmatically

    Design choice: we pass the full content of all 5 pages to give
    the LLM maximum context for synthesis. The temperature is kept
    low (0.2) to minimise hallucination beyond source material.
    """
    # Build source titles list for citation instructions
    source_titles = [page["title"] for page in pages]
    titles_str = ", ".join(f'"{t}"' for t in source_titles)

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
         "You are a senior market intelligence analyst preparing a "
         "professional industry report for a corporate strategy team.\n\n"
         "Generate a concise, executive-ready industry report that "
         "strictly follows the rules below.\n\n"
         "================================ INPUT\n"
         "You will receive the industry name and extracted text from "
         "five Wikipedia pages. Available sources: {source_titles}.\n"
         "You MUST base your report ONLY on those five sources.\n"
         "Do NOT use outside knowledge. Do NOT invent facts. Do NOT "
         "add statistics not present in the provided material.\n"
         "If information is missing, state: 'Data not available in "
         "retrieved sources.'\n\n"
         "================================ WORD LIMIT\n"
         "Maximum: {max_words} words. Target range: 430-480 words.\n"
         "If output exceeds {max_words} words, rewrite more concisely.\n\n"
         "================================ MANDATORY STRUCTURE\n"
         "Use markdown headings (##) and this exact order:\n\n"
         "## Executive Summary\n"
         "Key insight, strategic implication, and recommendation in "
         "2-3 sentences.\n\n"
         "## Industry Overview\n"
         "Definition, scope, and scale indicators.\n\n"
         "## Market Structure & Competitive Dynamics\n"
         "Key segments, major players, level of competition, "
         "differentiation factors, barriers to entry.\n\n"
         "## Growth Drivers\n"
         "Economic, technological, and behavioural factors.\n\n"
         "## Risks & Constraints\n"
         "Regulatory, structural, and operational risks.\n\n"
         "## Key Data\n"
         "Include a compact markdown table summarising the most "
         "decision-relevant quantitative figures found in the sources. "
         "If no quantitative data is available, state this explicitly.\n\n"
         "## Strategic Interpretation\n"
         "Explain what the findings mean for decision-makers. Do not "
         "repeat numbers. Interpret them.\n\n"
         "## Final Takeaway\n"
         "One strong concluding insight in 1-2 sentences.\n\n"
         "================================ ANALYTICAL STANDARDS\n"
         "The report must:\n"
         "- Synthesise information across sources\n"
         "- Compare information where possible\n"
         "- Highlight contradictions if present\n"
         "- Prioritise insight over description\n"
         "- Clearly distinguish facts vs interpretation\n"
         "- Cite sources inline using their titles\n\n"
         "================================ WRITING STYLE\n"
         "Tone: concise, analytical, objective, professional, confident.\n"
         "Avoid: fluff, generic phrases, marketing language, repetition.\n"
         "Each sentence must add value.\n\n"
         "================================ QUALITY CONTROL\n"
         "Before output, verify:\n"
         "- Under {max_words} words\n"
         "- Only source-supported claims\n"
         "- Logical structure\n"
         "- Executive readability\n"
         "- No repetition\n"
         "Do NOT include a word count line at the end.\n"
         "Output ONLY the final report."),
        ("human",
         "Industry: **{industry}**\n\nSources:\n{sources}"),
    ])

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "industry": industry,
        "sources": source_material,
        "max_words": MAX_REPORT_WORDS,
        "source_titles": titles_str,
    })

    # Remove any "Word count:" line the LLM might add
    lines = report.strip().split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith("word count")
        and not line.strip().lower().startswith("*word count")
    ]
    report = "\n".join(cleaned_lines).strip()

    # Programmatic word limit enforcement
    report = enforce_word_limit(report, HARD_WORD_LIMIT)

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
        "search_queries": [],
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
    st.session_state.search_queries = []


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
        st.markdown("### Configuration")

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
            "Get a free API key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)"
        )

        # Display pipeline progress
        st.divider()
        st.markdown("### Pipeline Status")
        step = st.session_state.get("current_step", 1)

        # Progress bar
        progress_map = {1: 0.0, 2: 0.33, 3: 0.66, 4: 1.0}
        st.progress(progress_map.get(step, 0.0))

        steps_info = [
            ("Industry Input", step >= 2),
            ("Source Retrieval", step >= 3),
            ("Report Generation", step >= 4),
        ]
        for i, (label, done) in enumerate(steps_info, 1):
            if done:
                st.markdown(f"~~Step {i}: {label}~~ :green[Done]")
            elif step == i:
                st.markdown(f"**Step {i}: {label}** :orange[In progress]")
            else:
                st.markdown(f"Step {i}: {label}")

    return selected_model, api_key


def render_step_1(llm: ChatGoogleGenerativeAI):
    """Step 1: Industry input and validation."""
    st.header("Step 1: Enter an Industry")
    st.markdown(
        "Provide the name of an industry or economic sector to research. "
        "The assistant will validate your input before proceeding."
    )

    # Text input
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
                handle_api_error(e, "Validation")
                return

        if result["is_valid"]:
            normalised = result["normalised"] or industry
            st.session_state.validated_industry = normalised
            st.session_state.current_step = 2
            st.success(
                f"Recognised industry: **{normalised}**"
            )
            if result["reason"]:
                st.caption(result["reason"])
            st.rerun()
        else:
            st.warning(
                "That doesn't appear to be a recognised industry. "
                "Please try again with a specific industry name "
                "(e.g. 'Renewable Energy' rather than 'energy')."
            )
            if result["reason"]:
                st.caption(result["reason"])

    elif not industry:
        st.info("Enter an industry name above to get started.")


def render_step_2(llm: ChatGoogleGenerativeAI):
    """Step 2: Retrieve and display 5 most relevant Wikipedia sources."""
    industry = st.session_state.validated_industry

    st.header("Step 2: Relevant Wikipedia Sources")

    if not st.session_state.wiki_pages:
        with st.spinner("Generating search queries and retrieving sources..."):
            try:
                # Stage A: Generate multiple search queries
                queries = generate_search_queries(llm, industry)
                st.session_state.search_queries = queries

                # Stage B: Broad retrieval across all queries
                raw_pages = retrieve_wikipedia_pages(industry, queries)

                if not raw_pages:
                    st.error(
                        "No Wikipedia pages found for this industry. "
                        "Try a broader or more common industry name."
                    )
                    return

                # Stage C: LLM-based relevance filtering
                top_pages = select_top_pages(llm, industry, raw_pages)
                st.session_state.wiki_pages = top_pages

            except Exception as e:
                handle_api_error(e, "Retrieval")
                return

    # Show search strategy
    if st.session_state.search_queries:
        with st.expander("Search strategy", expanded=False):
            st.markdown("Queries used to find sources:")
            for q in st.session_state.search_queries:
                st.markdown(f"- *{q}*")
            st.caption(
                f"Retrieved {len(st.session_state.wiki_pages)} most relevant "
                f"pages from initial candidate pool."
            )

    # Display the 5 selected sources
    pages = st.session_state.wiki_pages
    st.markdown(f"**Top {len(pages)} sources selected by relevance:**")

    for i, page in enumerate(pages, 1):
        st.markdown(f"{i}. [{page['title']}]({page['url']})")

    st.divider()

    if st.button("Generate Industry Report", type="primary"):
        st.session_state.current_step = 3
        st.rerun()


def render_step_3(llm: ChatGoogleGenerativeAI):
    """Step 3: Generate and display the industry report."""
    industry = st.session_state.validated_industry
    pages = st.session_state.wiki_pages

    st.header("Step 3: Industry Report")

    if not st.session_state.report:
        with st.spinner("Generating your industry report..."):
            try:
                report = generate_report(llm, industry, pages)
                st.session_state.report = report
                st.session_state.current_step = 4
            except Exception as e:
                handle_api_error(e, "Report generation")
                return

    # Display the report in a clean container
    st.markdown(st.session_state.report)

    # Word count badge
    wc = count_words(st.session_state.report)
    if wc <= HARD_WORD_LIMIT:
        st.caption(f"Word count: {wc} / {HARD_WORD_LIMIT}")
    else:
        st.caption(f":red[Word count: {wc} / {HARD_WORD_LIMIT} — over limit]")

    # Sources section
    st.divider()
    st.subheader("Sources")
    for i, page in enumerate(st.session_state.wiki_pages, 1):
        st.markdown(f"{i}. [{page['title']}]({page['url']})")

    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Research a different industry"):
            reset_pipeline()
            st.rerun()
    with col2:
        # Download report as text file
        report_text = (
            f"INDUSTRY REPORT: {industry}\n"
            f"{'=' * 50}\n\n"
            f"{st.session_state.report}\n\n"
            f"{'=' * 50}\n"
            f"SOURCES:\n"
        )
        for i, page in enumerate(st.session_state.wiki_pages, 1):
            report_text += f"{i}. {page['title']} - {page['url']}\n"

        st.download_button(
            label="Download report",
            data=report_text,
            file_name=f"{industry.lower().replace(' ', '_')}_report.txt",
            mime="text/plain",
        )


# ──────────────────────────────────────────────────────────────
# MAIN — application entry point
# ──────────────────────────────────────────────────────────────

def main():
    """Main application entry point.

    Orchestrates the three-step pipeline:
        1. Industry validation
        2. Wikipedia retrieval (multi-query + LLM ranking)
        3. Report generation with inline citations

    Each step only renders when its prerequisites are met,
    creating a clear, guided user experience.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )

    init_session_state()

    # Header
    st.title(APP_TITLE)
    st.caption(
        "AI-powered industry analysis from Wikipedia sources  |  "
        "Built with LangChain + Gemini"
    )

    # Sidebar
    selected_model, api_key = render_sidebar()

    # Gate: require API key
    if not api_key:
        st.info(
            "Enter your Google AI API key in the sidebar to begin.  \n"
            "Get a free key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)."
        )
        return

    # Initialise LLM
    llm = initialise_llm(selected_model, api_key)

    # Render pipeline steps
    step = st.session_state.current_step

    if step == 1:
        render_step_1(llm)
    elif step == 2:
        render_step_1(llm)
        st.divider()
        render_step_2(llm)
    elif step >= 3:
        render_step_1(llm)
        st.divider()
        render_step_2(llm)
        st.divider()
        render_step_3(llm)


if __name__ == "__main__":
    main()
