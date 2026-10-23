"""Streamlit UI for the GER-3620 RAG assistant.  Run:  streamlit run ui.py

UI only: all retrieval / answering logic lives in rag.py (this file just calls rag.ask()).
"""
import html
import time

import streamlit as st

from app import TEST_QUESTIONS
from rag import NOT_FOUND, ask, index_size

st.set_page_config(page_title="Gas Turbine Maintenance Assistant", page_icon="⚙️", layout="centered")

# Translucent colours + inherited text colour keep every element readable in light and dark themes.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --accent: #0d9488; --card-border: rgba(128,128,128,.22); --card-bg: rgba(128,128,128,.06); }
.stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 820px; }
.hero h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: -.02em; margin: 0 0 .25rem; padding: 0; }
.hero p { font-size: 1.05rem; opacity: .75; margin: 0 0 .8rem; }
.badge { display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .03em;
         padding: .22rem .7rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
         color: var(--accent); background: rgba(13,148,136,.12); border: 1px solid rgba(13,148,136,.35); }
.card { border: 1px solid var(--card-border); background: var(--card-bg); border-radius: 16px;
        padding: 1.3rem 1.5rem; margin: 1rem 0 .6rem; box-shadow: 0 6px 22px rgba(0,0,0,.08); }
.card .label { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
               opacity: .6; margin-bottom: .45rem; }
.card .answer { font-size: 1.12rem; line-height: 1.65; }
.pill { display: inline-block; font-size: .85rem; padding: .3rem .8rem; margin: .7rem .4rem 0 0;
        border-radius: 999px; border: 1px solid rgba(13,148,136,.4); background: rgba(13,148,136,.10); }
.pill b { color: var(--accent); }
.meta { font-size: .8rem; opacity: .6; margin-top: .8rem; }
.notfound { border-left: 5px solid #f59e0b; background: rgba(245,158,11,.10); }
.notfound .title { font-size: 1.15rem; font-weight: 700; margin-bottom: .3rem; }
.evidence { border-left: 3px solid var(--accent); padding: .4rem .9rem; margin: .2rem 0 .4rem;
            font-style: italic; line-height: 1.6; }
.evidence-page { font-size: .8rem; font-weight: 600; opacity: .7; }
.section-title { font-size: .95rem; font-weight: 600; margin: 1.4rem 0 .4rem; }
div[data-testid="stFormSubmitButton"] button { border-radius: 10px; font-weight: 600; }
/* never scroll sideways: long section names / table rows wrap instead */
html, body, .stApp { overflow-x: hidden; }
.pill { max-width: 100%; }
.pill, .answer, .evidence { overflow-wrap: anywhere; }

/* phones and small tablets */
@media (max-width: 768px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2.5rem; }
  .hero h1 { font-size: 1.55rem; }
  .hero p { font-size: .95rem; }
  .badge { font-size: .7rem; padding: .18rem .55rem; }
  .card { padding: 1rem 1.05rem; border-radius: 12px; margin: .8rem 0 .5rem; }
  .card .answer { font-size: 1rem; line-height: 1.6; }
  .pill { display: block; width: fit-content; font-size: .8rem; }
  /* sample-question chips stack full width; chip and history text wraps */
  [data-testid="stButtonGroup"] button { width: 100%; justify-content: flex-start; text-align: left; }
  [data-testid="stButtonGroup"] button p,
  button[data-testid="stBaseButton-tertiary"] p { white-space: normal; text-align: left; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def chunk_count() -> int:
    return index_size()


def run_question(question: str) -> None:
    """Ask the RAG pipeline and store the result (with timing) in the session history."""
    with st.spinner("Searching the manual..."):
        start = time.perf_counter()
        try:
            result = ask(question)
        except Exception as err:  # e.g. missing API key, network or quota error
            st.session_state.error = f"Could not answer right now: {err}"
            return
        seconds = time.perf_counter() - start
    st.session_state.error = None
    st.session_state.history.insert(0, {"question": question, "result": result, "seconds": seconds})
    st.session_state.current = 0


def pick_sample() -> None:
    """Sample chip clicked: queue the question and reset the chip so it can be clicked again."""
    st.session_state.pending = st.session_state.sample
    st.session_state.sample = None


def show_answer(entry: dict) -> None:
    r, esc = entry["result"], html.escape
    if r["answer"] == NOT_FOUND:
        st.markdown(
            f"""<div class="card notfound"><div class="label">Question</div><div>{esc(entry['question'])}</div>
            <div class="title" style="margin-top:.9rem">⚠️ Not found</div>
            <div>This question isn't answered in GE GER-3620 Rev P, so no answer is given.
            Try asking about inspections, maintenance factors, intervals or operating practices.</div>
            <div class="meta">⏱ {entry['seconds']:.1f} s</div></div>""",
            unsafe_allow_html=True,
        )
        return

    answer = esc(r["answer"]).replace("\n", "<br>")
    pills = "".join(
        f'<span class="pill"><b>📄 Page {s["page"]}</b> (printed p. {esc(s["page_label"])}) · {esc(s["section"])}</span>'
        for s in r["sources"]
    )
    st.markdown(
        f"""<div class="card"><div class="label">Question</div><div>{esc(entry['question'])}</div>
        <div class="label" style="margin-top:1rem">Answer</div><div class="answer">{answer}</div>
        <div class="label" style="margin-top:1rem">Reference</div>{pills}
        <div class="meta">⏱ {entry['seconds']:.1f} s · answered only from retrieved PDF excerpts</div></div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Evidence from document", expanded=True):
        st.markdown(
            f'<div class="evidence">“{esc(r["evidence"])}”</div>'
            f'<div class="evidence-page">Exact text from page {r["evidence_page"]} of GER-3620-P.pdf</div>',
            unsafe_allow_html=True,
        )


# ---------- state ----------
for key, default in {"history": [], "current": None, "pending": None, "error": None}.items():
    st.session_state.setdefault(key, default)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ About")
    st.write("Ask questions about GE gas turbine maintenance. Answers come **only** from the "
             "source manual and always cite the page, section and exact evidence.")
    st.markdown("### 📘 Source document")
    st.write("**GE GER-3620 Rev P**  \nHeavy-Duty Gas Turbine Operating and Maintenance "
             "Considerations · 60 pages")
    st.markdown("### 🔎 How it works")
    st.markdown(
        "1. **Chunk** the PDF by section, keeping page numbers\n"
        "2. **Embed** chunks with Gemini and store them in ChromaDB\n"
        "3. **Retrieve** the 5 most relevant chunks for your question\n"
        "4. **Answer** with Gemini using only those chunks, or say *Not found*"
    )
    try:
        st.metric("Chunks indexed", chunk_count())
    except Exception:
        st.warning("Index not found. Run `python ingest.py` first.")

# ---------- header ----------
st.markdown(
    """<div class="hero"><h1>Gas Turbine Maintenance Assistant</h1>
    <p>Grounded answers from the GE GER-3620 heavy-duty gas turbine O&amp;M manual, with page-level citations.</p>
    <span class="badge">RAG</span><span class="badge">ChromaDB</span>
    <span class="badge">LangChain</span><span class="badge">Gemini</span></div>""",
    unsafe_allow_html=True,
)

# ---------- question input ----------
with st.form("ask_form", clear_on_submit=False, border=False):
    question = st.text_input("Your question", placeholder="e.g. What does a maintenance factor of two mean?",
                             label_visibility="collapsed")
    submitted = st.form_submit_button("Ask", type="primary", width="stretch")

st.markdown('<div class="section-title">Try a sample question</div>', unsafe_allow_html=True)
st.pills("Sample questions", TEST_QUESTIONS, key="sample", on_change=pick_sample, label_visibility="collapsed")

if submitted and question.strip():
    run_question(question.strip())
elif st.session_state.pending:
    run_question(st.session_state.pending)
    st.session_state.pending = None

# ---------- answer ----------
if st.session_state.error:
    st.error(st.session_state.error)
elif st.session_state.current is not None:
    show_answer(st.session_state.history[st.session_state.current])

# ---------- recent questions (click to view again, no new API call) ----------
if st.session_state.history:
    st.markdown('<div class="section-title">Recent questions</div>', unsafe_allow_html=True)
    for i, entry in enumerate(st.session_state.history[:6]):
        icon = "⚠️" if entry["result"]["answer"] == NOT_FOUND else "✅"
        if st.button(f"{icon}  {entry['question']}", key=f"hist{i}", type="tertiary"):
            st.session_state.current = i
            st.rerun()
