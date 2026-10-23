"""Question answering over the GER-3620 vector store: retrieve top-k, then a strict Gemini prompt."""
import re
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config import CHAT_MODEL, FALLBACK_CHAT_MODELS, get_vectorstore

K = 5
# Cosine similarity of the best chunk below which the question is treated as off-topic and the LLM
# is not called. Calibrated on the test questions: in-document 0.71-0.81, off-topic 0.53-0.60.
SCORE_THRESHOLD = 0.62
NOT_FOUND = "Not found"

PROMPT = """You answer questions about ONE document: GE Vernova GER-3620 Rev P,
"Heavy-Duty Gas Turbine Operating and Maintenance Considerations".

Rules:
1. Use ONLY the numbered excerpts below. Never use outside knowledge, even if you are sure of the answer.
2. If the excerpts do not contain the answer, reply with exactly: Not found
3. Otherwise answer briefly (at most 4 short sentences or "-" bullets, plain text, no markdown) using
   the document's own terms and numbers. If the value depends on model or configuration, say so and
   give the main values. If the answer comes from a figure or table, name it (e.g. "Figure 36").
   Report numbers exactly as written, next to the model/condition they belong to; do not group them
   into classes the document does not state. If there are many values, give the overall range plus
   2-3 examples. Tables were extracted as plain text: if a row has fewer values than there are column
   headers, do not guess which column a value belongs to.
4. After the answer, add these two lines:
   EVIDENCE: <the shortest passage (max ~2 lines) copied EXACTLY, word for word, from an excerpt, that directly states the key fact in your answer; if the answer gives numbers, quote the line containing them (table rows may be quoted as-is)>
   SOURCES: <numbers of the excerpts you used, e.g. 1, 3>

Excerpts:
{context}

Question: {question}
"""


@lru_cache
def _db():
    return get_vectorstore()


@lru_cache
def _llm():
    """Main chat model; on quota/overload errors the fallback models are tried in order."""
    def make(model):
        return ChatGoogleGenerativeAI(model=model, temperature=0, timeout=90, max_retries=1)
    return make(CHAT_MODEL).with_fallbacks([make(m) for m in FALLBACK_CHAT_MODELS if m != CHAT_MODEL])


def _ref(doc) -> str:
    m = doc.metadata
    return f'Page {m["page"]} (printed p. {m["page_label"]}), Section "{m["section"]}"'


def _squash(text: str) -> str:
    """Normalise whitespace, case, dashes and line-break hyphens for verbatim comparison."""
    text = re.sub(r"-\s+", "-", text.replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", text).strip().lower()


def _evidence(quote: str, answer: str, docs) -> tuple[str, int]:
    """Return (snippet, page) where the snippet is guaranteed to exist in a retrieved chunk.

    The LLM's quote is used if it really appears in one of the chunks; otherwise fall back to the
    chunk line that shares the most words with the answer.
    """
    quote = quote.strip().strip('"“”').strip()
    for d in docs:
        if len(quote) > 15 and _squash(quote) in _squash(d.page_content):
            return quote, d.metadata["page"]
    words = set(re.findall(r"\w{4,}", answer.lower()))
    lines = [(l, d) for d in docs for l in d.page_content.splitlines()[1:] if len(l) > 20]  # skip header
    line, doc = max(lines, key=lambda x: len(words & set(re.findall(r"\w{4,}", x[0].lower()))))
    return line.strip(), doc.metadata["page"]


def index_size() -> int:
    """Number of chunks stored in the vector DB."""
    return _db()._collection.count()


def ask(question: str) -> dict:
    """Answer from the PDF only.

    Returns {"answer", "reference", "sources", "evidence", "evidence_page", "top_score"}; answer is
    exactly "Not found" (with empty reference/evidence) when the document does not support it.
    """
    hits = _db().similarity_search_with_relevance_scores(question, k=K)
    top_score = hits[0][1] if hits else 0.0
    not_found = {"answer": NOT_FOUND, "reference": "", "sources": [], "evidence": "", "evidence_page": None,
                 "top_score": top_score}
    if top_score < SCORE_THRESHOLD:  # clearly irrelevant: skip the LLM
        return not_found

    context = "\n\n".join(f"[{i}] {_ref(doc)}\n{doc.page_content}" for i, (doc, _) in enumerate(hits, 1))
    reply = _llm().invoke(PROMPT.format(context=context, question=question)).text.strip()

    # Split "answer / EVIDENCE: ... / SOURCES: ..." apart.
    ev = re.search(r"^\s*EVIDENCE:\s*(.*)$", reply, flags=re.I | re.M)
    src = re.search(r"^\s*SOURCES:\s*(.*)$", reply, flags=re.I | re.M)
    cut = min([m.start() for m in (ev, src) if m] or [len(reply)])
    answer = reply[:cut].strip()
    if not answer or answer.lower().startswith("not found"):
        return not_found

    used = [int(n) for n in re.findall(r"\d+", src[1])] if src else []
    used = [n for n in used if 1 <= n <= len(hits)] or [1]  # fall back to the best chunk
    docs = [hits[n - 1][0] for n in used]
    sources = list(dict.fromkeys((d.metadata["page"], d.metadata["page_label"], d.metadata["section"])
                                 for d in docs))  # dedupe, keep order
    evidence, evidence_page = _evidence(ev[1] if ev else "", answer, docs)
    return {
        "answer": answer,
        "reference": "; ".join(f'Page {p} (printed p. {lbl}), Section "{s}"' for p, lbl, s in sources),
        "sources": [{"page": p, "page_label": lbl, "section": s} for p, lbl, s in sources],
        "evidence": evidence,
        "evidence_page": evidence_page,
        "top_score": top_score,
    }
