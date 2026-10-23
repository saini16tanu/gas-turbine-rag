"""Build the vector store: PDF -> clean pages -> section-aware chunks -> Gemini embeddings -> Chroma.

Section headings are detected heuristically: a short, title-like body line (or 2-3 wrapped lines)
counts as a heading only if it matches an entry from the PDF's own table of contents. Headings must
appear in TOC order, which stops figure labels (e.g. "Combustion Inspection" in Figure 29) from being
mistaken for headings.
"""
import re
import shutil
import time
import warnings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, PDF_PATH, get_vectorstore

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader  # noqa: E402

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
BATCH_SIZE = 20       # chunks per embedding request
MAX_RETRIES = 6
HEADING_WINDOW = 4    # a new heading may be at most this many TOC entries ahead of the current one

TOC_LINE = re.compile(r"^(?P<title>\S.*?)\s+(?P<dots>(?:\.\s*){5,})\d+$")       # "Fuel . . . . . . 6"
CAPTION = re.compile(r"^Figure\s+((?:[A-Z]-)?\d+)\.\s+\S")            # "Figure 36. Baseline ..."
NOISE = re.compile(  # running header/footer and bare page numbers
    r"^(Doc Name ?:|Non-Public-|GE Vernova Power \||"
    r"Heavy-Duty Gas Turbine Operating and Maintenance Considerations$|(\d+|[ivx]+)$)"
)


def norm(text: str) -> str:
    """Normalise for heading comparison."""
    return re.sub(r"\s+", " ", text.replace("–", "-").replace("—", "-")).strip().lower()


def clean_line(line: str) -> str:
    line = re.sub(r"/uni([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), line)  # figure glyphs
    line = re.sub(r"\s+", " ", line).strip()
    return re.sub(r" ([.,;:)])", r"\1", line)  # "interval ." -> "interval.", "7E .03" -> "7E.03"


def parse_toc(pages):
    """Return TOC entries as [(display_section_name, normalised_title)] and the TOC page indexes."""
    raw, toc_pages = [], set()
    for i, page in enumerate(pages):
        lines = [l.strip() for l in page.page_content.splitlines()]
        found = []
        for j, line in enumerate(lines):
            m = TOC_LINE.match(line)
            if m:
                title = m["title"].strip()
                if title.startswith("(") and j:  # entry wrapped onto two lines
                    title = f"{lines[j - 1]} {title}"
                found.append((re.sub(r"\s+", " ", title), len(m["title"]) + m["dots"].count(".")))
        if len(found) >= 5:
            toc_pages.add(i)
            raw += found
    # Top-level TOC entries use a larger font, so fewer characters (title + leader dots) fit per line.
    cut = (min(n for _, n in raw) + max(n for _, n in raw)) / 2
    entries, parent = [], None
    for title, length in raw:
        if length < cut:
            parent, name = title, title
        else:
            name = f"{parent} > {title}" if parent else title
        entries.append((name, norm(title)))
    return entries, toc_pages


def match_heading(lines, k, titles, ptr):
    """If lines[k:k+n] (n = 1..3) is the next expected TOC heading, return (toc_index, n)."""
    for n in (1, 2, 3):
        cand = norm(" ".join(lines[k:k + n]))
        if len(cand) > 100:
            break
        # one entry back is allowed: two-column pages are sometimes extracted right column first
        for j in range(max(ptr - 1, 0), min(ptr + HEADING_WINDOW + 1, len(titles))):
            if titles[j] == cand:
                return j, n
    return None


def build_chunks():
    pages = PyPDFLoader(str(PDF_PATH)).load()
    toc, toc_pages = parse_toc(pages)
    titles = [t for _, t in toc]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks, ptr, section = [], 0, "Cover"

    for idx, page in enumerate(pages):
        lines = [clean_line(l) for l in page.page_content.splitlines()]
        lines = [l for l in lines if l and not NOISE.match(l)]
        captions = [l for l in lines if CAPTION.match(l)]
        if idx in toc_pages or len(captions) > 4:  # skip Contents / List of Figures (titles only)
            continue
        figures = ", ".join(f"Figure {CAPTION.match(c)[1]}" for c in captions)
        meta = {"source": PDF_PATH.name, "page": idx + 1, "page_label": page.metadata.get("page_label", ""),
                "figures": figures}

        def emit(buf, sec):
            text = "\n".join(buf).strip()
            if len(text) < 30:  # nothing but a heading line
                return
            header = "\n".join([f"Section: {sec}", *captions])  # gives each chunk its context
            for piece in splitter.split_text(text):
                chunks.append(Document(page_content=f"{header}\n{piece}", metadata={**meta, "section": sec}))

        # Walk the page; start a new segment at every detected heading so chunks never span sections.
        buf, k = [], 0
        while k < len(lines):
            hit = match_heading(lines, k, titles, ptr)
            if hit:
                emit(buf, section)
                ptr, n = hit
                section = toc[ptr][0]
                buf, k = [" ".join(lines[k:k + n])], k + n
            else:
                buf.append(lines[k])
                k += 1
        emit(buf, section)
    return chunks


def is_retryable(err: Exception) -> bool:
    msg = str(err)
    return any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"))


def main():
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from {PDF_PATH.name}")
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # always rebuild from scratch
    db = get_vectorstore()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [f"chunk-{start + i:04d}" for i in range(len(batch))]
        for attempt in range(MAX_RETRIES):
            try:
                db.add_documents(batch, ids=ids)
                break
            except Exception as err:  # back off on rate limits / transient errors only
                if attempt == MAX_RETRIES - 1 or not is_retryable(err):
                    raise
                wait = 5 * 2 ** attempt
                print(f"  rate limited on batch {start // BATCH_SIZE + 1}, retrying in {wait}s")
                time.sleep(wait)
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    print(f"Done. Vector store saved to {CHROMA_DIR.name}/")


if __name__ == "__main__":
    main()
