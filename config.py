"""Shared settings and factories used by ingest.py and rag.py."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # provides GOOGLE_API_KEY

PDF_PATH = ROOT / "data" / "GER-3620-P.pdf"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "ger3620"

# Checked Oct 2026: gemini-2.5-flash is closed to new keys; gemini-3.7/3.8-flash timed out (504).
# The free tier allows ~20 chat requests/day per model, so fallbacks are tried in order on errors.
# gemini-embedding-001 returns one vector per text, which batching needs.
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
FALLBACK_CHAT_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")


def get_vectorstore() -> Chroma:
    """Persistent Chroma collection using cosine similarity (relevance score = cosine)."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
