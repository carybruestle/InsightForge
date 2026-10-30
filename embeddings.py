from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """Extract text content from a single PDF file, one entry per page."""
    pdf_module = importlib.import_module("pypdf")
    pdf_reader_cls = getattr(pdf_module, "PdfReader")
    reader = pdf_reader_cls(str(pdf_path))
    page_text: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        page_text.append(text)

    return page_text


def split_page_texts(page_texts: list[str]) -> list[Any]:
    """Split extracted page texts into overlapping LangChain documents."""
    text_splitter_module = importlib.import_module("langchain_text_splitters")
    recursive_splitter_cls = getattr(
        text_splitter_module,
        "RecursiveCharacterTextSplitter",
    )
    splitter = recursive_splitter_cls(chunk_size=1024, chunk_overlap=64)
    return splitter.create_documents(page_texts)


def get_openai_embeddings() -> Any:
    """Create an embeddings client from the current environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to the project .env file.")

    openai_api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    embeddings_module = importlib.import_module("langchain_openai")
    embeddings_cls = getattr(embeddings_module, "OpenAIEmbeddings")

    embeddings_kwargs: dict[str, Any] = {"api_key": api_key}
    if openai_api_base:
        embeddings_kwargs["base_url"] = openai_api_base

    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if model:
        embeddings_kwargs["model"] = model

    return embeddings_cls(**embeddings_kwargs)


def create_vector_store(documents: list[Any]) -> Any:
    """Embed chunked documents and load them into a FAISS vector store."""
    vectorstores_module = importlib.import_module("langchain_community.vectorstores")
    faiss_cls = getattr(vectorstores_module, "FAISS")
    embeddings = get_openai_embeddings()
    return faiss_cls.from_documents(documents, embeddings)


def save_vector_store(store: Any, index_dir: Path, name: str) -> None:
    """Persist a FAISS vector store to disk under index_dir/<name>/."""
    store.save_local(str(index_dir / name))
    print(f"Saved FAISS index: {index_dir / name}")


def load_vector_store(index_dir: Path, name: str) -> Any:
    """Load a previously saved FAISS index from disk."""
    vectorstores_module = importlib.import_module("langchain_community.vectorstores")
    faiss_cls = getattr(vectorstores_module, "FAISS")
    return faiss_cls.load_local(
        str(index_dir / name),
        get_openai_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve(store: Any, query: str, k: int = 5) -> list[Any]:
    """Return the top-k most relevant documents for a query."""
    return store.similarity_search(query, k=k)


def convert_pdfs_to_vectorstores(data_dir: Path, index_dir: Path) -> dict[str, Any]:
    """Extract, split, embed, index, and persist every PDF in data_dir."""

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)

    vectorstores: dict[str, Any] = {}
    for pdf_file in pdf_files:
        try:
            page_texts = extract_text_from_pdf(pdf_file)
            documents = split_page_texts(page_texts)
            store = create_vector_store(documents)
            save_vector_store(store, index_dir, pdf_file.stem)
            vectorstores[pdf_file.name] = store
            print(f"Loaded {len(documents)} chunks into FAISS for: {pdf_file.name}")
        except Exception as exc:
            print(f"Failed to convert {pdf_file.name}: {exc}")

    return vectorstores


def update_vector_stores(
    data_dir: Path,
    index_dir: Path,
) -> dict[str, str]:
    """Sync vector stores with the current PDFs in data_dir.

    Returns a summary dict mapping each PDF stem to one of:
      'added'   - new PDF found, index created
      'removed' - PDF gone, index deleted
      'ok'      - already indexed, no change
    """
    import shutil

    index_dir.mkdir(parents=True, exist_ok=True)

    current_pdfs: set[str] = {p.stem for p in data_dir.glob("*.pdf")}
    indexed: set[str] = {p.name for p in index_dir.iterdir() if p.is_dir()}

    summary: dict[str, str] = {}

    # Remove indexes for PDFs that no longer exist.
    for removed in indexed - current_pdfs:
        shutil.rmtree(index_dir / removed)
        print(f"Removed stale index: {removed}")
        summary[removed] = "removed"

    # Build indexes for new PDFs.
    for added in sorted(current_pdfs - indexed):
        pdf_path = data_dir / f"{added}.pdf"
        try:
            page_texts = extract_text_from_pdf(pdf_path)
            documents = split_page_texts(page_texts)
            store = create_vector_store(documents)
            save_vector_store(store, index_dir, added)
            print(f"Added index for new PDF: {added}")
            summary[added] = "added"
        except Exception as exc:
            print(f"Failed to index {added}: {exc}")

    # Report unchanged PDFs.
    for unchanged in sorted(current_pdfs & indexed):
        summary[unchanged] = "ok"

    return summary


def main() -> None:
    data_dir = PROJECT_ROOT / "Data"
    index_dir = PROJECT_ROOT / "indexes"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    vectorstores = convert_pdfs_to_vectorstores(data_dir, index_dir)
    print(f"Done. Built and saved vector stores for {len(vectorstores)} PDF file(s).")


if __name__ == "__main__":
    main()
