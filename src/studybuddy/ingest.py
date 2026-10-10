"""Load source documents, split them by page, and create chunks."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".md", ".txt"}


def load_documents(paths: list[str | Path]) -> list[Document]:
    """Load files into one Document per PDF page (or per text file)."""
    docs: list[Document] = []
    for p in map(Path, paths):
        suffix = p.suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {p.name}")

        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
        else:
            loader = TextLoader(str(p), encoding="utf-8")

        for d in loader.load():
            if not d.page_content.strip():
                continue  # blank or scanned page: nothing to index
            d.metadata["source"] = p.name
            # PyPDFLoader pages start at 0; humans cite from 1
            d.metadata["page"] = int(d.metadata.get("page", 0)) + 1 if suffix == ".pdf" else None
            docs.append(d)
    return docs


def chunk_documents(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""],  # headings first
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def ingest(paths: list[str | Path]) -> list[Document]:
    """Load and chunk files. Raises ValueError if nothing readable was found."""
    chunks = chunk_documents(load_documents(paths))
    if not chunks:
        raise ValueError(
            "No readable text found. Scanned PDFs need OCR, which StudyBuddy doesn't support yet."
        )
    return chunks