"""Usage: uv run python scripts/tune_threshold.py notes.md [more files]
Edit the two lists below with questions about YOUR material."""

import sys

from studybuddy.ingest import ingest
from studybuddy.qa import retrieve
from studybuddy.vectorstore import build_vectorstore

IN_SCOPE = ["Which organelle produces energy?", "What is photosynthesis?"]
OUT_OF_SCOPE = ["What is the capital of Mongolia?", "Who won the 2010 World Cup?"]

vs = build_vectorstore(ingest(sys.argv[1:]))
for label, questions in (("IN scope", IN_SCOPE), ("OUT of scope", OUT_OF_SCOPE)):
    for q in questions:
        top = retrieve(vs, q, k=1, min_relevance=-1)  # -1 = keep everything
        print(f"{label:13} {top[0].metadata['relevance']:.3f}  {q}")