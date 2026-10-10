"""Usage: uv run python scripts/measure_quiz_validity.py notes.md [more files]"""

import sys

from studybuddy.ingest import ingest
from studybuddy.quiz import build_quiz
from studybuddy.vectorstore import build_vectorstore

vs = build_vectorstore(ingest(sys.argv[1:]))
batch = build_quiz(vs, n=10, seed=42)
print(f"attempted={batch.attempted} accepted={len(batch.items)} "
      f"rejected={batch.rejected} validity={batch.validity_rate:.0%}")
for item in batch.items:
    print("-", item.mcq.question)