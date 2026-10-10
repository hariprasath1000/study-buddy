"""SQLite-backed learner progress tracking."""
"""Quiz history and weak-topic tracking (SQLite)."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL DEFAULT (datetime('now')),
    quiz_id  TEXT NOT NULL,
    topic    TEXT NOT NULL,
    question TEXT NOT NULL,
    correct  INTEGER NOT NULL CHECK (correct IN (0, 1))
)
"""


@dataclass
class TopicStat:
    topic: str
    attempts: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0


@dataclass
class QuizResult:
    quiz_id: str
    ts: str
    total: int
    correct: int

    @property
    def pct(self) -> float:
        return 100 * self.correct / self.total if self.total else 0.0


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def record_attempts(db_path, quiz_id: str, rows: list[tuple[str, str, bool]]) -> None:
    """rows: (topic, question, was_correct). Values are bound as parameters, never formatted into SQL."""
    with closing(_connect(db_path)) as conn, conn:
        conn.executemany(
            "INSERT INTO attempts (quiz_id, topic, question, correct) VALUES (?, ?, ?, ?)",
            [(quiz_id, topic, question, int(ok)) for topic, question, ok in rows],
        )


def topic_stats(db_path) -> list[TopicStat]:
    """All topics, weakest first (ties: more attempts first)."""
    with closing(_connect(db_path)) as conn:
        data = conn.execute(
            "SELECT topic, COUNT(*), SUM(correct) FROM attempts GROUP BY topic"
        ).fetchall()
    stats = [TopicStat(topic, n, c) for topic, n, c in data]
    return sorted(stats, key=lambda s: (s.accuracy, -s.attempts, s.topic))


def weak_topics(db_path, min_attempts: int = 2, max_accuracy: float = 0.6) -> list[TopicStat]:
    return [
        s for s in topic_stats(db_path)
        if s.attempts >= min_attempts and s.accuracy < max_accuracy
    ]


def quiz_history(db_path) -> list[QuizResult]:
    """One row per quiz, oldest first."""
    with closing(_connect(db_path)) as conn:
        data = conn.execute(
            "SELECT quiz_id, MIN(ts), COUNT(*), SUM(correct) "
            "FROM attempts GROUP BY quiz_id ORDER BY MIN(id)"
        ).fetchall()
    return [QuizResult(q, ts, n, c) for q, ts, n, c in data]


def reset(db_path) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM attempts")