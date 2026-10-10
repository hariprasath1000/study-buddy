"""Streamlit user interface for StudyBuddy."""

import hashlib
import tempfile
from pathlib import Path

import streamlit as st
import uuid
from studybuddy.vectorstore import add_chunks, get_embeddings, save_vectorstore
import pandas as pd

from studybuddy import progress
from studybuddy.config import settings
from studybuddy.ingest import ingest
from studybuddy.qa import NO_ANSWER as NO_ANSWER_TEXT
from studybuddy.qa import answer_question
from studybuddy.quiz import build_quiz
from studybuddy.vectorstore import add_chunks, save_vectorstore

ALLOWED_TYPES = ["txt", "md", "pdf"]


# ---------- helpers ----------


def get_session_dir() -> Path:
    """One private temp folder per browser session (never shared between users)."""
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = tempfile.mkdtemp(prefix="studybuddy_")
    return Path(st.session_state.upload_dir)


def save_uploads(uploaded_files) -> list[Path]:
    """Save uploaded files into the session folder and return their paths."""
    upload_dir = get_session_dir()
    saved: list[Path] = []
    for f in uploaded_files:
        target = upload_dir / Path(f.name).name  # .name strips any path components
        target.write_bytes(f.getbuffer())
        saved.append(target)
    return saved


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_documents(paths: list[Path], on_progress=None) -> int:
    """Rebuild the session index from the current uploads.

    Returns the chunk count, or 0 if nothing changed since the last build.
    Embeddings come from the cache, so unchanged text is never re-embedded.
    """
    signature = frozenset((p.name, file_hash(p)) for p in paths)
    if signature == st.session_state.get("indexed_signature"):
        return 0

    chunks = ingest(paths)
    vs = add_chunks(None, chunks, on_progress=on_progress)  # fresh index, no stale chunks
    save_vectorstore(vs, get_session_dir() / "index")

    st.session_state.vectorstore = vs
    st.session_state.indexed_signature = signature
    return len(chunks)


@st.cache_resource(show_spinner="Loading embedding model (the first run downloads it)...")
def warm_up() -> bool:
    from studybuddy.vectorstore import get_embeddings

    get_embeddings().embed_query("warm up")
    return True


# ---------- tabs ----------


def render_upload_tab() -> None:
    st.subheader("Your study material")
    uploaded_files = st.file_uploader(
        "Choose files (TXT, Markdown, PDF)",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Upload at least one file to get started.")
        return

    paths = save_uploads(uploaded_files)
    st.write(f"{len(paths)} file(s) ready: " + ", ".join(p.name for p in paths))

    if st.button("Build index from uploaded files"):
        emb = get_embeddings()
        emb.reset_stats()
        bar = st.progress(0.0, text="Reading documents...")

        def on_progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Processing {done}/{total} chunks")

        try:
            n_chunks = index_documents(paths, on_progress=on_progress)
        except Exception as exc:
            bar.empty()
            st.error(f"Indexing failed: {exc}")
        else:
            bar.empty()
            if n_chunks == 0:
                st.info("Nothing changed since the last index, so nothing was re-processed.")
            else:
                st.success(
                    f"Indexed {n_chunks} chunks: {emb.hits} reused from the cache, "
                    f"{emb.misses} newly embedded."
                )


def render_ask_tab() -> None:
    vs = st.session_state.get("vectorstore")
    if vs is None:
        st.info("Upload and index your notes first.")
        return

    question = st.text_input("Ask a question about your notes")
    if st.button("Ask") and question.strip():
        with st.spinner("Searching your notes..."):
            try:
                result = answer_question(vs, question.strip())
            except Exception as exc:
                st.error(f"Couldn't get an answer: {exc}")
                return

        st.write(result.text)
        if result.sources:
            with st.expander(f"Sources ({len(result.sources)})"):
                for s in result.sources:
                    page = f", page {s.page}" if s.page else ""
                    st.markdown(f"**[{s.id}] {s.file}{page}**")
                    st.caption(s.excerpt + "...")
        elif result.text != NO_ANSWER_TEXT:
            st.warning("This answer has no verified citation. Treat it with caution.")


def render_quiz_tab() -> None:
    vs = st.session_state.get("vectorstore")
    if vs is None:
        st.info("Upload and index your notes first.")
        return

    topic = st.text_input("Topic (optional, leave empty for the whole material)")
    n = st.slider("Number of questions", 1, 10, 5)

    if st.button("Generate quiz"):
        with st.spinner("Writing and checking questions..."):
            try:
                st.session_state.quiz = build_quiz(vs, n, topic.strip() or None)
                st.session_state.quiz_id = st.session_state.get("quiz_id", 0) + 1
                st.session_state.quiz_uid = uuid.uuid4().hex  # <- new unique ID for this quiz
            except Exception as exc:
                st.error(f"Couldn't generate the quiz: {exc}")
                return

    batch = st.session_state.get("quiz")
    if batch is None:
        return
    if not batch.items:
        st.warning("No valid questions could be generated from this material. Try a different topic.")
        return

    st.caption(f"{len(batch.items)} questions. {batch.validity_rate:.0%} of attempts passed verification.")
    qid = st.session_state.quiz_id
    for i, item in enumerate(batch.items):
        st.markdown(f"**Q{i + 1}. {item.mcq.question}**")
        st.radio("Answer", item.mcq.options, index=None, key=f"q{qid}_{i}", label_visibility="collapsed")

    if st.button("Check answers"):
        score = 0
        rows = []
        for i, item in enumerate(batch.items):
            chosen = st.session_state.get(f"q{qid}_{i}")
            correct = item.mcq.options[item.mcq.correct_index]
            page = f", page {item.page}" if item.page else ""
            if chosen == correct:
                score += 1
                st.success(f"Q{i + 1}: correct")
            else:
                st.error(f"Q{i + 1}: the answer is **{correct}**")
            st.caption(f"{item.mcq.explanation} (Source: {item.file}{page})")
            if chosen is not None:  # skipped questions aren't evidence either way
                rows.append((item.topic, item.mcq.question, chosen == correct))
        st.subheader(f"Score: {score}/{len(batch.items)}")

        uid = st.session_state.get("quiz_uid")
        recorded = st.session_state.setdefault("recorded_quizzes", set())
        if rows and uid not in recorded:  # save each quiz only once
            progress.record_attempts(progress_db(), uid, rows)
            recorded.add(uid)
            st.caption("Results saved to your Progress tab.")


def render_progress_tab() -> None:
    db = progress_db()
    stats = progress.topic_stats(db)
    if not stats:
        st.info("Take a quiz and click **Check answers**. Your results will appear here.")
        return

    history = progress.quiz_history(db)
    total = sum(s.attempts for s in stats)
    correct = sum(s.correct for s in stats)

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions answered", total)
    c2.metric("Overall accuracy", f"{correct / total:.0%}")
    c3.metric("Quizzes taken", len(history))

    st.subheader("Needs revision")
    weak = progress.weak_topics(db)
    if weak:
        for s in weak:
            st.markdown(f"- **{s.topic}**: {s.correct}/{s.attempts} correct ({s.accuracy:.0%})")
        st.caption("Re-read these passages, then take another quiz.")
    else:
        st.success("No weak topics yet. A topic needs at least 2 answers to be counted.")

    st.subheader("Accuracy by topic")
    st.dataframe(
        pd.DataFrame(
            {
                "Topic": [s.topic for s in stats],
                "Answered": [s.attempts for s in stats],
                "Accuracy": [round(100 * s.accuracy) for s in stats],
            }
        ),
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", min_value=0, max_value=100, format="%d%%"
            )
        },
        hide_index=True,
        use_container_width=True,
    )

    if len(history) > 1:
        st.subheader("Score per quiz (%)")
        st.line_chart(pd.DataFrame({"Score": [q.pct for q in history]}))

    with st.expander("Reset progress"):
        if st.checkbox("I understand this deletes my history") and st.button("Delete all progress"):
            progress.reset(db)
            st.session_state.recorded_quizzes = set()
            st.rerun()
def progress_db() -> str:
    return settings.progress_db or str(get_session_dir() / "progress.db")


# ---------- app ----------


def main() -> None:
    st.set_page_config(page_title="StudyBuddy", page_icon="📚")
    st.title("StudyBuddy")
    st.caption("Ask questions, get quizzed, and see what to revise, all from your own notes.")
    warm_up()  # load the embedding model once, at page load

    tab_upload, tab_ask, tab_quiz, tab_progress = st.tabs(
        ["Upload", "Ask", "Quiz", "Progress"]
    )
    with tab_upload:
        render_upload_tab()
    with tab_ask:
        render_ask_tab()
    with tab_quiz:
        render_quiz_tab()
    with tab_progress:
        render_progress_tab()