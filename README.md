---
title: StudyBuddy
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# StudyBuddy 📚

**Upload your own notes or a textbook chapter, then ask questions, get quizzed, and see what to revise.**

StudyBuddy is a retrieval-augmented (RAG) study assistant. Every answer is grounded in *your* material and cites the exact file and page. Quiz questions are checked against their source passage before you ever see them, and your results feed a weak-topic tracker.

It is a learning and portfolio project, built end to end (ingestion, retrieval, LLM chains, evaluation, containerisation, CI/CD and deployment) using free-tier tools wherever possible.

---

## Features

- **Cited Q&A.** Answers carry numbered citations that map back to a file and page. Citations that don't match a retrieved passage are discarded, and uncited answers are flagged in the UI.
- **Honest "I don't know".** Two layers: a relevance gate before the LLM is called (saving quota), and a prompt rule that refuses when the passages don't contain the answer.
- **Verified quizzes.** Multiple-choice questions are generated as structured output, then a second LLM pass checks that the passage supports the marked answer and that no other option is also correct. The validity rate is shown.
- **Progress tracking.** Quiz results are stored in SQLite. The Progress tab shows overall accuracy, accuracy by topic, score history and topics that need revision.
- **Never embeds the same text twice.** A persistent embedding cache, keyed by a hash of the chunk text and model name, makes re-uploads and small edits fast.
- **Switchable LLM provider.** Gemini, Groq, OpenRouter or a local Ollama model, selected with one environment variable.
- **Privacy by default.** Each browser session gets its own private temp folder, and optional password protection guards a deployed instance.

---

## How it works

```
 Streamlit UI  (Upload · Ask · Quiz · Progress)
        │
        ▼
 Ingestion ── PDF / MD / TXT → pages → chunks (+ source, page, chunk_id)
        │
        ▼
 Embeddings (FastEmbed, bge-small-en-v1.5) ──► persistent cache (SQLite)
        │
        ▼
 FAISS index (one per session)
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
 Cited Q&A chain             Quiz generator
 retrieve → gate →           chunk → generate (structured) →
 answer → map citations      verify against source → accept / reject
                                  │
                                  ▼
                        SQLite progress store → Progress tab
```

### Design decisions

| Decision | Why |
|---|---|
| Local embeddings (FastEmbed, ONNX) | Free, no rate limits, small install. No PyTorch needed |
| Number the context passages and let the model cite `[n]` | Code maps numbers to real chunks, so invented citations can be detected |
| Relevance gate before the LLM | Irrelevant questions cost no quota |
| Second LLM pass to verify quiz questions | Generated questions are sometimes wrong or ambiguous |
| Math and counting in code, not the LLM | Scores and statistics must be exact |
| Rebuild the session index from current uploads | Edited or removed files never leave stale chunks behind |
| SQLite with parameter binding | Topic and question text comes from documents and LLM output, so it is treated strictly as data |

---

## Tech stack

| Area | Choice |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | FastEmbed with `BAAI/bge-small-en-v1.5` (local) |
| Vector search | FAISS (in-process) |
| LLM | Gemini, Groq, OpenRouter or Ollama (configurable) |
| Persistence | SQLite (progress and embedding cache) |
| Tests | pytest, including fake-LLM tests that run offline |
| Packaging | uv |
| Container | Docker |
| CI/CD | GitHub Actions |
| Hosting | Hugging Face Spaces (Docker, free CPU) |

---

## Quick start (local)

**Prerequisites:** Python (see `requires-python` in `pyproject.toml`), [uv](https://docs.astral.sh/uv/), and an API key for a free LLM tier, or [Ollama](https://ollama.com) for a fully local setup.

```bash
git clone https://github.com/YOUR_USER/studybuddy.git
cd studybuddy
uv sync
```

Create a `.env` file in the project root (never commit it):

```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash
GOOGLE_API_KEY=your_key_here
```

Run the app:

```bash
uv run streamlit run main.py
```

Open http://localhost:8501, upload a text-based PDF, Markdown or text file, click **Build index**, then use the **Ask** and **Quiz** tabs.

> Free-tier model names and limits change often. Check your provider's console for current model IDs before setting `LLM_MODEL`.

---

## Configuration

All settings are environment variables (or entries in `.env`).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `openrouter` or `ollama` |
| `LLM_MODEL` | `gemini-3.6-flash` | Model ID for the chosen provider |
| `GOOGLE_API_KEY` | – | Needed for `gemini` |
| `GROQ_API_KEY` | – | Needed for `groq` |
| `OPENROUTER_API_KEY` | – | Needed for `openrouter` |
| `APP_PASSWORD` | empty | If set, the app asks for it before showing anything |
| `PROGRESS_DB` | empty | Path to a SQLite file for **local single-user** persistence. Leave empty on public deployments |
| `EMBED_CACHE_DB` | `./.cache/embeddings.db` | Embedding cache location |
| `TOP_K` | `4` | Passages retrieved per question |
| `MIN_RELEVANCE` | `0.5` | Similarity threshold for the relevance gate. Tune it for your material |

### Choosing a relevance threshold

The default is a starting guess. Run `scripts/tune_threshold.py` with questions that are in and out of scope for your notes, then set `MIN_RELEVANCE` between the two groups of scores.

---

## Running with Docker

```bash
docker build -t studybuddy .
docker run --rm -p 8501:8501 --env-file .env -v studybuddy-data:/app/data studybuddy
```

- In `.env`, write values without quotes (`KEY=value`), because Docker's `--env-file` keeps quotes literally.
- The `studybuddy-data` volume keeps the embedding cache between runs.
- The embedding model is downloaded when the image is built, so startup doesn't wait on a download.

---

## Testing

```bash
uv run pytest -v
```

| Test file | What it covers |
|---|---|
| `test_ingest.py` | Loading, chunking, metadata, retrieval of the right file, index save and load |
| `test_qa.py` | Citation mapping, dropped invented citations, relevance gate skipping the LLM |
| `test_quiz.py` | Schema validation, verification gate, rejected questions, validity rate |
| `test_progress.py` | Topic statistics, weak-topic rules, history, reset, hostile input stored as data |
| `test_embed_cache.py` | Cache hits and misses, persistence across restarts, namespace isolation |
| `test_app.py` | The app starts without errors |

Most tests use a fake LLM and run offline. A single live-LLM test runs only when an API key is configured.

### Quality measurements

Fill these in after running the scripts on your own material:

| Metric | How to measure | Result |
|---|---|---|
| Retrieval hit@4 | Golden question set | _to fill in_ |
| Quiz validity rate (automatic) | `scripts/measure_quiz_validity.py` | _to fill in_ |
| Quiz questions judged good by hand | Read the accepted questions | _to fill in_ |

---

## Project structure

```
studybuddy/
├── main.py                    # entry point: streamlit run main.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .github/workflows/ci.yml   # tests → docker build → deploy to Hugging Face
├── src/studybuddy/
│   ├── config.py              # settings from environment
│   ├── ingest.py              # load + chunk documents
│   ├── embed_cache.py         # persistent embedding cache
│   ├── vectorstore.py         # embeddings and FAISS helpers
│   ├── qa.py                  # cited question answering
│   ├── quiz.py                # quiz generation and verification
│   ├── progress.py            # SQLite quiz history and weak topics
│   └── ui.py                  # Streamlit tabs
├── scripts/                   # threshold tuning, quiz validity measurement
└── tests/
```

---

## Deployment (Hugging Face Spaces)

The workflow in `.github/workflows/ci.yml` runs the tests, builds the Docker image, and, on pushes to `main`, pushes the repository to a Hugging Face Space, which builds and serves the Dockerfile.

1. Create a **Docker** Space (CPU Basic, free). Start it as **private**.
2. In the Space settings, add secrets `GOOGLE_API_KEY` (or your provider's key) and `APP_PASSWORD`, and variables `LLM_PROVIDER` and `LLM_MODEL`. Do not set `PROGRESS_DB`.
3. In the GitHub repository, add the secret `HF_TOKEN` (a Hugging Face token with write access) and the variable `HF_USER`.
4. Push to `main`.

### Free-tier realities

- The Space sleeps after a period of inactivity, so the first visit afterwards is slow.
- The Space disk is temporary. The embedding cache and any session data reset when the Space restarts.
- Anyone who can open a public Space spends **your** LLM quota, so keep it private or password-protected.
- Hosted free LLM tiers may use prompts to improve their models. Don't upload sensitive notes. Use a local Ollama model for private material.

---

## Known limitations

- **Text-based PDFs only.** Scanned or handwritten notes need OCR, which isn't supported yet.
- **Topics are file and page.** That is coarse, since one page can hold several ideas.
- **Progress is per browser session** on a deployed instance. Persistent history needs user accounts.
- **Small samples are noisy.** A topic needs at least two answers before it counts as weak.
- **LLM output can still be wrong.** Citations and verification reduce errors but don't eliminate them.

---

## Roadmap

- [ ] LangGraph tutor loop: ask, grade, explain from the source, update progress, choose the next question from weak topics
- [ ] Heading-based topics instead of file and page
- [ ] MCP server exposing `search_notes`, `make_quiz` and `weak_topics` as reusable tools
- [ ] CrewAI quiz writer plus fact-checker, only if measured quiz errors justify it
- [ ] Evaluation page with retrieval and answer-quality metrics
- [ ] Persistent multi-user storage

---

## License

Choose a licence (for example MIT) and add a `LICENSE` file.