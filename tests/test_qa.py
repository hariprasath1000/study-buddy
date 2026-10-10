import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from studybuddy import qa


class FakeVectorStore:
    def similarity_search_with_score(self, question: str, k: int):
        return [
            (
                Document(
                    page_content="Plants use sunlight to make food.",
                    metadata={"source": "biology.txt"},
                ),
                0.2,
            )
        ]


class FakeLLM:
    def invoke(self, prompt):
        return AIMessage(
            content=[
                {"type": "text", "text": "Plants make food using sunlight [1]."},
            ]
        )


def test_answer_question_accepts_list_content_blocks() -> None:
    result = qa.answer_question(
        FakeVectorStore(),
        "How do plants make food?",
        llm=FakeLLM(),
        min_relevance=0,
    )

    assert result.text == "Plants make food using sunlight [1]."
    assert result.grounded
    assert [source.file for source in result.sources] == ["biology.txt"]


def test_retrieve_filters_using_cosine_relevance_and_sets_metadata(monkeypatch):
    docs = [
        Document(page_content="near", metadata={"source": "one.txt"}),
        Document(page_content="far", metadata={"source": "two.txt"}),
    ]

    class Store:
        def similarity_search_with_score(self, question, k):
            assert k == 2
            return [(docs[0], 0.4), (docs[1], 1.2)]

    monkeypatch.setattr(qa.settings, "top_k", 2)
    monkeypatch.setattr(qa.settings, "min_relevance", 0.5)

    result = qa.retrieve(Store(), "question")

    assert result == [docs[0]]
    assert docs[0].metadata["relevance"] == 0.8
    assert "relevance" not in docs[1].metadata


def test_retrieve_honors_explicit_zero_threshold_and_k():
    class Store:
        def similarity_search_with_score(self, question, k):
            assert k == 3
            return [(Document(page_content="edge", metadata={"source": "x"}), 2.0)]

    assert len(qa.retrieve(Store(), "question", k=3, min_relevance=0)) == 1


def test_format_context_and_extract_sources_handle_pages_and_invalid_citations():
    docs = [
        Document(page_content="a" * 230, metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="short", metadata={"source": "b.txt", "page": None}),
    ]

    context = qa.format_context(docs)
    sources = qa.extract_sources("Use [2], [1][2], [0], and [9].", docs)

    assert "[1] (a.pdf, page 3)" in context
    assert "[2] (b.txt)\nshort" in context
    assert [(source.id, source.file, source.page) for source in sources] == [
        (1, "a.pdf", 3),
        (2, "b.txt", None),
    ]
    assert len(sources[0].excerpt) == 200


def test_answer_question_skips_model_when_no_documents():
    class EmptyStore:
        def similarity_search_with_score(self, question, k):
            return []

    class MustNotRun:
        def invoke(self, prompt):
            raise AssertionError("model should not be called")

    answer = qa.answer_question(EmptyStore(), "question", llm=MustNotRun())

    assert answer.text == qa.NO_ANSWER
    assert answer.sources == []
    assert not answer.grounded


def test_answer_question_marks_missing_citations_ungrounded():
    class UncitedLLM:
        def invoke(self, prompt):
            return AIMessage(content="An uncited answer.")

    answer = qa.answer_question(
        FakeVectorStore(), "question", llm=UncitedLLM(), min_relevance=0
    )

    assert answer.text == "An uncited answer."
    assert answer.sources == []
    assert not answer.grounded


@pytest.mark.parametrize(
    ("provider", "module", "class_name", "expected"),
    [
        ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", {"max_retries": 3}),
        ("groq", "langchain_groq", "ChatGroq", {"temperature": 0, "max_retries": 3}),
        (
            "openrouter",
            "langchain_openai",
            "ChatOpenAI",
            {
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": 0,
                "max_retries": 3,
            },
        ),
        ("ollama", "langchain_ollama", "ChatOllama", {"temperature": 0}),
    ],
)
def test_get_llm_configures_supported_providers(
    monkeypatch, provider, module, class_name, expected
):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{class_name: FakeClient}))
    monkeypatch.setattr(qa.settings, "llm_provider", provider)
    monkeypatch.setattr(qa.settings, "llm_model", "test-model")
    monkeypatch.setattr(qa.settings, "google_api_key", "google-test")
    monkeypatch.setattr(qa.settings, "groq_api_key", "groq-test")
    monkeypatch.setattr(qa.settings, "openrouter_api_key", "router-test")

    assert isinstance(qa.get_llm(), FakeClient)
    assert captured["model"] == "test-model"
    assert {key: captured[key] for key in expected} == expected


def test_get_llm_rejects_missing_key_and_unknown_provider(monkeypatch):
    monkeypatch.setattr(qa.settings, "llm_provider", "gemini")
    monkeypatch.setattr(qa.settings, "google_api_key", "")
    with pytest.raises(RuntimeError, match="No API key set"):
        qa.get_llm()

    monkeypatch.setattr(qa.settings, "llm_provider", "unknown")
    monkeypatch.setattr(type(qa.settings), "llm_ready", lambda self: True)
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        qa.get_llm()
