from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from studybuddy.qa import answer_question


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
    result = answer_question(
        FakeVectorStore(),
        "How do plants make food?",
        llm=FakeLLM(),
        min_relevance=0,
    )

    assert result.text == "Plants make food using sunlight [1]."
    assert result.grounded
    assert [source.file for source in result.sources] == ["biology.txt"]
