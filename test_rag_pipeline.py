import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# import your file
# from rag_pipeline import RAGPipeline, EnhancedQuery

# Example:
from rag_pipeline import RAGPipeline


@pytest.fixture
def mock_pipeline():
    """
    Create pipeline with mocked dependencies
    """

    with patch("your_file_name.ChatGoogleGenerativeAI") as mock_llm, \
         patch("your_file_name.Chroma") as mock_chroma, \
         patch("your_file_name.GoogleGenerativeAIEmbeddings"):

        # Mock vector store
        mock_vector_instance = MagicMock()
        mock_chroma.return_value = mock_vector_instance

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        pipeline = RAGPipeline()

        pipeline.vectore_store = mock_vector_instance
        pipeline.gen_model = mock_llm_instance

        return pipeline


def test_ingest(mock_pipeline):
    dataset = [
        "AI is changing healthcare",
        "Cloud computing is scalable"
    ]

    mock_pipeline.ingest(dataset)

    # Ensure add_documents called once
    assert mock_pipeline.vectore_store.add_documents.called

    # Get actual documents passed
    args, kwargs = mock_pipeline.vectore_store.add_documents.call_args

    docs = args[0]

    assert len(docs) == 2
    assert docs[0].page_content == "AI is changing healthcare"
    assert docs[1].page_content == "Cloud computing is scalable"


def test_retrieve_strategy_a(mock_pipeline):

    mock_results = [
        Document(page_content="AI healthcare result"),
        Document(page_content="Cloud result"),
    ]

    mock_pipeline.vectore_store.similarity_search.return_value = mock_results

    response = mock_pipeline.retrieve_strategy_a("AI")

    assert len(response) == 2
    assert "AI healthcare result" in response
    assert "Cloud result" in response

    mock_pipeline.vectore_store.similarity_search.assert_called_once_with(
        "AI",
        k=3
    )


def test_retrieve_strategy_b(mock_pipeline):

    # Mock structured output
    structured_mock = MagicMock()

    structured_mock.invoke.return_value.model_dump.return_value = {
        "query": "expanded AI healthcare query"
    }

    mock_pipeline.gen_model.with_structured_output.return_value = structured_mock

    # Mock vector search
    mock_pipeline.vectore_store.similarity_search.return_value = [
        Document(page_content="Expanded AI result")
    ]

    results, expanded_query = mock_pipeline.retrieve_strategy_b(
        "AI healthcare"
    )

    assert expanded_query == "expanded AI healthcare query"

    assert len(results) == 1
    assert results[0] == "Expanded AI result"

    mock_pipeline.vectore_store.similarity_search.assert_called_once_with(
        query="expanded AI healthcare query",
        k=3
    )


def test_retrieve_strategy_a_empty(mock_pipeline):

    mock_pipeline.vectore_store.similarity_search.return_value = []

    response = mock_pipeline.retrieve_strategy_a("random query")

    assert response == []


def test_retrieve_strategy_b_empty(mock_pipeline):

    structured_mock = MagicMock()

    structured_mock.invoke.return_value.model_dump.return_value = {
        "query": "expanded random query"
    }

    mock_pipeline.gen_model.with_structured_output.return_value = structured_mock

    mock_pipeline.vectore_store.similarity_search.return_value = []

    results, expanded_query = mock_pipeline.retrieve_strategy_b(
        "random query"
    )

    assert expanded_query == "expanded random query"
    assert results == []


def test_ingest_empty_list(mock_pipeline):

    mock_pipeline.ingest([])

    mock_pipeline.vectore_store.add_documents.assert_not_called()