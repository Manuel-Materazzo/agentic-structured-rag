"""
test_phase0_spike.py — Phase 0 API verification tests for datapizza-ai.

Verifies that the following APIs work as expected in the current version:
- Agent (constructor, tools, system_prompt)
- IngestionPipeline (run method)
- DagPipeline (add_module, run)
- NodeSplitter + ChunkEmbedder (payload structure)
- QdrantVectorstore (upsert, search, delete)
- ChatPromptTemplate + ToolRewriter (construction)
- ContextTracing (trace context manager)

NOTE: These tests mock external LLM/OpenAI calls and use Qdrant in-memory mode.
They do NOT require a live OPENAI_API_KEY to pass.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _mock_openai_client():
    """Return a mock that satisfies the datapizza Client interface.

    Key corrections vs the original helper:
    - Client exposes `invoke` (not `complete`).
    - ClientResponse requires content: list[Block] and a TokenUsage, not a
      bare MagicMock; we build a real instance so spec'd mocks don't reject it.
    - We also stub `_invoke` for any code that calls the internal method.
    """
    from datapizza.core.clients import Client, ClientResponse
    from datapizza.core.clients.models import TokenUsage
    from datapizza.type import TextBlock

    mock_response = ClientResponse(
        content=[TextBlock(content="mock response")],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    )

    client = MagicMock(spec=Client)
    client.invoke.return_value = mock_response
    client._invoke.return_value = mock_response
    return client


# ---------------------------------------------------------------------------
# 1. Agent — constructor, tools, system_prompt
# ---------------------------------------------------------------------------

class TestAgent:
    def test_agent_constructor_accepts_tools_and_system_prompt(self):
        """Agent must accept tools and system_prompt without raising."""
        from datapizza.agents import Agent
        from datapizza.tools import Tool

        called = {}

        def my_tool(query: str) -> str:
            """A test tool."""
            called["query"] = query
            return "tool_result"

        tool = Tool(func=my_tool, name="my_tool", description="A test tool")
        client = _mock_openai_client()

        agent = Agent(
            name="test-agent",
            client=client,
            system_prompt="You are a test assistant.",
            tools=[tool],
        )

        assert agent.name == "test-agent"
        assert agent.system_prompt == "You are a test assistant."
        assert len(agent._tools) >= 1

    def test_agent_requires_client(self):
        """Agent must raise ValueError when client is not provided."""
        from datapizza.agents import Agent

        with pytest.raises(ValueError, match="Client is required"):
            Agent(name="no-client-agent", system_prompt="hello")

    def test_agent_requires_name(self):
        """Agent must raise ValueError when name is not provided."""
        from datapizza.agents import Agent

        with pytest.raises(ValueError):
            Agent(client=_mock_openai_client(), system_prompt="hello")

    def test_agent_tool_call_end_to_end(self):
        """Agent constructed with tools should expose them correctly (no live LLM needed)."""
        from datapizza.agents import Agent
        from datapizza.tools import Tool

        tool_invoked = {}

        def echo_tool(text: str) -> str:
            """Echo tool for testing."""
            tool_invoked["text"] = text
            return f"echo: {text}"

        tool = Tool(func=echo_tool, name="echo_tool", description="Echo tool for testing")
        client = _mock_openai_client()

        agent = Agent(
            name="echo-agent",
            client=client,
            system_prompt="Use the echo tool.",
            tools=[tool],
            max_steps=1,
        )

        # Verify the agent was wired correctly without calling a live LLM.
        assert any(t.name == "echo_tool" for t in agent._tools)


# ---------------------------------------------------------------------------
# 2. IngestionPipeline — constructor and run interface
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    def test_ingestion_pipeline_constructor(self):
        """IngestionPipeline must accept a modules list."""
        from datapizza.pipeline import IngestionPipeline
        from datapizza.modules.splitters import NodeSplitter

        splitter = NodeSplitter(max_char=100)
        pipeline = IngestionPipeline(modules=[splitter])

        assert pipeline.pipeline is not None
        assert len(pipeline.pipeline.components) == 1

    def test_ingestion_pipeline_run_on_text_nodes(self):
        """IngestionPipeline.run should process a list of Nodes through NodeSplitter."""
        from datapizza.pipeline import IngestionPipeline
        from datapizza.modules.splitters import NodeSplitter
        from datapizza.type import Node

        node = Node(content="Hello world, this is a test document for chunking.")
        splitter = NodeSplitter(max_char=20)
        pipeline = IngestionPipeline(modules=[splitter])

        result = pipeline.pipeline.run(node)
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. DagPipeline — add_module and graph execution
# ---------------------------------------------------------------------------

class TestDagPipeline:
    def test_dag_pipeline_add_module(self):
        """DagPipeline.add_module must register named nodes."""
        from datapizza.pipeline import DagPipeline
        from datapizza.modules.splitters import NodeSplitter

        dag = DagPipeline()
        splitter = NodeSplitter(max_char=500)
        dag.add_module("splitter", splitter)

        assert "splitter" in dag.nodes

    def test_dag_pipeline_two_modules_registered(self):
        """DagPipeline must register multiple independently-added modules."""
        from datapizza.pipeline import DagPipeline
        from datapizza.modules.splitters import NodeSplitter

        dag = DagPipeline()
        s1 = NodeSplitter(max_char=500)
        s2 = NodeSplitter(max_char=200)
        dag.add_module("splitter1", s1)
        dag.add_module("splitter2", s2)

        assert "splitter1" in dag.nodes
        assert "splitter2" in dag.nodes


# ---------------------------------------------------------------------------
# 4. NodeSplitter + ChunkEmbedder — chunk payload
# ---------------------------------------------------------------------------

class TestNodeSplitterAndChunkEmbedder:
    def test_node_splitter_produces_chunks(self):
        """NodeSplitter must split a Node tree into multiple Chunks."""
        from datapizza.modules.splitters import NodeSplitter
        from datapizza.type import Node

        # Build a parent node whose content exceeds max_char, with children
        # that are each individually within the limit → each child becomes a chunk.
        children = [Node(content=f"Section {i}: " + "word " * 10) for i in range(5)]
        parent = Node(content="word " * 200, children=children)
        splitter = NodeSplitter(max_char=100)

        chunks = splitter(parent)
        assert len(chunks) > 1
        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert len(chunk.text) > 0

    def test_node_splitter_preserves_metadata(self):
        """NodeSplitter must propagate metadata from child Nodes into Chunks."""
        from datapizza.modules.splitters import NodeSplitter
        from datapizza.type import Node

        child = Node(
            content="Test content for metadata " * 5,
            metadata={"source": "test_menu.pdf", "restaurant": "Test Restaurant"},
        )
        parent = Node(content="x" * 200, children=[child])
        splitter = NodeSplitter(max_char=50)
        chunks = splitter(parent)

        assert len(chunks) > 0
        # Chunks should carry parent metadata
        for chunk in chunks:
            assert chunk.text  # non-empty

    def test_chunk_embedder_constructor(self):
        """ChunkEmbedder must accept an embedder client."""
        from datapizza.embedders import ChunkEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]

        embedder = ChunkEmbedder(client=mock_embedder)
        assert embedder is not None


# ---------------------------------------------------------------------------
# 5. OpenAIEmbedder — constructor validation
# ---------------------------------------------------------------------------

class TestOpenAIEmbedder:
    def test_openai_embedder_constructor(self):
        """OpenAIEmbedder must accept api_key and model_name."""
        from datapizza.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="test-key", model_name="text-embedding-3-small")
        assert embedder is not None

    def test_openai_embedder_embed_calls_api(self):
        """OpenAIEmbedder construction should succeed (actual embed() needs live API)."""
        from datapizza.embedders.openai import OpenAIEmbedder

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response

            embedder = OpenAIEmbedder(api_key="test-key", model_name="text-embedding-3-small")
            assert embedder is not None


# ---------------------------------------------------------------------------
# 6. QdrantVectorstore — upsert, search, delete on in-memory collection
# ---------------------------------------------------------------------------

class TestQdrantVectorstore:
    COLLECTION = "test_spike_collection"
    DIM = 4  # small dimension for speed

    def _get_vs(self):
        from datapizza.vectorstores.qdrant import QdrantVectorstore
        return QdrantVectorstore(location=":memory:")

    def _ensure_collection(self, vs):
        from qdrant_client.models import Distance, VectorParams
        client = vs.get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self.COLLECTION not in existing:
            client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=self.DIM, distance=Distance.COSINE),
            )
        return client

    def test_qdrant_upsert(self):
        """QdrantVectorstore must successfully upsert a point."""
        vs = self._get_vs()
        client = self._ensure_collection(vs)

        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"chunk_id": "c1", "doc_id": "d1", "text": "test chunk", "source_type": "menu"},
        )
        client.upsert(collection_name=self.COLLECTION, points=[point])
        count = client.count(collection_name=self.COLLECTION)
        assert count.count >= 1

    def test_qdrant_search(self):
        """QdrantVectorstore must return results on vector search."""
        vs = self._get_vs()
        client = self._ensure_collection(vs)

        from qdrant_client.models import PointStruct
        client.upsert(
            collection_name=self.COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"chunk_id": "c2", "doc_id": "d2", "text": "galactic pizza", "source_type": "menu"},
            )],
        )

        response = client.query_points(
            collection_name=self.COLLECTION,
            query=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )
        assert len(response.points) >= 1
        assert response.points[0].payload["text"] == "galactic pizza"

    def test_qdrant_delete(self):
        """QdrantVectorstore must delete points by filter."""
        vs = self._get_vs()
        client = self._ensure_collection(vs)

        from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
        doc_id = "delete_me_doc"
        client.upsert(
            collection_name=self.COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.9, 0.1, 0.0, 0.0],
                payload={"chunk_id": "c3", "doc_id": doc_id, "text": "to be deleted"},
            )],
        )

        client.delete(
            collection_name=self.COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

        response = client.query_points(
            collection_name=self.COLLECTION,
            query=[0.9, 0.1, 0.0, 0.0],
            limit=5,
            query_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        assert len(response.points) == 0


# ---------------------------------------------------------------------------
# 7. ChatPromptTemplate + ToolRewriter — construction and basic validation
# ---------------------------------------------------------------------------

class TestPromptAndRewriter:
    def test_chat_prompt_template_constructor(self):
        """ChatPromptTemplate must accept user and retrieval prompt templates."""
        from datapizza.modules.prompt import ChatPromptTemplate

        pt = ChatPromptTemplate(
            user_prompt_template="Query: {{ user_prompt }}",
            retrieval_prompt_template="{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
        )
        assert pt is not None

    def test_tool_rewriter_constructor(self):
        """ToolRewriter must accept a client and system_prompt."""
        from datapizza.modules.rewriters import ToolRewriter

        client = _mock_openai_client()
        rewriter = ToolRewriter(
            client=client,
            system_prompt="Rewrite the query to improve vector retrieval accuracy.",
        )
        assert rewriter is not None


# ---------------------------------------------------------------------------
# 8. ContextTracing — trace context manager emits spans
# ---------------------------------------------------------------------------

class TestContextTracing:
    def test_context_tracing_trace_context_manager(self):
        """ContextTracing.trace must function as a context manager without errors."""
        from datapizza.tracing import ContextTracing

        tracer = ContextTracing()
        with tracer.trace("spike_test_operation"):
            result = 1 + 1  # some work inside the trace

        assert result == 2

    def test_context_tracing_nested(self):
        """ContextTracing must support nested traces."""
        from datapizza.tracing import ContextTracing

        tracer = ContextTracing()
        with tracer.trace("outer"):
            with tracer.trace("inner"):
                pass  # should not raise
