"""
vector_store_agent.py — Micro-orchestrator for semantic retrieval and synthesis.

Delegates search to Qdrant via datapizza-ai primitives, but uses an LLM
to break down complex requests into multiple sub-queries, execute them,
and synthesize the raw chunks into a clean, structured response for the Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ingestion.structured_extraction import parse_json_response
from src.app.config import (
    ALL_COLLECTIONS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
)
from src.ingestion.knowledge_manager import KnowledgeManager

log = logging.getLogger(__name__)

# Planning prompt, break task into sub-queries
PLANNER_SYSTEM_PROMPT = """You are a semantic retrieval planner for a Vector Database (Qdrant).
Your responsibility is to receive a complex request and break it down into one or more precise semantic sub-queries.
Always aim to maximize semantic retrieval accuracy in a fictionary culinary database.

Available Collections:
- menu_index: Restaurant menus, dishes, ingredients, techniques.
- manual_index: Culinary manual, technique taxonomy, licenses, macro-categories.
- code_index: Codice Galattico, compliance rules, ingredient limits, license constraints.
- blog_index: Narrative blog posts, restaurant background stories, anomalies.

CRITICAL RULES:
- Analyze the request. If it requires looking up different types of information, create multiple sub-queries.
- Choose the best collection for each sub-query.
- Take a deep breath and think step by step before responding, but keep the rationale short and commit to a single interpretation. 
- Return ONLY a valid JSON object with this exact schema: {"queries": [{"collection": "name", "query": "text"}], "rationale": "brief explanation"}
- Keep queries concise and optimized for semantic search.
"""

# Syntesis prompt, remove useless noise from chunks
SYNTHESIZER_SYSTEM_PROMPT = """You are an evidence synthesizer for a RAG system.
You are given a user request and a set of raw text chunks retrieved from a vector database.
Your job is to read the chunks and extract ONLY the information relevant to the request.

CRITICAL RULES:
- Do not invent or assume information. Base your answer strictly on the provided chunks.
- If the chunks do not contain the answer, explicitly state "INSUFFICIENT DATA".
- Do not include raw chunk metadata or scores. Provide a clean, concise summary of the facts.
- Example: If asked for dish names and ingredients, return a list like: "- Piatto X: uses Ingredient Y (200g), Ingredient Z."
"""


@dataclass
class VectorStoreResult:
    request: str
    sub_queries: list[dict]
    raw_chunks_collected: int
    synthesized_answer: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "sub_queries": self.sub_queries,
            "raw_chunks_collected": self.raw_chunks_collected,
            "answer": self.synthesized_answer,
            "error": self.error,
        }


class VectorStoreAgent:
    """LLM-driven semantic agent that plans, retrieves, and synthesizes."""

    def __init__(
            self,
            knowledge_manager: KnowledgeManager | None = None,
            llm_client=None,
            score_threshold: float = 0.4,  # TODO: tune, currently low to avoid losing context on big chunks
    ):
        self._vectorstore = knowledge_manager.get_qdrant_vectorstore()
        self._llm_client = llm_client or self._get_llm_client()
        self._score_threshold = score_threshold

    def execute(self, request: str) -> VectorStoreResult:
        log.info("VectorStoreAgent received task: %s", request)

        try:
            # 1. Plan: Break request into sub queries
            sub_queries = self._plan_sub_queries(request)
            log.info("Planned %d sub-queries.", len(sub_queries))

            # 2. Execute queries and Collect chunks
            all_chunks = []
            for sq in sub_queries:
                log.info("Executing: %s on %s", sq['query'], sq['collection'])
                chunks = self._execute_search(sq['collection'], sq['query'])
                all_chunks.extend(chunks)

            # 3. Synthesize chunks to extract evidence
            synthesized = self._synthesize(request, all_chunks)
            log.info("Synthesis complete: %s", synthesized)

            return VectorStoreResult(
                request=request,
                sub_queries=sub_queries,
                raw_chunks_collected=len(all_chunks),
                synthesized_answer=synthesized
            )

        except Exception as exc:
            log.error("VectorStoreAgent failed: %s", exc, exc_info=True)
            return VectorStoreResult(
                request=request,
                sub_queries=[],
                raw_chunks_collected=0,
                synthesized_answer="",
                error=str(exc)
            )

    def _get_llm_client(self):
        from datapizza.clients.openai_like import OpenAILikeClient
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for the VectorStore agent.")
        return OpenAILikeClient(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    def _plan_sub_queries(self, request: str) -> list[dict]:
        """Phase 1: LLM breaks request in multiple sub-queries."""
        response = self._llm_client.invoke(
            input=f"REQUEST:\n{request}\n",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
        )
        try:
            parsed = parse_json_response(response.text)
            queries = parsed.get("queries", [])
            rationale = parsed.get("rationale", "")

            # log query and rationale
            log.info(f"Vector Store Agent planned queries: \"{queries}\" with the rationale: \"{rationale}\"")

            # Validation
            valid_queries = []
            for q in queries:
                coll = q.get("collection")
                txt = q.get("query")
                if coll in ALL_COLLECTIONS and txt:
                    valid_queries.append({"collection": coll, "query": txt})

            if not valid_queries and queries:
                # if LLM hallucinated collections but produced a valid query, fallback on menu_index
                first_query = queries[0].get("query", request)
                valid_queries.append({"collection": "menu_index", "query": first_query})

            return valid_queries or [{"collection": "menu_index", "query": request}]
        except Exception:
            log.warning(
                f"Vector Store Agent ran into an exception while parsing the planner LLM response. Raw response text: \"{response.text}\"")

    def _execute_search(self, collection: str, query: str) -> list[dict]:
        """Executes the query on Qdrant."""
        from datapizza.embedders.openai import OpenAIEmbedder

        # 1. Embed
        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)
        query_vector = embedder.embed(text=query)

        # 2. Retrieve
        results = self._vectorstore.search(query_vector=query_vector, collection_name=collection, k=3)

        # 3. Format
        chunks = []
        for res in results:
            score = getattr(res, "score", 0.0)
            payload = getattr(res, "metadata", getattr(res, "payload", {}))
            text = getattr(res, "text", payload.get("text", ""))
            if score >= self._score_threshold:
                chunks.append({
                    "score": float(score),
                    "text": text,
                    "source": payload.get("source_path", "unknown")
                })
        return chunks

    def _synthesize(self, request: str, all_chunks: list[dict]) -> str:
        """Phase 3: LLM sythetizes raw chunks into a clean response."""
        if not all_chunks:
            return "INSUFFICIENT DATA: No relevant chunks found in vector store."

        # Prepare context
        context = "\n\n".join([
            f"CHUNK {i + 1} (Source: {c['source']}):\n{c['text']}"
            for i, c in enumerate(all_chunks)
        ])

        response = self._llm_client.invoke(
            input=f"REQUEST:\n{request}\n\nRETRIEVED CHUNKS:\n{context}\n\n",
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return response.text
