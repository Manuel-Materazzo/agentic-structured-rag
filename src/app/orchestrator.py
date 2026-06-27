"""
orchestrator.py — LLM-driven Agentic Orchestrator.

The orchestrator autonomously decide when to call the SQL or Qdrant agents via tools,
respecting a bounded budget of steps.
"""

from __future__ import annotations

import logging


from app.agents.vector_store_agent import VectorStoreAgent
from datapizza.agents import Agent
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.tools import Tool

from ingestion.structured_extraction import parse_json_response
from app.agents.sql_agent import SQLAgent
from app.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL

log = logging.getLogger(__name__)

# TODO: Configurable
# TODO: dynamic tool definition in prompt
ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator for a hybrid RAG system (SQL + Vector).
Your goal is to answer the user's question by finding the correct list of dish names.
You have no home planet.

You have two sub agents available as tools:
1. call_sql_agent: Use this for deterministic lookups (exact ingredients, planets, licenses, distances, compliance rules).
2. call_qdrant_agent: Use this for semantic retrieval (narrative blogs, manual sections, ambiguous terms).


RULES FOR THE LOOP:
- Analyze the question. Formulate a plan and estimate a budget of max steps.
- Call the appropriate tool. Read the raw result.
- If the tool returns what you asked for, STOP calling tools and output the FINAL ANSWER.
- Only use the data returned by the tools, do not make assumptions.

IMPORTANT: The result of the tools IS the ground truth. If tools returns something, THOSE ARE THE FINAL ANSWER. Do not second-guess the database.


FINAL ANSWER FORMAT:
You must output ONLY a valid JSON object with this schema:
{"candidates": ["Dish Name 1", "Dish Name 2"]}
Do not add markdown, explanations, or any other text before or after the JSON.
"""


class Orchestrator:
    def __init__(
            self,
            sql_agent: SQLAgent | None = None,
            vector_store_agent: VectorStoreAgent | None = None,
            llm_client=None,
    ):
        self.sql_agent = sql_agent or SQLAgent()
        self.vector_store_agent = vector_store_agent or VectorStoreAgent()
        self._llm_client = llm_client or self._get_llm_client()

        # Build orchestrator agent
        self._agent = self._build_agent()

    def _get_llm_client(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for the orchestrator.")
        return OpenAILikeClient(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    # ------------------------------------------------------------------
    # Tools Definition
    # ------------------------------------------------------------------

    def call_sql_agent(self, request: str) -> str:
        """Tool: Delegates a natural language request to the SQL agent."""
        log.info("🔧 Orchestrator calling SQL Agent: %s", request)
        result = self.sql_agent.execute(request)
        if result.error:
            return f"SQL agent failed. Error: {result.error}"
        return f"SQL agent search successful. Found {len(result.rows)} results: {result.rows}"

    def call_vector_store_agent(self, request: str) -> str:
        """Tool: Delegates a semantic natural language query to the Vector Store agent."""
        log.info("🔧 Orchestrator calling Vector Store Agent: %s", request)
        result = self.vector_store_agent.execute(request)
        if result.error:
            return f"Vector Store agent failed. Error: {result.error}"
        return f"Vector Store agent search successful. Findings: {result.synthesized_answer}"

    # ------------------------------------------------------------------
    # Agent Building
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        # Register sub agent as tools for the orchestrator agent
        sql_tool = Tool(
            name="call_sql_agent",
            func=self.call_sql_agent
        )
        vector_store_tool = Tool(
            name="call_qdrant_agent",
            func=self.call_vector_store_agent
        )

        # Build agent
        return Agent(
            name="orchestrator",
            client=self._llm_client,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=[sql_tool, vector_store_tool],
            max_steps=10,  # Max handoff budget
            terminate_on_text=True,
        )

    # ------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------

    def execute(self, request: str) -> dict:
        log.info("Starting Orchestrator for request %s", request)

        task_input = f"Question: {request}"

        try:
            # Let the agent handle the internal ReAct loop
            result = self._agent.run(task_input=task_input, temperature=0.0)

            if not result or not getattr(result, "text", ""):
                log.error("Orchestrator agent returned empty result.")
                return {"candidates": [], "error": "Empty agent response"}

            log.info("Orchestrator agent response: %s", result.text)

            # Parse the final JSON output from the agent
            payload = parse_json_response(result.text)
            candidates = payload.get("candidates", [])

            if not isinstance(candidates, list):
                candidates = []

            log.info("Orchestrator extracted candidates: %s", candidates)

            return {
                "candidates": candidates,
                "agent_trace": getattr(result, "trace", None)
            }

        except Exception as e:
            log.error("Orchestrator crashed with error: %s", e, exc_info=True)
            return {"candidates": [], "error": str(e)}
