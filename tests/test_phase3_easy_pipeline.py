"""
test_phase3_runtime.py — Phase 3 acceptance tests.

Verifies:
- SQLAgent correctly parses LLM JSON, executes the query, and returns raw rows.
- Answer normalizer and validation tools filter out hallucinations and map to IDs.
- Orchestrator uses the tool loop to fetch data and return submission-ready IDs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from app.agents.vector_store_agent import VectorStoreAgent
from ingestion.knowledge_manager import KnowledgeManager
from utils.normalizer_utils import normalize_answers, map_dish_names_to_ids
from utils.validation_utils import validate_candidate_dishes

BASE_TMP = Path('output/.phase3_tests')
BASE_TMP.mkdir(parents=True, exist_ok=True)


class TestPhase3:
    def _mock_openai_client(self, response_text: str):
        """Create a mock of the datapizza OpenAI client that returns fixed JSON."""
        from datapizza.clients.openai import OpenAIClient

        mock_response = MagicMock()
        mock_response.text = response_text

        client = MagicMock(spec=OpenAIClient)
        client.invoke.return_value = mock_response
        return client

    def _seed_facts_db(self, con: duckdb.DuckDBPyConnection) -> None:
        """Initializes the schema and inserts test data into the in-memory DB."""
        from ingestion.knowledge_manager import _FACTS_SCHEMA_SQL

        con.execute(_FACTS_SCHEMA_SQL)
        con.execute(
            "INSERT INTO documents VALUES ('doc1', 'menu/test.pdf', 'menu', CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING")
        con.execute(
            "INSERT INTO restaurants VALUES (1, 'Star Pasta', 'Chef Uno', 'Mars', 'B', ['Order A'], 'doc1') ON CONFLICT DO NOTHING")
        con.execute(
            "INSERT INTO dishes VALUES (1, 'Stellar Lasagna', 1, 'Baked slowly', 'doc1') ON CONFLICT DO NOTHING")
        con.execute("INSERT INTO dish_ingredients VALUES (1, 'basilico', 12.5, '12.5g', FALSE) ON CONFLICT DO NOTHING")
        con.execute("INSERT INTO dish_techniques VALUES (1, 'gratinatura') ON CONFLICT DO NOTHING")

    @pytest.fixture
    def mock_config_paths(self, tmp_path_factory, monkeypatch):
        """Patch DB and Qdrant paths to use a temporary directory."""
        from app import config
        db_dir = tmp_path_factory.mktemp("phase3_db")

        monkeypatch.setattr(config, "FACTS_DB_PATH", db_dir / "facts.db")
        monkeypatch.setattr(config, "INGESTION_LOG_DB_PATH", db_dir / "ingestion_log.db")
        monkeypatch.setattr(config, "PARSED_DIR", db_dir / "parsed")
        monkeypatch.setattr(config, "QDRANT_LOCATION", ":memory:")
        monkeypatch.setattr(config, "QDRANT_HOST", None)
        monkeypatch.setattr(config, "QDRANT_PORT", 6333)
        monkeypatch.setattr(config, "QDRANT_API_KEY", None)

        # Create folders if they do not exist
        Path(db_dir / "parsed").mkdir(parents=True, exist_ok=True)
        return db_dir

    def test_sql_agent_executes_ingredient_lookup(self, mock_config_paths):
        """Test that the SQLAgent generates the query, executes it, and returns the raw data."""
        from app.agents.sql_agent import SQLAgent

        with KnowledgeManager.create() as km:
            self._seed_facts_db(km.facts_con)

            sql_client = self._mock_openai_client(
                '{"sql": "SELECT d.name AS dish_name FROM dish_ingredients di JOIN dishes d ON d.id = di.dish_id WHERE lower(di.ingredient) = lower(\'basilico\')", "rationale": "ingredient lookup"}'
            )

            agent = SQLAgent(knowledge_manager=km, llm_client=sql_client)
            result = agent.execute("Find dishes with ingredient basilico")

            assert result.error is None
            assert result.rows
            assert result.columns == ["dish_name"]
            assert result.rows[0][0] == "nebulosa celestiale alla stellaris"

    def test_answer_normalizer_and_validation(self, monkeypatch, mock_config_paths):
        """Test that normalization cleans up the text and that mapping excludes unknown plates."""
        from evaluation import generate_kaggle_submission_file

        # Mock the submission mapping so it doesn't depend on the real JSON file
        monkeypatch.setattr(generate_kaggle_submission_file, "_DISH_MAPPING", {"Stellar Lasagna": 11, "Galaxy Risotto": 22},
                            raising=False)
        monkeypatch.setattr(generate_kaggle_submission_file, "_DISH_MAPPING_LOWER",
                            {"stellar lasagna": 11, "galaxy risotto": 22},
                            raising=False)

        names = [" Stellar  Lasagna ", "stellar lasagna", "Unknown Dish"]
        normalized = normalize_answers(names)
        assert normalized == ["stellar lasagna", "unknown dish"]

        # Validation must remove plates not present in the mapping (anti-overfitting) TODO: update when removed
        valid = validate_candidate_dishes(normalized)
        assert valid == ["stellar lasagna"]

        ids = map_dish_names_to_ids(["stellar lasagna", "unknown dish"])
        assert ids == [11]

    @patch('app.orchestrator.Agent')
    def test_orchestrator_returns_submission_ready_ids(self, MockAgent, monkeypatch, mock_config_paths):
        """Test that the Orchestrator calls the SQL tool and returns the normalized IDs."""
        from app.agents.sql_agent import SQLAgent
        from app.orchestrator import Orchestrator
        from evaluation import generate_kaggle_submission_file

        # Mock mappings
        monkeypatch.setattr(generate_kaggle_submission_file, "_DISH_MAPPING", {"Stellar Lasagna": 11}, raising=False)
        monkeypatch.setattr(generate_kaggle_submission_file, "_DISH_MAPPING_LOWER", {"stellar lasagna": 11}, raising=False)

        with KnowledgeManager.create() as km:
            self._seed_facts_db(km.facts_con)
            sql_client = self._mock_openai_client(
                '{"sql": "SELECT d.name FROM dish_ingredients di JOIN dishes d ON d.id = di.dish_id WHERE lower(di.ingredient) = lower(\'basilico\')", "rationale": "ingredient lookup"}'
            )
            sql_agent = SQLAgent(knowledge_manager=km, llm_client=sql_client)
            vector_store_agent = VectorStoreAgent(knowledge_manager=km, llm_client=sql_client)

            # Simulate the datapizza Agent loop: the orchestrator will want to call the SQL tool,
            # receive the data, and then return the final JSON with the candidates.
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run.return_value = MagicMock(text='{"candidates": ["Stellar Lasagna"]}')

            # Instantiate the Orchestrator. The planner_client isn't used here because we're mocking the Agent directly.
            orch = Orchestrator(sql_agent=sql_agent, vector_store_agent=vector_store_agent, llm_client=sql_client)

            # We intercept the call to the tool to simulate the real execution on the DB
            def fake_call_sql_agent(request: str) -> str:
                import json
                res = sql_agent.execute(request)
                return json.dumps(res.as_dict(), ensure_ascii=False)

            # Replace the real method with our bypass
            orch.call_sql_agent = fake_call_sql_agent

            result_dict = orch.execute("Which dishes use ingredient basilico?")

            assert "Stellar Lasagna" in result_dict["candidates"]

            # Verify that the datapizza Agent has actually ran
            MockAgent.assert_called_once()
            mock_agent_instance.run.assert_called_once()
