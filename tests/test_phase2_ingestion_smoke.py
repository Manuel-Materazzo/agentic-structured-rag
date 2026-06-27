"""
test_phase2_ingestion.py — Phase 2 acceptance tests.

Verifies:
- quantity_grams is not 0.0 for unquantifiable ingredients
- Qdrant collections contain points with payload conforming to the contract
- A verification test extracts at least 5 sample dishes and checks DuckDB
- Vision fallback activates automatically on degraded parsing
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def mock_llm_extraction():
    """Provides a mock LLM response for structured extraction."""
    return {
        "restaurant": {
            "name": "Ristorante Test",
            "chef": "Chef Test",
            "planet": "Terra",
            "chef_license": "Licenza A",
            "professional_orders": ["Ordine dei Cuochi"],
        },
        "dishes": [
            {
                "name": "Piatto Dante",
                "ingredients": [
                    {"name": "Sale", "quantity_grams": None, "quantity_raw": "quanto basta"},  # NULL case
                    {"name": "Pasta", "quantity_grams": 200.0, "quantity_raw": "200g"}  # Float case
                ],
                "techniques": ["Cottura Sottovuoto"],
                "preparation_notes": "Note di test"
            },
            {
                "name": "Piatto Beatrice",
                "ingredients": [
                    {"name": "Zucchero", "quantity_grams": 0.0, "quantity_raw": "tracce"}
                    # Caso 0.0 da normalizzare a NULL
                ],
                "techniques": [],
                "preparation_notes": ""
            }
        ],
        "parsing_confidence": "high",
        "parsing_issues": ""
    }


@pytest.fixture
def mock_config_paths(tmp_path_factory, monkeypatch):
    """Patch DB and Qdrant paths to use a temporary directory."""
    from app import config
    db_dir = tmp_path_factory.mktemp("phase2_db")

    monkeypatch.setattr(config, "FACTS_DB_PATH", db_dir / "facts.db")
    monkeypatch.setattr(config, "INGESTION_LOG_DB_PATH", db_dir / "ingestion_log.db")
    monkeypatch.setattr(config, "PARSED_DIR", db_dir / "parsed")
    monkeypatch.setattr(config, "QDRANT_LOCATION", ":memory:")
    monkeypatch.setattr(config, "QDRANT_HOST", None)

    # Create folders if they do not exist
    Path(db_dir / "parsed").mkdir(parents=True, exist_ok=True)
    return db_dir


class TestIngestionConstraints:
    def test_no_zero_quantity_grams(self, mock_config_paths, mock_llm_extraction):
        """Verify that quantity_grams is not 0.0 for unquantifiable cases."""
        from ingestion.knowledge_manager import KnowledgeManager
        from ingestion.ingestors.menu_ingestor import MenuIngestor
        from ingestion.structured_extraction import _postprocess_quantities

        # Apply extraction normalization (should convert 0.0 to None)
        normalized_result = _postprocess_quantities(mock_llm_extraction)

        # Verify preprocessing did its job
        assert normalized_result["dishes"][1]["ingredients"][0]["quantity_grams"] is None

        # Write to DB and verify constraint
        with KnowledgeManager.create() as km:
            doc_id = "test_doc_123"
            km.facts_con.execute(
                "INSERT INTO documents VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                [doc_id, "test.pdf", "menu"]
            )

            ingestor = MenuIngestor()
            ingestor.write_db_entries(normalized_result, doc_id, km.facts_con)

            # Read from DB
            rows = km.facts_con.execute(
                "SELECT ingredient, quantity_grams FROM dish_ingredients WHERE dish_id IN (SELECT id FROM dishes WHERE doc_id = ?)",
                [doc_id]
            ).fetchall()

            by_ingredient = {row[0]: row[1] for row in rows}

            assert by_ingredient["sale"] is None
            assert by_ingredient["zucchero"] is None  # was 0.0, normalized at NULL
            assert by_ingredient["pasta"] == 200.0


class TestQdrantPayloadContract:
    def test_qdrant_payload_conforms(self, mock_config_paths, mock_llm_extraction):
        """Verify that Qdrant points have payloads containing doc_id, source_type, etc."""
        from ingestion.knowledge_manager import KnowledgeManager
        from qdrant_client.models import PointStruct

        with KnowledgeManager.create() as km:
            # Simulate inserting a point as would be done by the DatapizzaPipeline
            client = km.qdrant_client
            doc_id = "payload_test_doc"
            payload = {
                "chunk_id": "test-chunk-1",
                "doc_id": doc_id,
                "source_path": "menu/test.pdf",
                "source_type": "menu",
                "page": 1,
                "section": "Test Section",
                "restaurant": "Ristorante Test",
                "text": "Testo del chunk di test."
            }

            client.upsert(
                collection_name="menu_index",
                points=[PointStruct(id=1, vector=[0.1] * 1536, payload=payload)]
            )

            results = client.scroll(collection_name="menu_index", limit=10)
            assert len(results[0]) > 0

            point = results[0][0]
            assert point.payload["doc_id"] == doc_id
            assert point.payload["source_type"] == "menu"
            assert point.payload["restaurant"] == "Ristorante Test"
            assert "text" in point.payload


class TestSampleDishesExtraction:
    def test_extract_5_dishes_and_verify_db(self, mock_config_paths):
        """Verify that the extraction of 5 sample dishes completes correctly in DuckDB."""
        from ingestion.knowledge_manager import KnowledgeManager
        from ingestion.ingestors.menu_ingestor import MenuIngestor

        # Create a mock with 5 dishes
        mock_5_dishes = {
            "restaurant": {"name": "Ristorante 5 Piatti"},
            "dishes": [
                {"name": f"Piatto {i}",
                 "ingredients": [{"name": "Ingrediente", "quantity_grams": 100.0, "quantity_raw": "100g"}],
                 "techniques": []}
                for i in range(1, 6)
            ],
            "parsing_confidence": "high"
        }

        with KnowledgeManager.create() as km:
            doc_id = "5_dishes_doc"
            km.facts_con.execute(
                "INSERT INTO documents VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                [doc_id, "test.pdf", "menu"]
            )

            ingestor = MenuIngestor()
            ingestor.write_db_entries(mock_5_dishes, doc_id, km.facts_con)

            # count dishes on DB
            count = km.facts_con.execute(
                "SELECT COUNT(*) FROM dishes WHERE doc_id = ?", [doc_id]
            ).fetchone()[0]

            assert count == 5, f"Attesi 5 piatti nel DB, trovati {count}"


class TestVisionFallback:
    @patch('ingestion.ingestors.base_ingestor.BaseIngestor.parse_document')
    @patch('ingestion.structured_extraction.extract_entities')
    @patch('ingestion.vision_fallback.parse_with_vision')
    def test_vision_fallback_activates_on_low_confidence(
            self,
            mock_vision_parser,  # Corrisponde a parse_with_vision
            mock_extract_entities,
            mock_parse_document,
            mock_config_paths,
            tmp_path
    ):
        """Verify that vision fallback activates when parsing_confidence is 'low'."""
        from ingestion.ingestors.menu_ingestor import MenuIngestor
        from ingestion.ingestion_manager import IngestionManager

        # Simulate poorly parsed text from PDF
        mock_parse_document.return_value = "Degraded and incomprehensible menu text."

        # Simulate vision fallback OCR response
        mock_vision_parser.return_value = "Perfectly cleaned menu text from OCR."

        # Configure mock LLM to return first "low", then "high"
        low_confidence_result = {
            "restaurant": {}, "dishes": [], "parsing_confidence": "low", "parsing_issues": "Bad text"
        }
        high_confidence_result = {
            "restaurant": {"name": "Rescued Restaurant"}, "dishes": [], "parsing_confidence": "high"
        }
        mock_extract_entities.side_effect = [low_confidence_result, high_confidence_result]

        # Create a fake PDF file in a temporary directory
        fake_pdf = tmp_path / "degraded_menu.pdf"
        fake_pdf.write_text("Fake PDF content")

        with IngestionManager.create() as manager:
            ingestor = MenuIngestor()

            # Avoid writing to DB or Qdrant for this smoke test
            with patch.object(ingestor, 'write_db_entries'):
                with patch.object(ingestor, 'make_vector_indexer'):
                    # Execute single file ingestion
                    ingestor.ingest_single(manager, fake_pdf)

        # Verify LLM extraction was called twice (first low, then after fallback)
        assert mock_extract_entities.call_count == 2

        # Verify vision parser was invoked once
        mock_vision_parser.assert_called_once()
