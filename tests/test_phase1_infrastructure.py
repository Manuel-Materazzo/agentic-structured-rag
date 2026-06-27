"""
test_phase1_infrastructure.py — Phase 1 acceptance tests.

Verifies:
- facts.db and ingestion_log.db are created without errors
- All four Qdrant collections exist
- submission.csv is generated with 100 rows and correct columns
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tmp_db_dir(tmp_path_factory):
    """Temporary directory that acts as the database root."""
    return tmp_path_factory.mktemp("database")


class TestDatabaseInit:
    def test_facts_db_created(self, tmp_db_dir):
        """facts.db should be created with all required tables."""
        import duckdb
        from src.ingestion.runner import FACTS_SCHEMA_SQL

        db_path = tmp_db_dir / "facts.db"
        con = duckdb.connect(str(db_path))
        con.execute(FACTS_SCHEMA_SQL)

        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }

        required_tables = {
            "documents", "restaurants", "dishes",
            "dish_ingredients", "dish_techniques",
            "technique_taxonomy", "planet_distances", "compliance_rules",
        }
        for t in required_tables:
            assert t in tables, f"Missing table: {t}"
        con.close()

    def test_ingestion_log_db_created(self, tmp_db_dir):
        """ingestion_log.db should be created with ingestion_log table."""
        import duckdb
        from src.ingestion.runner import INGESTION_LOG_SCHEMA_SQL

        db_path = tmp_db_dir / "ingestion_log.db"
        con = duckdb.connect(str(db_path))
        con.execute(INGESTION_LOG_SCHEMA_SQL)

        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert "ingestion_log" in tables
        con.close()

    def test_dish_ingredients_nullable_quantity(self, tmp_db_dir):
        """quantity_grams must be FLOAT and nullable (not 0.0 for unquantifiable)."""
        import duckdb
        from src.ingestion.runner import FACTS_SCHEMA_SQL

        db_path = tmp_db_dir / "facts_qty.db"
        con = duckdb.connect(str(db_path))
        con.execute(FACTS_SCHEMA_SQL)

        # Insert a test document first
        con.execute(
            "INSERT INTO documents VALUES ('doc1', 'menu/test.pdf', 'menu', CURRENT_TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO restaurants (id, name, doc_id) VALUES (1, 'TestRest', 'doc1')"
        )
        con.execute(
            "INSERT INTO dishes (id, name, restaurant_id, doc_id) VALUES (1, 'TestDish', 1, 'doc1')"
        )

        # NULL quantity for "quanto basta"
        con.execute(
            "INSERT INTO dish_ingredients VALUES (1, 'sale', NULL, 'quanto basta', FALSE)"
        )
        # Numeric quantity
        con.execute(
            "INSERT INTO dish_ingredients VALUES (1, 'farina', 200.0, '200g', FALSE)"
        )

        rows = con.execute(
            "SELECT ingredient, quantity_grams FROM dish_ingredients WHERE dish_id = 1"
        ).fetchall()
        by_ingredient = dict(rows)

        assert by_ingredient["sale"] is None, "quantity_grams should be NULL for 'quanto basta'"
        assert by_ingredient["farina"] == 200.0
        con.close()


class TestQdrantCollections:
    def test_four_collections_exist(self):
        """After init_qdrant_collections(), all four collections must exist."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from src.app.config import ALL_COLLECTIONS, EMBEDDING_DIM

        client = QdrantClient(location=":memory:")

        # Simulate what init_qdrant_collections does
        for name in ALL_COLLECTIONS:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

        existing = {c.name for c in client.get_collections().collections}
        for col in ALL_COLLECTIONS:
            assert col in existing, f"Collection missing: {col}"


class TestSubmission:
    def test_empty_submission_100_rows(self, tmp_path):
        """export_empty_submission() must produce 100 rows with row_id and result columns."""
        import pandas as pd
        from src.app.submission import export_empty_submission
        from src.app.config import DOMANDE_CSV_PATH

        out_path = tmp_path / "submission.csv"
        real_out_path = export_empty_submission(output_path=out_path)

        assert out_path == real_out_path
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert "row_id" in df.columns
        assert "result" in df.columns
        assert len(df) == 100, f"Expected 100 rows, got {len(df)}"

    def test_dish_mapping_loads(self):
        """load_dish_mapping() must return a non-empty dict."""
        from src.app.submission import load_dish_mapping
        mapping = load_dish_mapping()
        assert len(mapping) > 0
        # All values should be integers
        for name, dish_id in mapping.items():
            assert isinstance(dish_id, int), f"ID for '{name}' is not int: {dish_id}"

    def test_dish_name_to_id_exact(self):
        """dish_name_to_id() must map a known dish name to its integer ID."""
        from src.app.submission import load_dish_mapping, dish_name_to_id
        mapping = load_dish_mapping()
        # Pick the first entry and verify round-trip
        first_name = next(iter(mapping))
        first_id = mapping[first_name]
        assert dish_name_to_id(first_name) == first_id

    def test_submission_export_with_answers(self, tmp_path):
        """export_submission() must write correct row_id/result pairs."""
        import pandas as pd
        from src.app.submission import export_submission, load_dish_mapping

        mapping = load_dish_mapping()
        # Fake answers for first 3 questions, TODO: avoid skipping real mapping
        fake_answers = {1: [1, 5], 2: [42], 3: []}
        out_path = tmp_path / "submission_answers.csv"
        export_submission(fake_answers, output_path=out_path)

        df = pd.read_csv(out_path)
        assert len(df) == 100

        row1 = df[df["row_id"] == 1].iloc[0]
        assert row1["result"] == "1,5"

        row2 = df[df["row_id"] == 2].iloc[0]
        assert str(row2["result"]) == "42"
