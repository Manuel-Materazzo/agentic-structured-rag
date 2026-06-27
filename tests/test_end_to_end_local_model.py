from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_SMOKE") != "1",
    reason="Set RUN_E2E_SMOKE=1 to run the local-model ingestion smoke test.",
)
def test_menu_ingestion_end_to_end_with_local_model(monkeypatch, tmp_path):
    """
    End-to-end smoke test for DB creation + menu ingestion.

    Uses a local OpenAI-compatible chat endpoint for structured extraction,
    while replacing the vector upsert step with a lightweight verifier so the
    test does not depend on an embeddings endpoint.

    Run on powershell with:
    $env:RUN_E2E_SMOKE="1"; $env:OPENAI_BASE_URL="http://localhost:1234/v1"; $env:OPENAI_API_KEY="local-key"; $env:LLM_MODEL="local-model"; python -m pytest tests\test_end_to_end_local_model.py
    """
    from src.app import config as app_config
    from src.ingestion import menu_ingestion, runner

    print(f"Using temp dir: {tmp_path}")

    # Keep artifacts isolated.
    db_dir = tmp_path / "database"
    parsed_dir = tmp_path / "parsed"
    output_dir = tmp_path / "output"
    db_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app_config, "DATABASE_DIR", db_dir, raising=False)
    monkeypatch.setattr(app_config, "FACTS_DB_PATH", db_dir / "facts.db", raising=False)
    monkeypatch.setattr(app_config, "INGESTION_LOG_DB_PATH", db_dir / "ingestion_log.db", raising=False)
    monkeypatch.setattr(app_config, "PARSED_DIR", parsed_dir, raising=False)
    monkeypatch.setattr(app_config, "OUTPUT_DIR", output_dir, raising=False)

    monkeypatch.setattr(runner, "FACTS_DB_PATH", app_config.FACTS_DB_PATH, raising=False)
    monkeypatch.setattr(runner, "INGESTION_LOG_DB_PATH", app_config.INGESTION_LOG_DB_PATH, raising=False)
    monkeypatch.setattr(runner, "PARSED_DIR", app_config.PARSED_DIR, raising=False)

    monkeypatch.setattr(menu_ingestion, "PARSED_DIR", parsed_dir, raising=False)

    class DummyPipeline:
        def run(self, file_path=None, metadata=None, **kwargs):
            assert file_path
            assert Path(file_path).exists()
            assert metadata["doc_id"]
            assert metadata["source_path"]
            assert metadata["source_type"] == "menu"
            assert metadata["restaurant"]
            return {"status": "ok"}

    monkeypatch.setattr(menu_ingestion, "_build_ingestion_pipeline", lambda vs, collection_name: DummyPipeline())

    facts_con, log_con = runner.init_all()
    try:
        pdf_path = next((menu_ingestion.MENU_DIR).glob("*.pdf"))
        doc_id = menu_ingestion.ingest_menu(pdf_path, facts_con, log_con, skip_if_complete=False)

        facts_count = facts_con.execute("SELECT COUNT(*) FROM documents WHERE doc_id = ?", [doc_id]).fetchone()[0]
        log_row = log_con.execute("SELECT status FROM ingestion_log WHERE doc_id = ?", [doc_id]).fetchone()

        assert facts_count == 1
        assert log_row is not None
        assert log_row[0] == "complete"
    finally:
        facts_con.close()
        log_con.close()
