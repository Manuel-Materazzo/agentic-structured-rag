from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest


def _get_embedder():
    from datapizza.embedders.openai import OpenAIEmbedder
    from src.app.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

    return OpenAIEmbedder(
        api_key=OPENAI_API_KEY or "local-key",
        base_url=OPENAI_BASE_URL,
        model_name=os.environ.get("LOCAL_EMBEDDING_MODEL", EMBEDDING_MODEL),
    )


def _make_qdrant_collection(embedder, collection_name: str):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(location=":memory:")
    dim = len(embedder.embed("vector dimension probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    return client, dim


def _upsert_roundtrip(client, collection_name: str, vector: list[float], payload: dict):
    from qdrant_client.models import PointStruct, FieldCondition, Filter, MatchValue

    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)],
    )

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=5,
    )
    assert response.points
    assert response.points[0].payload["doc_id"] == payload["doc_id"]
    assert response.points[0].payload["text"]

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=payload["doc_id"]))]
        ),
    )
    deleted = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=payload["doc_id"]))]
        ),
    )
    assert not deleted.points


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_QDRANT") != "1",
    reason="Set RUN_E2E_QDRANT=1 to run the source-by-source Qdrant smoke test.",
)
def test_menu_source_qdrant_roundtrip():
    from src.ingestion.structured_extraction import _parse_document

    embedder = _get_embedder()
    client, _ = _make_qdrant_collection(embedder, "menu_smoke")

    menu_pdf = next(Path("Dataset/knowledge_base/menu").glob("*.pdf"))
    text = _parse_document(str(menu_pdf), "menu")
    sample = text[:1200] or menu_pdf.name
    vector = embedder.embed(sample)
    _upsert_roundtrip(
        client,
        "menu_smoke",
        vector,
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": f"menu::{menu_pdf.stem}",
            "source_path": str(menu_pdf),
            "source_type": "menu",
            "page": 1,
            "section": "menu_smoke",
            "restaurant": None,
            "dish": None,
            "text": sample,
        },
    )


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_QDRANT") != "1",
    reason="Set RUN_E2E_QDRANT=1 to run the source-by-source Qdrant smoke test.",
)
def test_manual_source_qdrant_roundtrip():
    from src.ingestion.cook_manual_ingestion import _parse_manual_text
    from src.ingestion.shared import split_text

    embedder = _get_embedder()
    client, _ = _make_qdrant_collection(embedder, "manual_smoke")

    manual_pdf = Path("Dataset/knowledge_base/misc/Manuale di Cucina.pdf")
    text = _parse_manual_text(str(manual_pdf))
    sample = split_text(text, max_chars=1200)[0] if text else manual_pdf.name
    vector = embedder.embed(sample)
    _upsert_roundtrip(
        client,
        "manual_smoke",
        vector,
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": f"manual::{manual_pdf.stem}",
            "source_path": str(manual_pdf),
            "source_type": "manual",
            "page": None,
            "section": "manual_smoke",
            "restaurant": None,
            "dish": None,
            "text": sample,
        },
    )


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_QDRANT") != "1",
    reason="Set RUN_E2E_QDRANT=1 to run the source-by-source Qdrant smoke test.",
)
def test_code_source_qdrant_roundtrip():
    from src.ingestion.galactic_code_ingestion import _parse_code_text
    from src.ingestion.shared import split_text

    embedder = _get_embedder()
    client, _ = _make_qdrant_collection(embedder, "code_smoke")

    code_pdf = next(Path("Dataset/knowledge_base/codice_galattico").glob("*.pdf"))
    text = _parse_code_text(str(code_pdf))
    sample = split_text(text, max_chars=1200)[0] if text else code_pdf.name
    vector = embedder.embed(sample)
    _upsert_roundtrip(
        client,
        "code_smoke",
        vector,
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": f"code::{code_pdf.stem}",
            "source_path": str(code_pdf),
            "source_type": "codice",
            "page": None,
            "section": "code_smoke",
            "restaurant": None,
            "dish": None,
            "text": sample,
        },
    )


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_QDRANT") != "1",
    reason="Set RUN_E2E_QDRANT=1 to run the source-by-source Qdrant smoke test.",
)
def test_blog_source_qdrant_roundtrip():
    from src.ingestion.blog_ingestion import _parse_blog_sections
    from src.ingestion.shared import split_text

    embedder = _get_embedder()
    client, _ = _make_qdrant_collection(embedder, "blog_smoke")

    blog_file = next(p for p in Path("Dataset/knowledge_base/blogpost").rglob("*") if p.is_file())
    text = "\n\n".join(section_text for _, section_text in _parse_blog_sections(str(blog_file)))
    sample = split_text(text, max_chars=1200)[0] if text else blog_file.name
    vector = embedder.embed(sample)
    _upsert_roundtrip(
        client,
        "blog_smoke",
        vector,
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": f"blog::{blog_file.stem}",
            "source_path": str(blog_file),
            "source_type": "blog",
            "page": None,
            "section": "blog_smoke",
            "restaurant": None,
            "dish": None,
            "text": sample,
        },
    )


@pytest.mark.skipif(
    os.environ.get("RUN_E2E_QDRANT") != "1",
    reason="Set RUN_E2E_QDRANT=1 to run the source-by-source Qdrant smoke test.",
)
def test_distances_source_duckdb_roundtrip():
    import duckdb
    import tempfile

    from src.ingestion.distances_ingestion import ingest_distances
    from src.ingestion.runner import FACTS_SCHEMA_SQL

    with tempfile.TemporaryDirectory(dir="C:/tmp") as tmp_dir:
        con = duckdb.connect(str(Path(tmp_dir) / "facts.db"))
        con.execute(FACTS_SCHEMA_SQL)
        csv_path = Path("Dataset/knowledge_base/misc/Distanze.csv")
        count = ingest_distances(con, csv_path=csv_path)
        assert count > 0
        rows = con.execute("SELECT COUNT(*) FROM planet_distances").fetchone()[0]
        assert rows > 0
        con.close()
