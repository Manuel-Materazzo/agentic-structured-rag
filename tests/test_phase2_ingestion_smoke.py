from __future__ import annotations

import tempfile
from pathlib import Path


def test_manual_taxonomy_extraction_smoke():
    from src.ingestion.cook_manual_ingestion import _extract_taxonomy

    rows = _extract_taxonomy(
        "Forno Quantico  | Tecnica di cottura | Licenza B\n"
        "Linea di servizio - Categoria di supporto - Licenza C"
    )
    assert rows
    assert rows[0][0]
    assert rows[0][1]


def test_code_compliance_extraction_smoke():
    from src.ingestion.galactic_code_ingestion import _extract_compliance_rules

    rows = _extract_compliance_rules(
        "È vietato superare il limite massimo di 200g per il sale.\n"
        "Consentito solo con licenza A."
    )
    assert rows
    assert rows[0][0] == "norm_rule"


def test_blog_sectionization_smoke():
    from src.ingestion.blog_ingestion import _parse_blog_sections

    with tempfile.TemporaryDirectory(dir="C:/tmp") as tmp_dir:
        html = Path(tmp_dir) / "sample.html"
        html.write_text(
            "<html><body><h1>Titolo</h1><p>Primo paragrafo.</p><h2>Sotto</h2><p>Secondo paragrafo.</p></body></html>",
            encoding="utf-8",
        )
        sections = _parse_blog_sections(str(html))
    assert sections
    assert sections[0][0]
    assert sections[0][1]
