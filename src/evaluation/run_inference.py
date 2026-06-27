"""
run_inference.py — Esegue l'SQLAgent sulle domande del dataset e salva i risultati in CSV.
legge "domande_con_risposte.csv" e produce "inference_results.csv"
"""
import csv
import logging
from pathlib import Path

from evaluation.generate_kaggle_submission_file import get_dish_mapping
from ingestion.knowledge_manager import KnowledgeManager
from app.agents.sql_agent import SQLAgent
from utils.normalizer_utils import extract_dishes_from_rows

log = logging.getLogger(__name__)

DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("output")
INPUT_CSV = DATASET_DIR / "domande_con_risposte.csv"
OUTPUT_CSV = OUTPUT_DIR / "inference_results.csv"


def start_inference():
    dish_mappings: dict[str, int] = get_dish_mapping()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not INPUT_CSV.exists():
        log.error(f"File di input non trovato: {INPUT_CSV}")
        return

    log.info(f"Lettura del dataset da: {INPUT_CSV}")

    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "row_id", "difficoltà", "domanda", "ground_truth_text", "ground_truth_ids",
            "predicted_sql", "predicted_ids", "predicted_text"
        ])

        with KnowledgeManager.create() as knowledge_manager:
            log.info("✅ Data store initialized and available.")
            agent = SQLAgent(knowledge_manager=knowledge_manager)

            with open(INPUT_CSV, mode='r', encoding='utf-8') as in_f:
                reader = csv.DictReader(in_f)

                for idx, row in enumerate(reader, start=1):
                    row_id = row.get("row_id", idx)
                    difficolta = row.get("difficoltà", "")
                    domanda = row.get("domanda", "")
                    gt_text = row.get("risposta", "").lower()
                    gt_ids = row.get("result", "")

                    log.info(f"[{idx}] Processing row_id {row_id} ({difficolta}): {domanda[:50]}...")

                    try:
                        result = agent.execute(domanda)

                        if result.error:
                            log.warning(f"Query fallita per row_id {row_id}: {result.error}")
                            predicted_ids = ""
                            predicted_text = result.error
                        elif not result.rows:
                            log.info(f"Nessun risultato trovato per row_id {row_id}.")
                            predicted_ids = ""
                            predicted_text = "NO RESULTS"
                        else:
                            text_rows = [" | ".join(str(item) for item in r) for r in result.rows]
                            predicted_text = " \n ".join(text_rows)

                            # Extract ids by Fuzzy Matching to avoid typos
                            matches = extract_dishes_from_rows(result.rows, dish_mappings)
                            ids = [str(item[1]) for item in matches if item is not None]
                            predicted_ids = ",".join(ids)

                            log.info(f"-> Trovati {len(ids)} risultati.")

                    except Exception as e:
                        log.error(f"Errore inaspettato per row_id {row_id}: {e}")
                        predicted_ids = "EXCEPTION"
                        predicted_text = str(e)

                    writer.writerow([
                        row_id, difficolta, domanda, gt_text, gt_ids,
                        result.sql, predicted_ids, predicted_text
                    ])

    log.info(f"✅ Inferenza completata! Risultati salvati in: {OUTPUT_CSV}")


if __name__ == "__main__":
    start_inference()
