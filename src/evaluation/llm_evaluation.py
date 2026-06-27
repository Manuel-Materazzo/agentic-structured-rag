"""
llm_evaluation.py — Confronta le risposte generate (da run_inference.py) con la ground truth usando un LLM.
Legge 'inference_results.csv' e produce 'evaluated_results.csv'.
"""
import csv
import logging
from pathlib import Path

from ingestion.structured_extraction import parse_json_response
from app.config import LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

log = logging.getLogger(__name__)

DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("output")
INPUT_CSV = OUTPUT_DIR / "inference_results.csv"
OUTPUT_CSV = OUTPUT_DIR / "evaluated_results.csv"


def get_llm_client():
    from datapizza.clients.openai_like import OpenAILikeClient
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for evaluation.")
    return OpenAILikeClient(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL)


def evaluate_answers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not INPUT_CSV.exists():
        log.error(f"File di input non trovato: {INPUT_CSV}. Lancia prima run_inference.py")
        return

    llm = get_llm_client()
    log.info(f"Lettura dei risultati da: {INPUT_CSV}")

    system_prompt = """You are an evaluation agent for a RAG system.
You are given the Ground Truth (the correct answer) and the Predicted Answer (what the system found).
You must compare them and classify the result.
Predicted answer could contain other irrelevant data about each item, just ignore the extra details and focus on dish names.

CRITICAL RULES:
- Return ONLY a JSON object with keys: "evaluation", "reason".
- "evaluation" MUST be exactly one of: "PASS", "PARTIAL", "FAIL", "EMPTY", "ERROR".
- "PASS": The predicted answer matches the ground truth perfectly (ignoring order and extra details).
- "PARTIAL": The predicted answer contains some correct items but is missing others or contains wrong items.
- "FAIL": The predicted answer is completely wrong or totally irrelevant compared to the ground truth.
- "EMPTY": The predicted answer contains "NO RESULTS".
- "ERROR": The predicted answer contains an SQL error or exception.
- Do not add markdown or extra text.
"""

    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "row_id", "difficoltà", "domanda", "ground_truth_text",
            "predicted_text", "sql", "evaluation", "reason"
        ])

        with open(INPUT_CSV, mode='r', encoding='utf-8') as in_f:
            reader = csv.DictReader(in_f)

            for idx, row in enumerate(reader, start=1):
                row_id = row.get("row_id", idx)
                domanda = row.get("domanda", "")
                gt_text = row.get("ground_truth_text", "")
                pred_text = row.get("predicted_text", "")
                sql = row.get("sql", "")

                log.info(f"[{idx}] Evaluating row_id {row_id}...")

                # Gestione rapida dei casi vuoti o di errore senza chiamare l'LLM
                if pred_text == "NO RESULTS":
                    evaluation = "EMPTY"
                    reason = "Predicted text was empty."
                elif pred_text in ["ERROR", "EXCEPTION"] or "Error" in pred_text[:20]:
                    evaluation = "ERROR"
                    reason = pred_text
                else:
                    user_prompt = (
                        f"DOMANDA:\n{domanda}\n\n"
                        f"GROUND TRUTH (Correct Answer):\n{gt_text}\n\n"
                        f"PREDICTED ANSWER (To Evaluate):\n{pred_text}\n\n"
                        f"Classify the predicted answer."
                    )

                    try:
                        response = llm.invoke(input=user_prompt, system_prompt=system_prompt, temperature=0.0)
                        data = parse_json_response(response.text)
                        evaluation = data.get("evaluation", "FAIL")
                        reason = data.get("reason", "No reason provided.")
                    except Exception as e:
                        log.error(f"LLM Evaluation failed for row_id {row_id}: {e}")
                        evaluation = "FAIL"
                        reason = f"LLM Parsing Error: {str(e)}"

                log.info(f"-> Result: {evaluation}")

                writer.writerow([
                    row_id, row.get("difficoltà", ""), domanda, gt_text,
                    pred_text, sql, evaluation, reason
                ])

    log.info(f"✅ Valutazione completata! Risultati salvati in: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate_answers()
