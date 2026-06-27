#!/bin/bash
set -e

MODE=${MODE:-api}

case "$MODE" in

  ingestion)
    echo "📖 Running ingestion pipeline..."
    python src/ingestion.py
    echo "✅ Ingestion pipeline completed."
    ;;

  api)
    echo "🚀 Starting API server..."
    exec python src/api.py
    ;;

  inference)
    echo "🔍 Running full inference pipeline..."
    python src/evaluation/run_inference.py
    python src/evaluation/generate_kaggle_submission_file.py --answers_path output/inference_results.csv
    python src/evaluation/jaccard_evaluation.py --submission output/submission.csv
    echo "✅ Inference pipeline completed."
    ;;

  evaluate)
    echo "📊 Running Jaccard evaluation only..."
    python src/evaluation/jaccard_evaluation.py --submission output/submission.csv
    echo "✅ Evaluation completed."
    ;;

  *)
    echo "❌ Unknown MODE: $MODE"
    echo "Valid modes: ingestion | api | inference | evaluate"
    exit 1
    ;;
esac