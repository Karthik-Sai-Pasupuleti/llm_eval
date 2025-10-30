LLM Evaluation Pipeline — Drowsiness Classification
===================================================

This module provides a clean, reproducible pipeline to evaluate multiple LLMs (commercial and open-source) on a driver drowsiness classification task, log metrics to MLflow, and save predictions as artifacts.

What it does
------------
- Reads a CSV dataset with features and a ground-truth label.
- Invokes selected LLM(s) via LangChain using Pydantic-structured input and output.
- Writes a new CSV with predictions and any errors.
- Computes accuracy, precision, recall, and F1.
- Logs metrics, classification report, confusion matrix plot, and predictions to MLflow.

Features expected
-----------------
Canonical feature columns (case/space variations auto-mapped):
- `perclos`
- `blink_rate`
- `yawn_rate`
- `steering_entropy`
- `steering_reversal_rate`
- `std_lane_position` (maps from "standard deviation of lane position")

The label column is auto-detected among: `label`, `drowsy_level`, `drowsy level`, `drowsiness_level`, or you can pass `--label-col`.

Project layout
--------------
- `src/llm_eval/schema.py` — Pydantic models for input/output.
- `src/llm_eval/bot.py` — LLM prediction bot with structured output.
- `src/llm_eval/providers.py` — Provider/model factory (OpenAI, Azure, Anthropic, Google, Ollama).
- `src/llm_eval/prompts.py` — Prompt loading and template.
- `src/llm_eval/utils.py` — Data mapping, caching, helpers.
- `src/llm_eval/mlflow_utils.py` — MLflow metrics/artifacts helpers.
- `src/llm_eval/pipeline.py` — CLI pipeline to run evaluation.
- `configs/prompt_drowsiness.txt` — Default prompt (editable).
- `configs/models.yaml` — Example multi-model grid.

Install deps
------------
This project declares dependencies in `pyproject.toml`. Use your preferred tool (uv/pip/poetry). Example with pip:

```
pip install -e ./llm_eval
```

Environment variables
---------------------
Create a `.env` in repo root or export variables in your shell:

- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
- Anthropic: `ANTHROPIC_API_KEY`
- Google Gemini: `GOOGLE_API_KEY`
- MLflow: `MLFLOW_TRACKING_URI` (or pass `--mlflow-uri`)

Sample `.env` (edit with your keys):

```
OPENAI_API_KEY=sk-...
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
# AZURE_OPENAI_API_KEY=...
# OPENAI_API_VERSION=2024-06-01
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MLFLOW_TRACKING_URI=./.mlruns
```

Run the pipeline
----------------
Single model:

```
python -m llm_eval.pipeline \
  --data path/to/dataset.csv \
  --provider openai \
  --model gpt-4o-mini \
  --prompt llm_eval/configs/prompt_drowsiness.txt \
  --out-dir runs/outputs \
  --mlflow-experiment llm-eval
```

Multiple models via config:

```
python -m llm_eval.pipeline \
  --data path/to/dataset.csv \
  --config llm_eval/configs/models.yaml \
  --prompt llm_eval/configs/prompt_drowsiness.txt \
  --out-dir runs/outputs \
  --mlflow-experiment llm-eval
```

Outputs
-------
- `runs/outputs/*_predictions.csv` — Input rows + `prediction` and `error` columns.
- `runs/outputs/confusion_matrix.png` — Per-run confusion matrix (logged to MLflow).
- MLflow artifacts: predictions CSV, confusion matrix, classification report JSON.
- A JSON `summary_*.json` is written with per-model metrics and file paths.

Production-ready extras
-----------------------
- JSONL cache (`--cache runs/cache.jsonl`) to avoid duplicate LLM calls.
- Robust column mapping and label normalization.
- Clear separation of data schema, provider setup, and evaluation logic.
- Prompt can be swapped via `--prompt` for experimentation.

Notes
-----
- Labels are normalized to lowercase; allowed outputs: `low`, `moderate`, `high`.
- Errors during LLM calls are captured; those rows are excluded from metrics.
- If you prefer sequential secure runs, set `--limit` for a quick dry-run.
