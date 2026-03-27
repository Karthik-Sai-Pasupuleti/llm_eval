#!/usr/bin/env python3
"""
MLflow evaluation pipeline for Driver Drowsiness Detection using LLM-based bots.
Supports single-pass or multi-pass (consistent multi-run) evaluation with flag control.
"""

import os
import toml
import mlflow
import warnings
import tempfile
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

from utils import load_driver_data
from bot import BaseBot, Bot

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_model_configs(config_path: str) -> list[dict]:
    """Load model configurations from TOML file."""
    return toml.load(config_path)["models"]

def predict_drowsiness(
    df: pd.DataFrame,
    bot: BaseBot,
    multi_run: bool = False,
    num_runs: int = 4,
) -> tuple[dict, dict]:
    """
    Get LLM predictions for each driver sample.
    - If multi_run=True → perform multiple (num_runs) predictions per sample.
    - Otherwise → single prediction per sample.
    Handles both old and combined_participants.csv column naming conventions.
    """
    preds = {i: [] for i in range(1, num_runs + 1)} if multi_run else {1: []}
    reasoning = {i: [] for i in range(1, num_runs + 1)} if multi_run else {1: []}

    # Map column names to handle both conventions
    perclos_col = "metric_PERCLOS" if "metric_PERCLOS" in df.columns else "perclos"
    blink_rate_col = "metric_BlinkRate" if "metric_BlinkRate" in df.columns else "eye_blink_rate"
    yawning_col = "metric_YawnRate" if "metric_YawnRate" in df.columns else "yawning_rate"
    sdlp_col = "metric_SDLP" if "metric_SDLP" in df.columns else "std_lane_position"
    entropy_col = "metric_Entropy" if "metric_Entropy" in df.columns else "steering_entropy"
    steer_rate_col = "metric_SteeringRate" if "metric_SteeringRate" in df.columns else "steering_reversal_rate"
    hrv_rmssd_col = "HRV_RMSSD" if "HRV_RMSSD" in df.columns else "hrv_rmssd"
    hrv_sd1_col = "HRV_SD1" if "HRV_SD1" in df.columns else "hrv_sd1"

    for _, row in tqdm(df.iterrows(), desc="Predicting drowsiness", total=len(df)):
        input_data = Bot.Input(
            window_id=int(row["window_id"]),
            perclos=float(row[perclos_col]),
            blink_rate=float(row[blink_rate_col]),
            blink_duration_mean=float(row["blink_duration_mean"]),
            blink_duration_max=float(row["blink_duration_max"]),
            yawning_rate=float(row[yawning_col]),
            sdlp=float(row[sdlp_col]),
            steering_entropy=float(row[entropy_col]),
            steering_reversal_rate=float(row[steer_rate_col]),
            hrv_rmssd=float(row[hrv_rmssd_col]),
            hrv_sd1=float(row[hrv_sd1_col]),
        )

        iterations = range(1, num_runs + 1) if multi_run else range(1, 2)
        
        for run_id in iterations:
            try:
                result = bot.invoke(input_data)
                preds[run_id].append(result["drowsiness_level"])
                reasoning[run_id].append(result["reasoning"])
            except Exception as e:
                print(f"Run {run_id} failed for window {row.get('window_id')}: {e}")
                preds[run_id].append(None)
                reasoning[run_id].append(None)

    return preds, reasoning

def evaluate_predictions(
    df: pd.DataFrame,
    participant_id: str,
    artifact_dir: str,
) -> dict:
    """
    Evaluate predictions for one participant and log metrics.
    Returns a dict of {col: {accuracy, precision, recall, f1}} for later aggregation.
    Converts string labels (Low/Moderate/High) to numeric (1/2/3) if needed.
    Artifacts are written to artifact_dir (caller manages lifecycle).
    """
    y_true = df["drowsiness_level"]

    # Convert string labels to numeric if needed
    label_map = {"Low": 1, "Moderate": 2, "High": 3}
    if y_true.dtype == "object":
        y_true = y_true.map(label_map)

    run_cols = [col for col in df.columns if col.startswith("predicted_drowsiness_run")]
    if not run_cols:
        run_cols = ["predicted_drowsiness"]

    metric_sets = {}

    for col in run_cols:
        y_pred = df[col].reset_index(drop=True)
        valid_idx = [i for i, v in enumerate(y_pred) if isinstance(v, (int, float))]
        y_true_valid = y_true.iloc[valid_idx]
        y_pred_valid = y_pred.iloc[valid_idx]

        if y_pred_valid.empty:
            print(f"  No valid predictions for {participant_id}/{col}, skipping.")
            continue

        acc = accuracy_score(y_true_valid, y_pred_valid)
        prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)

        metric_sets[col] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        mlflow.log_metrics({
            f"{col}_accuracy": acc,
            f"{col}_precision": prec,
            f"{col}_recall": rec,
            f"{col}_f1": f1,
        })

        report_path = os.path.join(artifact_dir, f"{participant_id}_{col}_report.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_true_valid, y_pred_valid))
        mlflow.log_artifact(report_path)

        print(f"  [{participant_id}] Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    return metric_sets


def log_overall_metrics(all_metric_sets: list[dict], run_id: str):
    """Average metrics across all participants and log to the given run_id."""
    combined: dict[str, list] = {}
    for ms in all_metric_sets:
        for col, metrics in ms.items():
            for k, v in metrics.items():
                combined.setdefault(f"overall_{col}_{k}", []).append(v)

    averaged = {key: sum(vals) / len(vals) for key, vals in combined.items()}
    # Log explicitly to the model run — in MLflow 3.x the active-run context
    # is not reliably restored to the parent after nested runs exit.
    client = mlflow.MlflowClient()
    for key, value in averaged.items():
        client.log_metric(run_id, key, value)

    print("\nOverall metrics across all participants:")
    for k, v in averaged.items():
        print(f"  {k}: {v:.3f}")


def run_evaluation(
    participants: list[tuple[str, pd.DataFrame]],
    model_config_path: str,
    prompt_path: str,
    multi_run: bool = False,
    num_runs: int = 4,
    enable_history_: bool = True,
    history_limit_: int = 10,
    experiment_name: str = "Evaluations",
    prompt_name: str = "Prompt 1",
):
    """
    Groups runs under a single MLflow experiment ("Evaluations").
    Each model run sits directly under the experiment, with the prompt name and template saved as tags.
    History is cleared between participants; data is sorted by window_id within each.
    Logs per-participant metrics and overall aggregate metrics.
    """
    models = load_model_configs(model_config_path)
    prompt_template = toml.load(prompt_path)["driver_prompt"]["prompt"]

    # MLflow 3.x requires mlruns/models to exist for the file-store model registry
    # even when the model registry is not used. Pre-creating it avoids
    # the "invalid parent directory mlruns/models" error on set_experiment().
    mlruns_root = os.path.abspath("mlruns")
    os.makedirs(os.path.join(mlruns_root, "models"), exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_root}")

    mlflow.set_experiment(experiment_name)

    for model in tqdm(models, desc="Evaluating models"):
        model_run_name = model["name"]
        try:
            print(f"\nEvaluating {model_run_name} ({model['provider']}) across {len(participants)} participants")
            
            with mlflow.start_run(run_name=model_run_name) as model_run:
                mlflow.set_tags({
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template
                })
                mlflow.log_params({
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "temperature": model.get("temperature", 0.0),
                    "multi_run": multi_run,
                    "num_runs": num_runs,
                    "num_participants": len(participants),
                })

                config = Bot.BotConfig(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    prompt_template=prompt_template,
                    temperature=model.get("temperature", 0.0),
                )
                # Single bot reused across all participants; history cleared between them
                bot = BaseBot(config, enable_history=enable_history_, history_limit=history_limit_)

                all_metric_sets = []
                all_dfs = []

                with tempfile.TemporaryDirectory() as artifact_dir:
                    for participant_id, df in tqdm(participants, desc="Participants", leave=False):
                        print(f"\n  --- Participant: {participant_id} ---")
                        try:
                            with mlflow.start_run(run_name=f"Participant_{participant_id}", nested=True):
                                if "window_id" in df.columns:
                                    df = df.sort_values("window_id").reset_index(drop=True)

                                # Clear history before each participant so contexts don't bleed across
                                bot.clear_history()

                                preds_dict, reasoning_dict = predict_drowsiness(df, bot, multi_run=multi_run, num_runs=num_runs)

                                df["participant_id"] = participant_id
                                if multi_run:
                                    for run_id in range(1, num_runs + 1):
                                        df[f"predicted_drowsiness_run{run_id}"] = preds_dict[run_id]
                                        df[f"reasoning_run{run_id}"] = reasoning_dict[run_id]
                                else:
                                    df["predicted_drowsiness"] = preds_dict[1]
                                    df["reasoning"] = reasoning_dict[1]

                                ms = evaluate_predictions(df, participant_id, artifact_dir)
                                all_metric_sets.append(ms)
                                all_dfs.append(df)

                                csv_out = os.path.join(artifact_dir, f"{participant_id}_predictions.csv")
                                df.to_csv(csv_out, index=False)
                                mlflow.log_artifact(csv_out)

                        except Exception as e:
                            print(f"  Failed for participant {participant_id}: {e}")

                    # Log overall metrics across all participants
                    if all_metric_sets:
                        log_overall_metrics(all_metric_sets, model_run.info.run_id)

                    # Save combined CSV of all participants
                    if all_dfs:
                        combined_csv = os.path.join(artifact_dir, f"{model_run_name}_all_participants.csv")
                        pd.concat(all_dfs, ignore_index=True).to_csv(combined_csv, index=False)
                        mlflow.MlflowClient().log_artifact(model_run.info.run_id, combined_csv)

                print(f"\nCompleted evaluation for {model_run_name}")

        except Exception as e:
            print(f"Evaluation failed for {model_run_name}: {e}")


def load_combined_participants(csv_path: str) -> list[tuple[str, pd.DataFrame]]:
    """
    Load combined_participants.csv and split by participant ID.
    Returns list of (participant_id, participant_df) tuples.
    """
    df = load_driver_data(csv_path)
    participants = []

    for participant_id in sorted(df['participant'].unique()):
        participant_df = df[df['participant'] == participant_id].copy()
        participants.append((participant_id, participant_df))

    return participants


if __name__ == "__main__":

    # Load combined participants CSV and split by participant ID
    combined_csv_path = "/home/karthik/Desktop/llm_eval/combined_participants_cleaned.csv"
    participants_list = load_combined_participants(combined_csv_path)

    print(f"Found {len(participants_list)} participants in combined CSV:")
    for pid, pdf in participants_list:
        print(f"  {pid}: {len(pdf)} records")

    run_evaluation(
        participants=participants_list,
        model_config_path=r"src/configs/model_config.toml",
        prompt_path=r"src/configs/prompt.toml",
        multi_run=False,   # Set False for single-pass mode
        num_runs=4,        # Used only if multi_run=True
        enable_history_=False,  # Start fresh, no history carryover
        history_limit_=0,
        experiment_name="Evaluations",
        prompt_name="Prompt 1"
    )
