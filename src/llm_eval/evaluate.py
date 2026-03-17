#!/usr/bin/env python3
"""
MLflow evaluation pipeline for Driver Drowsiness Detection using LLM-based bots.
Supports single-pass or multi-pass (consistent multi-run) evaluation with flag control.
"""

import os
import toml
import mlflow
import warnings
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
    """
    preds = {i: [] for i in range(1, num_runs + 1)} if multi_run else {1: []}
    reasoning = {i: [] for i in range(1, num_runs + 1)} if multi_run else {1: []}

    for _, row in tqdm(df.iterrows(), desc="Predicting drowsiness", total=len(df)):
        input_data = Bot.Input(
            window_id=row["window_id"],
            perclos=row["perclos"],
            blink_rate=row["eye_blink_rate"],
            blink_duration_mean= row["blink_duration_mean"],
            blink_duration_max= row["blink_duration_max"],
            yawning_rate=row["yawning_rate"],
            sdlp=row["std_lane_position"],
            steering_entropy=row["steering_entropy"],
            steering_reversal_rate=row["steering_reversal_rate"],
            bpm=float(row["bpm"]) if "bpm" in row and pd.notna(row["bpm"]) else None,
            hrv_sdnn=float(row["hrv_sdnn"]) if "hrv_sdnn" in row and pd.notna(row["hrv_sdnn"]) else None,
            hrv_rmssd=float(row["hrv_rmssd"]) if "hrv_rmssd" in row and pd.notna(row["hrv_rmssd"]) else None,
            hrv_sd1=float(row["hrv_sd1"]) if "hrv_sd1" in row and pd.notna(row["hrv_sd1"]) else None,
            hrv_hf=float(row["hrv_hf"]) if "hrv_hf" in row and pd.notna(row["hrv_hf"]) else None,
            hrv_wavelet_entropy=float(row["hrv_wavelet_entropy"]) if "hrv_wavelet_entropy" in row and pd.notna(row["hrv_wavelet_entropy"]) else None,
            hrv_lfhf=float(row["hrv_lfhf"]) if "hrv_lfhf" in row and pd.notna(row["hrv_lfhf"]) else None,
        )

        if multi_run:
            for run_id in range(1, num_runs + 1):
                try:
                    result = bot.invoke(input_data)
                    preds[run_id].append(result["drowsiness_level"])
                    reasoning[run_id].append(result["reasoning"])
                except Exception as e:
                    print(f"Run {run_id} failed: {e}")
                    preds[run_id].append(None)
                    reasoning[run_id].append(None)
        else:
            try:
                result = bot.invoke(input_data)
                preds[1].append(result["drowsiness_level"])
                reasoning[1].append(result["reasoning"])
            except Exception as e:
                print(f"Prediction failed: {e}")
                preds[1].append(None)
                reasoning[1].append(None)

    return preds, reasoning

def evaluate_predictions(
    df: pd.DataFrame,
    model_name: str,
    participant_id: str,
) -> dict:
    """
    Evaluate predictions for one participant and log metrics prefixed by participant_id.
    Returns a dict of {col: {accuracy, precision, recall, f1}} for later aggregation.
    """
    y_true = df["drowsiness_level"]

    run_cols = [col for col in df.columns if col.startswith("predicted_drowsiness_run")]
    if not run_cols:
        run_cols = ["predicted_drowsiness"]

    os.makedirs("artifacts", exist_ok=True)
    metric_sets = {}

    for col in run_cols:
        y_pred = df[col]
        valid_idx = [i for i, v in enumerate(y_pred) if isinstance(v, (int, float))]
        y_true_valid = y_true.iloc[valid_idx]
        y_pred_valid = [y_pred[i] for i in valid_idx]

        if not y_pred_valid:
            print(f"  No valid predictions for {participant_id}/{col}, skipping.")
            continue

        acc = accuracy_score(y_true_valid, y_pred_valid)
        prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)

        metric_sets[col] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        # Log per-participant metrics with participant prefix
        mlflow.log_metrics({
            f"{participant_id}_{col}_accuracy": acc,
            f"{participant_id}_{col}_precision": prec,
            f"{participant_id}_{col}_recall": rec,
            f"{participant_id}_{col}_f1": f1,
        })

        report_path = f"artifacts/{model_name}_{participant_id}_{col}_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_true_valid, y_pred_valid))
        mlflow.log_artifact(report_path)

        print(f"  [{participant_id}] Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    return metric_sets


def log_overall_metrics(all_metric_sets: list[dict]):
    """Average metrics across all participants and log as overall_ metrics."""
    combined: dict[str, list] = {}
    for ms in all_metric_sets:
        for col, metrics in ms.items():
            for k, v in metrics.items():
                combined.setdefault(f"overall_{col}_{k}", []).append(v)

    averaged = {key: sum(vals) / len(vals) for key, vals in combined.items()}
    mlflow.log_metrics(averaged)

    print("\nOverall metrics across all participants:")
    for k, v in averaged.items():
        print(f"  {k}: {v:.3f}")


def run_evaluation(
    participants: list[tuple[str, str]],
    model_config_path: str,
    prompt_path: str,
    multi_run: bool = False,
    num_runs: int = 4,
    enable_history_: bool = True,
    history_limit_: int = 10,
):
    """
    One MLflow experiment per model, covering all participants sequentially.
    History is cleared between participants; data is sorted by window_id within each.
    Logs per-participant metrics and overall aggregate metrics.
    """
    models = load_model_configs(model_config_path)
    prompt_template = toml.load(prompt_path)["driver_prompt"]["prompt"]
    existing_experiments = {exp.name: exp.experiment_id for exp in mlflow.search_experiments()}

    for model in tqdm(models, desc="Evaluating models"):
        experiment_name = model["name"]
        try:
            if experiment_name in existing_experiments:
                print(f"\nExperiment '{experiment_name}' already exists — skipping.")
                continue

            print(f"\nEvaluating {experiment_name} ({model['provider']}) across {len(participants)} participants")
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=experiment_name):
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

                for participant_id, csv_path in tqdm(participants, desc="Participants", leave=False):
                    print(f"\n  --- Participant: {participant_id} ---")
                    try:
                        df = load_driver_data(csv_path)
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

                        ms = evaluate_predictions(df, experiment_name, participant_id)
                        all_metric_sets.append(ms)
                        all_dfs.append(df)

                        # Save per-participant artifact
                        os.makedirs("artifacts", exist_ok=True)
                        csv_out = f"artifacts/{experiment_name}_{participant_id}_predictions.csv"
                        df.to_csv(csv_out, index=False)
                        mlflow.log_artifact(csv_out)

                    except Exception as e:
                        print(f"  Failed for participant {participant_id}: {e}")

                # Log overall metrics across all participants
                if all_metric_sets:
                    log_overall_metrics(all_metric_sets)

                # Save combined CSV of all participants
                if all_dfs:
                    os.makedirs("artifacts", exist_ok=True)
                    combined_csv = f"artifacts/{experiment_name}_all_participants.csv"
                    pd.concat(all_dfs, ignore_index=True).to_csv(combined_csv, index=False)
                    mlflow.log_artifact(combined_csv)

                print(f"\nCompleted evaluation for {experiment_name}")

        except Exception as e:
            print(f"Evaluation failed for {experiment_name}: {e}")


def dataset_directories(root_dir: str) -> list[str]:
    """Recursively find all CSV files ending with 'Data.csv' under root_dir."""
    dataset_list = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("Data.csv"):
                dataset_list.append(os.path.join(subdir, file))
    return dataset_list


def participant_id_from_path(csv_path: str) -> str:
    """Extract participant identifier from the CSV path.

    E.g. '.../01_V_Data/V_Data.csv' → '01_V'
    """
    folder = os.path.basename(os.path.dirname(csv_path))
    return folder.replace("_Data", "")


if __name__ == "__main__":

    csv_paths = dataset_directories("/home/karthik/Desktop/llm_eval/Refined_Participants_Data")
    participants = [(participant_id_from_path(p), p) for p in sorted(csv_paths)]

    print("Found participants:")
    for pid, path in participants:
        print(f"  {pid}: {path}")

    run_evaluation(
        participants=participants,
        model_config_path=r"src/configs/model_config.toml",
        prompt_path=r"src/configs/prompt.toml",
        multi_run=False,   # Set False for single-pass mode
        num_runs=4,        # Used only if multi_run=True
        enable_history_=True,
        history_limit_=10,
    )

