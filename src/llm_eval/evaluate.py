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
            blink_duration_mean=row["blink_duration_mean"],
            blink_duration_max=row["blink_duration_max"],
            yawning_rate=row["yawning_rate"],
            sdlp=row["std_lane_position"],
            steering_entropy=row["steering_entropy"],
            steering_reversal_rate=row["steering_reversal_rate"],
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

def evaluate_predictions(df: pd.DataFrame, model_name: str):
    """Evaluate and log performance metrics for one or multiple runs."""
    y_true = df["drowsiness_level"]

    # Detect prediction columns automatically
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
            print(f"No valid predictions for {col}, skipping.")
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

        # Save detailed report
        report_path = f"artifacts/{model_name}_{col}_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_true_valid, y_pred_valid))
        mlflow.log_artifact(report_path)

        print(f"\n{model_name} | {col}")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1-Score:  {f1:.3f}")

    # Log overall mean if multiple runs
    if len(metric_sets) > 1:
        avg_metrics = {
            "accuracy_mean": sum(v["accuracy"] for v in metric_sets.values()) / len(metric_sets),
            "precision_mean": sum(v["precision"] for v in metric_sets.values()) / len(metric_sets),
            "recall_mean": sum(v["recall"] for v in metric_sets.values()) / len(metric_sets),
            "f1_mean": sum(v["f1"] for v in metric_sets.values()) / len(metric_sets),
        }
        mlflow.log_metrics(avg_metrics)
        print("\nAverage metrics across runs:")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.3f}")


def run_evaluation(
    csv_path: str,
    model_config_path: str,
    prompt_path: str,
    multi_run: bool = False,
    num_runs: int = 4,
):
    """
    Evaluate multiple models using MLflow tracking.
    Supports single-pass or multi-pass evaluation.
    """
    df = load_driver_data(csv_path)
    models = load_model_configs(model_config_path)
    prompt_template = toml.load(prompt_path)["driver_prompt"]["prompt"]

    existing_experiments = {exp.name: exp.experiment_id for exp in mlflow.search_experiments()}

    for model in tqdm(models, desc="Evaluating models"):
        try:
            model_name = f'{model["name"]}_with_history'
            print(f"\nChecking {model_name}...")

            if model_name in existing_experiments:
                print(f"Experiment '{model_name}' already exists — skipping.")
                continue

            print(f"Evaluating {model_name} ({model['provider']})")

            mlflow.set_experiment(model_name)
            df_ = df.copy()

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "temperature": model.get("temperature", 0.0),
                    "multi_run": multi_run,
                    "num_runs": num_runs,
                })

                config = Bot.BotConfig(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    prompt_template=prompt_template,
                    temperature=model.get("temperature", 0.0),
                )
                bot = BaseBot(config, enable_history = True, history_limit=10)

                preds_dict, reasoning_dict = predict_drowsiness(df_, bot, multi_run=multi_run, num_runs=num_runs)

                # Assign columns dynamically
                if multi_run:
                    for run_id in range(1, num_runs + 1):
                        df_[f"predicted_drowsiness_run{run_id}"] = preds_dict[run_id]
                        df_[f"reasoning_run{run_id}"] = reasoning_dict[run_id]
                else:
                    df_["predicted_drowsiness"] = preds_dict[1]
                    df_["reasoning"] = reasoning_dict[1]

                evaluate_predictions(df_, model_name)

                csv_out = f"artifacts/{model_name}_predictions.csv"
                os.makedirs("artifacts", exist_ok=True)
                df_.to_csv(csv_out, index=False)
                mlflow.log_artifact(csv_out)

                print(f" Completed evaluation for {model_name}")

        except Exception as e:
            print(f"Evaluation failed for {model.get('name', 'Unknown')}: {e}")



if __name__ == "__main__":
    run_evaluation(
        csv_path=r"/home/karthik/Desktop/drowsiness_detection_project/drowsiness_data/balu_processed_data.csv",
        model_config_path=r"/home/karthik/Desktop/drowsiness_detection_project/llm_eval/src/configs/model_config.toml",
        prompt_path=r"/home/karthik/Desktop/drowsiness_detection_project/llm_eval/src/configs/prompt.toml",
        multi_run=False,   # Set False for single-pass mode
        num_runs=4,       # Used only if multi_run=True
    )
