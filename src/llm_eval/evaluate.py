#!/usr/bin/env python3
"""
MLflow evaluation pipeline for Driver Drowsiness Detection using LLM-based bots.
Evaluates multiple models across consistent test samples and logs metrics to MLflow.
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


def predict_drowsiness(df: pd.DataFrame, bot: BaseBot) -> tuple[list, list, list]:
    """Get LLM predictions for each driver sample (3 runs per row for consistency)."""
    preds = {1: [], 2: [], 3: []}
    reasoning = {1: [], 2: [], 3: []}

    for _, row in df.iterrows():
        input_data = Bot.Input(
            window_id=row["window_id"],
            perclos=row["perclos"],
            blink_rate=row["eye_blink_rate"],
            blink_duration=row["blink_duration"],
            yawning_rate=row["yawning_rate"],
            sdlp=row["std_lane_position"],
            steering_entropy=row["steering_entropy"],
            steering_reversal_rate=row["steering_reversal_rate"],
        )

        for run_id in tqdm(range(1, 4), desc="Predicting drowsiness"):
            try:
                result = bot.invoke(input_data)
                level = result["drowsiness_level"]
                generated_reasoning = result["reasoning"]
                preds[run_id].append(level)
                reasoning[run_id].append(generated_reasoning)

                print(f"Run {run_id} Prediction: {level}")
            except Exception as e:
                print(f"Run {run_id} Prediction failed: {e}")
                preds[run_id].append(None)
                reasoning[run_id].append(None)

    return preds[1], preds[2], preds[3], reasoning[1], reasoning[2], reasoning[3]


def evaluate_predictions(df: pd.DataFrame, model_name: str):
    """Evaluate and log performance metrics for a given model."""
    y_true = df["label"]
    metrics_summary = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    os.makedirs("artifacts", exist_ok=True)

    for run_id in range(1, 4):
        y_pred = df[f"predicted_drowsiness_run{run_id}"]

        # Keep only valid numeric predictions
        valid_idx = [i for i, v in enumerate(y_pred) if isinstance(v, (int, float))]
        y_true_valid = y_true.iloc[valid_idx]
        y_pred_valid = [y_pred[i] for i in valid_idx]

        if not y_pred_valid:
            print(f"No valid predictions for Run {run_id}, skipping.")
            continue

        acc = accuracy_score(y_true_valid, y_pred_valid)
        prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)

        metrics_summary["accuracy"].append(acc)
        metrics_summary["precision"].append(prec)
        metrics_summary["recall"].append(rec)
        metrics_summary["f1"].append(f1)

        mlflow.log_metrics({
            f"accuracy_run{run_id}": acc,
            f"precision_run{run_id}": prec,
            f"recall_run{run_id}": rec,
            f"f1_run{run_id}": f1,
        })

        report_path = f"artifacts/{model_name}_run{run_id}_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_true_valid, y_pred_valid))
        mlflow.log_artifact(report_path)

        print(f"{model_name} | Run {run_id}")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1-Score:  {f1:.3f}")

    # Log averages
    if all(metrics_summary["accuracy"]):
        mlflow.log_metrics({
            "accuracy_mean": sum(metrics_summary["accuracy"]) / len(metrics_summary["accuracy"]),
            "precision_mean": sum(metrics_summary["precision"]) / len(metrics_summary["precision"]),
            "recall_mean": sum(metrics_summary["recall"]) / len(metrics_summary["recall"]),
            "f1_mean": sum(metrics_summary["f1"]) / len(metrics_summary["f1"]),
        })


def run_evaluation(csv_path: str, model_config_path: str, prompt_path: str):
    """Evaluate multiple models using MLflow tracking."""
    df = load_driver_data(csv_path)
    models = load_model_configs(model_config_path)
    prompt_template = toml.load(prompt_path)["driver_prompt"]["prompt"]

    for model in tqdm(models, desc="Evaluating models"):

        try:
            model_name = model["name"]
            print(f"\nEvaluating {model_name} ({model['provider']})")

            mlflow.set_experiment(model_name)
            df_ = df.copy()  # For quick testing; remove head() for full eval

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "temperature": model.get("temperature", 0.0),
                })

                config = Bot.BotConfig(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    prompt_template=prompt_template,
                    temperature=model.get("temperature", 0.0),
                )
                bot = BaseBot(config)

                preds1, preds2, preds3, reasoning1, reasoning2, reasoning3 = predict_drowsiness(df, bot)

                df_["predicted_drowsiness_run1"] = preds1
                df_["predicted_drowsiness_run2"] = preds2
                df_["predicted_drowsiness_run3"] = preds3
                df_["reasoning_run1"] = reasoning1
                df_["reasoning_run2"] = reasoning2
                df_["reasoning_run3"] = reasoning3

                evaluate_predictions(df_, model_name)

                csv_out = f"artifacts/{model_name}_predictions.csv"
                df_.to_csv(csv_out, index=False)
                mlflow.log_artifact(csv_out)

                print(f"Completed evaluation for {model_name}")
            
        except Exception as e:
            print(f"Evaluation failed for {model['name']}: {e}")


if __name__ == "__main__":
    run_evaluation(
        csv_path=r"C:\Users\pasupuleti\Desktop\group-project\experiments\drowsiness_data\drowsiness_dataset_cleaned.csv",
        model_config_path=r"C:\Users\pasupuleti\Desktop\group-project\experiments\llm_eval\src\configs\model_config.toml",
        prompt_path=r"C:\Users\pasupuleti\Desktop\group-project\experiments\llm_eval\src\configs\prompt.toml",
    )
