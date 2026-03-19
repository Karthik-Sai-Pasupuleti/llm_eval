#!/usr/bin/env python3
"""
VLM evaluation pipeline for driver drowsiness video windows.
Cycles through all participants under a root directory.
One MLflow experiment per model, covering all participants sequentially.
Sends a single collaged image per window to the VLM for ambiguity and facial behaviour detection.
Individual frames are saved separately to MLflow for human observer review.
"""

import os
import re
import cv2
import json
import toml
import base64
import mlflow
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from ollama import Client as OllamaClient

warnings.filterwarnings("ignore")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

VLM_PROMPT = """You are analyzing a contact sheet of 60 frames (arranged in a grid, left to right, top to bottom) from a 60-second driver-facing camera window. Each cell is one second apart.

Your task is to detect the following ONLY if clearly present:
1. Visual ambiguities: occlusions, sunglasses, hand covering mouth, hand covering face, poor lighting, motion blur, partial face obstruction.
2. Facial behaviour: yawning, head nodding, head tilting to one side, eyes closing, microsleep, prolonged eye closure.

Each frame is labelled with its frame_id in the top-left corner. When reporting detections, include the frame_ids where they occur.

Respond strictly in this JSON format:
{
  "ambiguities_detected": true or false,
  "ambiguity_frame_ids": [list of integer frame_ids where ambiguities are present, empty if none],
  "ambiguity_types": ["list of detected ambiguity types, empty if none"],
  "facial_behaviour_detected": true or false,
  "facial_behaviour_frame_ids": [list of integer frame_ids where facial behaviours are present, empty if none],
  "facial_behaviours": ["list of detected facial behaviours, empty if none"],
  "description": "brief one or two sentence summary of what is observed overall",
  "confidence": "low, medium, or high"
}

Only report what is clearly visible. Do not hallucinate. If nothing detected, set both flags to false and use empty lists."""


def extract_frames(video_path: str, num_frames: int = 60) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def center_crop_half(
    frame: np.ndarray,
    width_fraction: float = 0.5,
    height_fraction: float = 0.5,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    new_w = int(w * width_fraction)
    new_h = int(h * height_fraction)
    left = max(0, min((w - new_w) // 2 + offset_x, w - new_w))
    top = max(0, min((h - new_h) // 2 + offset_y, h - new_h))
    return frame[top:top + new_h, left:left + new_w]


def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")


def build_contact_sheet(
    frames: list[np.ndarray],
    cols: int = 10,
    thumb_width: int = 160,
) -> np.ndarray:
    thumbs = []
    for frame_id, f in enumerate(frames):
        h, w = f.shape[:2]
        scale = thumb_width / w
        thumb = cv2.resize(f, (thumb_width, int(h * scale)))
        cv2.putText(
            thumb, str(frame_id), (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        thumbs.append(thumb)

    rows = (len(thumbs) + cols - 1) // cols
    thumb_h, thumb_w = thumbs[0].shape[:2]
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(thumbs) % cols != 0:
        thumbs.append(blank)

    grid_rows = [np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid_rows)


def save_individual_frames(
    frames: list[np.ndarray],
    out_dir: str,
    highlight_frame_ids: set = None,
):
    highlight_frame_ids = highlight_frame_ids or set()
    os.makedirs(out_dir, exist_ok=True)

    for frame_id, frame in enumerate(frames):
        img = frame.copy()

        if frame_id in highlight_frame_ids:
            cv2.rectangle(
                img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 4
            )

        cv2.putText(
            img, f"frame_{frame_id}", (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        filename = f"frame_{str(frame_id).zfill(3)}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), img)


def parse_vlm_response(raw: str) -> dict:
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {
        "ambiguities_detected": None,
        "ambiguity_frame_ids": [],
        "ambiguity_types": [],
        "facial_behaviour_detected": None,
        "facial_behaviour_frame_ids": [],
        "facial_behaviours": [],
        "description": raw.strip(),
        "confidence": "low",
        "parse_error": True,
    }


def query_vlm(
    model_id: str,
    frames: list[np.ndarray],
    temperature: float = 0.1,
    cols: int = 10,
    thumb_width: int = 160,
) -> dict:
    client = OllamaClient(host=OLLAMA_HOST)

    collage = build_contact_sheet(frames, cols=cols, thumb_width=thumb_width)
    collage_b64 = frame_to_base64(collage)

    response = client.chat(
        model=model_id,
        options={"temperature": temperature},
        messages=[
            {
                "role": "user",
                "content": VLM_PROMPT,
                "images": [collage_b64],
            }
        ],
    )
    raw = response["message"]["content"]
    return parse_vlm_response(raw)


def load_model_configs(config_path: str) -> list[dict]:
    return toml.load(config_path)["models"]


def load_windows_from_csv(csv_path: str, video_dir: str) -> list[dict]:
    df = pd.read_csv(csv_path)

    if "video" not in df.columns:
        raise ValueError("CSV does not contain a 'video' column.")

    video_dir = Path(video_dir)
    extensions = [".mp4", ".avi", ".mkv", ".mov"]

    windows = []
    for _, row in df.iterrows():
        video_stem = str(row["video"])
        video_path = None

        for ext in extensions:
            candidate = video_dir / (video_stem + ext)
            if candidate.exists():
                video_path = str(candidate)
                break

        if video_path is None:
            candidate = video_dir / video_stem
            if candidate.exists():
                video_path = str(candidate)

        windows.append({
            "window_id": row["window_id"] if "window_id" in row.index else video_stem,
            "video_name": video_stem,
            "video_path": video_path,
            "drowsiness_level": row["drowsiness_level"] if "drowsiness_level" in row.index else (row["label"] if "label" in row.index else None),
        })

    found = sum(1 for w in windows if w["video_path"] is not None)
    missing = len(windows) - found
    print(f"  Matched {found}/{len(windows)} videos. {missing} missing.")

    return windows


def evaluate_participant(
    participant_id: str,
    windows: list[dict],
    model_id: str,
    model_name: str,
    out_dir: str,
    frames_base_dir: str,
    num_frames: int,
    temperature: float,
    contact_sheet_cols: int,
    thumb_width: int,
    crop_offset_x: int,
    crop_offset_y: int,
    save_frames: bool,
) -> list[dict]:
    """
    Run VLM inference on all windows for one participant.
    Logs per-participant metrics and artifacts to the active MLflow run.
    Returns list of result dicts.
    """
    results = []

    for window in tqdm(windows, desc=f"  Windows [{participant_id}]", leave=False):
        if window["video_path"] is None:
            results.append({
                "participant_id": participant_id,
                "window_id": window["window_id"],
                "video_name": window["video_name"],
                "drowsiness_level": window["drowsiness_level"],
                "ambiguities_detected": None,
                "ambiguity_frame_ids": [],
                "ambiguity_types": [],
                "facial_behaviour_detected": None,
                "facial_behaviour_frame_ids": [],
                "facial_behaviours": [],
                "description": None,
                "confidence": None,
                "error": "Video file not found",
            })
            continue

        try:
            frames = extract_frames(window["video_path"], num_frames=num_frames)
            if not frames:
                raise RuntimeError("No frames extracted.")

            cropped_frames = [
                center_crop_half(f, offset_x=crop_offset_x, offset_y=crop_offset_y)
                for f in frames
            ]

            output = query_vlm(
                model_id=model_id,
                frames=cropped_frames,
                temperature=temperature,
                cols=contact_sheet_cols,
                thumb_width=thumb_width,
            )
            output["participant_id"] = participant_id
            output["window_id"] = window["window_id"]
            output["video_name"] = window["video_name"]
            output["drowsiness_level"] = window["drowsiness_level"]
            output["error"] = None

            if save_frames:
                highlight_ids = set(
                    output.get("ambiguity_frame_ids", []) +
                    output.get("facial_behaviour_frame_ids", [])
                )
                window_frames_dir = os.path.join(
                    frames_base_dir,
                    participant_id,
                    f"window_{window['window_id']}"
                )
                save_individual_frames(
                    cropped_frames,
                    window_frames_dir,
                    highlight_frame_ids=highlight_ids,
                )
                mlflow.log_artifacts(
                    window_frames_dir,
                    artifact_path=f"frames/{participant_id}/window_{window['window_id']}"
                )

        except Exception as e:
            output = {
                "participant_id": participant_id,
                "window_id": window["window_id"],
                "video_name": window["video_name"],
                "drowsiness_level": window["drowsiness_level"],
                "ambiguities_detected": None,
                "ambiguity_frame_ids": [],
                "ambiguity_types": [],
                "facial_behaviour_detected": None,
                "facial_behaviour_frame_ids": [],
                "facial_behaviours": [],
                "description": None,
                "confidence": None,
                "error": str(e),
            }

        results.append(output)

    # Log per-participant metrics to active MLflow run
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    amb = sum(1 for r in results if r.get("ambiguities_detected") is True)
    face = sum(1 for r in results if r.get("facial_behaviour_detected") is True)
    processed = total - errors

    mlflow.log_metrics({
        f"{participant_id}_total_windows": total,
        f"{participant_id}_errors": errors,
        f"{participant_id}_windows_with_ambiguities": amb,
        f"{participant_id}_windows_with_facial_behaviour": face,
        f"{participant_id}_ambiguity_rate": amb / max(processed, 1),
        f"{participant_id}_facial_behaviour_rate": face / max(processed, 1),
    })

    # Save per-participant JSON artifact
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_{participant_id}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    mlflow.log_artifact(out_path)

    print(f"  [{participant_id}] {total} windows, {errors} errors, "
          f"{amb} with ambiguities, {face} with facial behaviour.")

    return results


def log_overall_metrics(all_results: list[dict], model_name: str, out_dir: str):
    """Aggregate metrics across all participants and log as overall_ metrics."""
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    amb = sum(1 for r in all_results if r.get("ambiguities_detected") is True)
    face = sum(1 for r in all_results if r.get("facial_behaviour_detected") is True)
    processed = total - errors

    mlflow.log_metrics({
        "overall_total_windows": total,
        "overall_errors": errors,
        "overall_windows_with_ambiguities": amb,
        "overall_windows_with_facial_behaviour": face,
        "overall_ambiguity_rate": amb / max(processed, 1),
        "overall_facial_behaviour_rate": face / max(processed, 1),
    })

    print(f"\nOverall: {total} windows, {errors} errors, "
          f"{amb} with ambiguities, {face} with facial behaviour.")

    # Save combined JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(out_dir, f"{model_name}_all_participants_{timestamp}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    mlflow.log_artifact(combined_path)


def dataset_directories(root_dir: str) -> list[str]:
    """Recursively find all CSV files ending with 'Data.csv' under root_dir."""
    dataset_list = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("Data.csv"):
                dataset_list.append(os.path.join(subdir, file))
    return sorted(dataset_list)


def participant_id_from_path(csv_path: str) -> str:
    """Extract participant identifier from CSV path.
    e.g. '.../01_V_Data/V_Data.csv' -> '01_V'
    """
    folder = os.path.basename(os.path.dirname(csv_path))
    return folder.replace("_Data", "")


def video_dir_from_csv_path(csv_path: str) -> str:
    """Infer video directory from CSV path.
    Looks for a 'video' or 'videos' subdirectory next to the CSV.
    """
    parent = Path(csv_path).parent
    for name in ["video", "videos"]:
        candidate = parent / name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"No video/videos directory found next to {csv_path}")


def run_vlm_evaluation(
    root_dir: str,
    model_config_path: str,
    num_frames: int = 60,
    out_dir: str = "vlm_outputs",
    mlflow_experiment_prefix: str = "vlm_eval",
    save_frames: bool = True,
    contact_sheet_cols: int = 10,
    thumb_width: int = 160,
    crop_offset_x: int = 0,
    crop_offset_y: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    models = load_model_configs(model_config_path)

    csv_paths = dataset_directories(root_dir)
    participants = [(participant_id_from_path(p), p) for p in csv_paths]

    print("Found participants:")
    for pid, path in participants:
        print(f"  {pid}: {path}")

    existing_experiments = {
        exp.name: exp.experiment_id for exp in mlflow.search_experiments()
    }

    for model in tqdm(models, desc="Evaluating VLMs"):
        model_name = f"{model['name']}_vlm"
        experiment_name = f"{mlflow_experiment_prefix}_{model_name}"

        print(f"\nChecking {experiment_name}...")
        if experiment_name in existing_experiments:
            print(f"Experiment '{experiment_name}' already exists — skipping.")
            continue

        print(f"Evaluating {experiment_name} ({model['model_id']}) across {len(participants)} participants")
        mlflow.set_experiment(experiment_name)

        frames_base_dir = os.path.join(out_dir, f"{model_name}_frames")
        all_results = []

        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_params({
                "model_id": model["model_id"],
                "temperature": model.get("temperature", 0.1),
                "num_frames_per_window": num_frames,
                "num_participants": len(participants),
                "root_dir": root_dir,
                "save_frames": save_frames,
                "crop_offset_x": crop_offset_x,
                "crop_offset_y": crop_offset_y,
            })

            for participant_id, csv_path in tqdm(participants, desc="Participants", leave=False):
                print(f"\n  --- Participant: {participant_id} ---")
                try:
                    video_dir = video_dir_from_csv_path(csv_path)
                    windows = load_windows_from_csv(csv_path, video_dir)

                    results = evaluate_participant(
                        participant_id=participant_id,
                        windows=windows,
                        model_id=model["model_id"],
                        model_name=model_name,
                        out_dir=out_dir,
                        frames_base_dir=frames_base_dir,
                        num_frames=num_frames,
                        temperature=model.get("temperature", 0.1),
                        contact_sheet_cols=contact_sheet_cols,
                        thumb_width=thumb_width,
                        crop_offset_x=crop_offset_x,
                        crop_offset_y=crop_offset_y,
                        save_frames=save_frames,
                    )
                    all_results.extend(results)

                except Exception as e:
                    print(f"  Failed for participant {participant_id}: {e}")

            if all_results:
                log_overall_metrics(all_results, model_name, out_dir)

        print(f"\nCompleted evaluation for {experiment_name}")


if __name__ == "__main__":
    run_vlm_evaluation(
        root_dir="/home/vanchha/Refined_Participants_Data",
        model_config_path="src/configs/vlm_model_config.toml",
        num_frames=60,
        out_dir="vlm_outputs",
        mlflow_experiment_prefix="vlm_eval",
        save_frames=True,
        contact_sheet_cols=10,
        thumb_width=160,
        crop_offset_x=0,
        crop_offset_y=0,
    )
