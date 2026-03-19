#!/usr/bin/env python3
"""
Build the final processed CSV from raw participant data.
Recalculates ocular, blink, and HRV metrics using participant-specific baselines
derived from the first 5 windows.
"""

import ast
import numpy as np
import pandas as pd
import neurokit2 as nk

# --- HARDCODED INPUT ---
INPUT_CSV = "/home/vanchha/Refined_Participants_Data/01_V_Data/V_Data.csv"
OUTPUT_CSV = "/home/vanchha/Refined_Participants_Data/01_V_Data/V_adaptive_threshold.csv"
BASELINE_WINDOWS = 5
FPS = 30  # camera frames per second, used for blink duration ms conversion


# ---------------------------------------------------------------------------
# Metric calculation helpers
# ---------------------------------------------------------------------------

def extract_baseline_ear(ear_arrays: list[np.ndarray]) -> float:
    """
    From the first BASELINE_WINDOWS windows of raw_ear, find the minimum
    non-zero value across all frames. This is the closed-eye (blink) threshold.
    """
    all_values = np.concatenate(ear_arrays)
    non_zero = all_values[all_values > 0]
    return float(np.min(non_zero))


def calculate_perclos(ear_array: np.ndarray, ear_threshold: float, min_consec_frames: int = 0) -> float:
    """
    PERCLOS: fraction of frames where EAR is below threshold (eyes closed).
    min_consec_frames: minimum consecutive frames to count as closure event (0 = no filter).
    """
    if len(ear_array) == 0:
        return 0.0
    below = ear_array < ear_threshold
    if min_consec_frames <= 1:
        return float(np.mean(below))
    # count only runs >= min_consec_frames
    closed_count = 0
    run = 0
    for val in below:
        if val:
            run += 1
        else:
            if run >= min_consec_frames:
                closed_count += run
            run = 0
    if run >= min_consec_frames:
        closed_count += run
    return float(closed_count / len(ear_array))


def calculate_blink_stats_all(ear_array: np.ndarray, ear_threshold: float) -> tuple[float, float, float]:
    """
    Returns (blink_duration_mean_ms, blink_duration_std_ms, blink_duration_max_ms).
    A blink is a contiguous run of frames where EAR < ear_threshold.
    Duration is converted to ms assuming FPS.
    """
    durations = []
    run = 0
    for val in ear_array:
        if val < ear_threshold:
            run += 1
        else:
            if run > 0:
                durations.append(run)
            run = 0
    if run > 0:
        durations.append(run)

    if not durations:
        return 0.0, 0.0, 0.0

    durations_ms = [d * (1000.0 / FPS) for d in durations]
    return float(np.mean(durations_ms)), float(np.std(durations_ms)), float(np.max(durations_ms))


def calculate_blink_rate(ear_array: np.ndarray, ear_threshold: float, window_duration_s: float = 60.0) -> float:
    """
    Blink rate = number of blink events per minute.
    """
    blink_count = 0
    in_blink = False
    for val in ear_array:
        if val < ear_threshold and not in_blink:
            blink_count += 1
            in_blink = True
        elif val >= ear_threshold:
            in_blink = False
    return float(blink_count / (window_duration_s / 60.0))


def calculate_hrv_features(ppg_array: np.ndarray, sampling_rate: int = 100) -> dict:
    """
    Use neurokit2 to compute HRV features from raw PPG signal.
    Returns a flat dict of HRV metrics; returns NaNs on failure.
    """
    nan_result = {
        "hrv_sdnn": np.nan, "hrv_rmssd": np.nan, "hrv_sd1": np.nan,
        "hrv_hf": np.nan, "hrv_wavelet_entropy": np.nan, "hrv_lfhf": np.nan,
        "metric_BPM_hrv": np.nan,
    }
    try:
        signals, info = nk.ppg_process(ppg_array, sampling_rate=sampling_rate)
        hrv = nk.hrv(signals, sampling_rate=sampling_rate, show=False)
        return {
            "hrv_sdnn": float(hrv["HRV_SDNN"].iloc[0]) if "HRV_SDNN" in hrv else np.nan,
            "hrv_rmssd": float(hrv["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv else np.nan,
            "hrv_sd1": float(hrv["HRV_SD1"].iloc[0]) if "HRV_SD1" in hrv else np.nan,
            "hrv_hf": float(hrv["HRV_HF"].iloc[0]) if "HRV_HF" in hrv else np.nan,
            "hrv_wavelet_entropy": float(hrv["HRV_WE"].iloc[0]) if "HRV_WE" in hrv else np.nan,
            "hrv_lfhf": float(hrv["HRV_LFHF"].iloc[0]) if "HRV_LFHF" in hrv else np.nan,
        }
    except Exception:
        return nan_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_final_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # Parse array columns stored as strings
    for col in ["raw_ear", "raw_mar", "raw_ppg", "metric_BPM"]:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    # --- drive_duration from initial_timestamp ---
    # Each window is 60s; duration = last_timestamp - first_timestamp within window
    # initial_timestamp is the first frame UNIX timestamp of the window
    # If a dedicated last_timestamp column exists use it, else approximate from FPS
    if "last_timestamp" in df.columns:
        df["drive_duration"] = df["last_timestamp"] - df["initial_timestamp"]
    else:
        # Approximate: window duration = len(raw_ear) / FPS
        df["drive_duration"] = df["raw_ear"].apply(lambda x: len(x) / FPS)

    # --- Participant-specific baseline from first BASELINE_WINDOWS windows ---
    baseline_df = df.iloc[:BASELINE_WINDOWS]
    ear_threshold = extract_baseline_ear(baseline_df["raw_ear"].tolist())
    mar_baseline = float(np.mean(np.concatenate(baseline_df["raw_mar"].tolist())))
    bpm_baseline = float(np.mean([np.mean(arr) for arr in baseline_df["metric_BPM"].tolist()]))
    ppg_baseline_mean = float(np.mean(np.concatenate(baseline_df["raw_ppg"].tolist())))

    print(f"Baselines -> EAR threshold: {ear_threshold:.4f} | MAR neutral: {mar_baseline:.4f} | "
          f"BPM neutral: {bpm_baseline:.2f} | PPG mean: {ppg_baseline_mean:.4f}")

    # --- Recalculate ocular metrics using baseline EAR threshold ---
    df["metric_PERCLOS"] = df["raw_ear"].apply(
        lambda x: calculate_perclos(np.array(x), ear_threshold=ear_threshold, min_consec_frames=0)
    )
    df[["blink_duration_mean", "blink_duration_std", "blink_duration_max"]] = df["raw_ear"].apply(
        lambda x: pd.Series(calculate_blink_stats_all(np.array(x), ear_threshold=ear_threshold))
    )
    df["metric_BlinkRate"] = df["raw_ear"].apply(
        lambda x: calculate_blink_rate(np.array(x), ear_threshold=ear_threshold)
    )

    # --- HRV from raw_ppg via neurokit2 ---
    hrv_rows = df["raw_ppg"].apply(lambda x: pd.Series(calculate_hrv_features(np.array(x))))
    for col in hrv_rows.columns:
        df[col] = hrv_rows[col]

    # --- metric_BPM: take mean of the per-window BPM array ---
    df["metric_BPM"] = df["metric_BPM"].apply(lambda x: float(np.mean(x)))

    # --- Select and order final columns ---
    final_cols = [
        "initial_timestamp", "window_id", "video", "participant_id",
        "Annotator_1", "Annotator_2", "Annotator_1_Notes", "Annotator_2_Notes",
        "drive_duration",
        "metric_PERCLOS", "metric_BlinkRate",
        "blink_duration_mean", "blink_duration_std", "blink_duration_max",
        "metric_YawnRate", "metric_Entropy", "metric_SteeringRate", "metric_SDLP",
        "metric_BPM",
        "raw_ear", "raw_mar", "raw_ppg",
    ]
    # Keep only columns that exist in df
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    df.to_csv(output_csv, index=False)
    print(f"Final CSV saved to: {output_csv}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    build_final_csv(INPUT_CSV, OUTPUT_CSV)
