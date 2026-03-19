#!/usr/bin/env python3
"""
Build the final processed CSV from raw participant data.
raw_ear, raw_mar, raw_ppg, metric_BPM are read from the .h5 file.
All other metrics and annotations are read from the CSV file.
 
EAR baseline (first 5 windows):
  - closed_threshold : min non-zero EAR (fully-closed eye floor)
  - open_threshold   : mean EAR of non-blink frames (open-eye baseline)
    Blink frames are identified as ±50ms around every local EAR minimum.
 
All ocular metrics are recalculated using open_threshold as ear_threshold.
Frames with EAR == 0 are undetected frames and treated the same as None.
"""
 
import ast
import numpy as np
import pandas as pd
import h5py
import neurokit2 as nk
 
# --- HARDCODED INPUT ---
INPUT_CSV  = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data.csv"
INPUT_H5   = "/home/vanchha/Refined_Participants_Data/02_MK_Data/session_data.h5"
OUTPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_final.csv"
 
BASELINE_WINDOWS  = 5
MIN_CONSEC_FRAMES = 0    # per specification
PPG_SR            = 100  # PPG sampling rate in Hz
 
 
# ---------------------------------------------------------------------------
# H5 loader
# ---------------------------------------------------------------------------
 
def load_raw_signals_from_h5(h5_path: str) -> pd.DataFrame:
    """
    Load raw_ear, raw_mar, raw_ppg, metric_BPM from .h5 file.
    Structure: window_N/raw_data/ear|mar|ppg|smooth_bpm
    metric_BPM stored as deduplicated array string per window.
    """
    records = []
    with h5py.File(h5_path, "r") as f:
        for window_key in f.keys():
            raw_data = f[window_key]["raw_data"]
            bpm_raw    = np.array(raw_data["smooth_bpm"])
            bpm_unique = np.unique(bpm_raw)
            bpm_unique = bpm_unique[bpm_unique > 0]
            records.append({
                "window_id":  int(window_key.replace("window_", "")),
                "raw_ear":    np.array(raw_data["ear"]),
                "raw_mar":    np.array(raw_data["mar"]),
                "raw_ppg":    np.array(raw_data["ppg"]),
                "metric_BPM": str(bpm_unique.tolist()),
            })
    return pd.DataFrame(records).sort_values("window_id").reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Baseline extraction
# ---------------------------------------------------------------------------
 
def extract_ear_thresholds(ear_arrays: list) -> tuple[float, float]:
    """
    Derive participant-specific EAR thresholds from the first BASELINE_WINDOWS windows.
 
    Steps:
    1. Find local minima (non-zero) — blink valleys.
    2. Mark ±50ms around each minimum as blink frames.
    3. closed_threshold = min non-zero EAR across all baseline frames.
    4. open_threshold   = mean EAR of non-blink, non-zero frames.
 
    Returns:
        closed_threshold : floor of a fully-closed eye
        open_threshold   : open-eye baseline — used as ear_threshold in all metrics
    """
    all_nonzero    = []
    open_eye_values = []
 
    for ear_array in ear_arrays:
        ear        = np.array(ear_array, dtype=float)
        valid_mask = ear > 0
        valid_vals = ear[valid_mask]
        if len(valid_vals) == 0:
            continue
 
        fps      = len(valid_vals) / 60.0
        half_win = max(1, round(0.05 * fps))  # 50ms in frames
 
        # Mark blink frames: ±half_win around each local minimum
        blink_mask = np.zeros(len(ear), dtype=bool)
        for i in range(1, len(ear) - 1):
            if ear[i] <= 0:
                continue
            if ear[i] < ear[i - 1] and ear[i] < ear[i + 1]:
                start = max(0, i - half_win)
                end   = min(len(ear), i + half_win + 1)
                blink_mask[start:end] = True
 
        all_nonzero.extend(valid_vals.tolist())
        open_frames = ear[valid_mask & ~blink_mask]
        open_eye_values.extend(open_frames.tolist())
 
    closed_threshold = float(np.min(all_nonzero)) if all_nonzero else 0.0
 
    if open_eye_values:
        open_threshold = float(np.mean(open_eye_values))
    else:
        open_threshold = float(np.mean(all_nonzero)) if all_nonzero else 0.0
 
    return closed_threshold, open_threshold
 
 
# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------
 
def _is_valid(ear) -> bool:
    """Frame is valid if it is not None and not zero (undetected)."""
    return ear is not None and ear > 0
 
 
def calculate_perclos(ear_values, ear_threshold: float, min_consec_frames: int) -> float:
    """
    PERCLOS: % of valid frames in contiguous runs >= min_consec_frames
    where EAR < ear_threshold.  Returns 0–100.
    Zero-valued frames are treated as undetected (excluded).
    """
    closed_frames = 0
    consec_count  = 0
 
    for ear in ear_values:
        if _is_valid(ear):
            if ear < ear_threshold:
                consec_count += 1
            else:
                if consec_count >= min_consec_frames:
                    closed_frames += consec_count
                consec_count = 0
        else:
            if consec_count >= min_consec_frames:
                closed_frames += consec_count
            consec_count = 0
 
    if consec_count >= min_consec_frames:
        closed_frames += consec_count
 
    total_frames = sum(_is_valid(e) for e in ear_values)
    if total_frames == 0:
        return 0.0
    return (closed_frames / total_frames) * 100.0
 
 
def calculate_blink_stats_all(ear_values, ear_threshold: float) -> tuple[float, float, float]:
    """
    Blink duration stats. A blink = contiguous run of valid frames where EAR < ear_threshold.
    FPS derived dynamically from valid frame count over 60s window.
    Returns (mean_s, std_s, max_s).
    """
    total_frames = sum(_is_valid(e) for e in ear_values)
    if total_frames == 0:
        return 0.0, 0.0, 0.0
 
    fps             = total_frames / 60.0
    blink_durations = []
    consec_count    = 0
 
    for ear in ear_values:
        if _is_valid(ear):
            if ear < ear_threshold:
                consec_count += 1
            else:
                if consec_count > 0:
                    blink_durations.append(consec_count / fps)
                    consec_count = 0
        else:
            if consec_count > 0:
                blink_durations.append(consec_count / fps)
                consec_count = 0
 
    if consec_count > 0:
        blink_durations.append(consec_count / fps)
 
    if not blink_durations:
        return 0.0, 0.0, 0.0
 
    return (
        float(np.mean(blink_durations)),
        float(np.std(blink_durations)),
        float(np.max(blink_durations)),
    )
 
 
def calculate_blink_rate(ear_values, ear_threshold: float) -> float:
    """
    Blink events per minute.
    A blink event starts when EAR drops below ear_threshold (from above).
    Zero / undetected frames reset the in_blink state.
    Window is 60s so blink_count == blinks per minute directly.
    """
    total_frames = sum(_is_valid(e) for e in ear_values)
    if total_frames == 0:
        return 0.0
 
    blink_count = 0
    in_blink    = False
 
    for ear in ear_values:
        if not _is_valid(ear):
            in_blink = False
        elif ear < ear_threshold and not in_blink:
            blink_count += 1
            in_blink = True
        elif ear >= ear_threshold:
            in_blink = False
 
    return float(blink_count)
 
 
def calculate_hrv_features(ppg_array, sampling_rate: int = PPG_SR) -> dict:
    """Compute all neurokit2 HRV features from raw PPG. Returns {} on failure."""
    try:
        signals, _ = nk.ppg_process(ppg_array, sampling_rate=sampling_rate)
        hrv = nk.hrv(signals, sampling_rate=sampling_rate, show=False)
        return {col: float(hrv[col].iloc[0]) for col in hrv.columns}
    except Exception as e:
        print(f"  HRV failed: {e}")
        return {}
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def build_final_csv(input_csv: str, input_h5: str, output_csv: str):
    df = pd.read_csv(input_csv)
 
    # Drop raw signal columns from CSV — all come from h5
    drop_cols = ["raw_ear", "raw_mar", "raw_ppg", "metric_BPM"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
 
    # Load raw signals from h5 and merge
    h5_df = load_raw_signals_from_h5(input_h5)
    df    = df.merge(h5_df, on="window_id", how="left")
    print(f"Loaded {len(df)} windows after merge.")
 
    # drive_duration: elapsed seconds since session start
    df["drive_duration"] = df["initial_timestamp"] - df["initial_timestamp"].iloc[0]
 
    # --- Participant-specific baselines from first BASELINE_WINDOWS windows ---
    baseline_rows = df.iloc[:BASELINE_WINDOWS]
 
    closed_threshold, open_threshold = extract_ear_thresholds(baseline_rows["raw_ear"].tolist())
 
    mar_baseline = float(np.mean(np.concatenate(
        [np.array(x, dtype=float) for x in baseline_rows["raw_mar"]]
    )))
    bpm_baseline = float(np.mean([
        np.mean(ast.literal_eval(v)) for v in baseline_rows["metric_BPM"]
    ]))
    ppg_baseline_mean = float(np.mean(np.concatenate(
        [np.array(x, dtype=float) for x in baseline_rows["raw_ppg"]]
    )))
 
    # Midpoint between fully-closed floor and open-eye baseline
    ear_threshold = (closed_threshold + open_threshold) / 2.0
 
    print(
        f"Baselines -> EAR closed: {closed_threshold:.4f} | EAR open: {open_threshold:.4f} | "
        f"EAR threshold (mid): {ear_threshold:.4f} | "
        f"MAR neutral: {mar_baseline:.4f} | BPM neutral: {bpm_baseline:.2f} | "
        f"PPG mean: {ppg_baseline_mean:.4f}"
    )
 
    # --- Recalculate ocular metrics ---
    df["metric_PERCLOS"] = df["raw_ear"].apply(
        lambda x: calculate_perclos(x, ear_threshold, MIN_CONSEC_FRAMES)
    )
    df[["blink_duration_mean", "blink_duration_std", "blink_duration_max"]] = df["raw_ear"].apply(
        lambda x: pd.Series(calculate_blink_stats_all(x, ear_threshold))
    )
    df["metric_BlinkRate"] = df["raw_ear"].apply(
        lambda x: calculate_blink_rate(x, ear_threshold)
    )
 
    # --- HRV from raw_ppg ---
    print("Computing HRV features (this may take a moment)...")
    hrv_rows = df["raw_ppg"].apply(
        lambda x: pd.Series(calculate_hrv_features(np.array(x, dtype=float)))
    ).fillna(np.nan)
    hrv_cols = list(hrv_rows.columns)
    for col in hrv_cols:
        df[col] = hrv_rows[col]
 
    # --- Final column ordering (raw signals kept as columns 20–22) ---
    base_cols = [
        "initial_timestamp", "window_id", "video", "participant_id",
        "Annotator_1", "Annotator_2", "Annotator_1_Notes", "Annotator_2_Notes",
        "drive_duration",
        "metric_PERCLOS", "metric_BlinkRate",
        "blink_duration_mean", "blink_duration_std", "blink_duration_max",
        "metric_YawnRate", "metric_Entropy", "metric_SteeringRate", "metric_SDLP",
        "metric_BPM",
        "raw_ear", "raw_mar", "raw_ppg",
    ]
    final_cols = [c for c in base_cols if c in df.columns] + hrv_cols
    df = df[final_cols]
 
    df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}  |  shape: {df.shape}  |  HRV features: {len(hrv_cols)}")
 
 
if __name__ == "__main__":
    build_final_csv(INPUT_CSV, INPUT_H5, OUTPUT_CSV)
 
 