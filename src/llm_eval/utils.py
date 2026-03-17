import pandas as pd
import toml


def load_prompts(prompt_path: str):
    """Load the system and user prompts from TOML."""
    config = toml.load(prompt_path)
    system_prompt = config["driver_prompt"]["system_prompt"]
    user_prompt = config["driver_prompt"]["user_prompt"]
    return system_prompt, user_prompt

def _parse_bpm_list(val) -> float | None:
    """Parse BPM column which is stored as a stringified list, return mean."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        lst = ast.literal_eval(str(val))
        return float(sum(lst) / len(lst))
    except Exception:
        return None

def load_driver_data(csv_path: str) -> pd.DataFrame:
    """Load driver behavior dataset and rename columns to match the prompt."""
    df = pd.read_csv(csv_path)

    # Rename columns from raw names to prompt-friendly names
    rename_map = {
        "metric_PERCLOS": "perclos",
        "metric_BlinkRate": "eye_blink_rate",
        "metric_YawnRate": "yawning_rate",
        "metric_Entropy": "steering_entropy",
        "metric_SteeringRate": "steering_reversal_rate",
        "metric_SDLP": "std_lane_position",
        "metric_BPM": "bpm",
        "HRV_SDNN": "hrv_sdnn",
        "HRV_RMSSD": "hrv_rmssd",
        "HRV_SD1": "hrv_sd1",
        "HRV_HF": "hrv_hf",
        "HRV_WaveletEntropy": "hrv_wavelet_entropy",
        "HRV_LFHF": "hrv_lfhf",
    }

    df = df.rename(columns=rename_map)
    if "bpm" in df.columns:
        df["bpm"] = df["bpm"].apply(_parse_bpm_list)

    label_map = {"Low": 1, "Moderate": 2, "High": 3}
    df["drowsiness_level"] = df["Annotator_1"].map(label_map)
    
    # Optional: check expected columns
    expected_cols = [
        "perclos",
        "eye_blink_rate",
        "yawning_rate",
        "steering_entropy",
        "steering_reversal_rate",
        "std_lane_position",
        "bpm",
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_sd1",
        "hrv_hf",
        "hrv_wavelet_entropy",
        "hrv_lfhf",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    return df
