import pandas as pd
import toml


def load_prompts(prompt_path: str):
    """Load the system and user prompts from TOML."""
    config = toml.load(prompt_path)
    system_prompt = config["driver_prompt"]["system_prompt"]
    user_prompt = config["driver_prompt"]["user_prompt"]
    return system_prompt, user_prompt



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
    }

    df = df.rename(columns=rename_map)

    # Optional: check expected columns
    expected_cols = [
        "perclos",
        "eye_blink_rate",
        "yawning_rate",
        "steering_entropy",
        "steering_reversal_rate",
        "std_lane_position",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    return df
