import pandas as pd
import os
from pathlib import Path

# Define the folder and output path
csv_folder = Path("/home/karthik/Desktop/llm_eval/final_csv")
output_file = Path("/home/karthik/Desktop/llm_eval/combined_participants.csv")

# Columns to keep
columns_to_keep = [
    "initial_timestamp",
    "window_id",
    "video",
    "Annotator_1",  # Will rename to drowsiness_level
    "metric_PERCLOS",
    "metric_BlinkRate",
    "blink_duration_mean",
    "blink_duration_std",
    "blink_duration_max",
    "metric_YawnRate",
    "metric_Entropy",
    "metric_SteeringRate",
    "metric_SDLP",
    "HRV_RMSSD",
    "HRV_SD1"
]

# Get all CSV files and sort them
csv_files = sorted(csv_folder.glob("*_Data*.csv"))
print(f"Found {len(csv_files)} CSV files")

combined_dfs = []

for csv_file in csv_files:
    # Extract participant ID from filename (e.g., "A_Data_final.csv" -> "A")
    participant_id = csv_file.stem.split("_")[0]

    print(f"Processing {csv_file.name} (Participant: {participant_id})")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Keep only specified columns
    df = df[columns_to_keep].copy()

    # Rename Annotator_1 to drowsiness_level
    df = df.rename(columns={"Annotator_1": "drowsiness_level"})

    # Add participant column
    df.insert(0, "participant", participant_id)

    combined_dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(combined_dfs, ignore_index=True)

# Save to CSV
combined_df.to_csv(output_file, index=False)

print(f"\n✓ Combined CSV created: {output_file}")
print(f"Total rows: {len(combined_df)}")
print(f"Columns: {list(combined_df.columns)}")
print(f"\nFirst few rows:")
print(combined_df.head())
