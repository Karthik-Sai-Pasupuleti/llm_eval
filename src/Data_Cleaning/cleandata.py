#!/usr/bin/env python3
"""
Filter the dataset to retain only a specific list of manually selected columns.
"""

import os
import pandas as pd

# --- Configuration ---
INPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_final.csv"
OUTPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_selected_features.csv"

# The exact list of columns to retain
COLUMNS_TO_KEEP = [
    "initial_timestamp",
    "window_id",
    "video",
    "Annotator_1",
    "Annotator_2",
    "Annotator_1_Notes",
    "Annotator_2_Notes",
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

def filter_dataset(input_csv: str, output_csv: str, columns_to_keep: list):
    # Read the original input dataset
    df = pd.read_csv(input_csv)
    
    # Verify which columns actually exist in the dataframe to prevent KeyErrors
    actual_columns = [col for col in columns_to_keep if col in df.columns]
    
    missing_columns = set(columns_to_keep) - set(actual_columns)
    if missing_columns:
        print(f"Warning: The following requested columns were not found in the input CSV: {missing_columns}")
    
    # Filter the dataframe
    filtered_df = df[actual_columns]
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Successfully saved filtered dataset with {len(actual_columns)} columns to:")
    print(output_csv)

if __name__ == "__main__":
    filter_dataset(INPUT_CSV, OUTPUT_CSV, COLUMNS_TO_KEEP)
