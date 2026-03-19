#!/usr/bin/env python3
"""
Filter the original dataset to retain only the top N HRV features from the first M principal components,
along with essential metadata, annotation, and specific metric/blink columns.
"""

import os
import pandas as pd

# --- Configuration ---
INPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_final.csv"
LOADINGS_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/PCA/pca_loadings_matrix.csv"
OUTPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_filtered_features.csv"

# User-defined variables
NUM_PCS = 7
TOP_N_FEATURES = 2  # Change this value to get top 1, 2, 3, etc., features per PC

# Columns that must always be retained
ALWAYS_KEEP = [
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
    "metric_SDLP"
]

def filter_top_features(input_csv: str, loadings_csv: str, output_csv: str, num_pcs: int, top_n_features: int):
    # 1. Read the PCA loadings matrix
    loadings_df = pd.read_csv(loadings_csv, index_col=0)
    
    # Filter the loadings matrix to exclude ALWAYS_KEEP features.
    # This ensures we are ONLY evaluating and selecting the top HRV features from the PCA.
    features_to_consider = [idx for idx in loadings_df.index if idx not in ALWAYS_KEEP]
    filtered_loadings_df = loadings_df.loc[features_to_consider]
    
    # 2. Extract the most important features (highest absolute loading) for the first `num_pcs` PCs
    top_features = set()
    for i in range(1, num_pcs + 1):
        pc_col = f"PC{i}"
        if pc_col in filtered_loadings_df.columns:
            # Find the top N indices (feature names) with the maximum absolute loading among HRV features
            pc_top_features = filtered_loadings_df[pc_col].abs().nlargest(top_n_features).index.tolist()
            top_features.update(pc_top_features)
        else:
            print(f"Warning: {pc_col} not found in the loadings matrix.")
    
    print(f"Identified {len(top_features)} unique top PCA (HRV) features across the first {num_pcs} PCs.")
    
    # 3. Read the original input dataset
    df = pd.read_csv(input_csv)
    
    # 4. Determine final columns to keep, checking if they exist in the input dataframe to avoid errors
    columns_to_keep = [col for col in ALWAYS_KEEP if col in df.columns]
    feature_columns = [col for col in top_features if col in df.columns]
    
    missing_keep = set(ALWAYS_KEEP) - set(columns_to_keep)
    if missing_keep:
        print(f"Notice: The following always-keep columns were not found in the input CSV: {missing_keep}")

    final_columns = columns_to_keep + feature_columns
    
    # 5. Filter and save
    filtered_df = df[final_columns]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully saved filtered dataset with {len(final_columns)} total columns to:")
    print(f"{output_csv}")
    
    print("\nTop PCA features retained:")
    for feat in sorted(feature_columns):
        print(f"  - {feat}")

if __name__ == "__main__":
    filter_top_features(INPUT_CSV, LOADINGS_CSV, OUTPUT_CSV, NUM_PCS, TOP_N_FEATURES)
