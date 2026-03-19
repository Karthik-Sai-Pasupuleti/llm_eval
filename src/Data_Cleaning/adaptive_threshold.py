#!/usr/bin/env python3
"""
Simple PCA on the final CSV metrics.
 
Outputs:
  - pca_2d.png                  : 2D scatter plot coloured by Annotator_1 drowsiness label
  - pca_variance.png            : explained variance bar chart
  - explained_variance.csv      : variance and cumulative variance per PC
  - pca_loadings_matrix.csv     : full feature vs PC loadings matrix
  - feature_importance_per_pc.csv : features ranked by absolute loading for each PC
  - pca_transformed_data.csv    : the transformed dataset including identifiers and labels
  - pc_bar_charts/              : folder containing individual bar charts for the top features of each PC
  - Console                     : top features per principal component (all PCs)
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
INPUT_CSV = "/home/vanchha/Refined_Participants_Data/02_MK_Data/MK_Data_final.csv"
OUTPUT_DIR = "/home/vanchha/Refined_Participants_Data/02_MK_Data/PCA"
 
# Columns to exclude from PCA (non-numeric, identifiers, raw signals, annotations)
EXCLUDE_COLS = [
    "initial_timestamp", "window_id", "video", "participant_id",
    "Annotator_1", "Annotator_2", "Annotator_1_Notes", "Annotator_2_Notes",
    "raw_ear", "raw_mar", "raw_ppg", "metric_BPM",
    "drive_duration",
]
 
LABEL_COL  = "Annotator_1"
N_TOP      = 10   # Increased to 10 to make the bar charts more informative
 
 
def run_pca(input_csv: str, output_dir: str):
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sub-directory for individual PC bar charts
    bar_charts_dir = os.path.join(output_dir, "pc_bar_charts")
    os.makedirs(bar_charts_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv)
 
    # Keep only rows with a valid label
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL].notna()].reset_index(drop=True)
        labels = df[LABEL_COL].astype(str)
    else:
        labels = None
 
    # Build feature matrix
    feature_df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.dropna(axis=1, how="all")
    feature_df = feature_df.fillna(feature_df.median())
 
    feature_names = feature_df.columns.tolist()
    print(f"Features used for PCA: {len(feature_names)}")
 
    X = StandardScaler().fit_transform(feature_df)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    pc_columns = [f"PC{i+1}" for i in range(len(pca.components_))]
 
    # --- 1. Export Explained Variance ---
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    exp_var_df = pd.DataFrame({
        "Principal_Component": pc_columns,
        "Explained_Variance_Ratio": pca.explained_variance_ratio_,
        "Cumulative_Variance_Ratio": cumvar
    })
    exp_var_path = os.path.join(output_dir, "explained_variance.csv")
    exp_var_df.to_csv(exp_var_path, index=False)
    print(f"Saved: {exp_var_path}")

    # --- 2. Export PCA Loadings Matrix ---
    loadings_df = pd.DataFrame(pca.components_.T, index=feature_names, columns=pc_columns)
    loadings_path = os.path.join(output_dir, "pca_loadings_matrix.csv")
    loadings_df.to_csv(loadings_path, index=True, index_label="Feature")
    print(f"Saved: {loadings_path}")

    # --- 3. Export Feature Importance Per PC ---
    imp_records = []
    for i, component in enumerate(pca.components_):
        sorted_idx = np.argsort(np.abs(component))[::-1]
        for rank, idx in enumerate(sorted_idx):
            imp_records.append({
                "PC": f"PC{i+1}",
                "Rank": rank + 1,
                "Feature": feature_names[idx],
                "Loading": component[idx],
                "Absolute_Loading": np.abs(component[idx])
            })
    feat_imp_df = pd.DataFrame(imp_records)
    feat_imp_path = os.path.join(output_dir, "feature_importance_per_pc.csv")
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"Saved: {feat_imp_path}")

    # --- 4. Export Transformed Data ---
    transformed_df = pd.DataFrame(X_pca, columns=pc_columns)
    if labels is not None:
        transformed_df[LABEL_COL] = labels.values
    
    for col in reversed(["initial_timestamp", "window_id", "participant_id", "video"]):
        if col in df.columns:
            transformed_df.insert(0, col, df[col].values)
            
    transformed_path = os.path.join(output_dir, "pca_transformed_data.csv")
    transformed_df.to_csv(transformed_path, index=False)
    print(f"Saved: {transformed_path}")
 
    # --- 2D scatter coloured by label ---
    fig, ax = plt.subplots(figsize=(10, 7))
    if labels is not None:
        for label in sorted(labels.unique()):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7, s=40)
        ax.legend(title=LABEL_COL, loc="best")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=40)
 
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA 2D Visualization")
    plt.tight_layout()
    pca_2d_path = os.path.join(output_dir, "pca_2d.png")
    plt.savefig(pca_2d_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {pca_2d_path}")
 
    # --- Explained variance bar chart ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance per PC")
    plt.tight_layout()
    pca_variance_path = os.path.join(output_dir, "pca_variance.png")
    plt.savefig(pca_variance_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {pca_variance_path}")
 
    # --- Save Individual Bar Charts for Top Features Per PC ---
    # Only plotting up to the 95% variance threshold to prevent generating too many images
    n_pcs_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"\nPCs to reach 95% variance: {n_pcs_95}")
    print(f"PC1 + PC2 cumulative variance: {cumvar[1] * 100:.1f}%")
    print(f"\nSaving top {N_TOP} feature bar charts for the first {n_pcs_95} PCs into {bar_charts_dir}/ ...")
    
    for i in range(n_pcs_95):
        component = pca.components_[i]
        top_idx = np.argsort(np.abs(component))[::-1][:N_TOP]
        
        loadings = component[top_idx]
        names = [feature_names[j] for j in top_idx]
        
        # Reverse to display the highest loading at the top of the horizontal bar chart
        names.reverse()
        loadings = loadings[::-1]
        
        # Color coding: blue for positive loadings, red for negative
        colors = ['steelblue' if val >= 0 else 'crimson' for val in loadings]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, loadings, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_xlabel("Loading Value")
        ax.set_title(f"PC{i+1} Top Features (Explains {pca.explained_variance_ratio_[i] * 100:.1f}%)")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add a vertical line at 0 for clarity
        ax.axvline(0, color='black', linewidth=1)
        
        plt.tight_layout()
        chart_path = os.path.join(bar_charts_dir, f"PC{i+1:02d}_top_features.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        
    print(f"Completed saving {n_pcs_95} bar charts.")
 
    # --- Console outputs ---
    print(f"\nTop {N_TOP} features per principal component:")
    print("=" * 60)
    for i, component in enumerate(pca.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:N_TOP]
        print(f"\nPC{i + 1}  (explains {pca.explained_variance_ratio_[i] * 100:.1f}%):")
        for j in top_idx:
            print(f"  {feature_names[j]:<45}  loading = {component[j]:+.4f}")
 
 
if __name__ == "__main__":
    run_pca(INPUT_CSV, OUTPUT_DIR)
