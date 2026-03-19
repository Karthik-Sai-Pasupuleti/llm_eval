import pandas as pd
import numpy as np
import ast
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================
INPUT_CSV_PATH = "/home/vanchha/Refined_Participants_Data/01_V_Data/V_Data.csv"
OUTPUT_DIR = "/home/vanchha/Refined_Participants_Data/01_V_Data/PCA"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_and_mean_bpm(array_str):
    """Parses the stringified metric_BPM array and returns the mean."""
    try:
        if isinstance(array_str, str):
            val_list = ast.literal_eval(array_str)
            if val_list:
                return np.mean(val_list)
        elif isinstance(array_str, list) and array_str:
            return np.mean(array_str)
    except (ValueError, SyntaxError):
        pass
    return np.nan

def run_pca():
    print("=== Starting PCA Process ===")
    _ensure_dir(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[ERROR] Could not find input file: {INPUT_CSV_PATH}")
        return

    print(f"[INFO] Loading data from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Process metric_BPM into a single continuous mean value per window for PCA
    if 'metric_BPM' in df.columns:
        df['mean_BPM'] = df['metric_BPM'].apply(parse_and_mean_bpm)

    # Define the core numerical features expected
    expected_feature_cols = [
        'metric_BlinkRate', 'metric_Entropy', 'metric_PERCLOS', 'metric_SDLP', 
        'metric_SteeringRate', 'metric_YawnRate', 'mean_BPM', 'HRV_SDNN', 
        'HRV_RMSSD', 'HRV_SD1', 'HRV_HF', 'HRV_WaveletEntropy', 'HRV_LFHF'
    ]

    # Dynamically select only features that actually exist in the dataframe
    actual_feature_cols = [col for col in expected_feature_cols if col in df.columns]
    
    # Filter dataframe to keep only rows with valid features and labels
    df_clean = df.dropna(subset=actual_feature_cols + ['Annotator_1']).copy()
    
    # If a column becomes completely NaN after row filtering, drop it completely to prevent PCA failure
    df_clean = df_clean.dropna(axis=1, how='all')
    final_feature_cols = [col for col in actual_feature_cols if col in df_clean.columns]
    
    if df_clean.empty or len(final_feature_cols) < 2:
        print("[ERROR] Not enough valid features/rows available after dropping missing values.")
        return

    X = df_clean[final_feature_cols]
    y = df_clean['Annotator_1']  # Used for visualization grouping

    print("[INFO] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Performing PCA calculations...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # ---------------- Plot 1: Explained Variance ----------------
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_explained_variance.png'))
    plt.close()

    # ---------------- Plot 2: 2D PCA Visualization ----------------
    plt.figure(figsize=(10, 8))
    # Map colors for the specific textual labels
    palette_colors = {'Low': '#2ca02c', 'Moderate': '#ff7f0e', 'High': '#d62728'}
    
    # Handle cases where Annotator labels might not strictly match the palette
    unique_labels = y.unique()
    current_palette = {label: palette_colors.get(label, '#333333') for label in unique_labels}
    
    sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=y, 
        palette=current_palette, 
        alpha=0.8,
        s=100
    )
    
    # If variance ratio is populated, plot it
    if len(explained_variance) > 1:
        plt.title(f'PCA: PC1 ({explained_variance[0]:.1%}) vs PC2 ({explained_variance[1]:.1%})')
    else:
        plt.title('PCA Visualization')
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_2d_visualization.png'))
    plt.close()

    # ---------------- Plot 3: Feature Importance (Loadings) ----------------
    loadings_abs = np.abs(pca.components_)
    
    # Generate feature importance bar plots for EVERY principal component
    for i in range(loadings_abs.shape[0]):  
        pc_name = f"PC{i+1}"
        vals = loadings_abs[i, :]
        s = pd.Series(vals, index=final_feature_cols).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=s.values, y=s.index, hue=s.index, palette="rocket", legend=False)
        plt.title(f"Feature Importance ({pc_name}) - Absolute PCA Loading")
        plt.xlabel("Absolute loading magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"pca_feature_importance_{pc_name}.png"))
        plt.close()

    # ---------------- Save Data Artifacts ----------------
    num_components = min(4, X_pca.shape[1])
    df_pca_train = pd.DataFrame(X_pca[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
    df_pca_train['Annotator_1'] = y.values
    if 'window_id' in df_clean.columns:
        df_pca_train['window_id'] = df_clean['window_id'].values
        
    pca_data_path = os.path.join(OUTPUT_DIR, 'PCA_transformed_data.csv')
    df_pca_train.to_csv(pca_data_path, index=False)

    # Use the dynamic final_feature_cols instead of the static list
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])], index=final_feature_cols)
    loadings_path = os.path.join(OUTPUT_DIR, 'pca_loadings_matrix.csv')
    loadings_df.to_csv(loadings_path)

    print(f"[INFO] PCA Process Complete. Plots and CSV files saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    run_pca()
