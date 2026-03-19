#!/usr/bin/env python3
"""
PCA analysis on the final processed CSV.
Plots 2D visualization colored by Annotator_1 drowsiness label,
and prints the most important features per principal component.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- HARDCODED INPUT ---
INPUT_CSV = "/home/karthik/Desktop/llm_eval/Refined_Participants_Data/01_V_Data/V_Data_final.csv"
OUTPUT_PLOT = "pca_2d_visualization.png"
OUTPUT_LOADINGS = "pca_loadings.csv"

NUMERIC_FEATURES = [
    "metric_PERCLOS", "metric_BlinkRate",
    "blink_duration_mean", "blink_duration_std", "blink_duration_max",
    "metric_YawnRate", "metric_Entropy", "metric_SteeringRate",
    "metric_SDLP", "metric_BPM",
]
LABEL_COL = "Annotator_1"
TOP_N_FEATURES = 3  # top features to print per PC


def run_pca(input_csv: str):
    df = pd.read_csv(input_csv)

    # Drop rows with missing features or label
    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    df = df.dropna(subset=features + [LABEL_COL])

    X = df[features].values
    y = df[LABEL_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # --- Explained variance ---
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    print("\nExplained variance per PC:")
    for i, (ev, cv) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {ev*100:.2f}%  (cumulative: {cv*100:.2f}%)")

    # --- Loadings: most important features per PC ---
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )
    loadings.to_csv(OUTPUT_LOADINGS)
    print(f"\nLoadings saved to: {OUTPUT_LOADINGS}")

    print(f"\nTop {TOP_N_FEATURES} features per principal component:")
    for col in loadings.columns:
        top = loadings[col].abs().nlargest(TOP_N_FEATURES)
        print(f"  {col}: {', '.join(top.index.tolist())}")

    # --- 2D visualization (PC1 vs PC2) ---
    labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for label, color in zip(labels, colors):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=str(label), color=color, alpha=0.7, edgecolors="k", linewidths=0.4, s=60)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.set_title("PCA 2D Visualization — Drowsiness Labels")
    ax.legend(title=LABEL_COL)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"\n2D plot saved to: {OUTPUT_PLOT}")

    # --- Scree plot ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(range(1, len(explained) + 1), explained * 100, color="steelblue", alpha=0.8)
    ax2.plot(range(1, len(explained) + 1), cumulative * 100, color="red", marker="o", label="Cumulative")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Scree Plot")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("pca_scree_plot.png", dpi=150)
    plt.close()
    print("Scree plot saved to: pca_scree_plot.png")


if __name__ == "__main__":
    run_pca(INPUT_CSV)
