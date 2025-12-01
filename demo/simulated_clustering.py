## 4) Create a Python script called `simulated_clustering.py`, inside the directory
## `demo/`, that uses only the clustering tools provided in the cluster_maker package 
## to analyse the dataset `simulated_data.csv`, inside the directory `data/`, and
## determine a plausible separation of the data into clusters.
##
## The script’s main purpose is to be convincing: your analysis should make it clear 
## why your chosen clustering is appropriate. Include meaningful plots that reveal the
## structure of the data, using both the plotting functions in cluster_maker and any
## supporting visualisations you choose to create with matplotlib.
## [15]

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"
# We have choosen to explore k = 2 to 6
K_RANGE = range(2, 7)


def main(args: List[str]) -> None:

    print("=== Simulated Data Clustering Analysis ===\n")
    
    if len(args) != 1:
        print("Usage: python simulated_clustering.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input file: {input_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    # The CSV is assumed to have two or more data columns
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols  # Use all numeric columns
    print(f"Features selected for analysis: {feature_cols}")

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    # ---------------------------------------------------------
    # PHASE 1: EXPLORATION (Finding the best k)
    # ---------------------------------------------------------

    print("\n--- Phase 1: Exploration (determining optimal k) ---")
    
    inertias = []
    silhouettes = []
    
    for k in K_RANGE:
        print(f"Testing configuration: k={k}...", end=" ")
        
        # We run the clustering just to get metrics.
        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=None,   # We don't need to save CSVs for these tests
            compute_elbow=False # We build our own manual elbow curve below
        )
        
        # Collect metrics
        metrics = result["metrics"]
        inertia = metrics["inertia"]
        sil = metrics.get("silhouette", 0)
        
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"Done. Inertia={inertia:.2f}, Silhouette={sil:.4f}")

    # ---------------------------------------------------------
    # PHASE 2: VISUAL PROOF (Why we choose k?)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Generating Decision Plots ---")

    # Create a dual-axis plot to show Elbow and Silhouette together
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Inertia (Elbow)
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (Lower is better)', color=color)
    ax1.plot(K_RANGE, inertias, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Silhouette on secondary axis
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score (Higher is better)', color=color)
    ax2.plot(K_RANGE, silhouettes, marker='s', linestyle='--', color=color, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Evaluation of Clustering Quality: Elbow Method vs Silhouette')
    fig.tight_layout()
    
    evaluation_plot_path = os.path.join(OUTPUT_DIR, f"{base}_model_evaluation_metrics.png")
    plt.savefig(evaluation_plot_path, dpi=150)
    print(f"Saved evaluation plot to: {evaluation_plot_path}")
    plt.close()

    # ---------------------------------------------------------
    # PHASE 3: FINAL EXECUTION (The Winner)
    # ---------------------------------------------------------
    
    # Logic: Pick the k with the highest Silhouette score
    best_idx = np.argmax(silhouettes)
    best_k = K_RANGE[best_idx]
    
    print(f"\n--- Phase 3: Final Model ---")
    print(f"Based on the analysis, the optimal separation is k={best_k}.")
    print(f"(Reason: Highest silhouette score of {silhouettes[best_idx]:.4f})")
    
    print(f"Running full clustering with k={best_k}...")
    
    final_result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=best_k,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, f"{base}_final_simulated_clusters.csv"),
        compute_elbow=False
    )

    print(f"Saved final clustered data to: {os.path.join(OUTPUT_DIR, f'{base}_final_simulated_clusters.csv')}")

    # Save the final cluster plot
    final_plot_path = os.path.join(OUTPUT_DIR, f"{base}_final_cluster_structure.png")
    if final_result["fig_cluster"]:
        final_result["fig_cluster"].savefig(final_plot_path, dpi=150)
        print(f"Saved final cluster structure plot to: {final_plot_path}")
    
    print("\n=== Analysis Completed Successfully ===")
    print(f"Check the '{OUTPUT_DIR}' folder for results.")

if __name__ == "__main__":
    main(sys.argv[1:])