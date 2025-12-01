
# demo_agglomerative.py

# Demonstration script for the new Agglomerative Clustering module.
# It tests the algorithm on a 'difficult' dataset to show robustness
# compared to standard approaches.


from __future__ import annotations

import os
import sys
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core import base
from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"
K_VALUES_TO_TEST = [2, 3, 4, 5]

def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python demo_agglomerative.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    print(f"Processing dataset: {base}")
    print(f"Features: {feature_cols}")
    print("-" * 60)

    # 2. Define the experiment
    # We will try a specific k that is likely suitable for the difficult dataset.
    # Assuming the difficult dataset might have complex structures, 
    # let's try k=3 and k=4 to see how it behaves.
    

    for k in K_VALUES_TO_TEST:
        print(f"\n>>> Running Agglomerative Clustering with k={k}...")
        try:
        # 3. Run the new algorithm via the interface
            result = run_clustering(
                input_path=input_path,
                feature_cols=feature_cols,
                algorithm="agglomerative",
                k=k,
                standardise=True,
                output_path=os.path.join(
                    OUTPUT_DIR, f"{base}_agglomerative_results_k{k}.csv"),
                compute_elbow=False
            )

            print(f"Clustering with k={k} finished successfully.")
            print(f"Clustered data saved to: "
                  f"{os.path.join(
                      OUTPUT_DIR, f'{base}_agglomerative_results_k{k}.csv')}")
        
            # 4. Show Metrics
            metrics = result["metrics"]
            print(f"\nEvaluation Metrics for k={k}:")
            print(f"  Inertia: {metrics['inertia']:.2f}")
            print(f"  Silhouette Score: "
                  f"{metrics.get('silhouette', 'N/A')}")

            # 5. Save the plot
            plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_plot_k{k}.png")
            if result["fig_cluster"]:
                result["fig_cluster"].savefig(plot_path, dpi=150)
                print(f"\nCluster plot for k={k} saved to: {plot_path}")
        
        except Exception as e:
            print(f"\nCRITICAL ERROR: The agglomerative module failed.")
            print(e)
            sys.exit(1)

    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main(sys.argv[1:])