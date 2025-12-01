## Mistakes in demo/cluster_plot.py

There were two different conceptual error in cluster_plot.py:

- The first problem with the original script was that it was supposed to give the inertia and silhouette metrics running a k-means algorithm with 4 different values of k, however, in the main function, run_clustering, it was given as a value for k, `k = min(k,3)`, being k = 2, 3, 4, 5 in a for loop. That made the real value of k to be `k = 3` when it was supposed to be `k = 4` and `k = 5`. The way to fix it is by just putting `k = k`, in the spot where `k = min(k,3)` was.

- The second problem with the original script was that the name given to the silhouette object in the metrics dictionay was `silhouette` not `silhouette_score`. By this error, the `if "silhouette" in metrics_df.columns:` line always was given a `False` making the silhoutte information never being plotted and printed. The way to fix this was changing the name to "silhouette".

## Summary of demo/cluster_plot.py

- First, it checks that it has been given only one argument. Then, it checks if there is a dataset that follows the path given by the argument.

- Secondly, it checks if the dataset has two numeric columns and store the first two in `feature_cols`.

- Then, for each value of k:
    - it executes `run_clustering` with the k-means algorithm, with the given dataset for values of k = 2,3,4,5. And also providing a different name to each iteration based on the base name of the dataset and the particular value of k. This function will compute the k-means algorithm to the dataset. All the information is stored in `result`.

    - With `results`, plots are created, and metrics are stored in `metrics_summary` and printed.

- Finally, a new CSV is created with all metrics, and in case of silhouette information existing on `metrics` dictionary, it creates and saves a plot for each value of k.

All the outputs are stored in a folder named "demo_output", that in case of not existing, it would be created.

## Short overview of cluster_maker package

These are the different modules in the `cluster_maker` package with a brief description:

# algorithms.py

Contains the core implementation of the K-means algorithm, including centroid initialization, point assignment, and iterative updates until convergence.

# data_analyser.py

Provides utilities for exploratory data analysis (EDA) on pandas DataFrames, such as calculating descriptive statistics and correlations.

# data_exporter.py

Handles the export of processed DataFrames to disk, supporting output formats such as CSV and formatted text files.

# dataframe_builder.py

Generates synthetic datasets based on user-defined cluster specifications (seeds), primarily used for testing and demonstrating the clustering algorithms.

# evaluation.py

Implements metrics for assessing cluster quality—such as Inertia and Silhouette scores—to assist in determining the optimal number of clusters.

# interface.py

Serves as the high-level orchestrator of the package. It integrates loading, preprocessing, clustering, evaluation, and plotting into a unified end-to-end workflow.

# plotting_clustered.py

Contains visualization functions to generate 2D scatter plots of clustered data and charts for evaluation metrics like the Elbow curve.

# preprocessing.py

Offers tools for data preparation, including feature selection and standardization (z-score normalization), ensuring data is numerically ready for processing.

