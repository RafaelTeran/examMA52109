## 5) Create a new module for the package cluster_maker, called `agglomerative.py`,
## that uses the agglomerative algorithm from scikit-learn, 
## sklearn.cluster.AgglomerativeClustering, to perform robust hierarchical clustering.
## The new module must integrate well with the rest of the package, following the same
## design principles and coding style. Once the new module is ready, write a script,
## called `demo_agglomerative.py`, inside the directory `demo/`, that proves the
## effectiveness of the new module on the dataset `difficult_dataset.csv`, available
## inside the directory `data/`. Your implementation should be robust, clearly 
## structured, and produce sensible clustering.
## [20]

from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Agglomerative Hierarchical Clustering using scikit-learn.

    Note: Agglomerative clustering does not produce centroids naturally.
    This function calculates the centroids as the mean of the points 
    in each cluster to maintain compatibility with the rest of the package.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    k : int
        The number of clusters to find.
    linkage : str, default 'ward'
        Which linkage criterion to use. The linkage criterion determines which 
        distance to use between sets of observation. The algorithm will merge 
        the pairs of cluster that minimize this criterion.
        - 'ward' minimizes the variance of the clusters being merged.
        - 'complete' or 'maximum' linkage uses the maximum distances between 
          all observations of the two sets.
        - 'average' uses the average of the distances of each observation of 
          the two sets.
        - 'single' uses the minimum of the distances between all observations 
          of the two sets.
    random_state : int, optional
        Not used by AgglomerativeClustering, but included for API consistency
        with other algorithms in this package.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
    centroids : ndarray of shape (k, n_features)
        Calculated centroids of the clusters.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    if k < 1:
        raise ValueError("k must be at least 1.")

    # 1. Initialize and Fit the model
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)

    # 2. Calculate Centroids manually for compatibility
    # (AgglomerativeClustering doesn't provide them)
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))

    for cluster_id in range(k):
        # Filter points belonging to this cluster
        mask = labels == cluster_id
        if np.any(mask):
            # Calculate mean position
            centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            # Handle edge case of empty cluster (unlikely in Agglomerative)
            # We assign it to a random point to avoid NaN
            # In a robust implementation, this might raise a warning.
            centroids[cluster_id] = X[np.random.choice(X.shape[0])]

    return labels, centroids