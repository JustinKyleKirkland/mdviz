"""
Clustering analysis module for mdviz

Provides various clustering algorithms for trajectory analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import warnings

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not found. Clustering functionality will be limited.")


class ClusterAnalyzer:
    """
    Clustering analysis for trajectory data.

    Supports multiple clustering algorithms:
    - K-means clustering
    - DBSCAN (density-based)
    - Agglomerative hierarchical clustering
    """

    def __init__(self, data: np.ndarray, feature_names: Optional[list] = None):
        """
        Initialize cluster analyzer.

        Parameters
        ----------
        data : np.ndarray
            Input data for clustering (n_samples, n_features)
        feature_names : list, optional
            Names of the features (e.g., ['PC1', 'PC2'])
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for clustering analysis")

        self.data = data
        self.feature_names = feature_names or [
            f"Feature_{i}" for i in range(data.shape[1])
        ]
        self.scaled_data = None
        self.cluster_labels = None
        self.cluster_model = None
        self.cluster_metrics = {}

    def preprocess_data(self, scale: bool = True) -> np.ndarray:
        """
        Preprocess data for clustering.

        Parameters
        ----------
        scale : bool, optional
            Whether to standardize the data (default: True)

        Returns
        -------
        np.ndarray
            Preprocessed data
        """
        if scale:
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(self.data)
            self.scaler = scaler
            print("Data standardized for clustering")
        else:
            self.scaled_data = self.data.copy()
            self.scaler = None

        return self.scaled_data

    def kmeans_clustering(
        self, n_clusters: int, random_state: int = 42, **kwargs
    ) -> np.ndarray:
        """
        Perform K-means clustering.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        random_state : int, optional
            Random state for reproducibility (default: 42)
        **kwargs
            Additional parameters for KMeans

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        if self.scaled_data is None:
            self.preprocess_data()

        self.cluster_model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
        )

        self.cluster_labels = self.cluster_model.fit_predict(self.scaled_data)
        self._calculate_metrics()

        print(f"K-means clustering completed with {n_clusters} clusters")
        print(f"Silhouette score: {self.cluster_metrics.get('silhouette', 'N/A'):.3f}")

        return self.cluster_labels

    def dbscan_clustering(
        self, eps: float = 0.5, min_samples: int = 5, **kwargs
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering.

        Parameters
        ----------
        eps : float, optional
            Maximum distance between samples (default: 0.5)
        min_samples : int, optional
            Minimum samples in a neighborhood (default: 5)
        **kwargs
            Additional parameters for DBSCAN

        Returns
        -------
        np.ndarray
            Cluster labels (-1 for noise points)
        """
        if self.scaled_data is None:
            self.preprocess_data()

        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.cluster_labels = self.cluster_model.fit_predict(self.scaled_data)

        n_clusters = len(set(self.cluster_labels)) - (
            1 if -1 in self.cluster_labels else 0
        )
        n_noise = list(self.cluster_labels).count(-1)

        self._calculate_metrics()

        print("DBSCAN clustering completed")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        if n_clusters > 1:
            print(
                f"Silhouette score: {self.cluster_metrics.get('silhouette', 'N/A'):.3f}"
            )

        return self.cluster_labels

    def hierarchical_clustering(
        self, n_clusters: int, linkage: str = "ward", **kwargs
    ) -> np.ndarray:
        """
        Perform agglomerative hierarchical clustering.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        linkage : str, optional
            Linkage criterion ('ward', 'complete', 'average', 'single')
        **kwargs
            Additional parameters for AgglomerativeClustering

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        if self.scaled_data is None:
            self.preprocess_data()

        self.cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, **kwargs
        )

        self.cluster_labels = self.cluster_model.fit_predict(self.scaled_data)
        self._calculate_metrics()

        print(f"Hierarchical clustering completed with {n_clusters} clusters")
        print(f"Silhouette score: {self.cluster_metrics.get('silhouette', 'N/A'):.3f}")

        return self.cluster_labels

    def find_optimal_k(
        self, k_range: range = range(2, 11), method: str = "silhouette"
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using various metrics.

        Parameters
        ----------
        k_range : range, optional
            Range of k values to test (default: 2-10)
        method : str, optional
            Method to use ('silhouette', 'calinski_harabasz', 'inertia')

        Returns
        -------
        tuple
            (optimal_k, scores_dict)
        """
        if self.scaled_data is None:
            self.preprocess_data()

        scores = {
            "k_values": list(k_range),
            "silhouette": [],
            "calinski_harabasz": [],
            "inertia": [],
        }

        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)

            # Calculate metrics
            if len(set(labels)) > 1:  # Need more than 1 cluster for silhouette score
                sil_score = silhouette_score(self.scaled_data, labels)
                ch_score = calinski_harabasz_score(self.scaled_data, labels)
            else:
                sil_score = -1
                ch_score = 0

            scores["silhouette"].append(sil_score)
            scores["calinski_harabasz"].append(ch_score)
            scores["inertia"].append(kmeans.inertia_)

        # Find optimal k based on method
        if method == "silhouette":
            optimal_idx = np.argmax(scores["silhouette"])
        elif method == "calinski_harabasz":
            optimal_idx = np.argmax(scores["calinski_harabasz"])
        elif method == "inertia":
            # For inertia, we look for the "elbow" - this is simplified
            inertias = scores["inertia"]
            diffs = np.diff(inertias)
            optimal_idx = np.argmax(diffs) + 1  # Simplified elbow method
        else:
            raise ValueError(f"Unknown method: {method}")

        optimal_k = scores["k_values"][optimal_idx]

        print(f"Optimal k found: {optimal_k} (method: {method})")

        return optimal_k, scores

    def _calculate_metrics(self):
        """Calculate clustering quality metrics."""
        if self.cluster_labels is None or self.scaled_data is None:
            return

        # Only calculate if we have more than 1 cluster
        unique_labels = set(self.cluster_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels or len(unique_labels) > 2:
            try:
                # Remove noise points for metric calculation
                mask = self.cluster_labels != -1
                if np.sum(mask) > 0:
                    clean_data = self.scaled_data[mask]
                    clean_labels = self.cluster_labels[mask]

                    if len(set(clean_labels)) > 1:
                        self.cluster_metrics["silhouette"] = silhouette_score(
                            clean_data, clean_labels
                        )
                        self.cluster_metrics["calinski_harabasz"] = (
                            calinski_harabasz_score(clean_data, clean_labels)
                        )
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Returns
        -------
        pd.DataFrame
            Summary statistics by cluster
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available")

        # Create DataFrame
        df = pd.DataFrame(self.data, columns=self.feature_names)
        df["cluster"] = self.cluster_labels

        # Summary statistics
        summary = (
            df.groupby("cluster")
            .agg(
                {
                    col: ["mean", "std", "min", "max", "count"]
                    for col in self.feature_names
                }
            )
            .round(3)
        )

        return summary

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers (for K-means).

        Returns
        -------
        np.ndarray or None
            Cluster centers if available
        """
        if hasattr(self.cluster_model, "cluster_centers_"):
            # Transform back to original scale if data was scaled
            centers = self.cluster_model.cluster_centers_
            if self.scaler is not None:
                centers = self.scaler.inverse_transform(centers)
            return centers
        return None

    def get_representative_frames(self, n_representatives: int = 1) -> Dict[int, list]:
        """
        Get representative frames for each cluster.

        Parameters
        ----------
        n_representatives : int, optional
            Number of representative frames per cluster (default: 1)

        Returns
        -------
        dict
            Dictionary mapping cluster_id -> list of frame indices
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available")

        representatives = {}

        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get points in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.scaled_data[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_data) == 0:
                continue

            # Find points closest to cluster center
            if hasattr(self.cluster_model, "cluster_centers_"):
                # K-means: use actual cluster center
                center = self.cluster_model.cluster_centers_[cluster_id]
            else:
                # Other methods: use centroid
                center = np.mean(cluster_data, axis=0)

            # Calculate distances to center
            distances = np.linalg.norm(cluster_data - center, axis=1)

            # Get n_representatives closest points
            closest_indices = np.argsort(distances)[:n_representatives]
            representatives[cluster_id] = cluster_indices[closest_indices].tolist()

        return representatives
