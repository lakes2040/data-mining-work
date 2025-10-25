"""Clustering analysis module for loan approval segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


@dataclass
class ClusteringResult:
    """Container holding clustering outputs."""

    model: KMeans
    labels: np.ndarray
    metrics: Dict[str, float]
    pca_components: np.ndarray
    cluster_centers_2d: np.ndarray


def evaluate_k_values(data: pd.DataFrame, k_range: Tuple[int, int]) -> pd.DataFrame:
    """Evaluate clustering quality metrics for a range of k values."""
    start, end = k_range
    results: List[Dict[str, float]] = []
    for k in range(start, end + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(data)
        inertia = model.inertia_
        silhouette = silhouette_score(data, labels) if k > 1 else np.nan
        db_index = davies_bouldin_score(data, labels) if k > 1 else np.nan
        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
            "davies_bouldin": db_index,
        })
    return pd.DataFrame(results)


def choose_best_k(metrics_df: pd.DataFrame) -> int:
    """Select the best k based on silhouette (max) and Davies-Bouldin (min)."""
    metrics_df = metrics_df.copy()
    filtered = metrics_df.dropna(subset=["silhouette", "davies_bouldin"]).copy()
    if filtered.empty:
        return int(metrics_df["k"].iloc[0])

    # Rank metrics
    filtered["silhouette_rank"] = filtered["silhouette"].rank(ascending=False)
    filtered["davies_rank"] = filtered["davies_bouldin"].rank(ascending=True)
    filtered["combined_rank"] = filtered[["silhouette_rank", "davies_rank"]].mean(axis=1)
    best_row = filtered.sort_values("combined_rank").iloc[0]
    return int(best_row["k"])


def run_kmeans(data: pd.DataFrame, k: int | None = None) -> ClusteringResult:
    """Fit a K-Means model and project results for visualization."""
    if k is None:
        metrics_df = evaluate_k_values(data, (2, 10))
        k = choose_best_k(metrics_df)
    else:
        metrics_df = evaluate_k_values(data, (k, k))

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(data)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(data)
    centers_2d = pca.transform(model.cluster_centers_)

    final_metrics = {
        "chosen_k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(data, labels) if k > 1 else np.nan,
        "davies_bouldin": davies_bouldin_score(data, labels) if k > 1 else np.nan,
    }

    return ClusteringResult(
        model=model,
        labels=labels,
        metrics=final_metrics,
        pca_components=components,
        cluster_centers_2d=centers_2d,
    )


def plot_clusters(result: ClusteringResult, output_path: str | None = None) -> None:
    """Visualize clustered points and centers in 2D space."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        result.pca_components[:, 0],
        result.pca_components[:, 1],
        c=result.labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.scatter(
        result.cluster_centers_2d[:, 0],
        result.cluster_centers_2d[:, 1],
        c="black",
        marker="x",
        s=150,
        label="Cluster Centers",
    )
    plt.title("K-Means Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def main() -> None:
    """Example usage running clustering on preprocessed data."""
    from data_preprocessing import preprocess_data

    artifacts = preprocess_data("loan_approval.csv")
    result = run_kmeans(artifacts.cleaned_data)
    print("Clustering metrics:", result.metrics)
    plot_clusters(result)


if __name__ == "__main__":
    main()
