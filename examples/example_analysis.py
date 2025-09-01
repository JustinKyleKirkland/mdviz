"""
Example usage of mdviz for MD trajectory analysis

This script demonstrates how to use mdviz for:
1. Loading and aligning a trajectory
2. Performing PCA analysis
3. Clustering the trajectory
4. Creating visualizations
5. Exporting representative structures
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import mdviz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mdviz import TrajectoryAnalyzer, ClusterAnalyzer, Visualizer

    print("✅ mdviz imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: This example uses synthetic data when mdviz imports fail.")


def run_example_analysis(topology_file: str, trajectory_file: str):
    """
    Run a complete trajectory analysis workflow.

    Parameters
    ----------
    topology_file : str
        Path to topology file (PSF, PDB, etc.)
    trajectory_file : str
        Path to trajectory file (DCD, XTC, etc.)
    """

    print("=== MD Trajectory Analysis with mdviz ===\n")

    # Step 1: Load and analyze trajectory
    print("1. Loading trajectory...")
    analyzer = TrajectoryAnalyzer(
        topology=topology_file,
        trajectory=trajectory_file,
        selection="protein and name CA",  # Alpha carbons only
    )

    # Get basic trajectory info
    info = analyzer.get_trajectory_info()
    print(f"   Loaded {info['n_frames']} frames")
    print(f"   Selected {info['n_atoms_selected']} atoms")
    print(
        f"   Time range: {info['time_range'][0]:.1f} - {info['time_range'][1]:.1f} ps\n"
    )

    # Step 2: Align trajectory and calculate RMSD
    print("2. Aligning trajectory and calculating RMSD...")
    analyzer.align_trajectory(
        reference_frame=0
    )  # Align trajectory (coords stored internally)
    rmsd_values = analyzer.calculate_rmsd(reference_frame=0)

    print(f"   Mean RMSD: {np.mean(rmsd_values):.2f} ± {np.std(rmsd_values):.2f} Å")
    print(f"   RMSD range: {np.min(rmsd_values):.2f} - {np.max(rmsd_values):.2f} Å\n")

    # Step 3: Perform PCA
    print("3. Performing PCA analysis...")
    pca_coords, pca_model = analyzer.perform_pca(n_components=3, align_first=False)

    print(f"   PC1 explains {pca_model.explained_variance_ratio_[0]:.3f} of variance")
    print(f"   PC2 explains {pca_model.explained_variance_ratio_[1]:.3f} of variance")
    print(f"   PC3 explains {pca_model.explained_variance_ratio_[2]:.3f} of variance")
    print(
        f"   Total explained variance: {sum(pca_model.explained_variance_ratio_):.3f}\n"
    )

    # Step 4: Clustering analysis
    print("4. Performing clustering analysis...")
    cluster_analyzer = ClusterAnalyzer(pca_coords[:, :2])  # Use first 2 PCs

    # Find optimal number of clusters
    optimal_k, scores = cluster_analyzer.find_optimal_k(k_range=range(2, 8))
    print(f"   Optimal number of clusters: {optimal_k}")

    # Perform clustering with optimal k
    cluster_labels = cluster_analyzer.kmeans_clustering(n_clusters=optimal_k)

    # Show cluster populations
    print("   Cluster populations:")
    for cluster_id in np.unique(cluster_labels):
        count = np.sum(cluster_labels == cluster_id)
        percentage = 100 * count / len(cluster_labels)
        print(f"     Cluster {cluster_id}: {count} frames ({percentage:.1f}%)")

    print()

    # Step 5: Find representative structures
    print("5. Finding representative structures...")
    representatives = cluster_analyzer.get_representative_frames(n_representatives=1)

    for cluster_id, frame_indices in representatives.items():
        print(f"   Cluster {cluster_id} representative: frame {frame_indices[0]}")

    print()

    # Step 6: Create visualizations
    print("6. Creating visualizations...")
    visualizer = Visualizer()

    # Create 2D PCA plot (for demonstration purposes)
    visualizer.plot_pca_2d(
        pca_coords[:, :2],
        cluster_labels,
        title="PCA Analysis with Clustering",
        interactive=False,
    )

    # Create RMSD plot (for demonstration purposes)
    time_values = np.arange(len(rmsd_values)) * info["dt"]
    visualizer.plot_rmsd(
        rmsd_values, time_values, title="RMSD vs Time", interactive=False
    )

    # Create cluster population plot (for demonstration purposes)
    visualizer.plot_cluster_summary(
        cluster_labels, title="Cluster Populations", interactive=False
    )

    # Create elbow curve (for demonstration purposes)
    visualizer.plot_elbow_curve(
        scores["k_values"],
        scores["silhouette"],
        metric="silhouette",
        title="Optimal Cluster Number (Silhouette Method)",
        interactive=False,
    )

    # Display plots
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.title("PCA Analysis with Clustering")
    # Note: This would normally show the PCA plot
    plt.text(
        0.5,
        0.5,
        "PCA Plot\n(2D visualization)",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

    plt.subplot(2, 2, 2)
    plt.plot(time_values, rmsd_values)
    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD vs Time")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique_labels.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Frames")
    plt.title("Cluster Populations")

    plt.subplot(2, 2, 4)
    plt.plot(scores["k_values"], scores["silhouette"], "bo-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal Cluster Number")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mdviz_analysis_results.png", dpi=300, bbox_inches="tight")
    print("   Saved analysis plots to 'mdviz_analysis_results.png'")

    # Step 7: Export representative structures
    print("\n7. Exporting representative structures...")

    try:
        import os

        output_dir = "representative_structures"
        os.makedirs(output_dir, exist_ok=True)

        for cluster_id, frame_indices in representatives.items():
            frame_idx = frame_indices[0]
            output_path = os.path.join(
                output_dir, f"cluster_{cluster_id}_frame_{frame_idx}.pdb"
            )
            analyzer.export_frame_pdb(frame_idx, output_path)

        print(f"   Exported representative structures to '{output_dir}/'")

    except Exception as e:
        print(f"   Warning: Could not export structures: {e}")

    print("\n=== Analysis Complete ===")

    # Return results for further analysis
    return {
        "analyzer": analyzer,
        "cluster_analyzer": cluster_analyzer,
        "visualizer": visualizer,
        "pca_coords": pca_coords,
        "cluster_labels": cluster_labels,
        "rmsd_values": rmsd_values,
        "representatives": representatives,
    }


def create_demo_data():
    """
    Create synthetic demo data for testing when real trajectory files are not available.
    """
    print("Creating synthetic demo data for testing...")

    # Generate synthetic PCA coordinates (simulating a trajectory)
    np.random.seed(42)
    n_frames = 1000

    # Create three "states" with different characteristics
    state1 = np.random.normal([0, 0], [1.0, 0.5], (300, 2))
    state2 = np.random.normal([3, 2], [0.8, 1.2], (400, 2))
    state3 = np.random.normal([-2, 3], [1.5, 0.7], (300, 2))

    pca_coords = np.vstack([state1, state2, state3])

    # Add some noise and transitions
    noise = np.random.normal(0, 0.1, pca_coords.shape)
    pca_coords += noise

    # Generate synthetic RMSD values
    rmsd_base = 2.0
    rmsd_trend = 0.5 * np.linspace(0, 1, n_frames)
    rmsd_noise = 0.3 * np.random.normal(0, 1, n_frames)
    rmsd_values = rmsd_base + rmsd_trend + rmsd_noise
    rmsd_values = np.abs(rmsd_values)  # Ensure positive values

    return pca_coords, rmsd_values


def run_demo_analysis():
    """
    Run analysis with synthetic demo data.
    """
    print("=== mdviz Demo Analysis (Synthetic Data) ===\n")

    # Create demo data
    pca_coords, rmsd_values = create_demo_data()

    print("Generated synthetic data:")
    print("  - {} frames".format(len(pca_coords)))
    print("  - 2 principal components")
    print(
        "  - RMSD range: {:.2f} - {:.2f} Å\n".format(
            np.min(rmsd_values), np.max(rmsd_values)
        )
    )

    # Clustering analysis
    print("Performing clustering analysis...")
    cluster_analyzer = ClusterAnalyzer(pca_coords, feature_names=["PC1", "PC2"])

    # Find optimal number of clusters
    optimal_k, scores = cluster_analyzer.find_optimal_k(k_range=range(2, 8))
    print(f"Optimal number of clusters: {optimal_k}\n")

    # Perform clustering
    cluster_labels = cluster_analyzer.kmeans_clustering(n_clusters=optimal_k)

    # Visualizations
    print("Creating visualizations...")

    # Create plots
    time_values = np.arange(len(rmsd_values)) * 0.1  # 0.1 ps timestep

    plt.figure(figsize=(15, 10))

    # PCA plot with clusters
    plt.subplot(2, 3, 1)
    unique_labels = np.unique(cluster_labels)
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        color = colors[i % len(colors)]
        plt.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=color,
            label=f"Cluster {label}",
            alpha=0.7,
            s=20,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Clustering")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # RMSD plot
    plt.subplot(2, 3, 2)
    plt.plot(time_values, rmsd_values)
    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD vs Time")
    plt.grid(True, alpha=0.3)

    # Cluster populations
    plt.subplot(2, 3, 3)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique_labels.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Frames")
    plt.title("Cluster Populations")

    # Add percentage labels
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        percentage = 100 * count / len(cluster_labels)
        plt.text(
            i,
            count + max(counts) * 0.01,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    # Elbow curve
    plt.subplot(2, 3, 4)
    plt.plot(scores["k_values"], scores["silhouette"], "bo-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs K")
    plt.grid(True, alpha=0.3)

    # Calinski-Harabasz score
    plt.subplot(2, 3, 5)
    plt.plot(scores["k_values"], scores["calinski_harabasz"], "ro-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title("Calinski-Harabasz Score vs K")
    plt.grid(True, alpha=0.3)

    # Inertia (within-cluster sum of squares)
    plt.subplot(2, 3, 6)
    plt.plot(scores["k_values"], scores["inertia"], "go-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Inertia vs K")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mdviz_demo_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved demo results to 'mdviz_demo_results.png'")

    # Print cluster summary
    print("\nCluster Summary:")
    summary = cluster_analyzer.get_cluster_summary()
    print(summary)

    print("\n=== Demo Analysis Complete ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        # Real trajectory analysis
        topology_file = sys.argv[1]
        trajectory_file = sys.argv[2]

        print(f"Running analysis on {trajectory_file}")
        results = run_example_analysis(topology_file, trajectory_file)

    else:
        # Demo analysis with synthetic data
        print("Usage: python example_analysis.py <topology> <trajectory>")
        print("Or run without arguments for demo with synthetic data\n")

        run_demo_analysis()
