#!/usr/bin/env python3
"""
Command-line demo for mdviz package.

This script can be run after installing mdviz to test the installation.
"""

import sys

def main():
    """Main demo function."""
    print("🧬 mdviz Demo")
    print("=" * 50)
    
    try:
        import mdviz
        print(f"✅ mdviz version {mdviz.__version__} imported successfully!")
        
        # Test basic imports (import them to verify they work)
        from mdviz import TrajectoryAnalyzer, ClusterAnalyzer, Visualizer
        print("✅ Core classes imported successfully!")
        print(f"   - TrajectoryAnalyzer: {TrajectoryAnalyzer.__name__}")
        print(f"   - ClusterAnalyzer: {ClusterAnalyzer.__name__}")
        print(f"   - Visualizer: {Visualizer.__name__}")
        
        # Run a simple demo with synthetic data
        print("\n📊 Running demo with synthetic data...")
        
        import numpy as np
        
        # Generate synthetic PCA data
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], [0.5, 0.5], (50, 2))
        cluster2 = np.random.normal([3, 0], [0.5, 0.5], (50, 2))
        cluster3 = np.random.normal([1.5, 3], [0.5, 0.5], (50, 2))
        pca_coords = np.vstack([cluster1, cluster2, cluster3])
        
        # Test clustering
        cluster_analyzer = ClusterAnalyzer(pca_coords, feature_names=['PC1', 'PC2'])
        cluster_labels = cluster_analyzer.kmeans_clustering(n_clusters=3)
        
        print(f"✅ Clustering completed! Found {len(np.unique(cluster_labels))} clusters")
        
        # Show cluster populations
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\n📈 Cluster populations:")
        for label, count in zip(unique_labels, counts):
            percentage = 100 * count / len(cluster_labels)
            print(f"   Cluster {label}: {count:3d} points ({percentage:5.1f}%)")
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo get started with your own data:")
        print("   from mdviz import TrajectoryAnalyzer")
        print("   analyzer = TrajectoryAnalyzer('topology.psf', 'trajectory.dcd')")
        print("\nFor more examples, see: https://github.com/JustinKyleKirkland/mdviz")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure mdviz is properly installed:")
        print("   pip install mdviz")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
