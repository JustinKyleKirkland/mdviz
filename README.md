# mdviz 🧬

**A simple, user-friendly toolkit for MD trajectory analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-tutorial-green.svg)](examples/mdviz_tutorial.ipynb)

mdviz makes molecular dynamics trajectory analysis accessible and straightforward. Built on top of proven libraries like MDAnalysis and scikit-learn, it provides a simple Python API for common analysis tasks.

## 🎯 Features

- **Simple API** for trajectory loading and preprocessing
- **Built-in PCA & clustering** (k-means, DBSCAN, hierarchical)
- **Interactive visualization** with Plotly and py3Dmol
- **Export options** for representative structures
- **Comprehensive analysis** from trajectories to publication-ready results

## 🚀 Quick Start

```python
from mdviz import TrajectoryAnalyzer, ClusterAnalyzer, Visualizer

# Load and analyze trajectory
analyzer = TrajectoryAnalyzer("topology.psf", "trajectory.dcd")
pca_coords, pca_model = analyzer.perform_pca(n_components=3)

# Perform clustering
cluster_analyzer = ClusterAnalyzer(pca_coords[:, :2])
cluster_labels = cluster_analyzer.kmeans_clustering(n_clusters=3)

# Create visualizations
visualizer = Visualizer()
fig = visualizer.plot_pca_2d(pca_coords[:, :2], cluster_labels)
fig.show()
```

## 📦 Installation

### Prerequisites

mdviz requires Python 3.8+ and the following core dependencies:

- MDAnalysis (trajectory handling)
- scikit-learn (PCA & clustering)
- numpy, pandas (data manipulation)
- plotly, matplotlib (visualization)
- py3Dmol (3D structure visualization)

### Install from source

```bash
git clone https://github.com/JustinKyleKirkland/mdviz.git
cd mdviz
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 📚 Documentation & Examples

### Tutorial Notebook
Start with the comprehensive tutorial: [`examples/mdviz_tutorial.ipynb`](examples/mdviz_tutorial.ipynb)

### Example Script
Run the demo analysis: [`examples/example_analysis.py`](examples/example_analysis.py)

```bash
# With your own trajectory files
python examples/example_analysis.py topology.psf trajectory.dcd

# With synthetic demo data
python examples/example_analysis.py
```

## 🧪 Core Components

### TrajectoryAnalyzer
Handle trajectory loading, alignment, and PCA analysis:

```python
analyzer = TrajectoryAnalyzer(
    topology="system.psf", 
    trajectory="trajectory.dcd",
    selection="protein and name CA"
)

# Align trajectory
aligned_coords = analyzer.align_trajectory()

# Calculate RMSD
rmsd_values = analyzer.calculate_rmsd()

# Perform PCA
pca_coords, pca_model = analyzer.perform_pca(n_components=3)

# Export representative frame
analyzer.export_frame_pdb(frame_idx=100, output_path="frame_100.pdb")
```

### ClusterAnalyzer
Perform various clustering algorithms:

```python
cluster_analyzer = ClusterAnalyzer(pca_coords)

# Find optimal number of clusters
optimal_k, scores = cluster_analyzer.find_optimal_k()

# K-means clustering
labels = cluster_analyzer.kmeans_clustering(n_clusters=optimal_k)

# DBSCAN clustering
labels = cluster_analyzer.dbscan_clustering(eps=0.5, min_samples=5)

# Hierarchical clustering
labels = cluster_analyzer.hierarchical_clustering(n_clusters=3)

# Get representative frames
representatives = cluster_analyzer.get_representative_frames()
```

### Visualizer
Create static and interactive visualizations:

```python
visualizer = Visualizer()

# 2D PCA plot
fig = visualizer.plot_pca_2d(pca_coords, cluster_labels, interactive=True)

# 3D PCA plot
fig = visualizer.plot_pca_3d(pca_coords, cluster_labels)

# RMSD over time
fig = visualizer.plot_rmsd(rmsd_values, time_values)

# Cluster populations
fig = visualizer.plot_cluster_summary(cluster_labels)

# Comprehensive dashboard
dashboard = visualizer.create_dashboard(pca_coords, cluster_labels, rmsd_values)
```

## 🔧 Advanced Usage

### Custom Atom Selections
```python
# Backbone atoms only
analyzer = TrajectoryAnalyzer("top.psf", "traj.dcd", selection="backbone")

# Specific residue range
analyzer = TrajectoryAnalyzer("top.psf", "traj.dcd", selection="resid 1:100 and name CA")

# Multiple chains
analyzer = TrajectoryAnalyzer("top.psf", "traj.dcd", selection="segid A B and protein")
```

### Clustering Optimization
```python
cluster_analyzer = ClusterAnalyzer(pca_coords)

# Test different methods for optimal k
optimal_k_sil, scores_sil = cluster_analyzer.find_optimal_k(method='silhouette')
optimal_k_ch, scores_ch = cluster_analyzer.find_optimal_k(method='calinski_harabasz')

# Compare different clustering algorithms
kmeans_labels = cluster_analyzer.kmeans_clustering(n_clusters=3)
dbscan_labels = cluster_analyzer.dbscan_clustering(eps=0.5)
hierarchical_labels = cluster_analyzer.hierarchical_clustering(n_clusters=3)
```

### Interactive Visualizations
```python
# Create interactive dashboard
visualizer = Visualizer()
dashboard = visualizer.create_dashboard(
    pca_coords=pca_coords,
    cluster_labels=cluster_labels,
    rmsd_values=rmsd_values,
    time_values=time_values
)
dashboard.show()

# 3D structure visualization
with open("structure.pdb", "r") as f:
    pdb_content = f.read()

viewer = visualizer.show_structure_3d(
    pdb_content, 
    style="cartoon", 
    color_scheme="chainbow"
)
viewer.show()
```

## 🧬 Supported File Formats

### Topology Files
- PDB, PSF, GRO, TOP, TPR

### Trajectory Files  
- DCD, XTC, TRR, NetCDF, H5MD

*Note: File format support depends on MDAnalysis capabilities*

## 🔬 Example Workflows

### Basic Analysis Pipeline
1. Load trajectory with `TrajectoryAnalyzer`
2. Align frames and calculate RMSD
3. Perform PCA to reduce dimensionality
4. Find optimal cluster number
5. Apply clustering algorithm
6. Visualize results
7. Export representative structures

### Conformational State Analysis
1. Focus on specific protein regions
2. Use PCA to identify major motions
3. Apply DBSCAN to find conformational states
4. Analyze state transitions over time
5. Export representative conformations

### Comparative Analysis
1. Load multiple trajectories
2. Align to common reference
3. Combine PCA coordinates
4. Cluster combined dataset
5. Compare populations across simulations

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific test categories:

```bash
# Test core functionality
python -m pytest tests/test_mdviz.py::TestTrajectoryAnalyzer

# Test clustering
python -m pytest tests/test_mdviz.py::TestClusterAnalyzer

# Test visualizations
python -m pytest tests/test_mdviz.py::TestVisualizer
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure tests pass (`python -m pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/JustinKyleKirkland/mdviz.git
cd mdviz
pip install -e ".[dev]"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

mdviz builds upon excellent open-source libraries:

- [MDAnalysis](https://www.mdanalysis.org/) - trajectory analysis
- [scikit-learn](https://scikit-learn.org/) - machine learning algorithms
- [Plotly](https://plotly.com/python/) - interactive visualizations
- [py3Dmol](https://github.com/avirshup/py3dmol) - 3D molecular visualization
- [NumPy](https://numpy.org/) & [pandas](https://pandas.pydata.org/) - data handling

## 📞 Support

- 📖 **Documentation**: See [`examples/mdviz_tutorial.ipynb`](examples/mdviz_tutorial.ipynb)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/JustinKyleKirkland/mdviz/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/JustinKyleKirkland/mdviz/discussions)
- 📧 **Contact**: [justin.kirkland.phd@gmail.com](mailto:justin.kirkland.phd@gmail.com)

---
