"""
mdviz: A simple, user-friendly toolkit for MD trajectory analysis

Provides easy-to-use tools for:
- Trajectory loading and preprocessing
- Principal Component Analysis (PCA)
- Clustering (k-means, DBSCAN, hierarchical)
- Interactive visualization
- Export of representative structures
"""

__version__ = "0.1.0"
__author__ = "Justin Kirkland"
__email__ = "justin.kirkland.phd@gmail.com"

from .trajectory import TrajectoryAnalyzer
from .clustering import ClusterAnalyzer
from .visualization import Visualizer
from .utils import align_trajectory, calculate_rmsd, export_structure

__all__ = [
    "TrajectoryAnalyzer",
    "ClusterAnalyzer",
    "Visualizer",
    "align_trajectory",
    "calculate_rmsd",
    "export_structure",
]
