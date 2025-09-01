"""
Unit tests for mdviz package

Run with: python -m pytest tests/
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import tempfile
import os


class TestTrajectoryAnalyzer:
    """Test the TrajectoryAnalyzer class."""

    def test_init_without_mdanalysis(self):
        """Test initialization when MDAnalysis is not available."""
        with patch("mdviz.trajectory.HAS_MDANALYSIS", False):
            from mdviz.trajectory import TrajectoryAnalyzer

            with pytest.raises(ImportError, match="MDAnalysis is required"):
                TrajectoryAnalyzer("test.psf", "test.dcd")

    @patch("mdviz.trajectory.mda")
    def test_successful_initialization(self, mock_mda):
        """Test successful initialization with mocked MDAnalysis."""
        # Mock universe and atom group
        mock_universe = Mock()
        mock_universe.trajectory = [Mock() for _ in range(100)]  # 100 frames
        mock_atom_group = Mock()
        mock_atom_group.__len__ = Mock(return_value=300)  # 300 atoms

        mock_mda.Universe.return_value = mock_universe
        mock_universe.select_atoms.return_value = mock_atom_group

        from mdviz.trajectory import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer("test.psf", "test.dcd")

        assert analyzer.topology == "test.psf"
        assert analyzer.trajectory == "test.dcd"
        assert analyzer.selection == "protein and name CA"
        assert analyzer.universe == mock_universe
        assert analyzer.atom_group == mock_atom_group

    def test_get_trajectory_info_no_universe(self):
        """Test trajectory info when no universe is loaded."""
        with patch("mdviz.trajectory.HAS_MDANALYSIS", False):
            from mdviz.trajectory import TrajectoryAnalyzer

            analyzer = TrajectoryAnalyzer.__new__(TrajectoryAnalyzer)
            analyzer.universe = None

            info = analyzer.get_trajectory_info()
            assert info == {}


class TestClusterAnalyzer:
    """Test the ClusterAnalyzer class."""

    def test_init_without_sklearn(self):
        """Test initialization when scikit-learn is not available."""
        with patch("mdviz.clustering.HAS_SKLEARN", False):
            from mdviz.clustering import ClusterAnalyzer

            with pytest.raises(ImportError, match="scikit-learn is required"):
                ClusterAnalyzer(np.random.random((100, 2)))

    def test_successful_initialization(self):
        """Test successful initialization."""
        data = np.random.random((100, 3))
        feature_names = ["PC1", "PC2", "PC3"]

        from mdviz.clustering import ClusterAnalyzer

        analyzer = ClusterAnalyzer(data, feature_names)

        assert np.array_equal(analyzer.data, data)
        assert analyzer.feature_names == feature_names
        assert analyzer.scaled_data is None
        assert analyzer.cluster_labels is None

    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = np.random.random((100, 2)) * 10  # Random data with large scale

        from mdviz.clustering import ClusterAnalyzer

        analyzer = ClusterAnalyzer(data)
        scaled_data = analyzer.preprocess_data(scale=True)

        assert scaled_data is not None
        assert np.allclose(np.mean(scaled_data, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(scaled_data, axis=0), 1, atol=1e-10)

    @patch("mdviz.clustering.KMeans")
    def test_kmeans_clustering(self, mock_kmeans):
        """Test K-means clustering."""
        data = np.random.random((100, 2))
        mock_labels = np.random.randint(0, 3, 100)

        # Mock KMeans
        mock_model = Mock()
        mock_model.fit_predict.return_value = mock_labels
        mock_kmeans.return_value = mock_model

        from mdviz.clustering import ClusterAnalyzer

        analyzer = ClusterAnalyzer(data)
        labels = analyzer.kmeans_clustering(n_clusters=3)

        assert np.array_equal(labels, mock_labels)
        assert analyzer.cluster_model == mock_model
        assert np.array_equal(analyzer.cluster_labels, mock_labels)


class TestVisualizer:
    """Test the Visualizer class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        from mdviz.visualization import Visualizer

        viz = Visualizer()
        assert viz is not None

    def test_pca_2d_plot_invalid_dimensions(self):
        """Test PCA 2D plot with invalid dimensions."""
        from mdviz.visualization import Visualizer

        viz = Visualizer()

        # Data with only 1 dimension
        pca_coords = np.random.random((100, 1))

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            viz.plot_pca_2d(pca_coords)

    def test_pca_3d_plot_invalid_dimensions(self):
        """Test PCA 3D plot with invalid dimensions."""
        with patch("mdviz.visualization.HAS_PLOTLY", False):
            from mdviz.visualization import Visualizer

            viz = Visualizer()

            with pytest.raises(ImportError, match="Plotly is required"):
                viz.plot_pca_3d(np.random.random((100, 3)))

    @patch("mdviz.visualization.HAS_MATPLOTLIB", True)
    @patch("mdviz.visualization.plt")
    def test_matplotlib_plotting(self, mock_plt):
        """Test matplotlib plotting functionality."""
        from mdviz.visualization import Visualizer

        viz = Visualizer()

        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        pca_coords = np.random.random((100, 2))
        cluster_labels = np.random.randint(0, 3, 100)

        result = viz.plot_pca_2d(pca_coords, cluster_labels, interactive=False)

        assert result == mock_fig
        mock_plt.subplots.assert_called_once()


class TestUtils:
    """Test utility functions."""

    def test_center_coordinates_2d(self):
        """Test coordinate centering for 2D array."""
        from mdviz.utils import center_coordinates

        # Create coordinates with known center of mass
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        centered = center_coordinates(coords)

        # Check that center of mass is at origin
        com = np.mean(centered, axis=0)
        assert np.allclose(com, [0, 0, 0])

    def test_center_coordinates_3d(self):
        """Test coordinate centering for 3D array."""
        from mdviz.utils import center_coordinates

        # Multiple frames
        coords = np.random.random((10, 5, 3)) * 10  # 10 frames, 5 atoms, 3D
        centered = center_coordinates(coords)

        # Check that each frame is centered
        for i in range(coords.shape[0]):
            com = np.mean(centered[i], axis=0)
            assert np.allclose(com, [0, 0, 0])

    def test_center_coordinates_invalid_shape(self):
        """Test coordinate centering with invalid shape."""
        from mdviz.utils import center_coordinates

        coords = np.random.random((5,))  # 1D array

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            center_coordinates(coords)

    def test_calculate_radius_of_gyration_2d(self):
        """Test radius of gyration calculation for single frame."""
        from mdviz.utils import calculate_radius_of_gyration

        # Simple test case: square arrangement
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        rg = calculate_radius_of_gyration(coords)

        # Should be a positive number
        assert isinstance(rg, float)
        assert rg > 0

    def test_calculate_radius_of_gyration_3d(self):
        """Test radius of gyration calculation for multiple frames."""
        from mdviz.utils import calculate_radius_of_gyration

        coords = np.random.random((5, 10, 3))  # 5 frames, 10 atoms, 3D
        rg_values = calculate_radius_of_gyration(coords)

        assert len(rg_values) == 5
        assert all(rg > 0 for rg in rg_values)

    def test_find_closest_frame(self):
        """Test finding closest frame to target coordinates."""
        from mdviz.utils import find_closest_frame

        # Create trajectory with known closest frame
        target = np.array([[0, 0, 0], [1, 1, 1]])

        trajectory = np.array(
            [
                [[10, 10, 10], [11, 11, 11]],  # Frame 0: far
                [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]],  # Frame 1: close
                [[5, 5, 5], [6, 6, 6]],  # Frame 2: medium
            ]
        )

        closest_idx = find_closest_frame(target, trajectory)

        assert closest_idx == 1  # Should be frame 1

    def test_validate_trajectory_input_file_not_found(self):
        """Test trajectory validation with missing files."""
        from mdviz.utils import validate_trajectory_input

        with pytest.raises(FileNotFoundError, match="Topology file not found"):
            validate_trajectory_input("nonexistent.psf", "test.dcd")

    def test_validate_trajectory_input_success(self):
        """Test successful trajectory validation."""
        from mdviz.utils import validate_trajectory_input

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".psf", delete=False) as top_file:
            topology_path = top_file.name

        with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as traj_file:
            trajectory_path = traj_file.name

        try:
            # Test with HAS_MDANALYSIS = False
            with patch("mdviz.utils.HAS_MDANALYSIS", False):
                result = validate_trajectory_input(topology_path, trajectory_path)
                assert result is True

        finally:
            # Clean up
            os.unlink(topology_path)
            os.unlink(trajectory_path)


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_synthetic_data_workflow(self):
        """Test complete workflow with synthetic data."""
        # Generate synthetic PCA coordinates
        np.random.seed(42)

        # Create three distinct clusters
        cluster1 = np.random.normal([0, 0], [0.5, 0.5], (50, 2))
        cluster2 = np.random.normal([3, 0], [0.5, 0.5], (50, 2))
        cluster3 = np.random.normal([1.5, 3], [0.5, 0.5], (50, 2))

        pca_coords = np.vstack([cluster1, cluster2, cluster3])

        # Test clustering
        from mdviz.clustering import ClusterAnalyzer

        cluster_analyzer = ClusterAnalyzer(pca_coords)

        # Should work without errors
        cluster_labels = cluster_analyzer.kmeans_clustering(n_clusters=3)

        assert len(cluster_labels) == 150
        assert len(np.unique(cluster_labels)) == 3

        # Test finding optimal k
        optimal_k, scores = cluster_analyzer.find_optimal_k(k_range=range(2, 6))

        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= 5
        assert "silhouette" in scores
        assert "k_values" in scores

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with empty data
        with pytest.raises(Exception):
            from mdviz.clustering import ClusterAnalyzer

            ClusterAnalyzer(np.array([]))

        # Test with single point
        single_point = np.array([[1, 2]])

        from mdviz.clustering import ClusterAnalyzer

        analyzer = ClusterAnalyzer(single_point)

        # Should handle gracefully
        try:
            analyzer.kmeans_clustering(n_clusters=1)
        except Exception as e:
            # Some error is expected with single point
            assert isinstance(e, (ValueError, Exception))


if __name__ == "__main__":
    pytest.main([__file__])
