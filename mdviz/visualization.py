"""
Visualization module for mdviz

Provides interactive and static plotting capabilities for trajectory analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Any
import warnings

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not found. Interactive visualization will be limited.")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not found. Static plotting will be limited.")

try:
    import py3Dmol

    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False
    warnings.warn("py3Dmol not found. 3D structure visualization will be limited.")


class Visualizer:
    """
    Visualization toolkit for MD trajectory analysis.

    Supports both interactive (Plotly) and static (Matplotlib) plotting,
    as well as 3D molecular structure visualization (py3Dmol).
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.default_colors = px.colors.qualitative.Set1 if HAS_PLOTLY else None

    def plot_pca_2d(
        self,
        pca_coords: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        title: str = "PCA Analysis",
        interactive: bool = True,
        **kwargs,
    ) -> Any:
        """
        Plot 2D PCA results.

        Parameters
        ----------
        pca_coords : np.ndarray
            PCA coordinates (n_samples, 2)
        cluster_labels : np.ndarray, optional
            Cluster labels for coloring points
        title : str, optional
            Plot title
        interactive : bool, optional
            Whether to use interactive (Plotly) or static (Matplotlib) plotting
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The plot object
        """
        if pca_coords.shape[1] < 2:
            raise ValueError("PCA coordinates must have at least 2 dimensions")

        if interactive and HAS_PLOTLY:
            return self._plot_pca_2d_plotly(pca_coords, cluster_labels, title, **kwargs)
        elif HAS_MATPLOTLIB:
            return self._plot_pca_2d_matplotlib(
                pca_coords, cluster_labels, title, **kwargs
            )
        else:
            raise ImportError("No plotting library available")

    def _plot_pca_2d_plotly(self, pca_coords, cluster_labels, title, **kwargs):
        """Create 2D PCA plot with Plotly."""
        df = pd.DataFrame(
            {
                "PC1": pca_coords[:, 0],
                "PC2": pca_coords[:, 1],
                "Frame": range(len(pca_coords)),
            }
        )

        if cluster_labels is not None:
            df["Cluster"] = cluster_labels.astype(str)
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="Cluster",
                hover_data=["Frame"],
                title=title,
                **kwargs,
            )
        else:
            fig = px.scatter(
                df, x="PC1", y="PC2", hover_data=["Frame"], title=title, **kwargs
            )

        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", hovermode="closest")

        return fig

    def _plot_pca_2d_matplotlib(self, pca_coords, cluster_labels, title, **kwargs):
        """Create 2D PCA plot with Matplotlib."""
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            cmap = plt.get_cmap("Set1")
            colors = cmap(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                ax.scatter(
                    pca_coords[mask, 0],
                    pca_coords[mask, 1],
                    c=[colors[i]],
                    label=f"Cluster {label}",
                    alpha=0.7,
                    s=kwargs.get("s", 20),
                )
            ax.legend()
        else:
            ax.scatter(
                pca_coords[:, 0], pca_coords[:, 1], alpha=0.7, s=kwargs.get("s", 20)
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_pca_3d(
        self,
        pca_coords: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        title: str = "3D PCA Analysis",
        **kwargs,
    ) -> Any:
        """
        Plot 3D PCA results (requires Plotly).

        Parameters
        ----------
        pca_coords : np.ndarray
            PCA coordinates (n_samples, 3)
        cluster_labels : np.ndarray, optional
            Cluster labels for coloring points
        title : str, optional
            Plot title
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure
            The 3D plot object
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for 3D plotting")

        if pca_coords.shape[1] < 3:
            raise ValueError("PCA coordinates must have at least 3 dimensions")

        df = pd.DataFrame(
            {
                "PC1": pca_coords[:, 0],
                "PC2": pca_coords[:, 1],
                "PC3": pca_coords[:, 2],
                "Frame": range(len(pca_coords)),
            }
        )

        if cluster_labels is not None:
            df["Cluster"] = cluster_labels.astype(str)
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Cluster",
                hover_data=["Frame"],
                title=title,
                **kwargs,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                hover_data=["Frame"],
                title=title,
                **kwargs,
            )

        fig.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
        )

        return fig

    def plot_rmsd(
        self,
        rmsd_values: np.ndarray,
        time_values: Optional[np.ndarray] = None,
        title: str = "RMSD vs Time",
        interactive: bool = True,
        **kwargs,
    ) -> Any:
        """
        Plot RMSD values over time.

        Parameters
        ----------
        rmsd_values : np.ndarray
            RMSD values
        time_values : np.ndarray, optional
            Time values (if None, uses frame numbers)
        title : str, optional
            Plot title
        interactive : bool, optional
            Whether to use interactive plotting
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The plot object
        """
        if time_values is None:
            time_values = np.arange(len(rmsd_values))

        if interactive and HAS_PLOTLY:
            df = pd.DataFrame({"Time": time_values, "RMSD": rmsd_values})

            fig = px.line(df, x="Time", y="RMSD", title=title, **kwargs)
            fig.update_layout(
                xaxis_title="Time (ps)" if time_values[1] > 1 else "Frame",
                yaxis_title="RMSD (Å)",
            )
            return fig

        elif HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
            ax.plot(time_values, rmsd_values, **kwargs)
            ax.set_xlabel("Time (ps)" if time_values[1] > 1 else "Frame")
            ax.set_ylabel("RMSD (Å)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        else:
            raise ImportError("No plotting library available")

    def plot_cluster_summary(
        self,
        cluster_labels: np.ndarray,
        title: str = "Cluster Population",
        interactive: bool = True,
        **kwargs,
    ) -> Any:
        """
        Plot cluster population summary.

        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster labels
        title : str, optional
            Plot title
        interactive : bool, optional
            Whether to use interactive plotting
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The plot object
        """
        # Count cluster populations
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        if interactive and HAS_PLOTLY:
            df = pd.DataFrame(
                {
                    "Cluster": unique_labels.astype(str),
                    "Population": counts,
                    "Percentage": 100 * counts / len(cluster_labels),
                }
            )

            fig = px.bar(
                df,
                x="Cluster",
                y="Population",
                title=title,
                hover_data=["Percentage"],
                **kwargs,
            )
            fig.update_layout(xaxis_title="Cluster", yaxis_title="Number of Frames")
            return fig

        elif HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

            # Add percentage labels on bars
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                percentage = 100 * count / len(cluster_labels)
                ax.text(
                    i,
                    count + max(counts) * 0.01,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                )

            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Frames")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            return fig

        else:
            raise ImportError("No plotting library available")

    def plot_elbow_curve(
        self,
        k_values: List[int],
        scores: List[float],
        metric: str = "inertia",
        title: Optional[str] = None,
        interactive: bool = True,
        **kwargs,
    ) -> Any:
        """
        Plot elbow curve for optimal cluster number selection.

        Parameters
        ----------
        k_values : list
            Number of clusters tested
        scores : list
            Corresponding metric scores
        metric : str, optional
            Metric name for labeling
        title : str, optional
            Plot title
        interactive : bool, optional
            Whether to use interactive plotting
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The plot object
        """
        if title is None:
            title = f"Elbow Curve ({metric.title()})"

        if interactive and HAS_PLOTLY:
            df = pd.DataFrame({"Number of Clusters": k_values, metric.title(): scores})

            fig = px.line(
                df,
                x="Number of Clusters",
                y=metric.title(),
                markers=True,
                title=title,
                **kwargs,
            )
            fig.update_layout(
                xaxis_title="Number of Clusters", yaxis_title=metric.title()
            )
            return fig

        elif HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
            ax.plot(k_values, scores, "bo-", **kwargs)
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel(metric.title())
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        else:
            raise ImportError("No plotting library available")

    def show_structure_3d(
        self,
        pdb_content: str,
        style: str = "cartoon",
        color_scheme: str = "chainbow",
        width: int = 600,
        height: int = 400,
    ) -> Any:
        """
        Display 3D molecular structure using py3Dmol.

        Parameters
        ----------
        pdb_content : str
            PDB file content as string
        style : str, optional
            Visualization style ('cartoon', 'stick', 'sphere', etc.)
        color_scheme : str, optional
            Color scheme ('chainbow', 'spectrum', etc.)
        width : int, optional
            Viewer width in pixels
        height : int, optional
            Viewer height in pixels

        Returns
        -------
        py3Dmol.view
            3D molecular viewer
        """
        if not HAS_PY3DMOL:
            raise ImportError("py3Dmol is required for 3D structure visualization")

        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_content, "pdb")

        if style == "cartoon":
            view.setStyle({style: {"colorscheme": color_scheme}})
        elif style == "stick":
            view.setStyle({style: {}})
        elif style == "sphere":
            view.setStyle({style: {"radius": 0.5}})
        else:
            view.setStyle({style: {}})

        view.zoomTo()
        return view

    def create_dashboard(
        self,
        pca_coords: np.ndarray,
        cluster_labels: np.ndarray,
        rmsd_values: Optional[np.ndarray] = None,
        time_values: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Create a comprehensive dashboard with multiple plots.

        Parameters
        ----------
        pca_coords : np.ndarray
            PCA coordinates
        cluster_labels : np.ndarray
            Cluster labels
        rmsd_values : np.ndarray, optional
            RMSD values over time
        time_values : np.ndarray, optional
            Time values

        Returns
        -------
        plotly.graph_objects.Figure
            Dashboard figure with subplots
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for dashboard creation")

        # Determine subplot layout
        if rmsd_values is not None:
            if pca_coords.shape[1] >= 3:
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "2D PCA",
                        "3D PCA",
                        "RMSD vs Time",
                        "Cluster Populations",
                    ),
                    specs=[
                        [{"type": "scatter"}, {"type": "scatter3d"}],
                        [{"type": "scatter"}, {"type": "bar"}],
                    ],
                )
            else:
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "2D PCA",
                        "RMSD vs Time",
                        "Cluster Populations",
                        "",
                    ),
                    specs=[
                        [{"type": "scatter"}, {"type": "scatter"}],
                        [{"type": "bar"}, None],
                    ],
                )
        else:
            if pca_coords.shape[1] >= 3:
                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=("2D PCA", "3D PCA", "Cluster Populations"),
                    specs=[
                        [{"type": "scatter"}, {"type": "scatter3d"}, {"type": "bar"}]
                    ],
                )
            else:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("2D PCA", "Cluster Populations"),
                    specs=[[{"type": "scatter"}, {"type": "bar"}]],
                )

        # Add 2D PCA plot
        unique_labels = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set1[: len(unique_labels)]

        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            fig.add_trace(
                go.Scatter(
                    x=pca_coords[mask, 0],
                    y=pca_coords[mask, 1],
                    mode="markers",
                    name=f"Cluster {label}",
                    marker=dict(color=colors[i % len(colors)]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Add 3D PCA plot if available
        if pca_coords.shape[1] >= 3:
            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                fig.add_trace(
                    go.Scatter3d(
                        x=pca_coords[mask, 0],
                        y=pca_coords[mask, 1],
                        z=pca_coords[mask, 2],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker=dict(color=colors[i % len(colors)]),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

        # Add RMSD plot if available
        if rmsd_values is not None:
            if time_values is None:
                time_values = np.arange(len(rmsd_values))

            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=rmsd_values,
                    mode="lines",
                    name="RMSD",
                    showlegend=False,
                ),
                row=2 if pca_coords.shape[1] >= 3 else 1,
                col=1 if pca_coords.shape[1] >= 3 else 2,
            )

        # Add cluster population plot
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        row_pos = 2 if rmsd_values is not None else 1
        col_pos = (
            2 if rmsd_values is not None else (3 if pca_coords.shape[1] >= 3 else 2)
        )

        fig.add_trace(
            go.Bar(
                x=unique_labels.astype(str),
                y=counts,
                name="Population",
                showlegend=False,
                marker=dict(color=colors[: len(unique_labels)]),
            ),
            row=row_pos,
            col=col_pos,
        )

        fig.update_layout(
            title="MD Trajectory Analysis Dashboard",
            height=800 if rmsd_values is not None else 400,
        )

        return fig
