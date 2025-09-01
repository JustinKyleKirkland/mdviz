"""
Trajectory analysis module for mdviz

Handles trajectory loading, preprocessing, and PCA analysis.
"""

import numpy as np
from typing import Tuple
import warnings

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not found. Some functionality will be limited.")

try:
    import mdtraj as md

    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TrajectoryAnalyzer:
    """
    Main class for trajectory analysis including loading, alignment, and PCA.

    This class provides a simple interface for loading molecular dynamics
    trajectories and performing common analysis tasks like alignment and
    principal component analysis.
    """

    def __init__(
        self, topology: str, trajectory: str, selection: str = "protein and name CA"
    ):
        """
        Initialize trajectory analyzer.

        Parameters
        ----------
        topology : str
            Path to topology file (PSF, PDB, etc.)
        trajectory : str
            Path to trajectory file (DCD, XTC, etc.)
        selection : str, optional
            Atom selection for analysis (default: "protein and name CA")
        """
        if not HAS_MDANALYSIS:
            raise ImportError("MDAnalysis is required for trajectory analysis")

        self.topology = topology
        self.trajectory = trajectory
        self.selection = selection
        self.universe = None
        self.aligned_coords = None
        self.pca_model = None
        self.pca_coords = None
        self.atom_group = None

        self._load_trajectory()

    def _load_trajectory(self):
        """Load trajectory using MDAnalysis."""
        try:
            self.universe = mda.Universe(self.topology, self.trajectory)
            self.atom_group = self.universe.select_atoms(self.selection)
            print(f"Loaded trajectory with {len(self.universe.trajectory)} frames")
            print(
                f"Selected {len(self.atom_group)} atoms with selection: '{self.selection}'"
            )
        except Exception as e:
            raise ValueError(f"Failed to load trajectory: {e}")

    def align_trajectory(self, reference_frame: int = 0) -> np.ndarray:
        """
        Align trajectory to a reference frame.

        Parameters
        ----------
        reference_frame : int, optional
            Frame index to use as reference (default: 0)

        Returns
        -------
        np.ndarray
            Aligned coordinates array (n_frames, n_atoms, 3)
        """
        if self.universe is None:
            raise ValueError("Trajectory not loaded")

        # Set reference
        self.universe.trajectory[reference_frame]
        ref_coords = self.atom_group.positions.copy()

        # Align all frames
        aligned_coords = []
        for ts in self.universe.trajectory:
            # Align current frame to reference
            align.alignto(self.atom_group, ref_coords)
            aligned_coords.append(self.atom_group.positions.copy())

        self.aligned_coords = np.array(aligned_coords)
        print(
            f"Aligned {len(aligned_coords)} frames to reference frame {reference_frame}"
        )

        return self.aligned_coords

    def calculate_rmsd(self, reference_frame: int = 0) -> np.ndarray:
        """
        Calculate RMSD for each frame relative to reference.

        Parameters
        ----------
        reference_frame : int, optional
            Frame index to use as reference (default: 0)

        Returns
        -------
        np.ndarray
            RMSD values for each frame
        """
        if self.universe is None:
            raise ValueError("Trajectory not loaded")

        # Set reference
        self.universe.trajectory[reference_frame]
        ref_coords = self.atom_group.positions.copy()

        rmsd_values = []
        for ts in self.universe.trajectory:
            rmsd_val = rms.rmsd(
                self.atom_group.positions, ref_coords, superposition=True
            )
            rmsd_values.append(rmsd_val)

        return np.array(rmsd_values)

    def perform_pca(
        self, n_components: int = 2, align_first: bool = True
    ) -> Tuple[np.ndarray, PCA]:
        """
        Perform Principal Component Analysis on trajectory.

        Parameters
        ----------
        n_components : int, optional
            Number of principal components (default: 2)
        align_first : bool, optional
            Whether to align trajectory before PCA (default: True)

        Returns
        -------
        tuple
            (pca_coordinates, pca_model)
        """
        if self.universe is None:
            raise ValueError("Trajectory not loaded")

        # Get coordinates
        if align_first and self.aligned_coords is None:
            coords = self.align_trajectory()
        elif self.aligned_coords is not None:
            coords = self.aligned_coords
        else:
            # Use unaligned coordinates
            coords = []
            for ts in self.universe.trajectory:
                coords.append(self.atom_group.positions.copy())
            coords = np.array(coords)

        # Flatten coordinates for PCA (n_frames, n_features)
        n_frames, n_atoms, _ = coords.shape
        coords_flat = coords.reshape(n_frames, -1)

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords_flat)

        # Perform PCA
        self.pca_model = PCA(n_components=n_components)
        self.pca_coords = self.pca_model.fit_transform(coords_scaled)

        # Store scaler for future use
        self.pca_model.scaler_ = scaler

        print(f"PCA completed with {n_components} components")
        print(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_}")
        print(
            f"Total explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}"
        )

        return self.pca_coords, self.pca_model

    def get_frame_coordinates(self, frame_idx: int) -> np.ndarray:
        """
        Get coordinates for a specific frame.

        Parameters
        ----------
        frame_idx : int
            Frame index

        Returns
        -------
        np.ndarray
            Coordinates for the specified frame
        """
        if self.universe is None:
            raise ValueError("Trajectory not loaded")

        self.universe.trajectory[frame_idx]
        return self.atom_group.positions.copy()

    def get_trajectory_info(self) -> dict:
        """
        Get basic information about the loaded trajectory.

        Returns
        -------
        dict
            Dictionary containing trajectory information
        """
        if self.universe is None:
            return {}

        return {
            "n_frames": len(self.universe.trajectory),
            "n_atoms_total": len(self.universe.atoms),
            "n_atoms_selected": len(self.atom_group),
            "selection": self.selection,
            "topology_file": self.topology,
            "trajectory_file": self.trajectory,
            "time_range": (
                self.universe.trajectory[0].time,
                self.universe.trajectory[-1].time,
            ),
            "dt": self.universe.trajectory.dt,
        }

    def export_frame_pdb(self, frame_idx: int, output_path: str):
        """
        Export a specific frame as PDB file.

        Parameters
        ----------
        frame_idx : int
            Frame index to export
        output_path : str
            Path for output PDB file
        """
        if self.universe is None:
            raise ValueError("Trajectory not loaded")

        self.universe.trajectory[frame_idx]

        # Write all atoms or just selection?
        writer = mda.Writer(output_path, self.universe.atoms.n_atoms)
        writer.write(self.universe)
        writer.close()

        print(f"Exported frame {frame_idx} to {output_path}")
