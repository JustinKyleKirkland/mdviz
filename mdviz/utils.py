"""
Utility functions for mdviz

Helper functions for common tasks in MD trajectory analysis.
"""

import numpy as np
from typing import Optional, Union, List
import warnings

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not found. Some utility functions will be limited.")


def align_trajectory(
    universe: "mda.Universe",
    selection: str = "protein and name CA",
    reference_frame: int = 0,
) -> np.ndarray:
    """
    Align trajectory to a reference frame.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The trajectory universe
    selection : str, optional
        Atom selection for alignment (default: "protein and name CA")
    reference_frame : int, optional
        Frame index to use as reference (default: 0)

    Returns
    -------
    np.ndarray
        Aligned coordinates array (n_frames, n_atoms, 3)
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis is required for trajectory alignment")

    atom_group = universe.select_atoms(selection)

    # Set reference
    universe.trajectory[reference_frame]
    ref_coords = atom_group.positions.copy()

    # Align all frames
    aligned_coords = []
    for ts in universe.trajectory:
        # Align current frame to reference
        align.alignto(atom_group, ref_coords)
        aligned_coords.append(atom_group.positions.copy())

    return np.array(aligned_coords)


def calculate_rmsd(
    universe: "mda.Universe",
    selection: str = "protein and name CA",
    reference_frame: int = 0,
) -> np.ndarray:
    """
    Calculate RMSD for each frame relative to reference.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The trajectory universe
    selection : str, optional
        Atom selection for RMSD calculation (default: "protein and name CA")
    reference_frame : int, optional
        Frame index to use as reference (default: 0)

    Returns
    -------
    np.ndarray
        RMSD values for each frame
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis is required for RMSD calculation")

    atom_group = universe.select_atoms(selection)

    # Set reference
    universe.trajectory[reference_frame]
    ref_coords = atom_group.positions.copy()

    rmsd_values = []
    for ts in universe.trajectory:
        rmsd_val = rms.rmsd(atom_group.positions, ref_coords, superposition=True)
        rmsd_values.append(rmsd_val)

    return np.array(rmsd_values)


def export_structure(
    universe: "mda.Universe", frame_idx: int, output_path: str, selection: str = "all"
) -> None:
    """
    Export a specific frame as PDB file.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The trajectory universe
    frame_idx : int
        Frame index to export
    output_path : str
        Path for output PDB file
    selection : str, optional
        Atom selection to export (default: "all")
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis is required for structure export")

    universe.trajectory[frame_idx]

    if selection == "all":
        atoms = universe.atoms
    else:
        atoms = universe.select_atoms(selection)

    writer = mda.Writer(output_path, atoms.n_atoms)
    writer.write(atoms)
    writer.close()

    print(f"Exported frame {frame_idx} to {output_path}")


def load_trajectory(
    topology: str, trajectory: str, selection: str = "protein and name CA"
) -> tuple:
    """
    Load trajectory and return universe and selected atoms.

    Parameters
    ----------
    topology : str
        Path to topology file
    trajectory : str
        Path to trajectory file
    selection : str, optional
        Atom selection (default: "protein and name CA")

    Returns
    -------
    tuple
        (universe, atom_group)
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis is required for trajectory loading")

    universe = mda.Universe(topology, trajectory)
    atom_group = universe.select_atoms(selection)

    print(f"Loaded trajectory with {len(universe.trajectory)} frames")
    print(f"Selected {len(atom_group)} atoms with selection: '{selection}'")

    return universe, atom_group


def calculate_pairwise_rmsd(coords: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise RMSD matrix between all frames.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array (n_frames, n_atoms, 3)

    Returns
    -------
    np.ndarray
        Pairwise RMSD matrix (n_frames, n_frames)
    """
    n_frames = coords.shape[0]
    rmsd_matrix = np.zeros((n_frames, n_frames))

    for i in range(n_frames):
        for j in range(i, n_frames):
            # Calculate RMSD between frames i and j
            rmsd_val = np.sqrt(np.mean(np.sum((coords[i] - coords[j]) ** 2, axis=1)))
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val

    return rmsd_matrix


def find_closest_frame(target_coords: np.ndarray, trajectory_coords: np.ndarray) -> int:
    """
    Find the frame in trajectory closest to target coordinates.

    Parameters
    ----------
    target_coords : np.ndarray
        Target coordinates (n_atoms, 3)
    trajectory_coords : np.ndarray
        Trajectory coordinates (n_frames, n_atoms, 3)

    Returns
    -------
    int
        Index of closest frame
    """
    n_frames = trajectory_coords.shape[0]
    rmsd_values = np.zeros(n_frames)

    for i in range(n_frames):
        rmsd_values[i] = np.sqrt(
            np.mean(np.sum((target_coords - trajectory_coords[i]) ** 2, axis=1))
        )

    return np.argmin(rmsd_values)


def center_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Center coordinates by removing center of mass.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array (n_frames, n_atoms, 3) or (n_atoms, 3)

    Returns
    -------
    np.ndarray
        Centered coordinates
    """
    if coords.ndim == 3:
        # Multiple frames
        centered = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            com = np.mean(coords[i], axis=0)
            centered[i] = coords[i] - com
        return centered
    elif coords.ndim == 2:
        # Single frame
        com = np.mean(coords, axis=0)
        return coords - com
    else:
        raise ValueError("Coordinates must be 2D or 3D array")


def calculate_radius_of_gyration(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Calculate radius of gyration for coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array (n_frames, n_atoms, 3) or (n_atoms, 3)
    masses : np.ndarray, optional
        Atomic masses (n_atoms,). If None, uses unit masses.

    Returns
    -------
    float or np.ndarray
        Radius of gyration value(s)
    """
    if coords.ndim == 3:
        # Multiple frames
        n_frames = coords.shape[0]
        rg_values = np.zeros(n_frames)

        for i in range(n_frames):
            rg_values[i] = _calculate_single_rg(coords[i], masses)

        return rg_values

    elif coords.ndim == 2:
        # Single frame
        return _calculate_single_rg(coords, masses)

    else:
        raise ValueError("Coordinates must be 2D or 3D array")


def _calculate_single_rg(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> float:
    """Calculate radius of gyration for single frame."""
    if masses is None:
        masses = np.ones(coords.shape[0])
    else:
        masses = np.asarray(masses)

    # Calculate center of mass
    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    # Calculate squared distances from COM
    diff = coords - com
    sq_distances = np.sum(diff**2, axis=1)

    # Calculate radius of gyration
    rg_sq = np.sum(masses * sq_distances) / total_mass

    return np.sqrt(rg_sq)


def save_representative_structures(
    universe: "mda.Universe",
    frame_indices: List[int],
    output_dir: str,
    prefix: str = "frame",
    selection: str = "all",
) -> List[str]:
    """
    Save multiple representative structures.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The trajectory universe
    frame_indices : list
        List of frame indices to export
    output_dir : str
        Output directory
    prefix : str, optional
        Filename prefix (default: "frame")
    selection : str, optional
        Atom selection to export (default: "all")

    Returns
    -------
    list
        List of output file paths
    """
    import os

    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis is required for structure export")

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for i, frame_idx in enumerate(frame_indices):
        filename = f"{prefix}_{frame_idx:04d}.pdb"
        output_path = os.path.join(output_dir, filename)

        export_structure(universe, frame_idx, output_path, selection)
        output_files.append(output_path)

    print(f"Saved {len(frame_indices)} representative structures to {output_dir}")

    return output_files


def validate_trajectory_input(topology: str, trajectory: str) -> bool:
    """
    Validate that trajectory files exist and are readable.

    Parameters
    ----------
    topology : str
        Path to topology file
    trajectory : str
        Path to trajectory file

    Returns
    -------
    bool
        True if files are valid
    """
    import os

    if not os.path.exists(topology):
        raise FileNotFoundError(f"Topology file not found: {topology}")

    if not os.path.exists(trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory}")

    # Try to load with MDAnalysis
    if HAS_MDANALYSIS:
        try:
            universe = mda.Universe(topology, trajectory)
            print("Trajectory validation successful:")
            print(f"  - {len(universe.trajectory)} frames")
            if universe.atoms is not None:
                print(f"  - {len(universe.atoms)} atoms")
            else:
                print("  - No atoms found in universe")
            return True
        except Exception as e:
            raise ValueError(f"Failed to load trajectory: {e}")
    else:
        print("Warning: MDAnalysis not available for validation")
        return True
