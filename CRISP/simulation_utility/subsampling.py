"""
CRISP/simulation_utility/subsampling.py

This module provides functionality for structure subsampling from molecular dynamics
trajectories using Farthest Point Sampling (FPS) with SOAP descriptors.
"""

import numpy as np
from ase.io import read, write
import fpsample
import glob
import os
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Union, List, Optional

__all__ = ['indices', 'compute_soap', 'create_repres', 'subsample']


def indices(atoms, ind: Union[str, List[Union[int, str]]]) -> np.ndarray:
    """
    Extract atom indices from an ASE Atoms object based on the input specifier.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic structure
    ind : Union[str, List[Union[int, str]]]
        Index specifier, can be:
        - "all" or None: all atoms
        - string ending with ".npy": load indices from NumPy file
        - integer or list of integers: direct atom indices
        - string or list of strings: chemical symbols to select
        
    Returns
    -------
    np.ndarray
        Array of selected indices
        
    Raises
    ------
    ValueError
        If the index type is invalid
    """
    # Select all atoms
    if ind == "all" or ind is None:
        return np.arange(len(atoms))
    
    # Load from NumPy file
    if isinstance(ind, str) and ind.endswith(".npy"):
        return np.load(ind, allow_pickle=True)
    
    # Convert single items to list
    if not isinstance(ind, list):
        ind = [ind]
    
    # Handle integer indices directly
    if any(isinstance(item, int) for item in ind):
        return np.array(ind)
    
    # Handle chemical symbols
    if any(isinstance(item, str) for item in ind):
        idx = []
        for symbol in ind:
            idx.append(np.where(np.array(atoms.get_chemical_symbols()) == symbol)[0])
        return np.concatenate(idx)
    
    raise ValueError("Invalid index type")


def compute_soap(
    structure,
    all_spec: List[str],
    rcut: float,
    idx: np.ndarray
) -> np.ndarray:
    """Compute SOAP descriptors for a given structure.
    
    Parameters
    ----------
    structure : ase.Atoms
        Atomic structure for which to compute SOAP descriptors
    all_spec : list
        List of chemical elements to include in the descriptor
    rcut : float
        Cutoff radius for the SOAP descriptor in Angstroms
    idx : numpy.ndarray
        Indices of atoms to use as centers for SOAP calculation
        
    Returns
    -------
    numpy.ndarray
        Average SOAP descriptor vector for the structure
    """
    periodic_cell = structure.cell.volume > 0
    soap = SOAP(
        species=all_spec,
        periodic=periodic_cell,
        r_cut=rcut,
        n_max=8,
        l_max=6,
        sigma=0.5,
        sparse=False
    )
    soap_ind = soap.create(structure, centers=idx)
    return np.mean(soap_ind, axis=0)


def create_repres(
    traj_path: List,
    rcut: float = 6,
    ind: Union[str, List[Union[int, str]]] = "all",
    n_jobs: int = -1
) -> np.ndarray:
    """Create SOAP representation vectors for a trajectory.
    
    Parameters
    ----------
    traj_path : list
        List of ase.Atoms objects representing a trajectory
    rcut : float, optional
        Cutoff radius for the SOAP descriptor in Angstroms (default: 6)
    ind : str, list, or None, optional
        Specification for which atoms to use as SOAP centers (default: "all")
    n_jobs : int, optional
        Number of parallel jobs to run; -1 uses all available cores (default: -1)
        
    Returns
    -------
    numpy.ndarray
        Array of SOAP descriptors for each frame in the trajectory
    """
    all_spec = traj_path[0].get_chemical_symbols()
    idx = indices(traj_path[0], ind=ind)

    repres = Parallel(n_jobs=n_jobs)(
        delayed(compute_soap)(structure, all_spec, rcut, idx) for structure in traj_path
    )

    return np.array(repres)


def subsample(
    traj_path: str,
    n_samples: int = 50,
    index_type: Union[str, List[Union[int, str]]] = "all",
    rcut: float = 6.0,
    file_format: Optional[str] = None,
    plot_subsample: bool = False,
    frame_skip: int = 1,
    output_dir: str = "subsampled_structures"
) -> None:
    """Subsample a trajectory using Farthest Point Sampling with SOAP descriptors.
    
    Parameters
    ----------
    traj_path : str
        Path pattern to trajectory file(s); supports globbing
    n_samples : int, optional
        Number of frames to select (default: 50)
    index_type : str, list, or None, optional
        Specification for which atoms to use for SOAP calculation (default: "all")
    rcut : float, optional
        Cutoff radius for SOAP in Angstroms (default: 6.0)
    file_format : str, optional
        File format for ASE I/O (default: None, auto-detect)
    plot_subsample : bool, optional
        Whether to generate a plot of FPS distances (default: False)
    frame_skip : int, optional
        Read every nth frame from the trajectory (default: 1)
    output_dir : str, optional
        Directory to save the subsampled structures (default: "subsampled_structures")
        
    Returns
    -------
    list
        List of selected ase.Atoms frames
        
    Notes
    -----
    The selected frames and plots are saved in the specified output directory
    """
    traj_files = glob.glob(traj_path)
    
    # Check if any matching files were found
    if not traj_files:
        raise ValueError(f"No files found matching pattern: {traj_path}")

    trajec = []
    for file in traj_files:
        if file_format is not None:
            trajec += read(file, index=f'::{frame_skip}', format=file_format)
        else:
            trajec += read(file, index=f'::{frame_skip}')

    if not isinstance(trajec, list): 
        trajec = [trajec]
    
    repres = create_repres(trajec, ind=index_type, rcut=rcut)
    
    # Ensure we don't request more samples than available frames
    n_samples = min(n_samples, len(trajec))
    
    perm = fpsample.fps_sampling(repres, n_samples, start_idx=0)

    fps_frames = []

    for str_idx, frame in enumerate(perm):
        new_frame = trajec[frame]
        fps_frames.append(new_frame)

    os.makedirs(output_dir, exist_ok=True)

    if plot_subsample:
        distance = []
        for i in range(1, len(perm)):
            distance.append(np.min(np.linalg.norm(repres[perm[:i]] - repres[perm[i]], axis=1)))

        plt.figure(figsize=(8, 6))
        plt.plot(distance, c="blue", linewidth=2)
        plt.ylim([0, 1.1 * max(distance)])
        plt.xlabel("Number of subsampled structures")
        plt.ylabel("Euclidean distance")
        plt.title("FPS Subsampling")
        plt.savefig(os.path.join(output_dir, "subsampled_convergence.png"), dpi=300)
        plt.show()
        plt.close()
        print(f"Saved convergence plot to {os.path.join(output_dir, 'subsampled_convergence.png')}")

    # Extract the base filename without path for output file using os.path for platform independence
    base_filename = os.path.basename(traj_files[0])
    output_file = os.path.join(output_dir, f"subsample_{base_filename}")
    
    try:
        write(output_file, fps_frames, format=file_format)
        print(f"Saved {len(fps_frames)} subsampled structures to {output_file}")
    except Exception as e:
        print(f"Error saving subsampled structures: {e}")

    return fps_frames
