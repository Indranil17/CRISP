"""
CRISP/simulation_utility/atomic_indices.py

This module extracts atomic indices from trajectory files
and identifying atom pairs within specified cutoff distances.
"""

import os
import numpy as np
import ase.io
import csv
from typing import Dict, Tuple, List, Optional

__all__ = ['atom_indices', 'run_atom_indices']

def atom_indices(
    traj_path: str,
    frame_index: int = 0,
    custom_cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[Dict[str, List[int]], np.ndarray, Dict[Tuple[str, str], List[Tuple[int, int, float]]]]:
    """Extract atom indices by chemical symbol and find atom pairs within specified cutoffs.
    
    Parameters
    ----------
    traj_path : str
        Path to the trajectory file in any format supported by ASE
    frame_index : int, optional
        Index of the frame to analyze (default: 0)
    custom_cutoffs : dict, optional
        Dictionary with atom symbol pairs as keys and cutoff distances as values
        Example: {('Si', 'O'): 2.0, ('Al', 'O'): 2.1}
        
    Returns
    -------
    indices_by_symbol : dict
        Dictionary with chemical symbols as keys and lists of atomic indices as values
    dist_matrix : numpy.ndarray
        Distance matrix between all atoms, accounting for periodic boundary conditions
    cutoff_indices : dict
        Dictionary with atom symbol pairs as keys and lists of (idx1, idx2, distance) tuples
        for atoms that are within the specified cutoff distance
    """
    try:
        new = ase.io.read(traj_path, index=frame_index)
        dist_matrix = new.get_all_distances(mic=True)
        symbols = new.get_chemical_symbols()

        unique_symbols = list(set(symbols))

        indices_by_symbol = {symbol: [] for symbol in unique_symbols}

        for idx, atom in enumerate(new):
            indices_by_symbol[atom.symbol].append(idx)

        cutoff_indices = {}

        if custom_cutoffs:
            for pair, cutoff in custom_cutoffs.items():
                symbol1, symbol2 = pair
                pair_indices_distances = []
                if symbol1 in indices_by_symbol and symbol2 in indices_by_symbol:
                    for idx1 in indices_by_symbol[symbol1]:
                        for idx2 in indices_by_symbol[symbol2]:
                            if dist_matrix[idx1, idx2] < cutoff:
                                pair_indices_distances.append(
                                    (idx1, idx2, dist_matrix[idx1, idx2])
                                )
                cutoff_indices[pair] = pair_indices_distances

        return indices_by_symbol, dist_matrix, cutoff_indices
    
    except Exception as e:
        raise ValueError(f"Error processing atomic structure: {e}. Check if the format is supported by ASE.")


def run_atom_indices(
    traj_path: str,
    output_dir: str,
    frame_index: int = 0,
    custom_cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> None:
    """Run atom index extraction and save results to files.
    
    Parameters
    ----------
    traj_path : str
        Path to the trajectory file in any format supported by ASE
    output_dir : str
        Directory where output files will be saved
    frame_index : int, optional
        Index of the frame to analyze (default: 0)
    custom_cutoffs : dict, optional
        Dictionary with atom symbol pairs as keys and cutoff distances as values
        
    Returns
    -------
    None
        Results are saved to the specified output directory:
        - lengths.npy: Dictionary of number of atoms per element
        - {symbol}_indices.npy: Numpy array of atom indices for each element
        - cutoff/{symbol1}-{symbol2}_cutoff.csv: CSV files with atom pairs within cutoff
    """
    try:
        try:
            traj = ase.io.read(traj_path, index=":")
            if isinstance(traj, list):
                traj_length = len(traj)
            else:
                traj_length = 1
        except TypeError:
            ase.io.read(traj_path)  
            traj_length = 1
            
        # Check if frame_index is within valid range
        if frame_index < 0 or frame_index >= traj_length:
            raise ValueError(
                f"Error: Frame index {frame_index} is out of range. "
                f"Valid range is 0 to {traj_length-1}."
            )
            
        print(f"Analyzing frame with index {frame_index} (out of {traj_length} frames)")
        
    except ValueError as e:
        raise e
        
    except Exception as e:
        raise ValueError(f"Error reading trajectory: {e}. Check if the format is supported by ASE.")
    
    indices, dist_matrix, cutoff_indices = atom_indices(traj_path, frame_index, custom_cutoffs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lengths = {symbol: len(indices[symbol]) for symbol in indices}
    np.save(os.path.join(output_dir, "lengths.npy"), lengths)

    for symbol, data in indices.items():
        np.save(os.path.join(output_dir, f"{symbol}_indices.npy"), data)
        print(f"Length of {symbol} indices: {len(data)}")

    print("Outputs saved.")

    cutoff_folder = os.path.join(output_dir, "cutoff")
    if not os.path.exists(cutoff_folder):
        os.makedirs(cutoff_folder)

    for pair, pair_indices_distances in cutoff_indices.items():
        symbol1, symbol2 = pair
        filename = f"{symbol1}-{symbol2}_cutoff.csv"
        filepath = os.path.join(cutoff_folder, filename)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"{symbol1} index", f"{symbol2} index", "distance"])
            writer.writerows(pair_indices_distances)
        print(f"Saved cutoff indices for {symbol1}-{symbol2} to {filepath}")