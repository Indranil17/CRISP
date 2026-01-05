"""
CRISP/data_analysis/prdf.py

This script performs Radial Distribution Function analysis on molecular dynamics trajectory data.
"""

import os
import numpy as np
import pickle
import math
from ase.io import read
from ase import Atoms
from typing import Optional, Union, Tuple, List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

__all__ = ['check_cell_and_r_max', 'compute_pairwise_rdf', 'plot_rdf', 'animate_rdf', 'analyze_rdf']

def check_cell_and_r_max(atoms: Atoms, rmax: float):
    """
    Check that the cell is large enough to contain a sphere of radius rmax.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with cell information
    rmax : float
        Maximum radius to consider
        
    Raises
    ------
    ValueError
        If cell is not defined or too small for requested rmax
    """
    if not atoms.cell.any():
        raise ValueError("RDF Error: The system's cell is not defined.")
    
    cell = atoms.cell
    try:
        lengths = cell.lengths()
        if np.min(lengths) < 2 * rmax:
            raise ValueError(f"RDF Error: Cell length {np.min(lengths)} is smaller than 2*rmax ({2*rmax}).")
    except AttributeError:
        volume = cell.volume
        required_volume = (4/3) * math.pi * rmax**3
        if volume < required_volume:
            raise ValueError(f"RDF Error: Cell volume {volume} is too small for rmax {rmax} (required >= {required_volume}).")

def compute_pairwise_rdf(atoms: Atoms,
                         ref_indices: List[int],
                         target_indices: List[int],
                         rmax: float, nbins: int,
                         volume: Optional[float] = None):
    """
    Compute pairwise radial distribution function between sets of atoms.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    ref_indices : List[int]
        Indices of reference atoms
    target_indices : List[int]
        Indices of target atoms
    rmax : float
        Maximum radius for RDF calculation
    nbins : int
        Number of bins for histogram
    volume : float, optional
        Custom normalization volume to use instead of cell volume.
        Useful for non-periodic systems or custom normalization.
        (default: None, uses atoms.get_volume())
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        RDF values and corresponding bin centers
    """
    N_total = len(atoms)
    dm = atoms.get_all_distances(mic=True)
    dr = float(rmax / nbins)
    volume = atoms.get_volume() if volume is None else volume

    
    if set(ref_indices) == set(target_indices):
        sub_dm = dm[np.ix_(ref_indices, target_indices)]
        sub_dm = np.triu(sub_dm, k=1)  # Exclude diagonal and use upper triangle
        distances = sub_dm[sub_dm > 0]
        
        N = len(ref_indices)
        
        # Division by 2 for same-species pairs
        norm = (4 * math.pi * dr * (N/volume) * N)/2
    else:
        sub_dm = dm[np.ix_(ref_indices, target_indices)]
        distances = sub_dm[sub_dm > 0]
        
        N_A = len(ref_indices)
        N_B = len(target_indices)
        
        norm = 4 * math.pi * dr * (N_A / volume) * N_B

    hist, bin_edges = np.histogram(distances, bins=nbins, range=(0, rmax))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    rdf = hist / (norm * (bin_centers**2))
                  
    return rdf, bin_centers

class Analysis:
    """
    Class for analyzing atomic trajectories and calculating RDFs.
    
    Parameters
    ----------
    images : List[Atoms]
        List of ASE Atoms objects representing trajectory frames
    """
    
    def __init__(self, images: List[Atoms]):
        self.images = images

    def _get_slice(self, imageIdx: Optional[Union[int, slice]]):
        """
        Convert image index to slice for selecting trajectory frames.
        
        Parameters
        ----------
        imageIdx : Optional[Union[int, slice]]
            Index or slice to select images
            
        Returns
        -------
        slice
            Slice object for image selection
        """
        if imageIdx is None:
            return slice(None)
        return imageIdx

    def get_rdf(self, 
                rmax: float,
                nbins: int = 100,
                imageIdx: Optional[Union[int, slice]] = None,
                atomic_indices: Optional[Tuple[List[int], List[int]]] = None,
                return_dists: bool = False):
        """
        Calculate radial distribution function for trajectory frames.
        
        Parameters
        ----------
        rmax : float
            Maximum radius for RDF calculation
        nbins : int, optional
            Number of bins for histogram (default: 100)
        imageIdx : Optional[Union[int, slice]], optional
            Index or slice to select images (default: None, all images)
        atomic_indices : Optional[Tuple[List[int], List[int]]], optional
            Tuple of (reference_indices, target_indices) for partial RDF
        return_dists : bool, optional
            Whether to return bin center distances (default: False)
            
        Returns
        -------
        List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
            List of RDF values or tuples (RDF, bin_centers) for each frame
        """
        sl = self._get_slice(imageIdx)
        images_to_process = self.images[sl]

        if atomic_indices is None:
            def process_image(image: Atoms):
                check_cell_and_r_max(image, rmax)
                full_indices = list(range(len(image)))
                rdf, bin_centers = compute_pairwise_rdf(image, full_indices, full_indices, rmax, nbins)
                return (rdf, bin_centers) if return_dists else rdf
        else:
            ref_indices, target_indices = atomic_indices
            def process_image(image: Atoms):
                check_cell_and_r_max(image, rmax)
                rdf, bin_centers = compute_pairwise_rdf(
                    image,  
                    ref_indices, 
                    target_indices,
                    rmax, 
                    nbins
                )
                return (rdf, bin_centers) if return_dists else rdf

        ls_rdf = Parallel(n_jobs=-1)(delayed(process_image)(image) for image in images_to_process)
        return ls_rdf

def plot_rdf(x_data_all, y_data_all, title=None, output_file=None):
    """
    Plot the average RDF with peak location marked by vertical line.
    
    Parameters
    ----------
    x_data_all : np.ndarray
        Distance values in Ångström
    y_data_all : List[np.ndarray]
        RDF values for each frame
    title : str, optional
        Custom title for the plot
    output_file : str, optional
        Path to save the plot (if None, just display)
        
    Returns
    -------
    None
    """
    # Average RDF across all frames
    y_data_avg = np.mean(y_data_all, axis=0)

    # Index of the maximum y value in the average RDF
    max_y_index = np.argmax(y_data_avg)
    max_y_x = x_data_all[max_y_index]
    max_y = y_data_avg[max_y_index]

    plt.figure(figsize=(10, 6))
    
    plt.plot(x_data_all, y_data_avg, linewidth=2, label='Average RDF')
    
    plt.axvline(x=max_y_x, color='red', linestyle='--', label=f'Peak at {max_y_x:.2f} Å')
    
    plt.xlabel('Distance (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title(title or 'Average Radial Distribution Function', fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.ylim(bottom=0, top=max_y * 1.2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()  # Added to display plot in addition to saving it
        plt.close()
    else:
        plt.show()


def animate_rdf(x_data_all, y_data_all, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_data_avg = np.mean(y_data_all, axis=0)

    plt.xlabel('Distance (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    
    max_y = max([np.max(y) for y in y_data_all] + [np.max(y_data_avg)]) * 1.1
    
    def update(frame):
        ax.clear()  # Clears the previous frame
        y = y_data_all[frame]
        
        ax.plot(x_data_all, y, linewidth=2, label='Current Frame')
        
        ax.plot(x_data_all, y_data_avg, linewidth=2, linestyle='--', 
                color='purple', label='Average RDF')
        
        max_y_index = np.argmax(y)
        max_y_x = x_data_all[max_y_index]
        
        ax.axvline(x=max_y_x, color='red', linestyle='--', label=f'Peak at {max_y_x:.2f} Å')
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_y)
        ax.set_title(f'Radial Distribution Function - Frame {frame}', fontsize=14)
        return ax,

    ani = FuncAnimation(fig, update, frames=range(len(y_data_all)), 
                       interval=200, blit=False)

    if output_file:
        html_file = os.path.splitext(output_file)[0] + ".html"
        html_code = ani.to_jshtml()
        with open(html_file, 'w') as f:
            f.write(html_code)
        print(f"Interactive animation saved to '{html_file}'")
        
        try:
            ani.save(output_file, writer='pillow', fps=5)
            print(f"GIF animation saved to '{output_file}'")
        except Exception as e:
            print(f"Warning: Could not save GIF animation: {e}")
        
        plt.tight_layout() 
        plt.close()
    else:
        plt.tight_layout()
    
    return ani

def analyze_rdf(use_prdf: bool,
                rmax: float,
                traj_path: str,
                nbins: int = 100,
                frame_skip: int = 10,
                output_filename: Optional[str] = None,
                atomic_indices: Optional[Tuple[List[int], List[int]]] = None,
                output_dir: str = 'custom_ase',
                create_plots: bool = False):  
    """
    Analyze trajectory and calculate radial distribution functions.
    
    Parameters
    ----------
    use_prdf : bool
        Whether to calculate partial RDF (True) or total RDF (False)
    rmax : float
        Maximum radius for RDF calculation
    traj_path : str
        Path to trajectory file
    nbins : int, optional
        Number of bins for histogram (default: 100)
    frame_skip : int, optional
        Number of frames to skip between analyses (default: 10)
    output_filename : Optional[str], optional
        Custom filename for output (default: None, auto-generated)
    atomic_indices : Optional[Tuple[List[int], List[int]]], optional
        Tuple of (reference_indices, target_indices) for partial RDF
    output_dir : str, optional
        Directory to save output files (default: 'custom_ase')
    create_plots : bool, optional
        Whether to create plots and animations of the RDF data (default: False)
        
    Returns
    -------
    dict
        Dictionary containing x_data (bin centers) and y_data_all (RDF values for each frame)
        
    Raises
    ------
    ValueError
        If no images found in trajectory or if atomic_indices is missing for PRDF
    """
    images = read(traj_path, index=f'::{frame_skip}')
    if not isinstance(images, list):
        images = [images]  
        
    if not images:
        raise ValueError("No images found in the trajectory.")
        
    # Check cell validity for the first image
    check_cell_and_r_max(images[0], rmax)
    
    analysis = Analysis(images)
    
    if use_prdf:
        if atomic_indices is None:
            raise ValueError("For partial RDF, atomic_indices must be provided.")
        ls_rdf = analysis.get_rdf(rmax, nbins, atomic_indices=atomic_indices, return_dists=True)
    else:
        ls_rdf = analysis.get_rdf(rmax, nbins, atomic_indices=None, return_dists=True)
    
    x_data_all = ls_rdf[0][1]
    y_data_all = [rdf for rdf, _ in ls_rdf]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not output_filename:
        if use_prdf:
            if atomic_indices:
                ref_str = f"{len(atomic_indices[0])}-atoms"
                target_str = f"{len(atomic_indices[1])}-atoms"
                base_name = f"prdf_{ref_str}_{target_str}"
            else:
                base_name = "prdf_custom_indices"
        else:
            base_name = "rdf_total"
    else:
        base_name = output_filename.rsplit('.', 1)[0] if '.' in output_filename else output_filename
    
    if output_dir:
        pickle_file = os.path.join(output_dir, f"{base_name}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump({'x_data': x_data_all, 'y_data_all': y_data_all}, f)
        
        print(f"Data saved in '{pickle_file}'")
        
        if create_plots:
            if use_prdf:
                if atomic_indices:
                    title = f"Partial RDF: {len(atomic_indices[0])} reference atoms, {len(atomic_indices[1])} target atoms"
                else:
                    title = "Partial RDF"
            else:
                title = "Total Radial Distribution Function"
                
            static_plot_file = os.path.join(output_dir, f"{base_name}_plot.png")
            plot_rdf(x_data_all, y_data_all, title=title, output_file=static_plot_file)
            print(f"Static plot saved in '{static_plot_file}'")
            
            if len(y_data_all) > 1:
                animation_file = os.path.join(output_dir, f"{base_name}_animation.gif")
                animate_rdf(x_data_all, y_data_all, output_file=animation_file)
                print(f"Animation saved in '{animation_file}'")
    
    return {'x_data': x_data_all, 'y_data_all': y_data_all}