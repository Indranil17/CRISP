"""
CRISP/data_analysis/msd.py

This module performs mean square displacement (MSD) analysis on molecular dynamics 
trajectory data for diffusion coefficient calculations.
"""

import ase.io
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from ase.units import fs
from ase.units import fs as fs_conversion
from ase.data import chemical_symbols
from scipy.optimize import curve_fit
import pandas as pd
import os
import traceback
from typing import List, Tuple, Union, Optional, Dict, Any
from joblib import Parallel, delayed, cpu_count

__all__ = ['read_trajectory_chunk', 'calculate_frame_msd', 'calculate_msd', 'calculate_save_msd', 'msd_analysis']


def read_trajectory_chunk(
    traj_path: str,
    index_slice: str,
    frame_skip: int = 1
) -> List:
    """Read a chunk of trajectory data in parallel.
    
    Parameters
    ----------
    traj_path : str
        Path to the trajectory file (supports any ASE-readable format)
    index_slice : str
        ASE index slice for reading a subset of frames
    frame_skip : int, optional
        Number of frames to skip (default: 1)
        
    Returns
    -------
    list
        List of ASE Atoms objects for the specified chunk
    """
    try:
        frames = ase.io.read(traj_path, index=index_slice)
        if not isinstance(frames, list):
            frames = [frames]
        
        frames = frames[::frame_skip]
        return frames
    except Exception as e:
        print(f"Error reading trajectory chunk {index_slice}: {e}")
        return []


def calculate_frame_msd(
    frame_idx: int,
    current_frame,
    reference_frame,
    atom_indices: List[int],
    msd_direction: bool = False
) -> Union[Tuple[int, float], Tuple[int, float, float, float]]:
    """
    Calculate MSD for a single frame.
    
    Parameters
    ----------
    frame_idx : int
        Index of the current frame
    current_frame : ase.Atoms
        Current frame
    reference_frame : ase.Atoms
        Reference frame
    atom_indices : list
        List of atom indices to include in MSD calculation
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
        
    Returns
    -------
    tuple
        If msd_direction is False: (frame_idx, msd_value)
        If msd_direction is True: (frame_idx, msd_x, msd_y, msd_z)
    """
    atom_positions_current = current_frame.positions[atom_indices]
    atom_positions_reference = reference_frame.positions[atom_indices]
    displacements = atom_positions_current - atom_positions_reference
    
    if not msd_direction:
        # Calculate total MSD
        msd_value = np.sum(np.square(displacements)) / (len(atom_indices))
        return frame_idx, msd_value
    else:
        # Directional MSDs
        msd_x = np.sum(displacements[:, 0]**2) / len(atom_indices)
        msd_y = np.sum(displacements[:, 1]**2) / len(atom_indices)
        msd_z = np.sum(displacements[:, 2]**2) / len(atom_indices)
        
        return frame_idx, msd_x, msd_y, msd_z

def calculate_msd(traj, timestep, atom_indices=None, ignore_n_images=0, n_jobs=-1, 
                 msd_direction=False, msd_direction_atom=None):
    """
    Calculate Mean Square Displacement (MSD) vs time using parallel processing.
    
    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory data
    timestep : float
        Simulation timestep
    atom_indices : numpy.ndarray, optional
        Indices of atoms to analyze (default: all atoms)
    ignore_n_images : int, optional
        Number of initial images to ignore (default: 0)
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1, use all available cores)
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
        If True and atom_indices is provided, directional MSD is calculated for those indices
    msd_direction_atom : str or int, optional
        Atom symbol or atomic number to filter for directional MSD (default: None)
        Only used when atom_indices is None
        
    Returns
    -------
    tuple or dict
        If atom_indices is provided: (msd_times, msd_x, msd_y, msd_z) if msd_direction=True
                                else (msd_values, msd_times) 
    If atom_indices is None: A dictionary with keys for each atom type
    
"""
    # Time values
    total_images = len(traj) - ignore_n_images
    timesteps = np.linspace(0, total_images * timestep, total_images+1)
    msd_times = timesteps[:] / fs_conversion  # Convert to femtoseconds
    
    # Reference frame
    reference_frame = traj[ignore_n_images]
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    direction_indices = None
    if msd_direction and msd_direction_atom is not None:
        atoms = traj[0]
        if isinstance(msd_direction_atom, str):
            # An atom symbol (e.g., 'O')
            symbols = atoms.get_chemical_symbols()
            direction_indices = [i for i, s in enumerate(symbols) if s == msd_direction_atom]
            print(f"Calculating directional MSD for {len(direction_indices)} {msd_direction_atom} atoms")
        elif isinstance(msd_direction_atom, int):
            # An atomic number (e.g., 8 for oxygen)
            atomic_numbers = atoms.get_atomic_numbers()
            direction_indices = [i for i, z in enumerate(atomic_numbers) if z == msd_direction_atom]
            symbol = chemical_symbols[msd_direction_atom]
            print(f"Calculating directional MSD for {len(direction_indices)} {symbol} atoms (Z={msd_direction_atom})")
    
    # MSD for those atoms
    if atom_indices is not None:
        do_direction = msd_direction
        
        # Parallelize MSD calculation
        results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_frame_msd)(
                i - ignore_n_images, 
                traj[i], 
                reference_frame, 
                atom_indices,
                do_direction
            )
            for i in range(ignore_n_images, len(traj))
        )
        
        # Sort results by frame index and extract MSD values
        results.sort(key=lambda x: x[0])
        
        if do_direction:
            print(f"Calculating directional MSD for {len(atom_indices)} specified atoms")
            msd_x = np.array([r[1] for r in results])
            msd_y = np.array([r[2] for r in results])
            msd_z = np.array([r[3] for r in results])
            return msd_times, msd_x, msd_y, msd_z
        else:
            msd_values = np.array([r[1] for r in results])
            return msd_values, msd_times[:]
    
    # MSD per atom type
    else:
        atoms = traj[0]
        symbols = atoms.get_chemical_symbols()
        unique_symbols = set(symbols)
        
        # A dictionary mapping symbols to their indices
        symbol_indices = {symbol: [i for i, s in enumerate(symbols) if s == symbol] 
                         for symbol in unique_symbols}
        
        # Overall MSD using all atoms
        all_indices = list(range(len(atoms)))
        
        overall_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_frame_msd)(
                i - ignore_n_images, 
                traj[i], 
                reference_frame, 
                all_indices,
                False  
            )
            for i in range(ignore_n_images, len(traj))
        )
        
        # Sort results by frame index and extract MSD values
        overall_results.sort(key=lambda x: x[0])
        overall_msd = np.array([r[1] for r in overall_results])
        
        # Dictionary to store MSD results
        result = {'overall': (overall_msd, msd_times)}
        
        # Calculate MSD for each atom type in parallel
        for symbol, indices in symbol_indices.items():
            print(f"Calculating MSD for {symbol} atoms...")
            calc_direction = msd_direction and (
                (isinstance(msd_direction_atom, str) and symbol == msd_direction_atom) or
                (isinstance(msd_direction_atom, int) and 
                 atoms.get_atomic_numbers()[indices[0]] == msd_direction_atom)
            )
            
            symbol_results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_frame_msd)(
                    i - ignore_n_images, 
                    traj[i], 
                    reference_frame, 
                    indices,
                    calc_direction
                )
                for i in range(ignore_n_images, len(traj))
            )
            
            # Sort results by frame index
            symbol_results.sort(key=lambda x: x[0])
            
            if calc_direction:
                msd_x = np.array([r[1] for r in symbol_results])
                msd_y = np.array([r[2] for r in symbol_results])
                msd_z = np.array([r[3] for r in symbol_results])
                
                # Fix: also store total and ensure (values, times) tuples
                total_values = msd_x + msd_y + msd_z
                result[symbol] = (total_values, msd_times)
                result[f'{symbol}_x'] = (msd_x, msd_times)
                result[f'{symbol}_y'] = (msd_y, msd_times)
                result[f'{symbol}_z'] = (msd_z, msd_times)
                
                print(f"Saved directional MSD data for {symbol} atoms")
            else:
                msd_values = np.array([r[1] for r in symbol_results])
                result[symbol] = (msd_values, msd_times)
        
        return result

def save_msd_data(msd_data, csv_file_path, output_dir="traj_csv_detailed"):
    """
    Save MSD data to CSV files.
    
    Parameters
    ----------
    msd_data : tuple or dict
        MSD data to be saved
    csv_file_path : str
        Path to the CSV file
    output_dir : str, optional
        Directory to save CSV files (default: "traj_csv_detailed")
        
    Returns
    -------
    list
        List of saved file paths
    """
    saved_files = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(csv_file_path)
    
    if isinstance(msd_data, tuple):
        if len(msd_data) == 2:
            msd_values, msd_times = msd_data
            
            csv_full_path = os.path.join(output_dir, base_filename)
            
            with open(csv_full_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])
                for time, msd in zip(msd_times, msd_values):
                    csv_writer.writerow([time, msd])
            
            print(f"MSD data has been saved to {csv_full_path}")
            saved_files.append(csv_full_path)
        
        elif len(msd_data) == 4:
            msd_times, msd_x, msd_y, msd_z = msd_data
            
            base_path, ext = os.path.splitext(base_filename)
            
            # Save total (x+y+z) to the base file
            total_values = msd_x + msd_y + msd_z
            total_path = os.path.join(output_dir, base_filename)
            with open(total_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])
                for time, msd in zip(msd_times, total_values):
                    csv_writer.writerow([time, msd])
            print(f"Total MSD data has been saved to {total_path}")
            saved_files.append(total_path)
            
            # Save directional components
            x_path = os.path.join(output_dir, f"{base_path}_x{ext}")
            with open(x_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])  
                for time, msd in zip(msd_times, msd_x):
                    csv_writer.writerow([time, msd])
            print(f"X-direction MSD data has been saved to {x_path}")
            saved_files.append(x_path)
            
            y_path = os.path.join(output_dir, f"{base_path}_y{ext}")
            with open(y_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])  
                for time, msd in zip(msd_times, msd_y):
                    csv_writer.writerow([time, msd])
            print(f"Y-direction MSD data has been saved to {y_path}")
            saved_files.append(y_path)
            
            z_path = os.path.join(output_dir, f"{base_path}_z{ext}")
            with open(z_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])  
                for time, msd in zip(msd_times, msd_z):
                    csv_writer.writerow([time, msd])
            print(f"Z-direction MSD data has been saved to {z_path}")
            saved_files.append(z_path)
    
    elif isinstance(msd_data, dict):
        base_name, ext = os.path.splitext(base_filename)
        
        if 'overall' in msd_data:
            overall_filename = f"{base_name}_overall{ext}"
            overall_path = os.path.join(output_dir, overall_filename)
            msd_values, msd_times = msd_data['overall']
            
            with open(overall_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])
                for time, msd in zip(msd_times, msd_values):
                    csv_writer.writerow([time, msd])
            
            print(f"Overall MSD data has been saved to {overall_path}")
            saved_files.append(overall_path)
        
        for symbol, data in msd_data.items():
            if symbol == 'overall':
                continue
                
            symbol_filename = f"{base_name}_{symbol}{ext}"
            symbol_path = os.path.join(output_dir, symbol_filename)
            msd_values, msd_times = data
            
            with open(symbol_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time (fs)', 'MSD'])
                for time, msd in zip(msd_times, msd_values):
                    csv_writer.writerow([time, msd])
            
            print(f"MSD data for {symbol} atoms has been saved to {symbol_path}")
            saved_files.append(symbol_path)
    
    return saved_files

def calculate_diffusion_coefficient(msd_times, msd_values, start_index=None, end_index=None, 
                                   with_intercept=False, plot_msd=False, dimension=3):
    """
    Calculate diffusion coefficient from MSD data in a general way for 1D, 2D, or 3D.
    
    Parameters
    ----------
    msd_times : numpy.ndarray
        Time values in femtoseconds.
    msd_values : numpy.ndarray
        Mean square displacement values.
    start_index : int, optional
        Starting index for the fit (default: 1/3 of data length).
    end_index : int, optional
        Ending index for the fit (default: None).
    with_intercept : bool, optional
        Whether to fit with intercept (default: False).
    plot_msd : bool, optional
        Whether to plot the fit (default: False).
    dimension : int, optional
        Dimensionality of the system (default: 3). Use 1 for 1D, 2 for 2D, 3 for 3D.
    
    Returns
    -------
    tuple
        (D, error) where D is the diffusion coefficient in cm²/s and error is the statistical error.
    """
    if start_index is None:
        start_index = len(msd_times) // 3  
    if end_index is None:
        end_index = len(msd_times)
    if start_index < 0 or end_index > len(msd_times):
        raise ValueError("Indices are out of bounds.")
    if start_index >= end_index:
        raise ValueError("Start index must be less than end index.")
    
    x_fit = msd_times[start_index:end_index]
    y_fit = msd_values[start_index:end_index]
    
    def linear_no_intercept(x, m):
        return m * x
    
    def linear_with_intercept(x, m, c):
        return m * x + c
    
    if with_intercept:
        params, covariance = curve_fit(linear_with_intercept, x_fit, y_fit)
        slope, intercept = params
        fit_func = lambda x: linear_with_intercept(x, slope, intercept)
    else:
        params, covariance = curve_fit(linear_no_intercept, x_fit, y_fit)
        slope = params[0]
        intercept = 0
        fit_func = lambda x: linear_no_intercept(x, slope)
    
    std_err = np.sqrt(np.diag(covariance))[0]
    
    # Calculate diffusion coefficient using D = slope / (2 * dimension)
    # Correct conversion from Å²/fs to cm²/s:
    # 1 Å = 10^-8 cm, 1 Å² = 10^-16 cm²
    # 1 fs = 10^-15 s
    # (Å²/fs) * (10^-16 cm²/Å²) / (10^-15 s/fs) = 10^-1 cm²/s
    conversion_angstrom2_fs_to_cm2_s = 0.1  
    D = slope / (2 * dimension) * conversion_angstrom2_fs_to_cm2_s
    error = std_err / (2 * dimension) * conversion_angstrom2_fs_to_cm2_s
    
    # goodness‐of‐fit R²
    y_model = fit_func(x_fit)
    ss_res  = np.sum((y_fit - y_model)**2)
    ss_tot  = np.sum((y_fit - np.mean(y_fit))**2)
    r2      = 1 - ss_res/ss_tot

    if plot_msd:
        plt.figure(figsize=(10, 6))
        # Convert time from fs to ps for plotting
        plt.scatter(msd_times/1000, msd_values, s=10, alpha=0.5, label='MSD data')
        plt.plot(x_fit/1000, fit_func(x_fit), 'r-', linewidth=2, 
                 label=f'D = {D:.2e} cm²/s')
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD (Å²)')
        plt.title('Mean Square Displacement vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('msd_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"R² = {r2:.4f}")
    
    return D, error

def plot_diffusion_time_series(msd_times, msd_values, min_window=10, with_intercept=False, csv_file=None, dimension=3):
    """
    Plot diffusion coefficient as a time series by calculating it over different time windows.
    
    Parameters
    ----------
    msd_times : numpy.ndarray
        Time values in femtoseconds
    msd_values : numpy.ndarray
        Mean square displacement values in Å²
    min_window : int, optional
        Minimum window size for calculating diffusion (default: 10)
    with_intercept : bool, optional
        Whether to fit with intercept (default: False)
    csv_file : str, optional
        Path to the CSV file, used for output filename (default: None)
    dimension : int, optional
        Dimensionality of the system: 1 for 1D, 2 for 2D, 3 for 3D (default: 3)
    
    Returns
    -------
    None
    """
    
    def linear_no_intercept(x, m):
        return m * x
        
    def linear_with_intercept(x, m, c):
        return m * x + c
    
    # Conversion from Å²/fs to Å²/ps
    # 1 ps = 1000 fs, so multiply by 1000
    conversion_fs_to_ps = 1000.0  
    
    diffusion_coeffs = []
    window_ends = []
        
    for end_idx in range(min_window + 1, len(msd_times)):
        x_fit = msd_times[:end_idx]
        y_fit = msd_values[:end_idx]
        
        try:
            if with_intercept:
                params, covariance = curve_fit(linear_with_intercept, x_fit, y_fit)
                slope = params[0]
            else:
                params, covariance = curve_fit(linear_no_intercept, x_fit, y_fit)
                slope = params[0]
                
            D = slope / (2 * dimension) * conversion_fs_to_ps
            
            diffusion_coeffs.append(D)
            window_ends.append(msd_times[end_idx-1])
        except:
            continue
    
    plt.figure(figsize=(10, 6))
    
    window_ends_ps = np.array(window_ends) / 1000.0
    
    # Plot diffusion coefficient vs. time
    plt.plot(window_ends_ps, diffusion_coeffs, 'b-', linewidth=2, label='Diffusion Coefficient')
    
    if len(diffusion_coeffs) > 1:
        avg_diffusion = np.mean(diffusion_coeffs)
        std_diffusion = np.std(diffusion_coeffs, ddof=1)  
        plt.axhline(y=avg_diffusion, color='r', linestyle='--', 
                   label=f'Average D = {avg_diffusion:.2e} ± {std_diffusion:.2e} Å²/ps')
        
        plt.axhspan(avg_diffusion - std_diffusion, avg_diffusion + std_diffusion, 
                   color='r', alpha=0.2)
    
    plt.xlabel('Time Window End (ps)')
    plt.ylabel('Diffusion Coefficient (Å²/ps)')
    plt.title(f'{dimension}D Diffusion Coefficient Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = 'diffusion_time_series.png'
    if csv_file:
        dir_name = os.path.dirname(csv_file)
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = os.path.join(dir_name, f"{base_name}_{dimension}D_diffusion_evolution.png")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Diffusion coefficient evolution plot saved to: {output_file}")

def calculate_save_msd(traj_path, timestep_fs, indices_path=None, 
                      ignore_n_images=0, output_file="msd_results.csv", 
                      frame_skip=1, n_jobs=-1, output_dir="traj_csv_detailed",
                      msd_direction=False, msd_direction_atom=None,
                      use_windowed=True, lag_times_fs=None):
    """
    Calculate MSD data and save to CSV file.
    
    Parameters
    ----------
    traj_path : str
        Path to the ASE trajectory file
    timestep_fs : float
        Simulation timestep in femtoseconds (fs)
    indices_path : str, optional
        Path to file containing atom indices (default: None)
    ignore_n_images : int, optional
        Number of initial images to ignore (default: 0)
    output_file : str, optional
        Output CSV file path (default: "msd_results.csv")
    frame_skip : int, optional
        Number of frames to skip between samples (default: 1)
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1, use all available cores)
    output_dir : str, optional
        Directory to save CSV files (default: "traj_csv_detailed")
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
    msd_direction_atom : str or int, optional
        Atom symbol or atomic number for directional analysis (default: None)
    use_windowed : bool, optional
        Whether to use the windowed approach for more robust statistics (default: True)
    lag_times_fs : list of float, optional
        List of lag times (in fs) for which to compute MSD (default: None, use all possible lags)
        
    Returns
    -------
    tuple or dict
        MSD values and corresponding time values
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    if indices_path is not None and not os.path.exists(indices_path):
        raise FileNotFoundError(f"Indices file not found: {indices_path}")

    try:
        traj = ase.io.read(traj_path, index=f'::{frame_skip}')
        if not isinstance(traj, list):
            traj = [traj]
        
        print(f"Loaded {len(traj)} frames after applying frame_skip={frame_skip}")

        traj_nodrift = []
        for frame in traj:
            new_frame = frame.copy()
            com = new_frame.get_center_of_mass()
            new_frame.set_positions(new_frame.get_positions() - com)
            traj_nodrift.append(new_frame)
        traj = traj_nodrift

        
    except Exception as e:
        print(f"Error loading trajectory file: {e}")
        return None, None

    atom_indices = None
    if indices_path:
        try:
            atom_indices = np.load(indices_path)
            print(f"Loaded {len(atom_indices)} atom indices")
        except Exception as e:
            print(f"Error loading atom indices: {e}")
            return None, None

    timestep = timestep_fs * fs
    print(f"Using timestep: {timestep_fs} fs")

    print("Calculating MSD using parallel processing...")
    if use_windowed:
        print("Using windowed approach for MSD calculation (averaging over all time origins)")
        if indices_path:
            msd_data = calculate_msd_windowed(
                traj=traj, 
                timestep=timestep,
                atom_indices=atom_indices,
                ignore_n_images=ignore_n_images,
                n_jobs=n_jobs,
                msd_direction=msd_direction,
                lag_times_fs=lag_times_fs
            )
        else:
            msd_data = calculate_msd_windowed(
                traj=traj, 
                timestep=timestep,
                atom_indices=atom_indices,
                ignore_n_images=ignore_n_images,
                n_jobs=n_jobs,
                msd_direction=msd_direction,
                msd_direction_atom=msd_direction_atom,
                lag_times_fs=lag_times_fs
            )
    else:
        print("Using single reference frame approach for MSD calculation")
        if indices_path:
            msd_data = calculate_msd(
                traj=traj, 
                timestep=timestep,
                atom_indices=atom_indices,
                ignore_n_images=ignore_n_images,
                n_jobs=n_jobs,
                msd_direction=msd_direction
            )
        else:
            msd_data = calculate_msd(
                traj=traj, 
                timestep=timestep,
                atom_indices=atom_indices,
                ignore_n_images=ignore_n_images,
                n_jobs=n_jobs,
                msd_direction=msd_direction,
                msd_direction_atom=msd_direction_atom
            )

    saved_files = save_msd_data(
        msd_data=msd_data, 
        csv_file_path=output_file,
        output_dir=output_dir
    )
    
    return msd_data

def analyze_from_csv(csv_file="msd_results.csv", fit_start=None, fit_end=None, dimension=3,
                    with_intercept=False, plot_msd=False, plot_diffusion=False,
                    use_block_averaging=False, n_blocks=10):
    """
    Analyze MSD data from a CSV file with block averaging by default.
    
    Parameters
    ----------
    csv_file : str, optional
        Path to the CSV file containing MSD data (default: "msd_results.csv")
    fit_start : int, optional
        Start index for fitting or visualization if using block averaging (default: None)
    fit_end : int, optional
        End index for fitting or visualization if using block averaging (default: None)
    dimension : int, optional
        Dimensionality of the system: 1 for 1D, 2 for 2D, 3 for 3D (default: 3)
    with_intercept : bool, optional
        Whether to fit with intercept (default: False)
    plot_msd : bool, optional
        Whether to plot MSD vs time (default: False)
    plot_diffusion : bool, optional
        Whether to plot diffusion coefficient as time series (default: False)
    use_block_averaging : bool, optional
        Whether to use block averaging for error estimation (default: True)
    n_blocks : int, optional
        Number of blocks for block averaging (default: 10)
    
    Returns
    -------
    tuple
        (D, error) where D is the diffusion coefficient in cm²/s and error is the statistical error
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded MSD data from {csv_file}")
        
        # Extract time and MSD values
        msd_times = df['Time (fs)'].values
        msd_values = df['MSD'].values
        
        # Diffusion coefficient
        if use_block_averaging:
            # Use fit_start and fit_end to select the fit zone
            fit_start_idx = fit_start if fit_start is not None else 0
            fit_end_idx = fit_end if fit_end is not None else len(msd_times)
            
            msd_times_fit = msd_times[fit_start_idx:fit_end_idx]
            msd_values_fit = msd_values[fit_start_idx:fit_end_idx]
            
            D, error = block_averaging_error(
                msd_times=msd_times_fit, 
                msd_values=msd_values_fit,
                n_blocks=n_blocks,
                dimension=dimension,
                with_intercept=with_intercept
            )
            
            print(f"Using block averaging method with {n_blocks} blocks")
            
            # Plot MSD with the block averaging diffusion coefficient
            if plot_msd:
                visualization_start = fit_start if fit_start is not None else int(len(msd_times) * 0.3)
                visualization_end = fit_end if fit_end is not None else int(len(msd_times) * 0.8)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(msd_times/1000, msd_values, s=10, alpha=0.5, label='MSD data')
                
                x_fit = msd_times[visualization_start:visualization_end]
                if with_intercept:
                    slope = 2 * dimension * D / 0.1  # Convert from cm²/s to Å²/fs
                    y_fit = msd_values[visualization_start:visualization_end]
                    residuals = y_fit - (slope * x_fit)
                    intercept = np.mean(residuals)
                    fit_line = slope * x_fit + intercept
                else:
                    slope = 2 * dimension * D / 0.1
                    fit_line = slope * x_fit
                
                plt.plot(x_fit/1000, fit_line, 'r-', linewidth=2, 
                        label=f'Block Avg: D = ({D:.2e} ± {error:.2e}) cm²/s')
                plt.xlabel('Time (ps)')
                plt.ylabel('MSD (Å²)')
                plt.title('Mean Square Displacement vs Time')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                dir_name = os.path.dirname(csv_file)
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                output_file = os.path.join(dir_name, f"{base_name}_msd_block_avg.png")
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"MSD plot saved to: {output_file}")
        else:
            D, error = calculate_diffusion_coefficient(
                msd_times=msd_times, 
                msd_values=msd_values, 
                start_index=fit_start, 
                end_index=fit_end, 
                with_intercept=with_intercept, 
                plot_msd=plot_msd,
                dimension=dimension
            )
        
        if plot_diffusion:
            plot_diffusion_time_series(msd_times, msd_values, 10, with_intercept, csv_file, dimension)
        
        method = "Block Averaging" if use_block_averaging else "Standard Fit"
        print(f"\nMSD Analysis Results ({method}):")
        if use_block_averaging:
            print(f"Diffusion Coefficient: D = {D:.4e} ± {error:.4e} cm²/s ({100*error/D:.1f}%)")
        else:
            print(f"Diffusion Coefficient: D = {D:.4e} cm²/s")
        
        return D, error
        
    except Exception as e:
        print(f"Error analyzing MSD data: {e}")
        traceback.print_exc()
        return None, None

def msd_analysis(traj_path, timestep_fs, indices_path=None, ignore_n_images=0,
                output_dir=None, frame_skip=10, fit_start=None, fit_end=None,
                with_intercept=False, plot_msd=False, save_csvs_in_subdir=False,
                msd_direction=False, msd_direction_atom=None, dimension=3,
                use_windowed=True, lag_times_fs=None):
    """
    Perform MSD analysis workflow: calculate MSD and save data.
    
    Parameters
    ----------
    traj_path : str
        Path to the ASE trajectory file
    timestep_fs : float
        Simulation timestep in femtoseconds (fs)
    indices_path : str, optional
        Path to file containing atom indices (default: None)
    ignore_n_images : int, optional
        Number of initial images to ignore (default: 0)
    output_dir : str, optional
        Directory to save output files (default: based on trajectory filename)
    frame_skip : int, optional
        Number of frames to skip between samples (default: 10)
    fit_start : int, optional
        Start index for fitting diffusion coefficient (default: None)
    fit_end : int, optional
        End index for fitting diffusion coefficient (default: None)
    with_intercept : bool, optional
        Whether to fit with intercept (default: False)
    plot_msd : bool, optional
        Whether to plot results (default: False)
    save_csvs_in_subdir : bool, optional
        Whether to save CSV files in a subdirectory (default: False)
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
    msd_direction_atom : str or int, optional
        Atom symbol or atomic number for directional analysis (default: None)
    dimension : int, optional
        Dimensionality of the system: 1 for 1D, 2 for 2D, 3 for 3D (default: 3)
    use_windowed : bool, optional
        Whether to use the windowed approach for more robust statistics (default: True)
    lag_times_fs : list of float, optional
        List of lag times (in fs) for which to compute MSD (default: None, use all possible lags)
        
    Returns
    -------
    dict
        Dictionary containing MSD values, times, output directory, and optionally diffusion coefficient
    """
    if output_dir is None:
        traj_basename = os.path.splitext(os.path.basename(traj_path))[0]
        output_dir = f"msd_{traj_basename}"
        print(f"Using trajectory-based output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "msd_data.csv")
    
    csv_dir = os.path.join(output_dir, "csv_data") if save_csvs_in_subdir else output_dir
    
    # Calculate and save MSD data (passing timestep directly in fs)
    msd_data = calculate_save_msd(
        traj_path=traj_path,
        timestep_fs=timestep_fs,
        indices_path=indices_path,
        ignore_n_images=ignore_n_images,
        output_file=csv_path,
        frame_skip=frame_skip,
        output_dir=csv_dir,
        msd_direction=msd_direction,
        msd_direction_atom=msd_direction_atom,
        use_windowed=use_windowed,
        lag_times_fs=lag_times_fs
    )
    
    if isinstance(msd_data, tuple) and msd_data[0] is None:
        print("Error: Failed to calculate MSD data")
        return {"error": "Failed to calculate MSD data"}
    
    # Extract MSD values and times for return value
    if isinstance(msd_data, dict):
        if 'overall' in msd_data:
            msd_values, msd_times = msd_data['overall']
        else:
            # Use the first atom type's data
            first_symbol = next(iter(msd_data))
            msd_values, msd_times = msd_data[first_symbol]
    else:
        if len(msd_data) == 4:
            msd_times, msd_x, msd_y, msd_z = msd_data
            msd_values = msd_x + msd_y + msd_z  # use total for downstream analysis
        else:
            msd_values, msd_times = msd_data
    
    result_dict = {
        "msd_values": msd_values,
        "msd_times": msd_times,
        "output_dir": output_dir
    }
    
    # Diffusion coefficient analyzing the CSV file
    if fit_start is not None or fit_end is not None or plot_msd:
        try:
            print("Calculating diffusion coefficient...")
            D, error = calculate_diffusion_coefficient(
                msd_times=msd_times,
                msd_values=msd_values,
                start_index=fit_start,
                end_index=fit_end,
                with_intercept=with_intercept,
                plot_msd=plot_msd,
                dimension=dimension
            )
            
            if D is not None:
                result_dict["diffusion_coefficient"] = D
                result_dict["error"] = error
                print(f"Calculated diffusion coefficient: {D:.2e} cm²/s")
        except Exception as e:
            print(f"Error calculating diffusion coefficient: {e}")
    
    return result_dict

def block_averaging_error(msd_times, msd_values, n_blocks=5, dimension=3, **kwargs):
    """
    Calculate diffusion coefficient error using block averaging.
    
    Parameters
    ----------
    msd_times : numpy.ndarray
        Time values in femtoseconds
    msd_values : numpy.ndarray
        MSD values
    n_blocks : int
        Number of blocks to divide the data into
    dimension : int
        Dimensionality of the system (default: 3)
    
    Returns
    -------
    tuple
        (mean_D, std_error_D) - mean diffusion coefficient and its standard error
    """
    # Block size
    block_size = len(msd_times) // n_blocks
    if block_size < 10:
        print(f"Warning: Block size is small ({block_size} points). Consider using fewer blocks.")
    
    # D for each block
    D_values = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else len(msd_times)
        
        if end_idx - start_idx < 10:  
            continue
            
        # Remove fit_start/end from kwargs for block fit            
        block_kwargs = {k: v for k, v in kwargs.items() if k not in ['start_index', 'end_index', 'fit_start', 'fit_end']}
            
        try:
            D, _ = calculate_diffusion_coefficient(
                msd_times[start_idx:end_idx], 
                msd_values[start_idx:end_idx],
                dimension=dimension,
                **block_kwargs
            )
            D_values.append(D)
        except Exception as e:
            print(f"Warning: Failed to fit block {i}: {e}")
    
    # Statistics
    D_values = np.array(D_values)
    mean_D = np.mean(D_values)
    std_D = np.std(D_values, ddof=1)  
    std_error_D = std_D / np.sqrt(len(D_values))  
    
    return mean_D, std_error_D

def calculate_frame_msd_windowed(positions_i, positions_j, atom_indices, msd_direction=False):
    """
    Calculate MSD between two frames for the windowed approach.
    
    Parameters
    ----------
    positions_i : numpy.ndarray
        Positions at time i
    positions_j : numpy.ndarray
        Positions at time j
    atom_indices : list
        List of atom indices to include in MSD calculation
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
        
    Returns
    -------
    float or tuple
        If msd_direction is False: msd_value
        If msd_direction is True: (msd_x, msd_y, msd_z)
    """
    atom_positions_i = positions_i[atom_indices]
    atom_positions_j = positions_j[atom_indices]
    displacements = atom_positions_j - atom_positions_i
    
    if not msd_direction:
        # Calculate total MSD
        msd_value = np.sum(np.square(displacements)) / (len(atom_indices))
        return msd_value
    else:
        # Directional MSDs
        msd_x = np.sum(displacements[:, 0]**2) / len(atom_indices)
        msd_y = np.sum(displacements[:, 1]**2) / len(atom_indices)
        msd_z = np.sum(displacements[:, 2]**2) / len(atom_indices)
        
        return msd_x, msd_y, msd_z

def calculate_msd_windowed(
    traj, timestep, atom_indices=None, ignore_n_images=0, n_jobs=-1, 
    msd_direction=False, msd_direction_atom=None, lag_times_fs=None
):
    """
    Calculate Mean Square Displacement (MSD) vs time using the windowed approach,
    averaging over all possible time origins.

    Parameters
    ----------
    traj : list of ase.Atoms
        Trajectory data
    timestep : float
        Simulation timestep
    atom_indices : numpy.ndarray, optional
        Indices of atoms to analyze (default: all atoms)
    ignore_n_images : int, optional
        Number of initial images to ignore (default: 0)
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1, use all available cores)
    msd_direction : bool, optional
        Whether to calculate directional MSD (default: False)
    msd_direction_atom : str or int, optional
        Atom symbol or atomic number to filter for directional MSD (default: None)
    lag_times_fs : list of float, optional
        List of lag times (in fs) for which to compute MSD (default: None, use all possible lags)

    Returns
    -------
    tuple or dict
        If atom_indices is provided: (msd_values, msd_times) or (msd_times, msd_x, msd_y, msd_z) if msd_direction=True
        If atom_indices is None: A dictionary with keys for each atom type
    """

    # Time values
    total_images = len(traj) - ignore_n_images
    timestep_fs = timestep / fs  # Convert timestep to fs
    n_frames = total_images
    positions = [traj[i].positions for i in range(ignore_n_images, len(traj))]
    positions = np.array(positions)

    # Determine lag times (in frames)
    if lag_times_fs is not None:
        lag_frames = [int(round(lag_fs / timestep_fs)) for lag_fs in lag_times_fs if lag_fs > 0]
        lag_frames = [lf for lf in lag_frames if 1 <= lf < n_frames]
    else:
        lag_frames = list(range(1, n_frames))

    msd_times = np.array(lag_frames) * timestep_fs  # in fs

    # MSD for specific atoms
    if atom_indices is not None:
        if msd_direction:
            msd_x = np.zeros(len(lag_frames))
            msd_y = np.zeros(len(lag_frames))
            msd_z = np.zeros(len(lag_frames))
            for idx, lag in enumerate(lag_frames):
                n_pairs = n_frames - lag
                results = Parallel(n_jobs=n_jobs)(
                    delayed(calculate_frame_msd_windowed)(
                        positions[i], positions[i + lag], atom_indices, True
                    ) for i in range(n_pairs)
                )
                msd_x[idx] = np.mean([r[0] for r in results])
                msd_y[idx] = np.mean([r[1] for r in results])
                msd_z[idx] = np.mean([r[2] for r in results])
            return msd_times, msd_x, msd_y, msd_z
        else:
            msd_values = np.zeros(len(lag_frames))
            for idx, lag in enumerate(lag_frames):
                n_pairs = n_frames - lag
                results = Parallel(n_jobs=n_jobs)(
                    delayed(calculate_frame_msd_windowed)(
                        positions[i], positions[i + lag], atom_indices, False
                    ) for i in range(n_pairs)
                )
                msd_values[idx] = np.mean(results)
            return msd_values, msd_times

    # MSD per atom type
    else:
        atoms = traj[0]
        symbols = atoms.get_chemical_symbols()
        unique_symbols = set(symbols)
        
        # A dictionary mapping symbols to their indices
        symbol_indices = {symbol: [i for i, s in enumerate(symbols) if s == symbol] 
                         for symbol in unique_symbols}
        
        # Overall MSD using all atoms
        all_indices = list(range(len(atoms)))
        
        # Initialize array for overall MSD
        overall_msd = np.zeros(len(lag_frames))
        
        # For each lag time, calculate the average MSD over all possible time origins
        for idx, lag in enumerate(lag_frames):
            n_pairs = n_frames - lag
            results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_frame_msd_windowed)(
                    positions[i], 
                    positions[i + lag],
                    all_indices,
                    False
                )
                for i in range(n_pairs)
            )
            overall_msd[idx] = np.mean(results)
        
        # Dictionary to store MSD results
        result = {'overall': (overall_msd[:], msd_times[:])}
        
        # Calculate MSD for each atom type
        for symbol, indices in symbol_indices.items():
            print(f"Calculating MSD for {symbol} atoms...")
            calc_direction = msd_direction and (
                (isinstance(msd_direction_atom, str) and symbol == msd_direction_atom) or
                (isinstance(msd_direction_atom, int) and 
                 atoms.get_atomic_numbers()[indices[0]] == msd_direction_atom)
            )
            
            if calc_direction:
                # Initialize arrays for directional MSD
                msd_x = np.zeros(len(lag_frames))
                msd_y = np.zeros(len(lag_frames))
                msd_z = np.zeros(len(lag_frames))
                
                # For each lag time, calculate the average MSD over all possible time origins
                for idx2, lag in enumerate(lag_frames):
                    n_pairs = n_frames - lag
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(calculate_frame_msd_windowed)(
                            positions[i], 
                            positions[i + lag],
                            indices,
                            True
                        )
                        for i in range(n_pairs)
                    )
                    msd_x[idx2] = np.mean([r[0] for r in results])
                    msd_y[idx2] = np.mean([r[1] for r in results])
                    msd_z[idx2] = np.mean([r[2] for r in results])
                
                total_values = msd_x + msd_y + msd_z
                result[symbol] = (total_values, msd_times[:])
                result[f'{symbol}_x'] = (msd_x, msd_times[:])
                result[f'{symbol}_y'] = (msd_y, msd_times[:])
                result[f'{symbol}_z'] = (msd_z, msd_times[:])
                
                print(f"Saved directional MSD data for {symbol} atoms")
            else:
                msd_values = np.zeros(len(lag_frames))
                for idx2, lag in enumerate(lag_frames):
                    n_pairs = n_frames - lag
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(calculate_frame_msd_windowed)(
                            positions[i], 
                            positions[i + lag],
                            indices,
                            False
                        )
                        for i in range(n_pairs)
                    )
                    msd_values[idx2] = np.mean(results)
                
                result[symbol] = (msd_values[:], msd_times[:])
        
        return result