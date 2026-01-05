"""
CRISP/simulation_utility/error_analysis.py

This module provides statistical error analysis tools for molecular dynamics simulations,
including autocorrelation and block averaging methods to estimate statistical errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
from typing import Dict, Optional, Any

__all__ = ['optimal_lag', 'vector_acf', 'autocorrelation_analysis', 'block_analysis']

def optimal_lag(acf_values: np.ndarray, threshold: float = 0.05) -> int:
    """Find the optimal lag time at which autocorrelation drops below threshold.
    
    Parameters
    ----------
    acf_values : numpy.ndarray
        Array of autocorrelation function values
    threshold : float, optional
        Correlation threshold below which data is considered uncorrelated (default: 0.05)
        
    Returns
    -------
    int
        Optimal lag time where autocorrelation drops below threshold
        
    Warns
    -----
    UserWarning
        If autocorrelation function does not converge within available data
    """
    for lag, value in enumerate(acf_values):
        if abs(value) < threshold:
            return lag
        
    acf_not_converged = (
        "Autocorrelation function is not converged. "
        f"Consider increasing the 'max_lag' parameter (current: {len(acf_values) - 1}) "
        "or extending the simulation length."
    )
    warnings.warn(acf_not_converged)
    
    return len(acf_values) - 1


def vector_acf(data, max_lag):
    """
    Calculate autocorrelation function for vector data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input vector data with shape (n_frames, n_dimensions)
    max_lag : int
        Maximum lag time to calculate autocorrelation for
        
    Returns
    -------
    numpy.ndarray
        Array of autocorrelation values from lag 0 to max_lag
    """
    n_frames = data.shape[0]
    m = np.mean(data, axis=0)
    data_centered = data - m
    norm0 = np.mean(np.sum(data_centered**2, axis=1))
    acf_vals = np.zeros(max_lag + 1)
    for tau in range(max_lag + 1):
        dots = np.sum(data_centered[:n_frames - tau] * data_centered[tau:], axis=1)
        acf_vals[tau] = np.mean(dots) / norm0
        
    return acf_vals


def autocorrelation_analysis(data, max_lag=None, threshold=0.05, plot_acf=False):
    """
    Perform autocorrelation analysis to estimate statistical errors.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data (1D array or multi-dimensional array)
    max_lag : int, optional
        Maximum lag time to calculate autocorrelation for (default: min(1000, N/10))
    threshold : float, optional
        Correlation threshold below which data is considered uncorrelated (default: 0.05)
    plot_acf : bool, optional
        Whether to generate an autocorrelation plot (default: False)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean: Mean value of data
        - acf_err: Error estimate from autocorrelation analysis
        - std: Standard deviation of data
        - tau_int: Integrated autocorrelation time
        - optimal_lag: Optimal lag time where autocorrelation drops below threshold
    """
    if data.ndim == 1:
        N = len(data)
        mean_value = np.mean(data)
        std_value = np.std(data, ddof=1)
        if max_lag is None:
            max_lag = min(1000, N // 10)
        acf_values = acf(data - mean_value, nlags=max_lag, fft=True)
        opt_lag = optimal_lag(acf_values, threshold)
    else:
        N = data.shape[0]
        mean_value = np.mean(data, axis=0)
        std_value = np.std(data, axis=0, ddof=1)
        if max_lag is None:
            max_lag = min(N // 20)
        acf_values = vector_acf(data, max_lag)
        opt_lag = optimal_lag(acf_values, threshold)
        
    tau_int = 0.5 + np.sum(acf_values[1:opt_lag + 1])
    autocorr_error = std_value * np.sqrt(2 * tau_int / N)
    
    if plot_acf:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(acf_values)), acf_values, linestyle='-', linewidth=2, color="blue", label='ACF')
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function (ACF)')
        plt.legend()
        plt.savefig("ACF_lag_analysis.png", dpi=300, bbox_inches='tight')

    return {"mean": mean_value, "acf_err": autocorr_error, "std": std_value, "tau_int": tau_int, "optimal_lag": opt_lag}


def block_analysis(data, convergence_tol=0.001, plot_blocks=False):
    """
    Perform block averaging analysis to estimate statistical errors.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    convergence_tol : float, optional
        Tolerance for determining convergence of standard error (default: 0.001)
    plot_blocks : bool, optional
        Whether to generate a block averaging plot (default: False)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean: Mean value of data
        - block_err: Error estimate from block averaging
        - std: Standard deviation of data
        - converged_blocks: Number of blocks at convergence
        
    Warns
    -----
    UserWarning
        If block averaging does not converge with the given tolerance
    """
    N = len(data)
    mean_value = np.mean(data)
    std_value = np.std(data, ddof=1)

    block_sizes = np.arange(1, N // 2)
    standard_errors = []

    for M in block_sizes:
        block_length = N // M

        truncated_data = data[:block_length * M]
        blocks = truncated_data.reshape(M, block_length)
        block_means = np.mean(blocks, axis=1)

        if len(block_means) > 1:
            std_error = np.std(block_means, ddof=1) / np.sqrt(M)
        else:
            continue
        
        standard_errors.append(std_error)

        if len(standard_errors) > 5:
            recent_errors = standard_errors[-5:]
            if np.max(recent_errors) - np.min(recent_errors) < convergence_tol:
                converged_blocks = M
                final_error = std_error
                break
    else:
        converged_blocks = block_sizes[-1]
        final_error = standard_errors[-1]
        warnings.warn("Block averaging did not fully converge. Consider increasing data length or lowering tolerance.")

    if plot_blocks:
        plt.figure(figsize=(8, 5))
        plt.plot(block_sizes[:len(standard_errors)], standard_errors, color="blue", label='Standard Error')
        plt.xlabel('Number of Blocks')
        plt.ylabel('Standard Error')
        plt.title('Block Averaging Convergence')
        plt.savefig("block_averaging_convergence.png", dpi=300, bbox_inches='tight')

    return {
        "mean": mean_value,
        "block_err": final_error,
        "std": std_value,
        "converged_blocks": converged_blocks
    }


def error_analysis(data, max_lag=None, threshold=0.05, convergence_tol=0.001, plot=False):
    """
    Perform comprehensive error analysis using both autocorrelation and block averaging.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    max_lag : int, optional
        Maximum lag time for autocorrelation (default: min(1000, N/10))
    threshold : float, optional
        Correlation threshold for autocorrelation analysis (default: 0.05)
    convergence_tol : float, optional
        Convergence tolerance for block averaging (default: 0.001)
    plot : bool, optional
        Whether to generate diagnostic plots (default: False)
        
    Returns
    -------
    dict
        Dictionary containing results from both methods:
        - mean: Mean value of data
        - std: Standard deviation of data
        - acf_results: Full results from autocorrelation analysis
        - block_results: Full results from block averaging analysis
    """
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Perform both types of analysis
    acf_results = autocorrelation_analysis(data, max_lag, threshold, plot_acf=plot)
    block_results = block_analysis(data, convergence_tol, plot_blocks=plot)
    
    # Combine results
    results = {
        "mean": acf_results["mean"],  
        "std": acf_results["std"],    
        "acf_results": acf_results,
        "block_results": block_results
    }
    
    return results


