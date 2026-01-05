"""
CRISP/data_analysis/clustering.py

This module performs cluster analysis on molecular dynamics trajectory data,
using DBSCAN algorithm to identify atom clusters and their properties.
"""

import numpy as np
from ase.io import read
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import pickle
import csv
import matplotlib.pyplot as plt
import os
from typing import Union, List, Optional, Tuple, Dict, Any

__all__ = ['analyze_frame', 'analyze_trajectory']


class analyze_frame:
    """
    Analyze atomic structures using DBSCAN clustering algorithm.
    
    Parameters
    ----------
    traj_path : str
        Path to trajectory file
    atom_indices : np.ndarray
        Array containing indices of atoms to analyze
    threshold : float
        DBSCAN eps parameter (distance threshold)
    min_samples : int
        DBSCAN min_samples parameter
    metric : str, optional
        Distance metric to use (default: 'precomputed')
    custom_frame_index : int, optional
        Specific frame to analyze (default: None, uses last frame)
    """
    def __init__(
        self,
        traj_path: str,
        atom_indices: Union[str, np.ndarray],
        threshold: float,
        min_samples: int,
        metric: str = 'precomputed',
        custom_frame_index: Optional[int] = None
    ) -> None:
        self.traj_path = traj_path
        if isinstance(atom_indices, str):
            atom_indices = np.load(atom_indices)
        self.atom_indices = atom_indices
        self.threshold = threshold
        self.min_samples = min_samples
        self.metric = metric
        self.custom_frame_index = custom_frame_index
        self.labels = None
        self.distance_matrix = None
        
    def read_custom_frame(self):
        """
        Read a specific frame or the last frame from the trajectory.
        
        Returns
        -------
        ase.Atoms or None
            Atomic structure or None if reading fails
        """
        try:
            if self.custom_frame_index is not None:
                frame = read(self.traj_path, index=self.custom_frame_index)
            else:
                frame = read(self.traj_path, index='-1')
            return frame
        except Exception as e:
            print(f"Error reading trajectory: {e}")
            return None
            
    def calculate_distance_matrix(self, atoms):
        """
        Calculate a distance matrix with periodic boundary conditions.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Distance matrix and positions array
            
        Raises
        ------
        ValueError
            If there are not enough atoms to form clusters
        """
        positions = atoms.positions[self.atom_indices]
        
        if len(self.atom_indices) < self.min_samples:
            raise ValueError(f"Not enough atoms ({len(self.atom_indices)}) to form clusters with min_samples={self.min_samples}")
        
        full_dm = atoms.get_all_distances(mic=True)
        n_atoms = len(self.atom_indices)
        self.distance_matrix = np.zeros((n_atoms, n_atoms))
        
        for i, idx_i in enumerate(self.atom_indices):
            for j, idx_j in enumerate(self.atom_indices):
                self.distance_matrix[i, j] = full_dm[idx_i, idx_j]
        
        return self.distance_matrix, positions
        
    def find_clusters(self):
        """
        Find clusters using DBSCAN algorithm.
        
        Returns
        -------
        np.ndarray
            Cluster labels for each input point
            
        Raises
        ------
        ValueError
            If distance matrix has not been calculated
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix must be calculated first")
            
        db = DBSCAN(
            eps=self.threshold, 
            min_samples=self.min_samples, 
            metric=self.metric
        ).fit(self.distance_matrix if self.metric == 'precomputed' else None)
        
        self.labels = db.labels_
        return self.labels

    def analyze_structure(self, save_html_path=None, output_dir=None):
        """
        Analyze the structure and find clusters.
        
        Parameters
        ----------
        save_html_path : str, optional
            Path to save HTML visualization
        output_dir : str, optional
            Directory to save all results
            
        Returns
        -------
        dict or None
            Dictionary with analysis results or None if analysis fails
        """
        frame = self.read_custom_frame()
        if frame is None:
            return None
        
        if output_dir is None and save_html_path is not None:
            output_dir = os.path.dirname(save_html_path)
        if not output_dir:
            output_dir = "clustering_results"
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving results to directory: {output_dir}")
        
        if save_html_path is None:
            base_name = os.path.splitext(os.path.basename(self.traj_path))[0]
            save_html_path = os.path.join(output_dir, f"{base_name}_clusters.html")
            
        distance_matrix, positions = self.calculate_distance_matrix(frame)
        self.find_clusters()
        
        # Silhouette score (excluding outliers)
        silhouette_avg = calculate_silhouette_score(self.distance_matrix, self.labels)

        create_html_visualization(
            positions=positions, 
            labels=self.labels,
            title='Interactive 3D Cluster Visualization', 
            save_path=save_html_path,
            cell_dimensions=frame.cell.lengths()  
        )

        # Extract cluster information
        cluster_info = extract_cluster_info(self.labels, self.atom_indices)
        num_clusters = cluster_info["num_clusters"]
        outlier_count = cluster_info["outlier_count"]
        avg_cluster_size = cluster_info["avg_cluster_size"]
        cluster_to_original = cluster_info["cluster_to_original"]

        print_cluster_summary(num_clusters, outlier_count, silhouette_avg, avg_cluster_size, cluster_to_original)
        
        frame_info_path = os.path.join(output_dir, "frame_data.txt")
        save_frame_info_to_file(
            frame_info_path, 
            self.threshold, 
            self.min_samples,
            num_clusters, 
            outlier_count, 
            silhouette_avg, 
            avg_cluster_size, 
            cluster_to_original, 
            self.labels, 
            self.atom_indices
        )
        
        pickle_path = os.path.join(output_dir, "single_frame_analysis.pkl")
        result_data = {
            "num_clusters": num_clusters,
            "outlier_count": outlier_count,
            "silhouette_avg": silhouette_avg,
            "avg_cluster_size": avg_cluster_size,
            "cluster_to_original": cluster_to_original,
            "labels": self.labels,
            "positions": positions,
            "parameters": {
                "threshold": self.threshold,
                "min_samples": self.min_samples,
                "trajectory": self.traj_path,
                "frame_index": self.custom_frame_index
            }
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(result_data, f)
        print(f"Full analysis data saved to: {pickle_path}")
        
        return result_data


def create_html_visualization(positions, labels, title, save_path, cell_dimensions=None):
    """
    Create and save a 3D HTML visualization of clusters.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of atom positions
    labels : np.ndarray
        Array of cluster labels
    title : str
        Title for the visualization
    save_path : str
        Path to save the HTML file
    cell_dimensions : np.ndarray, optional
        Cell dimensions from the simulation [a, b, c]
    """
    fig = go.Figure()
    
    for label in np.unique(labels):
        cluster_points = positions[labels == label]
        label_name = "Outliers" if label == -1 else f'Cluster {label}'
        marker_size = 5
        marker_color = 'gray' if label == -1 else None
        
        fig.add_trace(
            go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=marker_color
                ),
                name=label_name
            )
        )
    
    layout_args = {
        "title": title,
        "legend": dict(itemsizing='constant')
    }
    
    scene_dict = {
        "xaxis": dict(title='X'),
        "yaxis": dict(title='Y'),
        "zaxis": dict(title='Z')
    }
    
    if cell_dimensions is not None:
        scene_dict["xaxis"]["range"] = [0, cell_dimensions[0]]
        scene_dict["yaxis"]["range"] = [0, cell_dimensions[1]]
        scene_dict["zaxis"]["range"] = [0, cell_dimensions[2]]
        
        vertices = [
            [0, 0, 0], [cell_dimensions[0], 0, 0],
            [cell_dimensions[0], cell_dimensions[1], 0], [0, cell_dimensions[1], 0],
            [0, 0, cell_dimensions[2]], [cell_dimensions[0], 0, cell_dimensions[2]],
            [cell_dimensions[0], cell_dimensions[1], cell_dimensions[2]], [0, cell_dimensions[1], cell_dimensions[2]]
        ]
        
        # Define box edges
        i, j, k = [], [], []
        # Bottom face
        i.extend([0, 1, 2, 3, 0])
        j.extend([1, 2, 3, 0, 4])
        k.extend([0, 0, 0, 0, 0])
        # Top face
        i.extend([4, 5, 6, 7, 4])
        j.extend([5, 6, 7, 4, 0])
        k.extend([0, 0, 0, 0, 0])
        # Vertical edges
        i.extend([1, 2, 3])
        j.extend([5, 6, 7])
        k.extend([0, 0, 0])
        
        fig.add_trace(go.Scatter3d(
            x=[vertices[idx][0] for idx in i],
            y=[vertices[idx][1] for idx in j],
            z=[vertices[idx][2] for idx in k],
            mode='lines',
            line=dict(color='black', width=2),
            name='Unit Cell',
            showlegend=False
        ))
    
    layout_args["scene"] = scene_dict
    fig.update_layout(**layout_args)
    
    fig.write_html(save_path)
    print(f"3D visualization saved to {save_path}")


def calculate_silhouette_score(distance_matrix, labels):
    """
    Calculate silhouette score, handling edge cases.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix for points
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    float
        Silhouette score or 0 if calculation fails
    """
    try:
        non_outlier_mask = labels != -1
        if np.sum(non_outlier_mask) > 1:
            # Extract the sub-matrix for non-outlier points
            filtered_matrix = distance_matrix[np.ix_(non_outlier_mask, non_outlier_mask)]
            filtered_labels = labels[non_outlier_mask]
            return silhouette_score(filtered_matrix, filtered_labels, metric='precomputed')
        return 0
    except ValueError:
        return 0


def extract_cluster_info(labels, atom_indices):
    """
    Extract cluster information from labels.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    atom_indices : np.ndarray
        Original atom indices
        
    Returns
    -------
    dict
        Dictionary with cluster information
    """
    cluster_indices = {}
    cluster_sizes = {}
    cluster_to_original = {}

    for cluster_id in np.unique(labels):
        if cluster_id != -1:  # Only count actual clusters (not outliers)
            cluster_indices[cluster_id] = np.where(labels == cluster_id)[0]
            cluster_sizes[cluster_id] = len(cluster_indices[cluster_id])
            cluster_to_original[cluster_id] = atom_indices[cluster_indices[cluster_id]]

    outlier_count = np.sum(labels == -1)
    num_clusters = len([label for label in np.unique(labels) if label != -1])
    
    # Calculate average cluster size
    avg_cluster_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0

    return {
        "num_clusters": num_clusters,
        "outlier_count": outlier_count,
        "avg_cluster_size": avg_cluster_size,
        "cluster_sizes": cluster_sizes,
        "cluster_to_original": cluster_to_original
    }


def print_cluster_summary(num_clusters, outlier_count, silhouette_avg, avg_cluster_size, cluster_to_original):
    """
    Print a summary of clustering results.
    
    Parameters
    ----------
    num_clusters : int
        Number of clusters found
    outlier_count : int
        Number of outliers
    silhouette_avg : float
        Average silhouette score
    avg_cluster_size : float
        Average cluster size
    cluster_to_original : dict
        Mapping from cluster IDs to original atom indices
        
    Returns
    -------
    None
    """
    print(f"\nNumber of Clusters: {num_clusters}")
    print(f"Number of Outliers: {outlier_count}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Average Cluster Size: {avg_cluster_size:.2f}")
    print("Cluster Information:")
    
    for cluster_id, atoms in cluster_to_original.items():
        print(f"  Cluster {cluster_id}: {len(atoms)} points")


def save_frame_info_to_file(file_path, threshold, min_samples, num_clusters, outlier_count, 
                           silhouette_avg, avg_cluster_size, cluster_to_original, labels, atom_indices):
    """
    Save detailed frame information to a text file.
    
    Parameters
    ----------
    file_path : str
        Path to save the text file
    threshold : float
        DBSCAN eps parameter
    min_samples : int
        DBSCAN min_samples parameter
    num_clusters : int
        Number of clusters found
    outlier_count : int
        Number of outliers
    silhouette_avg : float
        Average silhouette score
    avg_cluster_size : float
        Average cluster size
    cluster_to_original : dict
        Mapping from cluster IDs to original atom indices
    labels : np.ndarray
        Cluster labels
    atom_indices : np.ndarray
        Original atom indices
        
    Returns
    -------
    None
    """
    with open(file_path, 'w') as f:
        f.write(f"DBSCAN Clustering Analysis Results\n")
        f.write(f"================================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Threshold (eps): {threshold}\n")
        f.write(f"  Min Samples: {min_samples}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Number of Clusters: {num_clusters}\n")
        f.write(f"  Number of Outliers: {outlier_count}\n")
        f.write(f"  Silhouette Score: {silhouette_avg:.4f}\n")
        f.write(f"  Average Cluster Size: {avg_cluster_size:.2f}\n\n")
        
        f.write(f"Detailed Cluster Information:\n")
        for cluster_id, indices in cluster_to_original.items():
            f.write(f"  Cluster {cluster_id}: {len(indices)} points\n")
            f.write(f"    Original atom indices: {indices.tolist()}\n\n")
        
        f.write(f"Outlier Information:\n")
        outlier_indices = atom_indices[labels == -1]
        f.write(f"  {len(outlier_indices)} outliers\n")
        if len(outlier_indices) > 0:
            f.write(f"    Original atom indices: {outlier_indices.tolist()}\n")
    
    print(f"Detailed frame data saved to: {file_path}")


def analyze_trajectory(traj_path, indices_path, threshold, min_samples, frame_skip=10,
                     output_dir="clustering_results", save_html_visualizations=True):
    """
    Analyze an entire trajectory with DBSCAN clustering.
    
    Parameters
    ----------
    traj_path : str
        Path to trajectory file (supports any ASE-readable format like XYZ)
    indices_path : Union[str, List[int], np.ndarray]
        Either a path to numpy file containing atom indices to analyze,
        or a direct list/array of atom indices
    threshold : float
        DBSCAN eps parameter (distance threshold)
    min_samples : int
        DBSCAN min_samples parameter
    frame_skip : int, optional
        Number of frames to skip (default: 10)
    output_dir : str, optional
        Directory to save output files (default: "clustering_results")
    save_html_visualizations : bool, optional
        Whether to save HTML visualizations for first and last frames (default: True)
        
    Returns
    -------
    list
        List of analysis results for each frame
    """
    if isinstance(indices_path, str):
        atom_indices = np.load(indices_path)
        print(f"Loaded {len(atom_indices)} atoms for clustering from {indices_path}")
    else:
        atom_indices = np.array(indices_path)
        print(f"Using {len(atom_indices)} directly provided atom indices for clustering")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Loading trajectory from {traj_path} (using every {frame_skip}th frame)...")
        trajectory = read(traj_path, index=f'::{frame_skip}')
        if not isinstance(trajectory, list):
            trajectory = [trajectory]
        print(f"Loaded {len(trajectory)} frames from trajectory")
    except Exception as e:
        print(f"Error reading trajectory: {e}")
        return []
    
    results = []
    
    print(f"Analyzing {len(trajectory)} frames...")
    
    for i, frame in enumerate(trajectory):
        try:
            frame_number = i * frame_skip
            
            full_dm = frame.get_all_distances(mic=True)
            n_atoms = len(atom_indices)
            distance_matrix = np.zeros((n_atoms, n_atoms))
            
            for i_local, idx_i in enumerate(atom_indices):
                if idx_i >= len(frame):
                    print(f"Warning: Atom index {idx_i} out of range for frame with {len(frame)} atoms. Skipping.")
                    continue
                for j_local, idx_j in enumerate(atom_indices):
                    if idx_j >= len(frame):
                        continue
                    distance_matrix[i_local, j_local] = full_dm[idx_i, idx_j]
            
            db = DBSCAN(
                eps=threshold, 
                min_samples=min_samples, 
                metric='precomputed'
            ).fit(distance_matrix)
            
            labels = db.labels_
            
            # Extract positions for visualization
            positions = frame.positions[atom_indices]
            
            # Calculate silhouette score
            silhouette_avg = calculate_silhouette_score(distance_matrix, labels)
            
            # Extract cluster information
            cluster_info = extract_cluster_info(labels, atom_indices)
            num_clusters = cluster_info["num_clusters"]
            outlier_count = cluster_info["outlier_count"]
            avg_cluster_size = cluster_info["avg_cluster_size"]
            cluster_to_original = cluster_info["cluster_to_original"]
            
            if save_html_visualizations and (i == 0 or i == len(trajectory) - 1):
                frame_prefix = "first" if i == 0 else "last"
                html_path = os.path.join(output_dir, f"{frame_prefix}_frame_clusters.html")
                create_html_visualization(
                    positions=positions, 
                    labels=labels, 
                    title=f"Frame {frame_number} Clusters", 
                    save_path=html_path,
                    cell_dimensions=frame.cell.lengths()  
                )
            
            results.append([frame_number, num_clusters, outlier_count, silhouette_avg, avg_cluster_size])

        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            results.append([i * frame_skip, 0, 0, 0.0, 0.0])
    
    print(f"Trajectory analysis complete: {len(results)} frames processed")
    
    if not results:
        print("Warning: No results were generated from trajectory analysis")
        return []
        
    return results


def save_analysis_results(analysis_results, output_dir="clustering_results", output_prefix="clustering_results"):
    """
    Save analysis results to CSV, TXT, and PKL files in the specified output directory.
    
    Parameters
    ----------
    analysis_results : list
        List of analysis results for each frame
    output_dir : str, optional
        Directory to save output files (default: "clustering_results")
    output_prefix : str, optional
        Prefix for output file names (default: "clustering_results")
        
    Returns
    -------
    str
        Path to the saved pickle file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv_file = os.path.join(output_dir, f"{output_prefix}.csv")
    output_txt_file = os.path.join(output_dir, f"{output_prefix}.txt")
    output_pickle_file = os.path.join(output_dir, f"{output_prefix}.pkl")

    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Frame Number", 
            "Number of Clusters",
            "Number of Outliers",
            "Silhouette Score", 
            "Average Cluster Size"
        ])
        for result in analysis_results:
            csv_writer.writerow(result)

    with open(output_txt_file, 'w') as f:
        # Calculate averages
        frame_numbers = [result[0] for result in analysis_results]
        num_clusters = [result[1] for result in analysis_results]
        outlier_counts = [result[2] for result in analysis_results]
        silhouette_scores = [result[3] for result in analysis_results]
        avg_cluster_sizes = [result[4] for result in analysis_results]
        
        avg_num_clusters = np.mean(num_clusters)
        avg_outlier_count = np.mean(outlier_counts)
        avg_silhouette = np.mean(silhouette_scores)
        avg_cluster_size = np.mean(avg_cluster_sizes)
        
        f.write(f"DBSCAN Clustering Analysis Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Average Values Across All Frames:\n")
        f.write(f"  Average Number of Clusters: {avg_num_clusters:.2f}\n")
        f.write(f"  Average Number of Outliers: {avg_outlier_count:.2f}\n")
        f.write(f"  Average Silhouette Score: {avg_silhouette:.4f}\n")
        f.write(f"  Average Cluster Size: {avg_cluster_size:.2f}\n\n")
        
        f.write(f"Analysis Results by Frame:\n")
        for result in analysis_results:
            frame_number, num_clusters, outlier_count, silhouette_avg, avg_cluster_size = result
            f.write(f"Frame {frame_number}:\n")
            f.write(f"  Number of Clusters: {num_clusters}\n")
            f.write(f"  Number of Outliers: {outlier_count}\n")
            f.write(f"  Silhouette Score: {silhouette_avg:.4f}\n")
            f.write(f"  Average Cluster Size: {avg_cluster_size:.2f}\n\n")

    with open(output_pickle_file, 'wb') as picklefile:
        pickle.dump(analysis_results, picklefile)

    print(f"Analysis results saved to directory: {output_dir}")
    
    return output_pickle_file


def plot_analysis_results(pickle_file, output_dir=None):
    """
    Plot analysis results from a pickle file and save to specified directory.
    
    Parameters
    ----------
    pickle_file : str
        Path to pickle file containing analysis results
    output_dir : str, optional
        Directory to save output files
        
    Returns
    -------
    None
    """
    with open(pickle_file, 'rb') as f:
        analysis_results = pickle.load(f)

    # Extract data for plotting
    frame_numbers = [result[0] for result in analysis_results]
    num_clusters = [result[1] for result in analysis_results]
    outlier_counts = [result[2] for result in analysis_results]
    silhouette_scores = [result[3] for result in analysis_results]
    avg_cluster_sizes = [result[4] for result in analysis_results]
    
    # Calculate averages
    avg_num_clusters = np.mean(num_clusters)
    avg_outlier_count = np.mean(outlier_counts)
    avg_silhouette = np.mean(silhouette_scores)
    avg_avg_cluster_size = np.mean(avg_cluster_sizes)

    fig, axs = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    # Plot for avg_cluster_sizes
    axs[0].plot(frame_numbers, avg_cluster_sizes, color='red', linestyle='-', linewidth=2)
    axs[0].axhline(y=avg_avg_cluster_size, color='darkred', linestyle='--', alpha=0.7, 
                 label=f'Average: {avg_avg_cluster_size:.2f}')
    axs[0].set_ylabel('Average Cluster Size', color='red', fontsize=16)
    axs[0].tick_params(axis='y', labelcolor='red', labelsize=14)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=12)

    # Plot for number of clusters
    axs[1].plot(frame_numbers, num_clusters, color='blue', linestyle='-', linewidth=2)
    axs[1].axhline(y=avg_num_clusters, color='darkblue', linestyle='--', alpha=0.7,
                 label=f'Average: {avg_num_clusters:.2f}')
    axs[1].set_ylabel('Number of Clusters', color='blue', fontsize=16)
    axs[1].tick_params(axis='y', labelcolor='blue', labelsize=14)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=12)

    # Plot for outlier counts
    axs[2].plot(frame_numbers, outlier_counts, color='purple', linestyle='-', linewidth=2)
    axs[2].axhline(y=avg_outlier_count, color='darkviolet', linestyle='--', alpha=0.7,
                 label=f'Average: {avg_outlier_count:.2f}')
    axs[2].set_ylabel('Number of Outliers', color='purple', fontsize=16)
    axs[2].tick_params(axis='y', labelcolor='purple', labelsize=14)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=12)

    # Plot for silhouette scores
    axs[3].plot(frame_numbers, silhouette_scores, color='orange', linestyle='-', linewidth=2)
    axs[3].axhline(y=avg_silhouette, color='darkorange', linestyle='--', alpha=0.7,
                 label=f'Average: {avg_silhouette:.4f}')
    axs[3].set_ylabel('Silhouette Score', color='orange', fontsize=16)
    axs[3].tick_params(axis='y', labelcolor='orange', labelsize=14)
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(fontsize=12)
    axs[3].set_xlabel('Frame Number', fontsize=16)

    plt.suptitle('Clustering Analysis Results', y=0.98, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if output_dir is None:
        output_dir = os.path.dirname(pickle_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_base = os.path.splitext(os.path.basename(pickle_file))[0]
    plot_file = os.path.join(output_dir, f"{output_base}_plot.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Analysis plot saved to: {plot_file}")


def cluster_analysis(traj_path, indices_path, threshold, min_samples=2, 
                    mode='single', output_dir="clustering", custom_frame_index=None,
                    frame_skip=10, output_prefix="clustering_results"):
    """
    Analyze molecular structures with DBSCAN clustering.
    
    Parameters
    ----------
    traj_path : str
        Path to trajectory file (supports any ASE-readable format like XYZ)
    indices_path : Union[str, List[int], np.ndarray]
        Either a path to numpy file containing atom indices to analyze,
        or a direct list/array of atom indices
    threshold : float
        DBSCAN clustering threshold (eps parameter)
    min_samples : int, optional
        Minimum number of samples in a cluster for DBSCAN (default: 2)
    mode : str, optional
        Analysis mode: 'single' for single frame, 'trajectory' for whole trajectory (default: 'single')
    output_dir : str, optional
        Directory to save output files (default: "clustering")
    custom_frame_index : int, optional
        Specific frame number to analyze in 'single' mode. If None, the last frame is analyzed
    frame_skip : int, optional
        Skip frames in trajectory analysis (default: 10)
    output_prefix : str, optional
        Prefix for output file names in trajectory analysis (default: "clustering_results")
        
    Returns
    -------
    dict or list
        Analysis result (dict for single frame, list for trajectory)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: {output_dir}")
    
    if isinstance(indices_path, str):
        atom_indices = np.load(indices_path)
        print(f"Loaded {len(atom_indices)} atoms for clustering from {indices_path}")
    else:
        atom_indices = np.array(indices_path)
        print(f"Using {len(atom_indices)} directly provided atom indices for clustering")
    
    if mode == 'single':
        # Create a mode-specific subdirectory
        mode_dir = os.path.join(output_dir, "single_frame")
        os.makedirs(mode_dir, exist_ok=True)
        
        # Analyze a single frame
        analyzer = analyze_frame(
            traj_path, 
            atom_indices, 
            threshold, 
            min_samples,
            metric='precomputed',
            custom_frame_index=custom_frame_index
        )
        
        analysis_result = analyzer.analyze_structure(output_dir=mode_dir)
        return analysis_result
        
    else:
        mode_dir = os.path.join(output_dir, "trajectory")
        os.makedirs(mode_dir, exist_ok=True)
        
        analysis_results = analyze_trajectory(
            traj_path, 
            atom_indices,  
            threshold, 
            min_samples, 
            frame_skip,
            output_dir=mode_dir,
            save_html_visualizations=True
        )
        
