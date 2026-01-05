"""
CRISP/data_analysis/contact_coordination.py

This module performs contact and correlation on molecular dynamics trajectory data.
"""

from ase.io import read
import numpy as np
from joblib import Parallel, delayed
import pickle
from typing import Union, List, Dict, Tuple, Optional, Any
import os
import matplotlib.pyplot as plt
import itertools
from ase.data import vdw_radii, atomic_numbers, chemical_symbols
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
__all__ = ['indices', 'coordination_frame', 'coordination', 'contacts_frame', 'contacts']

def indices(atoms, ind: Union[str, List[Union[int, str]]]) -> np.ndarray:
    """
    Return array of atom indices from an ASE Atoms object based on the input specifier.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic structure
    ind : Union[str, List[Union[int, str]]]
        Index specifier, can be "all", .npy file, integer(s), or chemical symbol(s)
        
    Returns
    -------
    np.ndarray
        Array of selected indices
        
    Raises
    ------
    ValueError
        If the index type is invalid
    """
    if ind == "all" or ind is None:
        return np.arange(len(atoms))
    if isinstance(ind, str) and ind.endswith(".npy"):
        return np.load(ind, allow_pickle=True)
    if not isinstance(ind, list):
        ind = [ind]
    if any(isinstance(item, int) for item in ind):
        return np.array(ind)
    if any(isinstance(item, str) for item in ind):
        idx = []
        if isinstance(ind, str):
            ind = [ind]
        for symbol in ind:
            idx.append(np.where(np.array(atoms.get_chemical_symbols()) == symbol)[0])
        return np.concatenate(idx)
    raise ValueError("Invalid index type")


def coordination_frame(atoms, central_atoms, target_atoms, custom_cutoffs=None, mic=True):
    """
    Calculate coordination numbers for central atoms based on interatomic distances and cutoff criteria.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic structure
    central_atoms : Union[str, List[Union[int, str]]]
        Specifier for central atoms
    target_atoms : Union[str, List[Union[int, str]]]
        Specifier for target atoms that interact with central atoms
    custom_cutoffs : Optional[Dict[Tuple[str, str], float]]
        Dictionary with custom cutoff distances for atom pairs
    mic : bool
        Whether to use the minimum image convention
        
    Returns
    -------
    Dict[int, int]
        Dictionary mapping central atom indices to their coordination numbers
    """
    indices_central = indices(atoms, central_atoms)
    indices_target = indices(atoms, target_atoms)

    dm = atoms.get_all_distances(mic=mic)
    np.fill_diagonal(dm, np.inf)

    sub_dm = dm[np.ix_(indices_central, indices_target)]

    central_atomic_numbers = np.array(atoms.get_atomic_numbers())[indices_central]
    target_atomic_numbers = np.array(atoms.get_atomic_numbers())[indices_target]

    central_vdw_radii = vdw_radii[central_atomic_numbers]
    target_vdw_radii = vdw_radii[target_atomic_numbers]

    cutoff_matrix = 0.6 * (central_vdw_radii[:, np.newaxis] + target_vdw_radii[np.newaxis, :])

    if custom_cutoffs is not None:
        cutoff_atomic_numbers = [tuple(sorted(atomic_numbers[symbol] for symbol in pair)) for pair in
                                 list(custom_cutoffs.keys())]
        cutoff_values = list(custom_cutoffs.values())

        cutoff_matrix_indices = [[tuple(sorted([i, j])) for j in target_atomic_numbers] for i in central_atomic_numbers]

        for i, central_atom in enumerate(cutoff_matrix_indices):
            for j, bond in enumerate(central_atom):
                if bond in cutoff_atomic_numbers:
                    cutoff_matrix[i, j] = cutoff_values[cutoff_atomic_numbers.index(bond)]

    coordination_numbers = np.sum(sub_dm < cutoff_matrix, axis=1)
    coordination_dict_frame = dict(zip(indices_central, coordination_numbers))
    return coordination_dict_frame


def get_avg_percentages(coordination_data, atom_type, plot_cn=False, output_dir="./"):
    """
    Compute average percentages of each coordination number over all frames.
    
    Parameters
    ----------
    coordination_data : Dict[int, List[int]]
        Dictionary mapping atom indices to lists of coordination numbers
    atom_type : Optional[str]
        Chemical symbol of target atoms or None
    plot_cn : bool, optional
        Boolean to indicate if a time series plot should be generated
    output_dir : str, optional
        Directory where output files will be saved
        
    Returns
    -------
    Dict[int, List[float]]
        Dictionary mapping each coordination number to a list of average percentages per frame
    """
    coord_types = set(itertools.chain.from_iterable(coordination_data.values()))
    coord_types = sorted(coord_types)

    num_frames = len(next(iter(coordination_data.values())))
    avg_percentages = {coord_type: [] for coord_type in coord_types}

    for frame_idx in range(num_frames):
        frame_data = [values[frame_idx] for values in coordination_data.values()]
        total_atoms = len(frame_data)
        for coord_type in coord_types:
            count = frame_data.count(coord_type)
            avg_percentage = count / total_atoms * 100
            avg_percentages[coord_type].append(avg_percentage)

    frames = list(range(len(next(iter(avg_percentages.values())))))
    coord_types = list(avg_percentages.keys())

    if plot_cn:
        colors = plt.get_cmap('tab10', len(coord_types)).colors
        markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'p', '*', '+', 'x'])
        plt.figure(figsize=(10, 6))
        for i, coord_type in enumerate(coord_types):
            plt.plot(frames, avg_percentages[coord_type], label=f'CN={coord_type}',
                     color=colors[i], marker=next(markers), markevery=max(1, len(frames) // 20))
        for i, coord_type in enumerate(coord_types):
            mean_value = sum(avg_percentages[coord_type]) / len(avg_percentages[coord_type])
            plt.axhline(y=mean_value, color=colors[i], linestyle='--', alpha=0.7,
                        label=f'Mean CN={coord_type}: {mean_value:.1f}%')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Percentage of Atoms (%)', fontsize=12)
        if atom_type is not None:
            plt.title(f'Coordination Analysis: {atom_type} Atoms', fontsize=14)
        else:
            plt.title('Coordination Analysis', fontsize=14)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "CN_time_series.png"), dpi=300, bbox_inches='tight')
        plt.show()  
        plt.close()

    return avg_percentages


def plot_coordination_distribution(avg_percentages, atom_type, plot_cn, output_dir="./", output_file="CN_distribution"):
    """
    Plot a pie chart showing the overall average distribution of coordination numbers.
    
    Parameters
    ----------
    avg_percentages : Dict[int, List[float]]
        Dictionary of average percentages per coordination number
    atom_type : Optional[str]
        Chemical symbol for target atoms
    plot_cn : bool
        Boolean to indicate if the plot should be generated
    output_dir : str, optional
        Directory where output files will be saved
    output_file : str, optional
        Filename for saving the plot
        
    Returns
    -------
    Dict[int, float]
        Dictionary of overall average coordination percentages
    """
    overall_avg_percentages = plot_coordination_distribution_plotly(avg_percentages, atom_type, plot_cn, 
                                                                   output_dir=output_dir, 
                                                                   output_file=output_file)
    
    # Generate a static version
    if plot_cn:
        overall_avg_percentages = {coord_type: sum(percentages) / len(percentages) 
                                  for coord_type, percentages in avg_percentages.items()}
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sorted_data = sorted(overall_avg_percentages.items())
        coord_types = [item[0] for item in sorted_data]
        percentages = [item[1] for item in sorted_data]
        
        colors = plt.cm.tab10.colors[:len(coord_types)]
        
        # Create pie chart 
        wedges, _ = ax.pie(percentages, 
                          wedgeprops=dict(width=0.5, edgecolor='w'),
                          startangle=90,
                          colors=colors)
        
        legend_labels = [f'CN={coord_type}: {pct:.1f}%' for coord_type, pct in sorted_data]
        
        ax.legend(wedges, legend_labels, 
                  title="Coordination Numbers",
                  loc="center left", 
                  bbox_to_anchor=(1, 0.5),
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        ax.axis('equal')
        
        if atom_type:
            plt.title(f'Average Distribution of {atom_type} Atom Coordination', fontsize=14)
        else:
            plt.title(f'Average Distribution of Atom Coordination', fontsize=14)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{output_file}.png"), dpi=300, bbox_inches='tight')
        plt.show()  
        plt.close()
    
    return overall_avg_percentages


def plot_coordination_distribution_plotly(avg_percentages, atom_type, plot_cn, output_dir="./", output_file="CN_distribution"):
    """
    Plot an interactive pie chart showing the overall average distribution of coordination numbers.
    
    Parameters
    ----------
    avg_percentages : Dict[int, List[float]]
        Dictionary of average percentages per coordination number
    atom_type : Optional[str]
        Chemical symbol for target atoms
    plot_cn : bool
        Boolean to indicate if the plot should be generated
    output_dir : str, optional
        Directory where output files will be saved
    output_file : str, optional
        Filename for saving the plot
        
    Returns
    -------
    Dict[int, float]
        Dictionary of overall average coordination percentages
    """
    overall_avg_percentages = {coord_type: sum(percentages) / len(percentages) for coord_type, percentages in
                               avg_percentages.items()}

    if plot_cn:
        sorted_data = sorted(overall_avg_percentages.items())
        coord_types = [f"CN={item[0]}" for item in sorted_data]
        percentages = [item[1] for item in sorted_data]
        
        hover_info = [f"CN={coord_type}: {pct:.2f}%" for coord_type, pct in sorted_data]
        
        fig = go.Figure(data=[go.Pie(
            labels=coord_types,
            values=percentages,
            hole=0.4,  
            textinfo='label+percent',
            hoverinfo='text',
            hovertext=hover_info,
            textfont=dict(size=12),
            marker=dict(
                colors=[f'rgb{tuple(int(c*255) for c in plt.cm.tab10(i)[:3])}' for i in range(len(coord_types))],
                line=dict(color='white', width=2)
            ),
        )])
        
        if atom_type:
            title_text = f'Average Distribution of {atom_type} Atom Coordination'
        else:
            title_text = 'Average Distribution of Atom Coordination'
            
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=16)
            ),
            legend=dict(
                orientation='h',
                xanchor='center',
                x=0.5,
                y=-0.1
            ),
            height=600,
            width=800
        )
        
        html_path = os.path.join(output_dir, f"{output_file}.html")
        fig.write_html(html_path)
        print(f"Interactive coordination distribution chart saved to {html_path}")

    return overall_avg_percentages


def log_coordination_data(distribution, avg_percentages, atom_type, avg_cn=None, std_cn=None, output_dir="./"):
    """
    Log coordination analysis statistics to a text file.
    
    Parameters
    ----------
    distribution : Dict[int, float]
        Dictionary of overall average coordination percentages
    avg_percentages : Dict[int, List[float]]
        Dictionary of percentage values per frame for each coordination number
    atom_type : Optional[str]
        Chemical symbol of target atoms or None
    avg_cn : float, optional
        Average coordination number
    std_cn : float, optional
        Standard deviation of coordination number
    output_dir : str, optional
        Directory where the statistics file will be saved
        
    Returns
    -------
    None
    """
    if atom_type is not None:
        stats_file = f"CN_{atom_type}_statistics.txt"
    else:
        stats_file = "CN_statistics.txt"

    stats_file = os.path.join(output_dir, stats_file)
    with open(stats_file, 'w') as f:
        if atom_type is not None:
            f.write(f"Coordination Analysis for {atom_type} Atoms\n")
        else:
            f.write("Coordination Analysis\n")
        f.write("======================================\n\n")
        
        if avg_cn is not None and std_cn is not None:
            f.write(f"Average Coordination Number: {avg_cn:.2f} ± {std_cn:.2f}\n\n")
        
        f.write("Overall Average Percentages:\n")
        
        # Standard deviations for each coordination number
        std_devs = {cn: np.std(values) for cn, values in avg_percentages.items()}
        
        for coord_type, avg_percentage in sorted(distribution.items()):
            std_dev = std_devs[coord_type]
            f.write(f"  CN={coord_type}: {avg_percentage:.2f}% ± {std_dev:.2f}%\n")
        
        most_common_cn = max(distribution.items(), key=lambda x: x[1])[0]
        f.write(f"\nMost common coordination number: {most_common_cn}\n")
        
    print(f"Coordination statistics saved to {stats_file}")


def coordination(traj_path, central_atoms, target_atoms, custom_cutoffs, frame_skip=10, 
                plot_cn=False, output_dir="./"):
    """
    Process a trajectory file to compute coordination numbers for each frame.
    
    Parameters
    ----------
    traj_path : str
        Path to the trajectory file
    central_atoms : Union[str, List[Union[int, str]]]
        Specifier for central atoms being analyzed
    target_atoms : Union[str, List[Union[int, str]]]
        Specifier for target atoms that interact with central atoms
    custom_cutoffs : Dict[Tuple[str, str], float]
        Dictionary with custom cutoff distances
    frame_skip : int, optional
        Interval for skipping frames (default: 10)
    plot_cn : bool, optional
        Boolean to indicate if plots should be generated (default: False)
    output_dir : str, optional
        Directory where output files will be saved (default: "./")
        
    Returns
    -------
    List
        List containing coordination dictionary, average percentages, distribution, atom type, and avg CN
    """
    os.makedirs(output_dir, exist_ok=True)

    trajectory = read(traj_path, index=f"::{frame_skip}")
    coordination_dict = {}

    results = Parallel(n_jobs=-1)(
        delayed(coordination_frame)(atoms, central_atoms, target_atoms, custom_cutoffs)
        for atoms in trajectory
    )

    for frame_dict in results:
        for key, value in frame_dict.items():
            coordination_dict.setdefault(key, []).append(value)

    
    if isinstance(central_atoms, str):
        if central_atoms.endswith('.npy'):
            # Extract just the basename without extension for .npy files
            atom_type = os.path.splitext(os.path.basename(central_atoms))[0]
        else:
            atom_type = central_atoms
    elif isinstance(central_atoms, int):
        atom_type = chemical_symbols[central_atoms]
    else:
        atom_type = None

    avg_percentages = get_avg_percentages(coordination_dict, atom_type, plot_cn, output_dir=output_dir)
    
    distribution = plot_coordination_distribution(avg_percentages, atom_type, plot_cn, output_dir=output_dir)
    
    all_cn_values = []
    for atom_idx, cn_list in coordination_dict.items():
        all_cn_values.extend(cn_list)
    
    all_cn_values = np.array(all_cn_values)
    avg_cn = np.mean(all_cn_values)
    std_cn = np.std(all_cn_values)
    
    weighted_avg_cn = sum(cn * (pct/100) for cn, pct in distribution.items())
    
    log_coordination_data(distribution, avg_percentages, atom_type, 
                         avg_cn=avg_cn, std_cn=std_cn, output_dir=output_dir)
    
    print("\nCoordination Statistics Summary:")
    print("=" * 40)
    
    std_devs = {cn: np.std(values) for cn, values in avg_percentages.items()}
    
    for coord_type, avg_percentage in sorted(distribution.items()):
        std_dev = std_devs[coord_type]
        print(f"CN={coord_type}: {avg_percentage:.2f}% ± {std_dev:.2f}%")
    
    most_common_cn = max(distribution.items(), key=lambda x: x[1])[0]
    print(f"\nMost common coordination number: {most_common_cn}")
    
    print(f"\nAverage Coordination Number: {avg_cn:.2f} ± {std_cn:.2f}")
    
    return [coordination_dict, avg_percentages, distribution, atom_type, avg_cn, std_cn]


def contacts_frame(atoms, central_atoms, target_atoms, custom_cutoffs, mic=True):
    """
    Processes a single atoms frame to compute the sub-distance matrix and the corresponding cutoff matrix.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic structure
    central_atoms : Union[str, List[Union[int, str]]]
        Selection criteria for central atoms being analyzed
    target_atoms : Union[str, List[Union[int, str]]]
        Selection criteria for target atoms that interact with central atoms
    custom_cutoffs : Dict[Tuple[str, str], float]
        Dictionary mapping atom pairs to custom cutoff values
    mic : bool, optional
        Whether to use minimum image convention (default: True)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Sub-distance matrix, cutoff matrix, central atom indices, and target atom indices
    """
    indices_central = indices(atoms, central_atoms)
    indices_target = indices(atoms, target_atoms)
    
    dm = atoms.get_all_distances(mic=mic)
    np.fill_diagonal(dm, np.inf)  # Avoids self-interactions
    
    has_overlap = np.intersect1d(indices_central, indices_target).size > 0
    
    if has_overlap:
        mask = np.zeros_like(dm, dtype=bool)
        for i_idx, i in enumerate(indices_central):
            for j_idx, j in enumerate(indices_target):
                # We only count each pair once by enforcing i < j
                if i < j:
                    mask[i, j] = True
                    
        filtered_dm = np.where(mask, dm, np.inf)
        sub_dm = filtered_dm[np.ix_(indices_central, indices_target)]
    else:
        sub_dm = dm[np.ix_(indices_central, indices_target)]

    # Get atomic numbers and van der Waals radii
    central_atomic_numbers = np.array(atoms.get_atomic_numbers())[indices_central]
    target_atomic_numbers = np.array(atoms.get_atomic_numbers())[indices_target]

    central_vdw_radii = vdw_radii[central_atomic_numbers]
    target_vdw_radii = vdw_radii[target_atomic_numbers]

    cutoff_matrix = 0.6 * (central_vdw_radii[:, np.newaxis] + target_vdw_radii[np.newaxis, :])

    if custom_cutoffs is not None:
        cutoff_atomic_numbers = [
            tuple(sorted(atomic_numbers[symbol] for symbol in pair))
            for pair in list(custom_cutoffs.keys())
        ]
        cutoff_values = list(custom_cutoffs.values())

        cutoff_matrix_indices = [
            [tuple(sorted([i, j])) for j in target_atomic_numbers]
            for i in central_atomic_numbers
        ]
        for i, central_atom in enumerate(cutoff_matrix_indices):
            for j, bond in enumerate(central_atom):
                if bond in cutoff_atomic_numbers:
                    cutoff_matrix[i, j] = cutoff_values[cutoff_atomic_numbers.index(bond)]

    return sub_dm, cutoff_matrix, indices_central, indices_target


def plot_contact_heatmap(contact_matrix, frame_skip, time_step, x_labels, y_labels, atom_type, output_dir="./"):
    """
    Plots and saves an interactive heatmap showing contact times between central and target atoms.
    
    Parameters
    ----------
    contact_matrix : np.ndarray
        Boolean 3D array of contacts
    frame_skip : int
        Number of frames skipped when processing the trajectory
    time_step : float
        Time step between frames (used to convert counts to time)
    x_labels : List[str]
        Labels for the target atoms (x-axis)
    y_labels : List[str]
        Labels for the central atoms (y-axis)
    atom_type : Optional[str] 
        Central atom type (used for naming the file)
    output_dir : str, optional
        Directory where the output file will be saved (default: "./")
        
    Returns
    -------
    None
    """
    contact_times_matrix = np.sum(contact_matrix, axis=0) * frame_skip * time_step / 1000
    
    hover_text = []
    for i, central_label in enumerate(y_labels):
        hover_row = []
        for j, target_label in enumerate(x_labels):
            hover_row.append(f"Central: {central_label}<br>" +
                           f"Target: {target_label}<br>" +
                           f"Contact time: {contact_times_matrix[i, j]:.2f} ps")
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=contact_times_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        hoverinfo='text',
        text=hover_text,
        colorbar=dict(title='Contact Time (ps)')
    ))
    
    fig.update_layout(
        title=dict(text='Heatmap of contact times within cutoffs', font=dict(size=16)),
        xaxis=dict(title='Target Atoms', tickfont=dict(size=10)),
        yaxis=dict(title='Central Atoms', tickfont=dict(size=10)),
        width=900,
        height=700,
        autosize=True
    )
    
    if atom_type is not None:
        html_filename = os.path.join(output_dir, f"{atom_type}_heatmap_contacts.html")
    else:
        html_filename = os.path.join(output_dir, "heatmap_contacts.html")
    
    fig.write_html(html_filename)
    print(f"Interactive contact heatmap saved to {html_filename}")


def plot_distance_heatmap(sub_dm_total, x_labels, y_labels, atom_type, output_dir="./"):
    """
    Plots and saves an interactive heatmap of average distances between central and target atoms.
    
    Parameters
    ----------
    sub_dm_total : np.ndarray
        3D numpy array containing sub-distance matrices for each frame
    x_labels : List[str]
        Labels for the target atoms (x-axis)
    y_labels : List[str]
        Labels for the central atoms (y-axis)
    atom_type : Optional[str]
        Central atom type (used for naming the file)
    output_dir : str, optional
        Directory where the output file will be saved (default: "./")
        
    Returns
    -------
    None
    """
    average_distance_matrix = np.mean(sub_dm_total, axis=0)
    std_distance_matrix = np.std(sub_dm_total, axis=0)
    
    hover_text = []
    for i, central_label in enumerate(y_labels):
        hover_row = []
        for j, target_label in enumerate(x_labels):
            hover_row.append(
                f"Central: {central_label}<br>" +
                f"Target: {target_label}<br>" +
                f"Avg distance: {average_distance_matrix[i, j]:.3f} Å<br>" +
                f"Std deviation: {std_distance_matrix[i, j]:.3f} Å<br>" +
                f"Min: {np.min(sub_dm_total[:, i, j]):.3f} Å<br>" +
                f"Max: {np.max(sub_dm_total[:, i, j]):.3f} Å"
            )
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=average_distance_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        hoverinfo='text',
        text=hover_text,
        colorbar=dict(title='Distance (Å)')
    ))
    
    fig.update_layout(
        title=dict(text='Distance matrix of selected atoms', font=dict(size=16)),
        xaxis=dict(title='Target Atoms', tickfont=dict(size=10)),
        yaxis=dict(title='Central Atoms', tickfont=dict(size=10)),
        width=900,
        height=700,
        autosize=True
    )
    
    if atom_type is not None:
        html_filename = os.path.join(output_dir, f"{atom_type}_heatmap_distance.html")
    else:
        html_filename = os.path.join(output_dir, "heatmap_distance.html")
    
    fig.write_html(html_filename)
    print(f"Interactive distance heatmap saved to {html_filename}")


def plot_contact_distance(sub_dm_total, contact_matrix, time_step, frame_skip, output_dir="./"):
    """
    Plots and saves a time series of the average contact distance over the trajectory using Plotly
    and also saves a static Matplotlib version as PNG.
    
    Parameters
    ----------
    sub_dm_total : np.ndarray
        3D numpy array of sub-distance matrices for each frame
    contact_matrix : np.ndarray
        Boolean 3D numpy array indicating which distances are considered contacts
    time_step : float
        Time between frames
    frame_skip : int
        Number of frames skipped when processing
    output_dir : str, optional
        Directory where the output file will be saved (default: "./")
        
    Returns
    -------
    None
    """
    contact_distance = np.where(contact_matrix, sub_dm_total, np.nan)
    contact_count = np.sum(contact_matrix, axis=(1, 2))/np.shape(sub_dm_total)[1]
    average_distance_contacts = np.nanmean(contact_distance, axis=(1, 2))

    x = np.arange(len(average_distance_contacts)) * time_step * frame_skip / 1000
    
    valid_indices = ~np.isnan(average_distance_contacts)
    interpolated = np.interp(x, x[valid_indices], average_distance_contacts[valid_indices])
    mean_distance = np.mean(interpolated)
    mean_count = np.mean(contact_count)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, 
        y=interpolated, 
        mode='lines+markers',
        name='Avg Dist',
        line=dict(color='blue'),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]],
        y=[mean_distance, mean_distance],
        mode='lines',
        name=f'Mean Dist: {mean_distance:.2f} Å',
        line=dict(color='blue', dash='dash'),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, 
        y=contact_count, 
        mode='lines+markers',
        name='Contact Count',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]],
        y=[mean_count, mean_count],
        mode='lines',
        name=f'Mean Count: {mean_count:.1f}',
        line=dict(color='red', dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Average Distance of Contacts & Contact Count',
        xaxis=dict(
            title='Time (ps)',
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(  
                text='Distance (Å)',
                font=dict(color='blue')
            ),
            tickfont=dict(color='blue'),
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis2=dict(
            title=dict(  
                text='Contact count per atom',
                font=dict(color='red')
            ),
            tickfont=dict(color='red'),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=900,
        height=600
    )
    
    html_filename = os.path.join(output_dir, "average_contact_analysis.html")
    fig.write_html(html_filename)
    print(f"Interactive contact analysis chart saved to {html_filename}")
    

    fig_mpl, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(x, interpolated, 'o-', color='blue', label='Avg Dist', 
                     markersize=4, markevery=max(1, len(x)//20))
    ax1.axhline(y=mean_distance, color='blue', linestyle='--', 
                label=f'Mean Distance: {mean_distance:.2f} Å')
    
    line2 = ax2.plot(x, contact_count, 'o-', color='red', label='Contact Count', 
                     markersize=4, markevery=max(1, len(x)//20))
    ax2.axhline(y=mean_count, color='red', linestyle='--', 
                label=f'Mean Count: {mean_count:.1f}')
    
    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('Distance (Å)', color='blue', fontsize=12)
    ax2.set_ylabel('Contact count per atom', color='red', fontsize=12)
    plt.title('Average Distance of Contacts & Contact Count', fontsize=14)
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    all_lines = line1 + line2 + [
        plt.Line2D([0], [0], color='blue', linestyle='--'),
        plt.Line2D([0], [0], color='red', linestyle='--')
    ]
    all_labels = [l.get_label() for l in line1 + line2] + [
        f'Mean Dist: {mean_distance:.2f} Å',
        f'Mean Count: {mean_count:.1f}'
    ]
    
    fig_mpl.legend(all_lines, all_labels, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout(pad=1.2)
    
    png_filename = os.path.join(output_dir, "average_contact_analysis.png")
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.show()  
    plt.close()
    
    print(f"Static contact analysis chart saved to {png_filename}")


def save_matrix_data(sub_dm_total, contact_matrix, output_dir="./"):
    """
    Saves the sub-distance matrices and contact matrix to npy files.
    
    Parameters
    ----------
    sub_dm_total : np.ndarray
        3D numpy array of sub-distance matrices
    contact_matrix : np.ndarray
        Boolean 3D numpy array of contact information
    output_dir : str, optional
        Directory where the output files will be saved (default: "./")
        
    Returns
    -------
    None
    """
    np.save(os.path.join(output_dir, "sub_dm_total.npy"), sub_dm_total)
    np.save(os.path.join(output_dir, "contact_matrix.npy"), contact_matrix)


def contacts(traj_path, central_atoms, target_atoms, custom_cutoffs, frame_skip=10,
             plot_distance_matrix=False, plot_contacts=False, time_step=None, save_data=False,
             output_dir="./", mic=True):
    """
    Processes a molecular trajectory file to compute contacts between central and target atoms.
    
    Parameters
    ----------
    traj_path : str
        Path to the trajectory file
    central_atoms : Union[str, List[Union[int, str]]]
        Criteria for selecting central atoms being analyzed
    target_atoms : Union[str, List[Union[int, str]]]
        Criteria for selecting target atoms that interact with central atoms
    custom_cutoffs : Dict[Tuple[str, str], float]
        Dictionary with custom cutoff values for specific atom pairs
    frame_skip : int, optional
        Number of frames to skip (default: 10)
    plot_distance_matrix : bool, optional
        Boolean flag to plot average distance heatmap (default: False)
    plot_contacts : bool, optional
        Boolean flag to plot contact times heatmap (default: False)
    time_step : float, optional
        Time between frames in fs (required for contact heatmap)
    save_data : bool, optional
        Boolean flag to save matrices as npy files (default: False)
    output_dir : str, optional
        Directory where output files will be saved (default: "./")
    mic : bool, optional
        Whether to use minimum image convention (default: True)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        3D numpy array of sub-distance matrices and Boolean 3D numpy array of contacts
    """
    os.makedirs(output_dir, exist_ok=True)

    sub_dm_list = []
    trajectory = read(traj_path, index=f"::{frame_skip}")
    
    results = Parallel(n_jobs=-1)(
        delayed(contacts_frame)(atoms, central_atoms, target_atoms, custom_cutoffs, mic=mic)
        for atoms in trajectory
    )

    sub_dm_list, cutoff_matrices, indices_central, indices_target = zip(*results)
    cutoff_matrix = cutoff_matrices[0]
    sub_dm_total = np.array(sub_dm_list)

    atom_type = (central_atoms if isinstance(central_atoms, str)
                 else chemical_symbols[central_atoms] if isinstance(central_atoms, int)
                 else None)
                 
    y_labels = [f"{trajectory[0].get_chemical_symbols()[i]}({i})" for i in indices_central[0]]
    x_labels = [f"{trajectory[0].get_chemical_symbols()[i]}({i})" for i in indices_target[0]]

    contact_matrix = sub_dm_total < cutoff_matrix

    if plot_contacts and time_step is not None:
        plot_contact_heatmap(contact_matrix, frame_skip, time_step, x_labels, y_labels, atom_type,
                             output_dir=output_dir)
        plot_contact_distance(sub_dm_total, contact_matrix, time_step, frame_skip, output_dir=output_dir)

    if plot_distance_matrix:
        plot_distance_heatmap(sub_dm_total, x_labels, y_labels, atom_type, output_dir=output_dir)

    if save_data:
        save_matrix_data(sub_dm_total, contact_matrix, output_dir=output_dir)

    return sub_dm_total, contact_matrix
