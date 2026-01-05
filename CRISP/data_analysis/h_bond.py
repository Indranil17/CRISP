"""
CRISP/data_analysis/h_bond.py

This script performs hydrogen bond analysis on molecular dynamics trajectory data.
"""
from ase.io import read
import numpy as np
import csv
from joblib import Parallel, delayed
import argparse
import os
from typing import Union, List, Optional, Tuple, Any, Dict
import matplotlib.pyplot as plt
from ase.data import vdw_radii, atomic_numbers, chemical_symbols
import seaborn as sns
import itertools
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'svg'
pio.renderers.default = 'notebook'

__all__ = ['indices', 'count_hydrogen_bonds', 'aggregate_data', 'hydrogen_bonds']


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
        for symbol in ind:
            idx.append(np.where(np.array(atoms.get_chemical_symbols()) == symbol)[0])
        return np.concatenate(idx)
    
    raise ValueError("Invalid index type")


def count_hydrogen_bonds(
    atoms,
    acceptor_atoms: List[str] = None,
    angle_cutoff: float = 120,
    h_bond_cutoff: float = 2.4,
    bond_cutoff: float = 1.6,
    mic: bool = True,
    single_h_bond: bool = False
) -> Tuple[Dict[int, List], int]:
    """Count hydrogen bonds in an atomic structure.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic structure
    acceptor_atoms : List[str], optional
        List of acceptor atom symbols (default: ["N", "O", "F"])
    angle_cutoff : float, optional
        Minimum angle in degrees for hydrogen bond (default: 120)
    h_bond_cutoff : float, optional
        Maximum distance for hydrogen bond in Angstroms (default: 2.4)
    bond_cutoff : float, optional
        Maximum distance for covalent bond in Angstroms (default: 1.6)
    mic : bool, optional
        Use minimum image convention (default: True)
    single_h_bond : bool, optional
        Count only atoms with single hydrogen bonds (default: False)
        
    Returns
    -------
    Tuple[Dict[int, List], int]
        Dictionary mapping hydrogen indices to acceptor bonds and count of H-bonds
    """
    if acceptor_atoms is None:
        acceptor_atoms = ["N", "O", "F"]
    
    indices_hydrogen = indices(atoms, "H")
    indices_acceptor = indices(atoms, acceptor_atoms)

    dm = atoms.get_all_distances(mic=mic)
    np.fill_diagonal(dm, np.inf)

    sub_dm = dm[indices_hydrogen, :][:, indices_acceptor]

    hb_hyd = indices_hydrogen[np.where(sub_dm < h_bond_cutoff)[0]]
    hb_acc = indices_acceptor[np.where(sub_dm < h_bond_cutoff)[1]]

    distances = sub_dm[np.where(sub_dm < h_bond_cutoff)]

    hydrogen_dict = {}

    for hydrogen, acceptor, distance in zip(hb_hyd, hb_acc, distances):
        if hydrogen not in hydrogen_dict:
            hydrogen_dict[hydrogen] = []
        hydrogen_dict[hydrogen].append([acceptor, distance])

    hydrogen_dict = {hydrogen: sorted(acceptors, key=lambda x: x[1]) for hydrogen, acceptors in hydrogen_dict.items()}
    
    for hydrogen, bonds in hydrogen_dict.items():
        if len(bonds) > 0 and bonds[0][1] < bond_cutoff:
            filtered_bonds = [bonds[0]]
            for acceptor_h_bond in bonds[1:]:
                angle = atoms.get_angle(bonds[0][0], hydrogen, acceptor_h_bond[0], mic=mic)
                if angle >= angle_cutoff:
                    acceptor_h_bond.append(angle)
                    filtered_bonds.append(acceptor_h_bond)
            hydrogen_dict[hydrogen] = filtered_bonds
        else:
            hydrogen_dict[hydrogen] = []
        
    for idx in indices_hydrogen:
        if idx not in hydrogen_dict:
            hydrogen_dict[idx] = []

    if single_h_bond:
        num_hydrogen_bonds = sum(1 for bonds in hydrogen_dict.values() if len(bonds) > 1)
    else:
        num_hydrogen_bonds = sum(len(bonds[1:]) for bonds in hydrogen_dict.values())

    return hydrogen_dict, num_hydrogen_bonds


def aggregate_data(
    data: List[Dict[int, List]],
    index_map: Dict[int, int],
    N: int
) -> np.ndarray:
    """Aggregate hydrogen bond data across multiple frames.
    
    Parameters
    ----------
    data : List[Dict[int, List]]
        List of hydrogen bond dictionaries from each frame
    index_map : Dict[int, int]
        Mapping from global to local indices
    N : int
        Number of unique atoms
        
    Returns
    -------
    np.ndarray
        Aggregated hydrogen bond count array
    """
    """
    Aggregates hydrogen bond data to create correlation matrix and network graph.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing hydrogen bond data
    index_map : dict
        Mapping from atom indices to array indices
    N : int
        Number of atoms to include in correlation matrix
        
    Returns
    -------
    Tuple[np.ndarray, nx.Graph, List]
        Correlation matrix, NetworkX graph, and list of all pairs
    """
    # Aggregates hydrogen bond data
    node_frequency = {node: 0 for node in index_map.keys()}
    edge_weight = {}
    all_pairs = []

    for frame, group in data.groupby('Frame'):
        pairs = group[['Donor', 'Acceptor']].values
        all_pairs.extend(pairs)
        
        for donor, acceptor in pairs:
            if donor in index_map and acceptor in index_map:
                node_frequency[donor] += 1
                node_frequency[acceptor] += 1
                edge = tuple(sorted([donor, acceptor]))
                edge_weight[edge] = edge_weight.get(edge, 0) + 1

    G = nx.Graph()

    for node, freq in node_frequency.items():
        G.add_node(node, size=freq)
    
    for (donor, acceptor), weight in edge_weight.items():
        G.add_edge(donor, acceptor, weight=weight)

    corr_matrix = np.zeros((N, N), dtype=int)
    for (donor, acceptor), weight in edge_weight.items():
        donor_idx = index_map[donor]
        acceptor_idx = index_map[acceptor]
        corr_matrix[donor_idx, acceptor_idx] = weight
        corr_matrix[acceptor_idx, donor_idx] = weight

    return corr_matrix, G, all_pairs


def process_frame(data, donor_acceptor_indices, frame_index, index_map, N):
    """
    Processes data for a specific frame to create correlation matrix and network graph.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing hydrogen bond data
    donor_acceptor_indices : np.ndarray
        Array of donor/acceptor atom indices
    frame_index : int
        Frame index to process
    index_map : dict
        Mapping from atom indices to array indices
    N : int
        Number of atoms to include in correlation matrix
        
    Returns
    -------
    Tuple[np.ndarray, nx.Graph, List]
        Correlation matrix, NetworkX graph, and list of all pairs
    """
    frame_data = data[data['Frame'] == frame_index]
    
    has_distance = 'Distance' in frame_data.columns
    
    pairs = frame_data[['Donor', 'Acceptor']].values
    distances = frame_data['Distance'].values if has_distance else [1.0] * len(pairs)
    
    corr_matrix = np.zeros((N, N), dtype=float)  

    for i, (donor, acceptor) in enumerate(pairs):
        if donor in index_map and acceptor in index_map:
            donor_idx = index_map[donor]
            acceptor_idx = index_map[acceptor]
            distance = float(distances[i]) if has_distance and distances[i] != "N/A" else 0
            
            # Stores the distance in the correlation matrix
            if corr_matrix[donor_idx, acceptor_idx] == 0:
                corr_matrix[donor_idx, acceptor_idx] = distance
                corr_matrix[acceptor_idx, donor_idx] = distance
            else:
                # If multiple bonds exist, uses the average distance
                corr_matrix[donor_idx, acceptor_idx] = (corr_matrix[donor_idx, acceptor_idx] + distance) / 2
                corr_matrix[acceptor_idx, donor_idx] = (corr_matrix[acceptor_idx, donor_idx] + distance) / 2

    G = nx.Graph()
    for i, donor_idx in enumerate(range(len(donor_acceptor_indices))):
        for j, acceptor_idx in enumerate(range(len(donor_acceptor_indices))):
            if corr_matrix[i, j] > 0:
                G.add_edge(donor_acceptor_indices[i], donor_acceptor_indices[j],
                           weight=1,  # default weight of 1 for edge thickness
                           distance=corr_matrix[i, j])  # Store the actual distance

    return corr_matrix, G, pairs


def visualize_hydrogen_bonds_matrix(corr_matrix, donor_acceptor_indices=None, frame_index=None, average=False, output_dir=None):
    """
    Visualizes the hydrogen bond correlation matrix using Matplotlib.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix of hydrogen bonds
    donor_acceptor_indices : np.ndarray, optional
        Array of donor/acceptor atom indices
    frame_index : int, optional
        Frame index for title
    average : bool, optional
        Whether this is an average across frames
    output_dir : str, optional
        Directory to save output file
        
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 8))
    
    if np.issubdtype(corr_matrix.dtype, np.integer):
        fmt = "d"  # Integer format
    else:
        fmt = ".2f"  
    
    sns.heatmap(corr_matrix, annot=True, fmt=fmt, cmap='viridis', 
                xticklabels=donor_acceptor_indices, yticklabels=donor_acceptor_indices)
    
    if average:
        plt.title("Average Hydrogen Bond Correlation Matrix Across All Frames")
        filename = "hbond_correlation_matrix_average.png"
    else:
        plt.title(f"Hydrogen Bond Correlation Matrix for Frame {frame_index}")
        filename = f"hbond_correlation_matrix_frame_{frame_index}.png"
    
    plt.xlabel("Atom Index")
    plt.ylabel("Atom Index")
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.show()  
        plt.close()
        print(f"Correlation matrix saved as '{os.path.join(output_dir, filename)}'")
    else:
        plt.show()


def visualize_hydrogen_bonds_plotly(G, donor_acceptor_indices=None, frame_index=None, average=False, output_dir=None):
    """
    Visualizes the hydrogen bond network using Plotly.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph of hydrogen bonds
    donor_acceptor_indices : np.ndarray, optional
        Array of donor/acceptor atom indices
    frame_index : int, optional
        Frame index for title
    average : bool, optional
        Whether this is an average across frames
    output_dir : str, optional
        Directory to save output file
        
    Returns
    -------
    None
    """
    seed = 42
    pos = nx.spring_layout(G, seed=seed)
    
    node_size = [G.nodes[node].get('size', 20) for node in G.nodes()]
    node_color = [G.degree(node) for node in G.nodes()]
    edge_width = [G[u][v].get('weight', 1) * 0.5 for u, v in G.edges()]
    
    edge_distances = []
    for u, v in G.edges():
        distance = G[u][v].get('distance', None)
        if distance is not None and distance != "N/A":
            edge_distances.append(f"{distance:.2f} Å")
        else:
            edge_distances.append("N/A")

    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]

    edge_trace = []
    for i, (u, v) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_info = f'Bond {u}-{v}: {edge_distances[i]}'
        
        edge_trace.append(go.Scatter(
            x=[x0, x1], 
            y=[y0, y1], 
            mode='lines', 
            line=dict(width=edge_width[i], color='Magenta'),
            name=edge_info,
            hoverinfo='text',
            text=edge_info
        ))

    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes, mode='markers', 
        marker=dict(
            size=node_size, 
            color=node_color, 
            colorscale='Viridis',
            colorbar=dict(
                thickness=15, 
                title=dict(text='Node Connections', side='right'),
                xanchor='left'
            )
        ),
        text=[str(node) for node in G.nodes()],
        textposition='top center', 
        name='Donor/Acceptor Atoms'
    )
    
    title = "Average Hydrogen Bond Network Across All Frames" if average else f"Hydrogen Bond Network for Frame {frame_index}"
    
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            title=dict(text=f'<br>{title}', font=dict(size=16)),
            showlegend=False, 
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Network graph visualization of hydrogen bonds",
                showarrow=False, 
                xref="paper", 
                yref="paper", 
                x=0.005, 
                y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    if output_dir:
        filename = "hbond_network_average.html" if average else f"hbond_network_frame_{frame_index}.html"
        fig.write_html(os.path.join(output_dir, filename))
        print(f"Figure saved as '{os.path.join(output_dir, filename)}'")
    else:
        fig.show()


def visualize_hydrogen_bonds_matrix_plotly(corr_matrix, donor_acceptor_indices=None, frame_index=None, average=False, output_dir=None):
    """
    Visualizes the hydrogen bond correlation matrix using Plotly.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix of hydrogen bonds
    donor_acceptor_indices : np.ndarray, optional
        Array of donor/acceptor atom indices
    frame_index : int, optional
        Frame index for title
    average : bool, optional
        Whether this is an average across frames
    output_dir : str, optional
        Directory to save output file
        
    Returns
    -------
    None
    """
    # Format hover text based on matrix values
    hover_text = [[f"Donor: {donor_acceptor_indices[i]}<br>Acceptor: {donor_acceptor_indices[j]}<br>Value: {val:.2f}" 
                 if isinstance(val, float) else f"Donor: {donor_acceptor_indices[i]}<br>Acceptor: {donor_acceptor_indices[j]}<br>Value: {val}"
                 for j, val in enumerate(row)] for i, row in enumerate(corr_matrix)]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[str(idx) for idx in donor_acceptor_indices],
        y=[str(idx) for idx in donor_acceptor_indices],
        colorscale='Viridis',
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(title='No. of H-Bond')
    ))
    
    title = "Average Hydrogen Bond Correlation Matrix Across All Frames" if average else f"Hydrogen Bond Correlation Matrix for Frame {frame_index}"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title='Atom Index', tickfont=dict(size=10)),
        yaxis=dict(title='Atom Index', tickfont=dict(size=10)),
        width=900,
        height=800,
        autosize=True
    )
    
    if output_dir:
        filename = "hbond_correlation_matrix_average.html" if average else f"hbond_correlation_matrix_frame_{frame_index}.html"
        fig.write_html(os.path.join(output_dir, filename))
        print(f"Interactive correlation matrix saved as '{os.path.join(output_dir, filename)}'")
    else:
        fig.show()


def visualize_hydrogen_bonds(csv_file, indices_path, frame_index=None, average=False, output_dir=None):
    """
    Visualizes hydrogen bonds from a CSV file containing donor-acceptor pairs.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file with hydrogen bond data
    indices_path : str
        Path to NumPy array file with donor/acceptor indices
    frame_index : int, optional
        Frame index to visualize (required if average=False)
    average : bool, optional
        Whether to visualize average across all frames
    output_dir : str, optional
        Directory to save output files
        
    Returns
    -------
    None
    """
    data = pd.read_csv(csv_file)
    donor_acceptor_indices = np.load(indices_path)    
    index_map = {idx: i for i, idx in enumerate(donor_acceptor_indices)}
    N = len(donor_acceptor_indices)
    
    if average:
        corr_matrix, G, pairs = aggregate_data(data, index_map, N)
        visualize_hydrogen_bonds_matrix_plotly(corr_matrix, donor_acceptor_indices=donor_acceptor_indices, average=True, output_dir=output_dir)
        visualize_hydrogen_bonds_plotly(G, average=True, output_dir=output_dir)
    else:
        if frame_index is None:
            raise ValueError("frame_index must be provided when average=False")
        
        corr_matrix, G, pairs = process_frame(data, donor_acceptor_indices, frame_index, index_map, N)
        visualize_hydrogen_bonds_matrix_plotly(corr_matrix, donor_acceptor_indices=donor_acceptor_indices, frame_index=frame_index, output_dir=output_dir)
        visualize_hydrogen_bonds_plotly(G, frame_index=frame_index, output_dir=output_dir)


def hydrogen_bonds(traj_path, frame_skip=10, acceptor_atoms=["N","O","F"], angle_cutoff=120, 
                  h_bond_cutoff=2.4, bond_cutoff=1.6, mic=True, single_h_bond=False, 
                  output_dir="./", time_step=None, plot_count=False, plot_heatmap=False, 
                  plot_graph_frame=True, plot_graph_average=False, indices_path=None, 
                  graph_frame_index=0):
    """
    Analyze hydrogen bonds in a molecular dynamics trajectory.
    
    Parameters
    ----------
    traj_path : str
        Path to trajectory file
    frame_skip : int, optional
        Number of frames to skip (default: 10)
    acceptor_atoms : List[str], optional
        List of element symbols that can be acceptors (default: ["N","O","F"])
    angle_cutoff : float, optional
        Minimum angle in degrees for hydrogen bond (default: 120)
    h_bond_cutoff : float, optional
        Maximum distance in Å for hydrogen bond (default: 2.4)
    bond_cutoff : float, optional
        Maximum distance in Å for covalent bond (default: 1.6)
    mic : bool, optional
        Whether to use minimum image convention (default: True)
    single_h_bond : bool, optional
        Whether to count only first hydrogen bond per atom (default: False)
    output_dir : str, optional
        Directory to save output files (default: "./")
    time_step : float, optional
        Simulation time step for plotting (default: None)
    plot_count : bool, optional
        Whether to plot hydrogen bond count (default: False)
    plot_heatmap : bool, optional
        Whether to plot 2D histogram (default: False)
    plot_graph_frame : bool, optional
        Whether to plot interactive hydrogen bond network graph for specific frame (default: True)
    plot_graph_average : bool, optional
        Whether to plot interactive average hydrogen bond network graph (default: False)
    indices_path : str, optional
        Path to NumPy array with donor/acceptor atom indices for graph plotting.
        If None, the unique indices will be automatically extracted and saved (default: None)
    graph_frame_index : int, optional
        Frame index to use for graph visualization (default: 0)
        
    Returns
    -------
    List[int]
        List of hydrogen bond counts per frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f'hydrogen_bonds_{frame_skip}skips.csv')
    total_bonds_filename = os.path.join(output_dir, f'total_hydrogen_bonds_per_frame_{frame_skip}skips.csv')
    
    trajectory = read(traj_path, index=f"::{frame_skip}")
    
    all_data = Parallel(n_jobs=-1)(
        delayed(count_hydrogen_bonds)(
            atoms, acceptor_atoms=acceptor_atoms, angle_cutoff=angle_cutoff, h_bond_cutoff=h_bond_cutoff,
            bond_cutoff=bond_cutoff, mic=True, single_h_bond=single_h_bond
        ) for i, atoms in enumerate(trajectory)
    )

    h_bonds_per_frame = [num_bonds for _, num_bonds in all_data]
    frame_dict_list = [frame_dict for frame_dict, _ in all_data]
    data_dict = {i*frame_skip: d for i, d in enumerate(frame_dict_list)}
    
    # Donor-acceptor distances for each frame
    donor_acceptor_distances = {}
    for frame_idx, frame in enumerate(list(data_dict.keys())):
        frame_atoms = trajectory[frame_idx]
        dm = frame_atoms.get_all_distances(mic=mic)
        
        donor_acceptor_distances[frame] = {}
        
        for hydrogen in data_dict[frame]:
            if len(data_dict[frame][hydrogen]) > 1:
                donor = data_dict[frame][hydrogen][0][0]
                donor_acceptor_distances[frame][hydrogen] = {}
                
                for sublist in data_dict[frame][hydrogen][1:]:
                    acceptor = sublist[0]
                    # Get direct donor-acceptor distance (e.g., O-O distance in water)
                    donor_acceptor_dist = dm[donor, acceptor]
                    donor_acceptor_distances[frame][hydrogen][acceptor] = donor_acceptor_dist

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Frame', 'Hydrogen', 'Donor', 'Acceptor(s)', 
            'Donor-Hydrogen Distance', 'Hydrogen-Acceptor(s) Distance(s)', 
            'Donor-Acceptor(s) Distance(s)', 'Angle(A-H-D)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for frame in list(data_dict.keys()):
            for hydrogen in data_dict[frame]:
                donor = data_dict[frame][hydrogen][0][0] if len(data_dict[frame][hydrogen]) > 0 else ""
                donor_hydrogen_dist = data_dict[frame][hydrogen][0][1] if len(data_dict[frame][hydrogen]) > 0 else ""
                
                if len(data_dict[frame][hydrogen]) > 1:
                    acceptors = [sublist[0] for sublist in data_dict[frame][hydrogen][1:]]
                    acceptors_hydrogen_dist = [sublist[1] for sublist in data_dict[frame][hydrogen][1:]]
                    angles = [sublist[2] for sublist in data_dict[frame][hydrogen][1:]]
                    
                    # Get donor-acceptor distances
                    donor_acceptor_dist = [donor_acceptor_distances[frame][hydrogen].get(acc, "N/A") for acc in acceptors]
                else:
                    acceptors = ""
                    acceptors_hydrogen_dist = ""
                    donor_acceptor_dist = ""
                    angles = ""

                row = {
                    'Frame': frame,
                    'Hydrogen': hydrogen,
                    'Donor': donor,
                    'Acceptor(s)': acceptors,
                    'Donor-Hydrogen Distance': donor_hydrogen_dist,
                    'Hydrogen-Acceptor(s) Distance(s)': acceptors_hydrogen_dist,
                    'Donor-Acceptor(s) Distance(s)': donor_acceptor_dist,
                    'Angle(A-H-D)': angles,
                }
                
                writer.writerow(row)
        
    
    with open(total_bonds_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Total Hydrogen Bonds'])
        for frame_idx, num_bonds in enumerate(h_bonds_per_frame):
            writer.writerow([frame_idx * frame_skip, num_bonds])

    if time_step is not None and plot_count:
        plot_hydrogen_count(h_bonds_per_frame, frame_skip, time_step, output_dir)

    if plot_heatmap:
        plot_2Dheatmap(data_dict, output_dir)
        
    if plot_graph_frame or plot_graph_average:
        network_csv = os.path.join(output_dir, "hydrogen_bonds_network.csv")
        
        unique_atoms = set()
        
        with open(network_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Donor', 'Acceptor', 'Distance'])
            for frame in list(data_dict.keys()):
                for hydrogen in data_dict[frame]:
                    if len(data_dict[frame][hydrogen]) > 1:
                        donor = data_dict[frame][hydrogen][0][0]
                        unique_atoms.add(donor)  # Added donor to unique atoms
                        
                        for i, sublist in enumerate(data_dict[frame][hydrogen][1:]):
                            acceptor = sublist[0]
                            unique_atoms.add(acceptor)  # Added acceptor to unique atoms
                            
                            # Include the donor-acceptor distance in the network file
                            distance = donor_acceptor_distances[frame][hydrogen].get(acceptor, "N/A")
                            writer.writerow([frame, donor, acceptor, distance])
        
        if indices_path is None:
            indices_path = os.path.join(output_dir, "donor_acceptor_indices.npy")
            unique_atoms_array = np.array(sorted(list(unique_atoms)), dtype=int)
            np.save(indices_path, unique_atoms_array)
            print(f"Generated and saved {len(unique_atoms_array)} unique donor/acceptor atom indices to {indices_path}")
        
        if plot_graph_frame:
            print(f"Generating hydrogen bond network visualizations for frame {graph_frame_index}...")
            visualize_hydrogen_bonds(network_csv, indices_path, 
                                     frame_index=graph_frame_index, average=False, 
                                     output_dir=output_dir)
        
        if plot_graph_average:
            print("Generating average hydrogen bond network visualization...")
            visualize_hydrogen_bonds(network_csv, indices_path, 
                                     average=True, output_dir=output_dir)
    
    return h_bonds_per_frame


def plot_hydrogen_count(h_bonds_per_frame, frame_skip, time_step, output_dir):
    """
    Plot hydrogen bond count over time.
    
    Parameters
    ----------
    h_bonds_per_frame : List[int]
        Number of hydrogen bonds per frame
    frame_skip : int
        Number of frames skipped in trajectory analysis
    time_step : float
        Time step between frames in fs
    output_dir : str
        Directory to save output file
        
    Returns
    -------
    None
    """
    x = np.arange(len(h_bonds_per_frame)) * time_step * frame_skip / 1000

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x, h_bonds_per_frame, '-', color="blue", label="H-bond count")
    ax1.axhline(np.mean(h_bonds_per_frame), linestyle="--", color="blue", label=f"Mean: {np.mean(h_bonds_per_frame):.2f}")
    ax1.set_xlabel("Time [ps]", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    plt.title("Hydrogen bond count", fontsize=14)
    fig.legend(loc="center right", bbox_to_anchor=(1.1, 0.5))
    filename = os.path.join(output_dir, "h_bond_count.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Hydrogen bond count plot saved to '{filename}'")


def plot_2Dheatmap(data_dict, output_dir):
    angles_list = []
    dist_list = []
    
    for frame in list(data_dict.keys()):
            for hydrogen in data_dict[frame]:                
                if len(data_dict[frame][hydrogen]) > 1:
                    dist_list.append([sublist[1] for sublist in data_dict[frame][hydrogen][1:]])
                    angles_list.append([sublist[2] for sublist in data_dict[frame][hydrogen][1:]])

    angles_list = list(itertools.chain(*angles_list))
    dist_list = list(itertools.chain(*dist_list))
    
    hb = plt.hist2d(angles_list, dist_list, bins=30, cmap="viridis")

    plt.colorbar(hb[3], label="Count")
    
    plt.xlabel("Donor-Hydrogen-Acceptor Angle [°]",fontsize=12)
    plt.ylabel("Acceptor-Hydrogen Distance [Å]",fontsize=12)
    plt.title("2D Histogram of H-bonds parameters",fontsize=14)

    filename = os.path.join(output_dir, "h_bond_structure.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.show() 
    plt.close()
    print(f"H-bond structure 2D histogram saved to '{filename}'")
