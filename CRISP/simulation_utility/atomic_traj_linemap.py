"""
CRISP/simulation_utility/atomic_traj_linemap.py

This module provides functionality for visualizing atomic trajectories from molecular dynamics 
simulations using interactive 3D plots.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Union
from ase.io import read
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
__all__ = ['plot_atomic_trajectory', 'VDW_RADII', 'ELEMENT_COLORS']
# Dictionary of van der Waals radii for all elements in Ångström
VDW_RADII = {
    # Period 1
    'H': 1.20, 'He': 1.40, 
    # Period 2
    'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    # Period 3
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    # Period 4
    'K': 2.75, 'Ca': 2.31, 'Sc': 2.30, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05, 'Mn': 2.05,
    'Fe': 2.05, 'Co': 2.00, 'Ni': 2.00, 'Cu': 2.00, 'Zn': 2.10, 'Ga': 1.87, 'Ge': 2.11,
    'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    # Period 5
    'Rb': 3.03, 'Sr': 2.49, 'Y': 2.40, 'Zr': 2.30, 'Nb': 2.15, 'Mo': 2.10, 'Tc': 2.05,
    'Ru': 2.05, 'Rh': 2.00, 'Pd': 2.05, 'Ag': 2.10, 'Cd': 2.20, 'In': 2.20, 'Sn': 2.17,
    'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    # Period 6
    'Cs': 3.43, 'Ba': 2.68, 'La': 2.50, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45, 'Pm': 2.43,
    'Sm': 2.42, 'Eu': 2.40, 'Gd': 2.38, 'Tb': 2.37, 'Dy': 2.35, 'Ho': 2.33, 'Er': 2.32,
    'Tm': 2.30, 'Yb': 2.28, 'Lu': 2.27, 'Hf': 2.25, 'Ta': 2.20, 'W': 2.10, 'Re': 2.05,
    'Os': 2.00, 'Ir': 2.00, 'Pt': 2.05, 'Au': 2.10, 'Hg': 2.05, 'Tl': 2.20, 'Pb': 2.30,
    'Bi': 2.30, 'Po': 2.00, 'At': 2.00, 'Rn': 2.00,
    # Period 7
    'Fr': 3.50, 'Ra': 2.80, 'Ac': 2.60, 'Th': 2.40, 'Pa': 2.30, 'U': 2.30, 'Np': 2.30,
    'Pu': 2.30, 'Am': 2.30, 'Cm': 2.30, 'Bk': 2.30, 'Cf': 2.30, 'Es': 2.30, 'Fm': 2.30,
    'Md': 2.30, 'No': 2.30, 'Lr': 2.30, 'Rf': 2.30, 'Db': 2.30, 'Sg': 2.30, 'Bh': 2.30,
    'Hs': 2.30, 'Mt': 2.30, 'Ds': 2.30, 'Rg': 2.30, 'Cn': 2.30, 'Nh': 2.30, 'Fl': 2.30,
    'Mc': 2.30, 'Lv': 2.30, 'Ts': 2.30, 'Og': 2.30
}


ELEMENT_COLORS = {
    # Common elements
    'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green',
    'Na': 'purple', 'Mg': 'pink', 'Al': 'gray', 'Si': 'yellow', 'P': 'orange',
    'S': 'yellow', 'Cl': 'green', 'K': 'purple', 'Ca': 'gray', 'Fe': 'orange',
    'Cu': 'orange', 'Zn': 'gray',
    # Additional common elements with colors
    'Br': 'brown', 'I': 'purple', 'Li': 'purple', 'B': 'olive',
    'He': 'cyan', 'Ne': 'cyan', 'Ar': 'cyan', 'Kr': 'cyan', 'Xe': 'cyan',
    'Mn': 'gray', 'Co': 'blue', 'Ni': 'green', 'Pd': 'gray', 'Pt': 'gray', 
    'Au': 'gold', 'Hg': 'silver', 'Pb': 'darkgray', 'Ag': 'silver',
    'Ti': 'gray', 'V': 'gray', 'Cr': 'gray', 'Zr': 'gray', 'Mo': 'gray', 
    'W': 'gray', 'U': 'green'
}

def plot_atomic_trajectory(
    traj_path: str,
    indices_path: Union[str, List[int]],
    output_dir: str,
    output_filename: str = "trajectory_plot.html",
    frame_skip: int = 100,
    plot_title: str = None,
    show_plot: bool = False,
    atom_size_scale: float = 1.0,
    plot_lines: bool = False  
):
    """
    Create a 3D visualization of atom trajectories with all atom types displayed.
    
    Parameters
    ----------
    traj_path : str
        Path to the ASE trajectory file (supports any ASE-readable format like XYZ)
    indices_path : str or List[int]
        Either a path to numpy file containing atom indices to plot trajectories for,
        or a direct list of atom indices
    output_dir : str
        Directory where the output visualization will be saved
    output_filename : str, optional
        Filename for the output visualization (default: "trajectory_plot.html")
    frame_skip : int, optional
        Use every nth frame from the trajectory (default: 100)
    plot_title : str, optional
        Custom title for the plot (default: auto-generated based on atom types)
    show_plot : bool, optional
        Whether to display the plot interactively (default: False)
    atom_size_scale : float, optional
        Scale factor for atom sizes in the visualization (default: 1.0)
    plot_lines : bool, optional
        Whether to connect trajectory points with lines (default: False)
        
    Returns
    -------
    plotly.graph_objects.Figure
        The generated plotly figure object that can be further customized
        
    Notes
    -----
    This function creates an interactive 3D visualization showing:
    1. All atoms from the first frame, colored by element
    2. Trajectory paths for selected atoms throughout all frames
    3. Annotations for the start and end positions of traced atoms
    
    The output is saved as an HTML file which can be opened in any web browser.
    """
    print(f"Loading trajectory from {traj_path} (using every {frame_skip}th frame)...")
    traj = read(traj_path, index=f'::{frame_skip}')
    
    # Convert to list if not already (happens with single frame)
    if not isinstance(traj, list):
        traj = [traj]
    
    print(f"Loaded {len(traj)} frames from trajectory")
    
    if isinstance(indices_path, str):
        selected_indices = np.load(indices_path)
        print(f"Loaded {len(selected_indices)} atoms for trajectory plotting from {indices_path}")
    else:
        selected_indices = np.array(indices_path)
        print(f"Using {len(selected_indices)} directly provided atom indices for trajectory plotting")
    
    box = traj[0].cell.lengths()
    print(f"Simulation box dimensions: {box} Å")
    
    atom_types = {}
    max_index = max([atom.index for atom in traj[0]])
    print(f"Analyzing atom types in first frame (total atoms: {len(traj[0])}, max index: {max_index})...")
    
    for atom in traj[0]:
        symbol = atom.symbol
        if symbol not in atom_types:
            atom_types[symbol] = []
        atom_types[symbol].append(atom.index)
    
    print(f"Found {len(atom_types)} atom types: {', '.join(atom_types.keys())}")
    
    fig = go.Figure()
    
    use_same_color = len(selected_indices) > 5
    colors = ['blue'] * len(selected_indices) if use_same_color else [
        'blue', 'green', 'red', 'orange', 'purple'
    ][:len(selected_indices)]
    
    for symbol, indices in atom_types.items():
        positions = np.array([traj[0].positions[i] for i in indices])
        
        # Skip if no atoms of this type
        if len(positions) == 0:
            continue
        
        size = VDW_RADII.get(symbol, 1.0) * 3.0 * atom_size_scale
        color = ELEMENT_COLORS.get(symbol, 'gray')
        
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            name=f'{symbol} Atoms',
            marker=dict(
                size=size,
                color=color,
                symbol='circle',
                opacity=0.7,
                line=dict(color='black', width=0.5)
            )
        ))
    
    selected_positions = {idx: [] for idx in selected_indices}
    for atoms in traj:
        for idx in selected_indices:
            if idx < len(atoms):
                selected_positions[idx].append(atoms.positions[idx])
            else:
                print(f"Warning: Index {idx} is out of range")
    
    annotations = []
    for i, idx in enumerate(selected_indices):
        if not selected_positions[idx]:  
            continue
            
        pos = np.array(selected_positions[idx])
        color = colors[i % len(colors)]
        
        # Add annotations for first and last frames 
        first_frame_pos = pos[0]
        last_frame_pos = pos[-1]
        
        annotations.extend([
            dict(
                x=first_frame_pos[0],
                y=first_frame_pos[1],
                z=first_frame_pos[2],
                text=f'Start {idx}',
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-20,
                arrowcolor=color,
                font=dict(color=color, size=10)
            ),
            dict(
                x=last_frame_pos[0],
                y=last_frame_pos[1],
                z=last_frame_pos[2],
                text=f'End {idx}',
                showarrow=True,
                arrowhead=2,
                ax=-20,
                ay=-20,
                arrowcolor=color,
                font=dict(color=color, size=10)
            )
        ])
        
        if plot_lines:
            # Add trajectory line and markers (original behavior)
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='lines+markers',
                name=f'Atom {idx}',
                line=dict(width=3, color=color),
                marker=dict(size=4, color=color),
            ))
        else:
            # Scatter-only mode: just showing points for each frame
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='markers',
                name=f'Atom {idx}',
                marker=dict(
                    size=5,
                    color=color,
                    symbol='circle',
                    opacity=0.8,
                    line=dict(color='black', width=0.5)
                )
            ))
    
    if not plot_title:
        atom_types_str = ', '.join(atom_types.keys())
        plot_title = f'Atomic Trajectories in {atom_types_str} System'
    
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            xaxis=dict(range=[0, box[0]]),
            yaxis=dict(range=[0, box[1]]),
            zaxis=dict(range=[0, box[2]]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        scene_annotations=annotations  
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    fig.write_html(output_path)
    print(f"Plot has been saved to {output_path}")
    
    if show_plot:
        fig.show()
    
    return fig