"""
CRISP/data_analysis/volumetric_density.py

This module performs volumetric density distribution analysis on specific atoms 
in molecular dynamics trajectory data, creating 3D visualizations of atom density maps.
"""

import numpy as np
import plotly.graph_objects as go
from ase.io import read
from typing import Optional, Dict, List, Union
import os
__all__ = ['create_density_map', 'VDW_RADII', 'ELEMENT_COLORS']
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

# Default color palette for atom types
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


def create_density_map(
    traj_path: str,
    indices_path: str,
    frame_skip: int = 100,
    threshold: float = 0.05,
    absolute_threshold: bool = False,
    opacity: float = 0.2,
    atom_size_scale: float = 3.0,
    output_dir: str = ".",
    output_file: Optional[str] = None,
    colorscale: str = 'Plasma',
    plot_title: str = 'Density Distribution of Selected Atoms',
    nbins: int = 50,
    omit_static_indices: Optional[Union[str, List[int], np.ndarray]] = None,
    save_density: bool = False,
    density_output_file: Optional[str] = None,
    show_projections: bool = False,
    projection_opacity: float = 0.7,
    projection_offset: float = 2.0,
    save_projection_images: bool = False,
    projection_image_dpi: int = 300
) -> go.Figure:
    """
    Create a 3D visualization of atom density with molecular structure and save to HTML.
    
    This function analyzes the spatial distribution of selected atoms across a 
    trajectory, creating a volumetric density map showing where specified atoms 
    tend to be located over time.
    
    Parameters
    ----------
    traj_path : str
        Path to the ASE trajectory file
    indices_path : str
        Path to numpy file containing atom indices to analyze
    frame_skip : int, optional
        Only read every nth frame from trajectory (default: 100)
    threshold : float, optional
        Density value threshold (default: 0.05)
    absolute_threshold : bool, optional
        If True, threshold is an absolute count value;
        If False (default), threshold is relative to maximum (0-1 scale)
    opacity : float, optional
        Transparency of the density visualization (default: 0.2, range: 0.0-1.0)
    atom_size_scale : float, optional
        Scale factor for atom sizes (default: 3.0)
    output_dir : str, optional
        Directory to save output files (default: current directory)
    output_file : str, optional
        Filename for HTML output (default: auto-generated from trajectory name)
    colorscale : str, optional
        Colorscale for density plot ('Plasma', 'Viridis', 'Blues', etc.)
    plot_title : str, optional
        Title for the visualization
    nbins : int, optional
        Number of bins for density grid in each dimension (default: 50)
    omit_static_indices: str, list or np.ndarray, optional
        Indices of atoms to omit from the static structure visualization.
        Can be a path to a numpy file (like indices_path) or an array of indices.
        If None, the selected_indices (from indices_path) will be used.
    save_density : bool, optional
        If True, saves the density data to a file (default: False)
    density_output_file : str, optional
        Filename for saving density data (default: auto-generated from trajectory name)
    show_projections : bool, optional
        Whether to show 2D projections of the density (default: False)
    projection_opacity : float, optional
        Opacity of projection surfaces (default: 0.7)
    projection_offset : float, optional
        Distance to offset projections from the main volume (default: 2.0)
    save_projection_images : bool, optional
        Whether to save 2D projections as separate PNG files (default: False)
    projection_image_dpi : int, optional
        Resolution of saved projection images (default: 300)
    
    Returns
    -------
    plotly.graph_objects.Figure
        The generated figure object
    
    Notes
    -----
    The function creates a 3D histogram of selected atom positions across the
    trajectory and visualizes it as an isosurface volume plot, overlaid with
    the reference molecular structure and unit cell boundaries.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_file is None:
        traj_basename = os.path.splitext(os.path.basename(traj_path))[0]
        output_file = f"{traj_basename}_density_map.html"
    
    output_path = os.path.join(output_dir, output_file)
    
    threshold = max(0.0, threshold)  # Ensures threshold is at least 0
    opacity = max(0.0, min(1.0, opacity))  # Ensures opacity is between 0 and 1
    
    print(f"Loading trajectory from {traj_path} (using every {frame_skip}th frame)...")
    trajectory = read(traj_path, index=f"::{frame_skip}")
    selected_indices = np.load(indices_path)
    print(f"Loaded {len(trajectory)} frames, {len(selected_indices)} selected indices")
    
    # Get a reference frame for the static structure
    static_frame = trajectory[0]
    cell = static_frame.get_cell()

    # Define cell boundaries (assuming orthogonal cell)
    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = cell[0, 0], cell[1, 1], cell[2, 2]

    print("Extracting selected atom positions from trajectory...")
    positions = []
    for frame in trajectory:
        frame.wrap()  
        for idx in selected_indices:
            positions.append(frame.positions[idx])
    positions = np.array(positions)

    # Create a 3D grid using np.histogramdd
    print("Creating density grid...")
    edges_x = np.linspace(xmin, xmax, nbins)
    edges_y = np.linspace(ymin, ymax, nbins)
    edges_z = np.linspace(zmin, zmax, nbins)
    H, edges = np.histogramdd(positions, bins=[edges_x, edges_y, edges_z])

    if save_density:
        if density_output_file is None:
            traj_basename = os.path.splitext(os.path.basename(traj_path))[0]
            density_output_file = f"{traj_basename}_density_data.npz"
        
        density_path = os.path.join(output_dir, density_output_file)
        
        np.savez(
            density_path,
            density=H,                # Actual density histogram
            edges=edges,              # Bin edges
            cell=cell,                # Unit cell
            nbins=nbins,              # Number of bins
            selected_indices=selected_indices  # Which atoms were used
        )
        
        print(f"Density data saved to: {density_path}")

    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    z_centers = (edges[2][:-1] + edges[2][1:]) / 2
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # Choose whether to use absolute or relative thresholds
    if absolute_threshold:
        vol = H
        isomin = threshold  # absolute count threshold
        isomax = H.max()    # maximum raw count
        threshold_type = "absolute"
    else:
        # Normalize the histogram for relative thresholds
        vol = H / H.max()
        isomin = threshold  # relative threshold (0-1)
        isomax = 1.0        # maximum normalized value
        threshold_type = "relative"
    
    print(f"Creating visualization with {threshold_type} threshold={threshold}, opacity={opacity}")
    print(f"Density range: {H.min()} to {H.max()} counts")
    
    fig = go.Figure()

    # Add volume trace for density visualization
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol.flatten(),
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,
        surface_count=20,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Density Volume'
    ))

    if show_projections:
        print("Adding 2D projections of density data...")
        
        # Calculate projection data by summing along each axis
        xy_projection = np.sum(H, axis=2)  # Sum along z-axis
        xz_projection = np.sum(H, axis=1)  # Sum along y-axis
        yz_projection = np.sum(H, axis=0)  # Sum along x-axis
        
        # Normalize if using relative threshold
        if not absolute_threshold:
            max_val = max(xy_projection.max(), xz_projection.max(), yz_projection.max())
            if max_val > 0:
                xy_projection = xy_projection / max_val
                xz_projection = xz_projection / max_val
                yz_projection = yz_projection / max_val
                
        # Define offset positions for projections
        # XY projection (bottom)
        z_offset = np.ones((len(x_centers), len(y_centers))) * (np.min(z_centers) - projection_offset)
        # YZ projection (left)
        x_offset = np.ones((len(y_centers), len(z_centers))) * (np.min(x_centers) - projection_offset)
        # XZ projection (back)
        y_offset = np.ones((len(x_centers), len(z_centers))) * (np.max(y_centers) + projection_offset)
        
        xy_text = [[f'x: {x_centers[i]:.2f}<br>y: {y_centers[j]:.2f}<br>Density: {xy_projection[i,j]:.2f}' 
                   for j in range(len(y_centers))] for i in range(len(x_centers))]
        
        yz_text = [[f'y: {y_centers[i]:.2f}<br>z: {z_centers[j]:.2f}<br>Density: {yz_projection[i,j]:.2f}' 
                   for j in range(len(z_centers))] for i in range(len(y_centers))]
        
        xz_text = [[f'x: {x_centers[i]:.2f}<br>z: {z_centers[j]:.2f}<br>Density: {xz_projection[i,j]:.2f}' 
                   for j in range(len(z_centers))] for i in range(len(x_centers))]
        
        # Add XY projection (floor)
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
        fig.add_trace(go.Surface(
            z=z_offset,
            x=xx,
            y=yy,
            surfacecolor=xy_projection,
            colorscale=colorscale,
            opacity=projection_opacity,
            showscale=False,
            text=xy_text,
            hoverinfo='text',
            name='XY Projection (Floor)'
        ))
        
        # Add YZ projection (left wall)
        yy, zz = np.meshgrid(y_centers, z_centers, indexing='ij')
        fig.add_trace(go.Surface(
            z=zz,
            x=x_offset,
            y=yy,
            surfacecolor=yz_projection,
            colorscale=colorscale,
            opacity=projection_opacity,
            showscale=False,
            text=yz_text,
            hoverinfo='text',
            name='YZ Projection (Left Wall)'
        ))
        
        # Add XZ projection (back wall)
        xx, zz = np.meshgrid(x_centers, z_centers, indexing='ij')
        fig.add_trace(go.Surface(
            z=zz,
            x=xx,
            y=y_offset,
            surfacecolor=xz_projection,
            colorscale=colorscale,
            opacity=projection_opacity,
            showscale=False,
            text=xz_text,
            hoverinfo='text',
            name='XZ Projection (Back Wall)'
        ))

        if save_projection_images:
            output_basename = os.path.splitext(output_file)[0]
            
            # Save XY projection (top-down view)
            xy_fig = go.Figure(data=go.Heatmap(
                z=xy_projection.T,  
                x=x_centers,
                y=y_centers,
                colorscale=colorscale,
                colorbar=dict(
                    title='Density',
                    thickness=20
                )
            ))
            xy_fig.update_layout(
                title=f"{plot_title} - XY Projection (Top View)",
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                width=800,
                height=700
            )
            xy_path = os.path.join(output_dir, f"{output_basename}_xy_projection.png")
            xy_fig.write_image(xy_path, scale=1, width=800, height=700, engine="kaleido")
            print(f"XY projection saved to: {xy_path}")
            
            # Save YZ projection (side view)
            yz_fig = go.Figure(data=go.Heatmap(
                z=yz_projection.T,  
                x=y_centers,
                y=z_centers,
                colorscale=colorscale,
                colorbar=dict(
                    title='Density',
                    thickness=20
                )
            ))
            yz_fig.update_layout(
                title=f"{plot_title} - YZ Projection (Side View)",
                xaxis_title='Y (Å)',
                yaxis_title='Z (Å)',
                width=800,
                height=700
            )
            yz_path = os.path.join(output_dir, f"{output_basename}_yz_projection.png")
            yz_fig.write_image(yz_path, scale=1, width=800, height=700, engine="kaleido")
            print(f"YZ projection saved to: {yz_path}")
            
            # Save XZ projection (front view)
            xz_fig = go.Figure(data=go.Heatmap(
                z=xz_projection.T,  
                x=x_centers,
                y=z_centers,
                colorscale=colorscale,
                colorbar=dict(
                    title='Density',
                    thickness=20
                )
            ))
            xz_fig.update_layout(
                title=f"{plot_title} - XZ Projection (Front View)",
                xaxis_title='X (Å)',
                yaxis_title='Z (Å)',
                width=800,
                height=700
            )
            xz_path = os.path.join(output_dir, f"{output_basename}_xz_projection.png")
            xz_fig.write_image(xz_path, scale=1, width=800, height=700, engine="kaleido")
            print(f"XZ projection saved to: {xz_path}")

    # Only add atomic structure visualization if projections are not enabled
    if not show_projections:
        symbols = static_frame.get_chemical_symbols()
        positions = static_frame.positions
        unique_elements = list(set(symbols))
        
        indices_to_omit = selected_indices 
        
        if omit_static_indices is not None:
            if isinstance(omit_static_indices, str):
                # Treat as path to numpy file
                indices_to_omit = np.load(omit_static_indices)
            else:
                indices_to_omit = np.array(omit_static_indices)
            print(f"Omitting {len(indices_to_omit)} custom indices from static structure visualization")

        for element in unique_elements:
            element_indices = [i for i, symbol in enumerate(symbols) if symbol == element and i not in indices_to_omit]
            
            if not element_indices:
                continue
                
            element_positions = positions[element_indices]
            
            # Use ELEMENT_COLORS for color and VDW_RADII for size
            color = ELEMENT_COLORS.get(element, 'gray')
            
            # Calculate marker size based on van der Waals radius (with scaling factor)
            size = VDW_RADII.get(element, 1.0) * atom_size_scale
            
            # Uniform size for all atoms of this element
            sizes = [size] * len(element_indices)
            
            fig.add_trace(go.Scatter3d(
                x=element_positions[:, 0],
                y=element_positions[:, 1],
                z=element_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.8,
                    line=dict(color='black', width=0.5)
                ),
                name=element
            ))
    else:
        print("Skipping atom visualization since projections are enabled")

    if np.all(static_frame.get_pbc()):
        cell = static_frame.get_cell()
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        box_coords = np.dot(corners, cell)
        
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        
        for i, j in edges:
            fig.add_trace(go.Scatter3d(
                x=[box_coords[i, 0], box_coords[j, 0]],
                y=[box_coords[i, 1], box_coords[j, 1]],
                z=[box_coords[i, 2], box_coords[j, 2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

    threshold_info = f"(Threshold: {threshold} {'counts' if absolute_threshold else 'relative'})"
    full_title = f"{plot_title} {threshold_info}"

    fig.update_layout(
        title=full_title,
        scene=dict(
            xaxis=dict(title='X (Å)'),
            yaxis=dict(title='Y (Å)'),
            zaxis=dict(title='Z (Å)'),
            aspectmode='data' if not show_projections else 'cube'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    if show_projections:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            args=[{"visible": [True, True, True, True]}],  # Only 4 traces: volume + 3 projections
                            label="All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False, False, False]}],
                            label="Volume Only",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, True, False, False]}],
                            label="XY Projection",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False, True, False]}],
                            label="YZ Projection",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False, False, True]}],
                            label="XZ Projection",
                            method="update"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

    fig.write_html(output_path)
    print(f"Visualization saved as HTML file: {output_path}")
    
    return fig