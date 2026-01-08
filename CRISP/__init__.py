"""A python package for Post-simulation analysis and visualization"""

__all__ = [
    "cli",
    "data_analysis",
    "simulation_utility",
]

from ._version import __version__

from . import cli
from . import data_analysis
from . import simulation_utility

# Re-export commonly used classes and functions for convenience
# This allows both import styles:
# Old style: from CRISP.data_analysis.prdf import Analysis
# New style: from CRISP import Analysis

# Data analysis imports
from .data_analysis.prdf import (
    check_cell_and_r_max,
    compute_pairwise_rdf,
    plot_rdf,
    animate_rdf,
    analyze_rdf,
    Analysis
)

from .data_analysis.clustering import (
    analyze_frame, 
    analyze_trajectory, 
    cluster_analysis, 
    save_analysis_results, 
    plot_analysis_results,
    cluster_analysis
)

from .data_analysis.contact_coordination import (
    indices, 
    coordination_frame, 
    coordination, 
    contacts_frame, 
    contacts
)

from .data_analysis.h_bond import (
    indices, 
    count_hydrogen_bonds, 
    aggregate_data, 
    hydrogen_bonds
)

from .data_analysis.msd import (
    read_trajectory_chunk, 
    calculate_frame_msd, 
    calculate_msd, 
    calculate_save_msd, 
    msd_analysis
)

from .data_analysis.volumetric_atomic_density import (
    create_density_map, 
    VDW_RADII, 
    ELEMENT_COLORS
)

# Simulation utility imports
from .simulation_utility.atomic_indices import (
    atom_indices, 
    run_atom_indices
)

from .simulation_utility.atomic_traj_linemap import (
    plot_atomic_trajectory, 
    VDW_RADII, 
    ELEMENT_COLORS
)

from .simulation_utility.error_analysis import (
    optimal_lag, 
    vector_acf, 
    autocorrelation_analysis, 
    block_analysis
)

from .simulation_utility.interatomic_distances import (
    indices, 
    distance_calculation, 
    save_distance_matrices, 
    calculate_interatomic_distances
)

from .simulation_utility.subsampling import (
    indices, 
    compute_soap, 
    create_repres, 
    subsample
)
