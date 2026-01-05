"""
Comprehensive tests for atomic_traj_linemap module using real trajectory data.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from ase.io import read, write
from ase import Atoms
from CRISP.simulation_utility.atomic_traj_linemap import (
    plot_atomic_trajectory,
    VDW_RADII,
    ELEMENT_COLORS
)


@pytest.fixture
def real_trajectory():
    """Return path to the actual trajectory file in the data folder."""
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    traj_path = os.path.join(data_folder, 'wrapped_traj.traj')
    if os.path.exists(traj_path):
        return traj_path
    else:
        pytest.skip("Real trajectory file not found")


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_trajectory():
    """Create a small test trajectory."""
    temp_dir = tempfile.mkdtemp()
    traj_path = os.path.join(temp_dir, 'test.traj')
    
    frames = []
    for i in range(5):
        atoms = Atoms('H2O', positions=[
            [0 + 0.1*i, 0, 0],
            [1 + 0.1*i, 0, 0],
            [0.5 + 0.1*i, 0.5, 0]
        ])
        atoms.cell = [10, 10, 10]
        atoms.pbc = True
        frames.append(atoms)
    
    write(traj_path, frames)
    yield traj_path
    shutil.rmtree(temp_dir)


class TestPlotAtomicTrajectoryBasic:
    """Test basic functionality of plot_atomic_trajectory."""
    
    def test_plot_with_real_trajectory_all_atoms(self, real_trajectory, temp_output_dir):
        """Test plotting with real trajectory data."""
        # Read trajectory to get number of atoms
        traj = read(real_trajectory, index='0')
        n_atoms = len(traj)
        
        # Plot all atoms
        fig = plot_atomic_trajectory(
            real_trajectory,
            indices_path=list(range(min(3, n_atoms))),  # Plot first 3 atoms
            output_dir=temp_output_dir,
            output_filename='test_plot.html',
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None
        assert os.path.exists(os.path.join(temp_output_dir, 'test_plot.html'))
    
    def test_plot_with_real_trajectory_frame_skip(self, real_trajectory, temp_output_dir):
        """Test frame skipping with real trajectory."""
        traj = read(real_trajectory, index='0')
        n_atoms = len(traj)
        
        fig = plot_atomic_trajectory(
            real_trajectory,
            indices_path=list(range(min(2, n_atoms))),
            output_dir=temp_output_dir,
            frame_skip=2,
            show_plot=False
        )
        
        assert fig is not None
    
    def test_plot_with_custom_title(self, small_trajectory, temp_output_dir):
        """Test with custom plot title."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0, 1],
            output_dir=temp_output_dir,
            plot_title="Custom Trajectory Title",
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None
        assert 'Custom Trajectory Title' in fig.layout.title.text
    
    def test_plot_with_atom_size_scale(self, small_trajectory, temp_output_dir):
        """Test with different atom size scales."""
        for scale in [0.5, 1.0, 2.0]:
            fig = plot_atomic_trajectory(
                small_trajectory,
                indices_path=[0],
                output_dir=temp_output_dir,
                atom_size_scale=scale,
                frame_skip=1,
                show_plot=False
            )
            
            assert fig is not None
    
    def test_plot_with_lines_enabled(self, small_trajectory, temp_output_dir):
        """Test trajectory with lines connecting positions."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0, 1],
            output_dir=temp_output_dir,
            plot_lines=True,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None


class TestPlotAtomicTrajectoryIndices:
    """Test different ways of specifying atom indices."""
    
    def test_plot_with_list_indices(self, small_trajectory, temp_output_dir):
        """Test with list of integer indices."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0, 2],
            output_dir=temp_output_dir,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None
    
    def test_plot_with_numpy_file_indices(self, small_trajectory, temp_output_dir):
        """Test with numpy file containing indices."""
        # Create numpy indices file
        indices_file = os.path.join(temp_output_dir, 'indices.npy')
        np.save(indices_file, np.array([0, 1]))
        
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=indices_file,
            output_dir=temp_output_dir,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None
    
    def test_plot_with_single_atom(self, small_trajectory, temp_output_dir):
        """Test plotting single atom trajectory."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[1],
            output_dir=temp_output_dir,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None


class TestPlotAtomicTrajectoryOutputFiles:
    """Test output file creation and handling."""
    
    def test_output_file_created(self, small_trajectory, temp_output_dir):
        """Test that output HTML file is created."""
        output_file = 'custom_output.html'
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0],
            output_dir=temp_output_dir,
            output_filename=output_file,
            frame_skip=1,
            show_plot=False
        )
        
        output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(output_path)
        
        # Check file is not empty
        assert os.path.getsize(output_path) > 0
    
    def test_output_directory_created(self, small_trajectory):
        """Test that output directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        new_output_dir = os.path.join(temp_dir, 'new_subdir')
        
        try:
            fig = plot_atomic_trajectory(
                small_trajectory,
                indices_path=[0],
                output_dir=new_output_dir,
                frame_skip=1,
                show_plot=False
            )
            
            assert os.path.exists(new_output_dir)
        finally:
            shutil.rmtree(temp_dir)


class TestPlotAtomicTrajectoryFrameSkip:
    """Test frame skipping functionality."""
    
    def test_frame_skip_variations(self, small_trajectory, temp_output_dir):
        """Test different frame skip values."""
        for skip in [1, 2, 3]:
            fig = plot_atomic_trajectory(
                small_trajectory,
                indices_path=[0],
                output_dir=temp_output_dir,
                frame_skip=skip,
                show_plot=False
            )
            
            assert fig is not None
    
    def test_large_frame_skip(self, real_trajectory, temp_output_dir):
        """Test with very large frame skip."""
        traj = read(real_trajectory, index='0')
        
        fig = plot_atomic_trajectory(
            real_trajectory,
            indices_path=[0],
            output_dir=temp_output_dir,
            frame_skip=100,
            show_plot=False
        )
        
        assert fig is not None


class TestPlotAtomicTrajectoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_frame_trajectory(self, temp_output_dir):
        """Test with single frame trajectory."""
        temp_dir = tempfile.mkdtemp()
        traj_path = os.path.join(temp_dir, 'single_frame.traj')
        
        try:
            atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]])
            atoms.cell = [10, 10, 10]
            write(traj_path, atoms)
            
            fig = plot_atomic_trajectory(
                traj_path,
                indices_path=[0],
                output_dir=temp_output_dir,
                frame_skip=1,
                show_plot=False
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_empty_indices_list(self, small_trajectory, temp_output_dir):
        """Test with empty indices list."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[],
            output_dir=temp_output_dir,
            frame_skip=1,
            show_plot=False
        )
        
        # Should still create plot with all atoms in first frame
        assert fig is not None
    
    def test_very_small_atom_size(self, small_trajectory, temp_output_dir):
        """Test with very small atom size scale."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0],
            output_dir=temp_output_dir,
            atom_size_scale=0.1,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None
    
    def test_very_large_atom_size(self, small_trajectory, temp_output_dir):
        """Test with very large atom size scale."""
        fig = plot_atomic_trajectory(
            small_trajectory,
            indices_path=[0],
            output_dir=temp_output_dir,
            atom_size_scale=5.0,
            frame_skip=1,
            show_plot=False
        )
        
        assert fig is not None


class TestVDWRadiiAndColors:
    """Test VDW radii and element color dictionaries."""
    
    def test_vdw_radii_common_elements(self):
        """Test VDW radii for common elements."""
        assert 'H' in VDW_RADII
        assert 'C' in VDW_RADII
        assert 'N' in VDW_RADII
        assert 'O' in VDW_RADII
        
        assert VDW_RADII['H'] > 0
        assert VDW_RADII['O'] > 0
    
    def test_element_colors_common_elements(self):
        """Test element colors for common elements."""
        assert 'H' in ELEMENT_COLORS
        assert 'C' in ELEMENT_COLORS
        assert 'N' in ELEMENT_COLORS
        assert 'O' in ELEMENT_COLORS
        
        assert ELEMENT_COLORS['H'] == 'white'
        assert ELEMENT_COLORS['O'] == 'red'
    
    def test_vdw_radii_all_periods(self):
        """Test that VDW radii covers multiple periods."""
        # Period 1-7 elements
        test_elements = ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
        for elem in test_elements:
            assert elem in VDW_RADII
            assert VDW_RADII[elem] > 0


class TestPlotAtomicTrajectoryIntegration:
    """Integration tests with real trajectory data."""
    
    def test_complete_workflow(self, real_trajectory, temp_output_dir):
        """Test complete workflow from trajectory to visualization."""
        # Read trajectory to get info
        traj = read(real_trajectory, index='0')
        n_atoms = len(traj)
        
        # Select subset of atoms
        indices = list(range(min(3, n_atoms)))
        
        # Create visualization
        fig = plot_atomic_trajectory(
            real_trajectory,
            indices_path=indices,
            output_dir=temp_output_dir,
            output_filename='workflow_test.html',
            frame_skip=2,
            plot_title='Integration Test',
            atom_size_scale=1.5,
            plot_lines=True,
            show_plot=False
        )
        
        # Verify output
        assert fig is not None
        assert os.path.exists(os.path.join(temp_output_dir, 'workflow_test.html'))
        
        # Verify figure has data
        assert len(fig.data) > 0
    
    def test_multiple_plots_same_directory(self, small_trajectory, temp_output_dir):
        """Test creating multiple plots in same directory."""
        for i in range(3):
            fig = plot_atomic_trajectory(
                small_trajectory,
                indices_path=[i % 3],
                output_dir=temp_output_dir,
                output_filename=f'plot_{i}.html',
                frame_skip=1,
                show_plot=False
            )
            
            assert fig is not None
        
        # Check all files created
        assert os.path.exists(os.path.join(temp_output_dir, 'plot_0.html'))
        assert os.path.exists(os.path.join(temp_output_dir, 'plot_1.html'))
        assert os.path.exists(os.path.join(temp_output_dir, 'plot_2.html'))


class TestPlotAtomicTrajectoryRealData:
    """Test specifically with real trajectory data from data folder."""
    
    def test_oxygen_atoms_trajectory(self, real_trajectory, temp_output_dir):
        """Test plotting oxygen atom trajectories."""
        # Read first frame to identify oxygen atoms
        atoms = read(real_trajectory, index='0')
        symbols = atoms.get_chemical_symbols()
        
        # Find oxygen atoms
        o_indices = [i for i, s in enumerate(symbols) if s == 'O']
        
        if len(o_indices) > 0:
            fig = plot_atomic_trajectory(
                real_trajectory,
                indices_path=o_indices[:min(3, len(o_indices))],
                output_dir=temp_output_dir,
                plot_title='Oxygen Trajectories',
                frame_skip=5,
                show_plot=False
            )
            
            assert fig is not None
    
    def test_hydrogen_atoms_trajectory(self, real_trajectory, temp_output_dir):
        """Test plotting hydrogen atom trajectories."""
        atoms = read(real_trajectory, index='0')
        symbols = atoms.get_chemical_symbols()
        
        # Find hydrogen atoms
        h_indices = [i for i, s in enumerate(symbols) if s == 'H']
        
        if len(h_indices) > 0:
            fig = plot_atomic_trajectory(
                real_trajectory,
                indices_path=h_indices[:min(2, len(h_indices))],
                output_dir=temp_output_dir,
                plot_title='Hydrogen Trajectories',
                frame_skip=10,
                plot_lines=True,
                show_plot=False
            )
            
            assert fig is not None
    
    def test_real_trajectory_with_various_parameters(self, real_trajectory, temp_output_dir):
        """Test real trajectory with various parameter combinations."""
        atoms = read(real_trajectory, index='0')
        
        # Test with different combinations
        configs = [
            {'frame_skip': 1, 'atom_size_scale': 0.8, 'plot_lines': False},
            {'frame_skip': 5, 'atom_size_scale': 1.2, 'plot_lines': True},
            {'frame_skip': 10, 'atom_size_scale': 1.5, 'plot_lines': False},
        ]
        
        for i, config in enumerate(configs):
            fig = plot_atomic_trajectory(
                real_trajectory,
                indices_path=[0],
                output_dir=temp_output_dir,
                output_filename=f'config_{i}.html',
                show_plot=False,
                **config
            )
            
            assert fig is not None
