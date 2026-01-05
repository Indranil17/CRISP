"""Extended tests for volumetric_atomic_density module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write

from CRISP.data_analysis.volumetric_atomic_density import (
    create_density_map,
    VDW_RADII,
    ELEMENT_COLORS,
)


class TestVolumetricDensityBasic:
    """Basic volumetric density functionality tests."""
    
    def test_vdw_radii_dict_exists(self):
        """Test that VDW_RADII dictionary is properly defined."""
        assert isinstance(VDW_RADII, dict)
        assert len(VDW_RADII) > 100
        assert 'H' in VDW_RADII
        assert 'C' in VDW_RADII
        assert 'O' in VDW_RADII
        assert VDW_RADII['H'] == 1.20
        assert VDW_RADII['C'] == 1.70
    
    def test_element_colors_dict_exists(self):
        """Test that ELEMENT_COLORS dictionary is properly defined."""
        assert isinstance(ELEMENT_COLORS, dict)
        assert len(ELEMENT_COLORS) > 20
        assert 'H' in ELEMENT_COLORS
        assert 'C' in ELEMENT_COLORS
        assert 'O' in ELEMENT_COLORS
    
    def test_create_density_map_basic(self):
        """Test basic density map creation."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a simple trajectory
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            # Write trajectory with multiple frames
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            # Save indices
            np.save(indices_file, np.array([0, 1]))
            
            # Create density map
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                output_dir=temp_dir
            )
            
            assert fig is not None
            assert hasattr(fig, 'add_trace')
        finally:
            shutil.rmtree(temp_dir)
    
    def test_create_density_map_with_custom_parameters(self):
        """Test density map creation with custom parameters."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([1, 2]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                threshold=0.1,
                absolute_threshold=False,
                opacity=0.5,
                atom_size_scale=2.0,
                nbins=30,
                colorscale='Viridis',
                plot_title='Custom Density Map',
                output_dir=temp_dir,
                output_file='custom_density.html'
            )
            
            assert fig is not None
            output_file = os.path.join(temp_dir, 'custom_density.html')
            assert os.path.exists(output_file)
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityAbsoluteThreshold:
    """Test absolute threshold functionality."""
    
    @pytest.mark.parametrize("abs_threshold", [0.0, 1.0, 5.0, 10.0])
    def test_absolute_threshold_variations(self, abs_threshold):
        """Test density map with different absolute thresholds."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                threshold=abs_threshold,
                absolute_threshold=True,
                nbins=20,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityOpacity:
    """Test opacity parameter."""
    
    @pytest.mark.parametrize("opacity", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_opacity_variations(self, opacity):
        """Test density map with different opacity values."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                opacity=opacity,
                nbins=20,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityBins:
    """Test number of bins parameter."""
    
    @pytest.mark.parametrize("nbins", [10, 20, 40, 50])
    def test_nbins_variations(self, nbins):
        """Test density map with different bin counts."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=nbins,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensitySaveDensity:
    """Test saving density data functionality."""
    
    def test_save_density_data(self):
        """Test saving density data to file."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            density_file = os.path.join(temp_dir, 'density.npz')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                save_density=True,
                density_output_file='density.npz',
                output_dir=temp_dir
            )
            
            assert fig is not None
            assert os.path.exists(density_file)
            
            # Load and verify density data
            data = np.load(density_file)
            assert 'density' in data
            assert 'edges' in data
            assert 'cell' in data
            assert 'nbins' in data
            assert 'selected_indices' in data
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityProjections:
    """Test projection functionality."""
    
    def test_projections_enabled(self):
        """Test density map with 2D projections enabled."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                show_projections=True,
                projection_opacity=0.7,
                projection_offset=2.0,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityColorscales:
    """Test different colorscale options."""
    
    @pytest.mark.parametrize("colorscale", ['Plasma', 'Viridis', 'Blues', 'Reds'])
    def test_colorscale_variations(self, colorscale):
        """Test density map with different colorscales."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                colorscale=colorscale,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_atom_system(self):
        """Test density map with single atom."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_large_cell(self):
        """Test density map with large unit cell."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            indices_file = os.path.join(temp_dir, 'indices.npy')
            
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([50, 50, 50])
            atoms.set_pbc([True, True, True])
            
            write(traj_file, atoms)
            write(traj_file, atoms)
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                output_dir=temp_dir
            )
            
            assert fig is not None
        finally:
            shutil.rmtree(temp_dir)


class TestVolumetricDensityIntegration:
    """Integration tests for volumetric density."""
    
    def test_complete_workflow(self):
        """Test complete density analysis workflow."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'workflow.traj')
            indices_file = os.path.join(temp_dir, 'workflow_indices.npy')
            
            # Create multi-frame trajectory
            for i in range(5):
                atoms = Atoms('H2O', positions=[
                    [0.0 + i*0.1, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [0.24, 0.93, 0.0]
                ])
                atoms.set_cell([10, 10, 10])
                atoms.set_pbc([True, True, True])
                write(traj_file, atoms, append=(i > 0))
            
            np.save(indices_file, np.array([0, 1]))
            
            fig = create_density_map(
                traj_path=traj_file,
                indices_path=indices_file,
                frame_skip=1,
                nbins=20,
                save_density=True,
                output_dir=temp_dir,
                show_projections=True
            )
            
            assert fig is not None
            
            # Verify output file was created
            html_file = os.path.join(temp_dir, 'workflow_density_map.html')
            assert os.path.exists(html_file)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
