"""Tests for interatomic distances calculation module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from CRISP.simulation_utility.interatomic_distances import (
    indices,
    distance_calculation,
    save_distance_matrices,
    calculate_interatomic_distances,
)

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_atoms():
    """Create sample atoms object."""
    if not ASE_AVAILABLE:
        return None
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ])
    return Atoms('H4', positions=positions, cell=[10, 10, 10], pbc=True)


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestIndicesFunction:
    """Test the indices extraction function."""
    
    def test_indices_all_atoms(self, sample_atoms):
        """Test extracting all atom indices."""
        result = indices(sample_atoms, "all")
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 3]))
    
    def test_indices_none_defaults_to_all(self, sample_atoms):
        """Test that None defaults to all atoms."""
        result = indices(sample_atoms, None)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
    
    def test_indices_with_list_integers(self, sample_atoms):
        """Test extracting specific atom indices by integers."""
        result = indices(sample_atoms, [0, 2])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([0, 2]))
    
    def test_indices_with_chemical_symbols(self, sample_atoms):
        """Test extracting atoms by chemical symbol."""
        result = indices(sample_atoms, ["H"])
        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # All atoms are H


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestDistanceCalculation:
    """Test distance calculation function."""
    
    def test_distance_calculation_invalid_file(self):
        """Test with nonexistent trajectory file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            distance_calculation(
                traj_path="nonexistent_trajectory.traj",
                frame_skip=1,
                index_type="all"
            )
    
    def test_distance_calculation_module_import(self):
        """Test that module can be imported."""
        from CRISP.simulation_utility import interatomic_distances
        assert hasattr(interatomic_distances, 'distance_calculation')
        assert hasattr(interatomic_distances, 'calculate_interatomic_distances')
        assert hasattr(interatomic_distances, 'indices')


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestSaveDistanceMatrices:
    """Test saving distance matrices."""
    
    def test_save_distance_matrices_basic(self, temp_output_dir, sample_atoms):
        """Test saving distance matrices."""
        # Create sample distance matrices
        dm1 = sample_atoms.get_all_distances(mic=True)
        dm2 = sample_atoms.get_all_distances(mic=True)
        
        full_dms = [dm1, dm2]
        sub_dms = [dm1[:2, :2], dm2[:2, :2]]
        
        save_distance_matrices(
            full_dms=full_dms,
            sub_dms=sub_dms,
            index_type="all",
            output_dir=temp_output_dir
        )
        
        # Check that output file was created
        output_file = os.path.join(temp_output_dir, "distance_matrices.pkl")
        assert os.path.exists(output_file)


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestCalculateInteratomicDistances:
    """Test main wrapper function."""
    
    def test_calculate_interatomic_distances_basic(self, sample_atoms, temp_output_dir):
        """Test basic interatomic distance calculation."""
        # Create a temporary trajectory file
        traj_file = os.path.join(temp_output_dir, "test.traj")
        sample_atoms.write(traj_file)
        
        result = calculate_interatomic_distances(
            traj_path=traj_file,
            frame_skip=1,
            index_type="all",
            output_dir=temp_output_dir,
            save_results=True
        )
        
        assert isinstance(result, dict)
        assert "full_dms" in result
        assert isinstance(result["full_dms"], list)
        assert len(result["full_dms"]) > 0
    
    def test_calculate_with_atom_indices(self, sample_atoms, temp_output_dir):
        """Test calculation with specific atom indices."""
        traj_file = os.path.join(temp_output_dir, "test.traj")
        sample_atoms.write(traj_file)
        
        result = calculate_interatomic_distances(
            traj_path=traj_file,
            frame_skip=1,
            index_type=[0, 1],
            output_dir=temp_output_dir,
            save_results=True
        )
        
        assert "full_dms" in result
        assert "sub_dms" in result


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestInteratomicDistancesEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_atom(self, temp_output_dir):
        """Test with single atom."""
        if not ASE_AVAILABLE:
            pytest.skip("ASE not available")
        
        atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        traj_file = os.path.join(temp_output_dir, "single_atom.traj")
        atoms.write(traj_file)
        
        result = calculate_interatomic_distances(
            traj_path=traj_file,
            frame_skip=1,
            output_dir=temp_output_dir,
            save_results=False
        )
        
        # Single atom should have 1x1 distance matrix
        assert result["full_dms"][0].shape == (1, 1)
    
    def test_invalid_trajectory_file(self, temp_output_dir):
        """Test with invalid trajectory file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            calculate_interatomic_distances(
                traj_path="nonexistent.traj",
                frame_skip=1,
                output_dir=temp_output_dir
            )
    
    def test_distance_matrix_symmetry(self, sample_atoms, temp_output_dir):
        """Test that distance matrices are symmetric."""
        traj_file = os.path.join(temp_output_dir, "test.traj")
        sample_atoms.write(traj_file)
        
        result = calculate_interatomic_distances(
            traj_path=traj_file,
            frame_skip=1,
            output_dir=temp_output_dir,
            save_results=False
        )
        
        dm = result["full_dms"][0]
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dm, dm.T, decimal=5)
    
    def test_distance_matrix_diagonal_zeros(self, sample_atoms, temp_output_dir):
        """Test that distance matrix diagonal is zero."""
        traj_file = os.path.join(temp_output_dir, "test.traj")
        sample_atoms.write(traj_file)
        
        result = calculate_interatomic_distances(
            traj_path=traj_file,
            frame_skip=1,
            output_dir=temp_output_dir,
            save_results=False
        )
        
        dm = result["full_dms"][0]
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dm), np.zeros(len(dm)), decimal=5)
