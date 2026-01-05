"""Extended comprehensive tests for Contact Coordination module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write
from CRISP.data_analysis.contact_coordination import (
    indices,
    coordination_frame,
    coordination,
    contacts,
    contacts_frame,
)


class TestCoordinationIndices:
    """Test the indices helper function."""
    
    def test_indices_all_string(self):
        """Test 'all' returns all atoms."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, "all")
        
        assert len(result) == 3
        assert np.array_equal(result, np.array([0, 1, 2]))
    
    def test_indices_none_returns_all(self):
        """Test None returns all atoms."""
        atoms = Atoms('H4O2', positions=np.random.rand(6, 3))
        result = indices(atoms, None)
        
        assert len(result) == 6
        assert np.array_equal(result, np.arange(6))
    
    def test_indices_single_integer(self):
        """Test single integer."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, 1)
        
        assert len(result) == 1
        assert result[0] == 1
    
    def test_indices_list_integers(self):
        """Test list of integers."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, [0, 2])
        
        assert len(result) == 2
        assert np.array_equal(result, np.array([0, 2]))
    
    def test_indices_single_symbol(self):
        """Test single chemical symbol."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, 'H')
        
        assert len(result) == 2
    
    def test_indices_list_symbols(self):
        """Test list of symbols."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, ['H', 'O'])
        
        assert len(result) == 3
    
    def test_indices_oxygen_only(self):
        """Test getting oxygen only."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = indices(atoms, 'O')
        
        assert len(result) == 1
    
    def test_indices_invalid_raises_error(self):
        """Test invalid index raises error."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        with pytest.raises(ValueError):
            indices(atoms, {'dict': 'value'})


class TestCoordinationFrame:
    """Test single frame coordination calculation."""
    
    def test_coordination_frame_basic(self):
        """Test basic coordination frame calculation."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10, 10, 10]
        atoms.pbc = False
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(v, (int, np.integer)) for v in result.values())
    
    def test_coordination_frame_with_indices(self):
        """Test coordination with index specifiers."""
        atoms = Atoms('H4O2', positions=np.random.rand(6, 3) * 2)
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(
            atoms, 
            central_atoms=[0, 1],  # First two atoms as central
            target_atoms=[2, 3, 4, 5]  # Rest as target
        )
        
        assert isinstance(result, dict)
        assert len(result) == 2
    
    def test_coordination_frame_with_custom_cutoffs(self):
        """Test with custom cutoff distances."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10, 10, 10]
        
        custom_cutoffs = {('O', 'H'): 1.5}
        
        result = coordination_frame(
            atoms,
            central_atoms='O',
            target_atoms='H',
            custom_cutoffs=custom_cutoffs
        )
        
        assert isinstance(result, dict)
    
    def test_coordination_frame_with_mic(self):
        """Test with minimum image convention."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10, 10, 10]
        atoms.pbc = True
        
        result_mic = coordination_frame(
            atoms, 
            central_atoms='O',
            target_atoms='H',
            mic=True
        )
        
        result_no_mic = coordination_frame(
            atoms,
            central_atoms='O',
            target_atoms='H',
            mic=False
        )
        
        assert isinstance(result_mic, dict)
        assert isinstance(result_no_mic, dict)
    
    def test_coordination_frame_single_central_atom(self):
        """Test with single central atom."""
        atoms = Atoms('H3O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(atoms, central_atoms=3, target_atoms='H')
        
        assert len(result) == 1
        assert 3 in result
    
    def test_coordination_frame_multiple_central_atoms(self):
        """Test with multiple central atoms."""
        atoms = Atoms('H4O2', positions=np.random.rand(6, 3) * 2)
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)
        assert all(isinstance(k, (int, np.integer)) for k in result.keys())
    
    def test_coordination_frame_zero_coordination(self):
        """Test with isolated atoms (zero coordination)."""
        atoms = Atoms('HO', positions=[[0, 0, 0], [100, 100, 100]])
        atoms.cell = [200, 200, 200]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        # Result should be a dict, values should be 0 (isolated atom)
        assert isinstance(result, dict)
        if len(result) > 0:
            assert all(v == 0 for v in result.values())
    
    def test_coordination_frame_identical_central_and_target(self):
        """Test when central and target atoms are the same."""
        atoms = Atoms('O2', positions=[[0, 0, 0], [1.5, 0, 0]])
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='O')
        
        assert isinstance(result, dict)


class TestCoordinationTrajectory:
    """Test trajectory-based coordination analysis."""
    
    @pytest.fixture
    def coordination_trajectory(self):
        """Create trajectory for coordination tests."""
        temp_dir = tempfile.mkdtemp()
        traj_path = os.path.join(temp_dir, 'coord_traj.traj')
        
        frames = []
        for i in range(5):
            atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            atoms.cell = [10, 10, 10]
            atoms.pbc = True
            atoms.positions += np.random.rand(3, 3) * 0.2
            frames.append(atoms)
        
        write(traj_path, frames)
        yield traj_path
        shutil.rmtree(temp_dir)
    
    def test_coordination_trajectory_basic(self, coordination_trajectory):
        """Test basic trajectory analysis."""
        result = coordination(
            coordination_trajectory,
            central_atoms='O',
            target_atoms='H',
            custom_cutoffs=None,
            frame_skip=1
        )
        
        assert result is not None
    
    def test_coordination_frame_skip(self, coordination_trajectory):
        """Test with frame skipping."""
        result = coordination(
            coordination_trajectory,
            central_atoms='O',
            target_atoms='H',
            custom_cutoffs=None,
            frame_skip=2
        )
        
        assert result is not None
    
    def test_coordination_custom_cutoff(self, coordination_trajectory):
        """Test with custom cutoff."""
        result = coordination(
            coordination_trajectory,
            central_atoms='O',
            target_atoms='H',
            custom_cutoffs={('O', 'H'): 1.5},
            frame_skip=1
        )
        
        assert result is not None
    
    def test_coordination_nonexistent_file(self):
        """Test with non-existent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            coordination('/nonexistent/path.traj', 'O', 'H', None)
    
    def test_coordination_with_output_dir(self, coordination_trajectory):
        """Test with output directory."""
        output_dir = tempfile.mkdtemp()
        
        result = coordination(
            coordination_trajectory,
            central_atoms='O',
            target_atoms='H',
            custom_cutoffs=None,
            output_dir=output_dir
        )
        
        assert result is not None
        shutil.rmtree(output_dir)


class TestCoordinationParameterVariations:
    """Test coordination analysis with various parameters."""
    
    @pytest.fixture
    def multi_atom_trajectory(self):
        """Create trajectory with multiple atom types."""
        temp_dir = tempfile.mkdtemp()
        traj_path = os.path.join(temp_dir, 'multi_coord.traj')
        
        frames = []
        for i in range(4):
            atoms = Atoms('H4O2N2', positions=np.random.rand(8, 3) * 3)
            atoms.cell = [15, 15, 15]
            atoms.pbc = True
            frames.append(atoms)
        
        write(traj_path, frames)
        yield traj_path
        shutil.rmtree(temp_dir)
    
    def test_coordination_different_central_atoms(self, multi_atom_trajectory):
        """Test with different central atom types."""
        for central in ['O', 'N', 'H']:
            result = coordination(
                multi_atom_trajectory,
                central_atoms=central,
                target_atoms='H',
                custom_cutoffs=None
            )
            assert result is not None
    
    def test_coordination_different_target_atoms(self, multi_atom_trajectory):
        """Test with different target atom types."""
        for target in ['O', 'N', 'H']:
            result = coordination(
                multi_atom_trajectory,
                central_atoms='O',
                target_atoms=target,
                custom_cutoffs=None
            )
            assert result is not None
    
    def test_coordination_multiple_target_atoms(self, multi_atom_trajectory):
        """Test with multiple target atom types."""
        result = coordination(
            multi_atom_trajectory,
            central_atoms='O',
            target_atoms=['H', 'N'],
            custom_cutoffs=None
        )
        
        assert result is not None


class TestCoordinationIntegration:
    """Integration tests for coordination analysis."""
    
    def test_coordination_workflow_water_network(self):
        """Test coordination in water-like network."""
        # Create water-like network
        positions = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],  # Water 1
            [3, 0, 0], [4, 0, 0], [3, 1, 0],  # Water 2
            [1.5, 1.5, 0], [2.5, 1.5, 0], [1.5, 2.5, 0],  # Water 3
        ]
        atoms = Atoms('H2OH2OH2O', positions=positions)
        atoms.cell = [10, 10, 10]
        atoms.pbc = False
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, np.integer)) for v in result.values())
    
    def test_coordination_frame_consistency(self):
        """Test that same frame gives same coordination."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10, 10, 10]
        
        result1 = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        result2 = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert result1 == result2
    
    def test_coordination_with_custom_indices(self):
        """Test coordination using custom atom indices."""
        atoms = Atoms('H4O2', positions=np.random.rand(6, 3) * 2)
        atoms.cell = [10, 10, 10]
        
        # Get H atoms via indices function
        h_indices = indices(atoms, 'H')
        o_indices = indices(atoms, 'O')
        
        # Convert numpy int64 to regular Python int to avoid type issues
        result = coordination_frame(atoms, central_atoms=[int(i) for i in o_indices], target_atoms=[int(i) for i in h_indices])
        
        assert isinstance(result, dict)


class TestCoordinationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_coordination_single_atom(self):
        """Test with single atom."""
        atoms = Atoms('O', positions=[[0, 0, 0]])
        
        result = coordination_frame(atoms, central_atoms=0, target_atoms='H')
        
        assert 0 in result
        assert result[0] == 0
    
    def test_coordination_isolated_atoms(self):
        """Test with isolated atoms."""
        atoms = Atoms('HO', positions=[[0, 0, 0], [100, 100, 100]])
        atoms.cell = [200, 200, 200]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        # Oxygen should have 0 coordination
        assert all(v == 0 for v in result.values()) or len(result) == 0
    
    def test_coordination_no_target_atoms(self):
        """Test when target atoms don't exist."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='X')
        
        # Should handle gracefully
        assert isinstance(result, dict)
    
    def test_coordination_very_large_system(self):
        """Test with large system."""
        symbols = ['H'] * 50 + ['O'] * 50
        positions = np.random.rand(100, 3) * 10
        atoms = Atoms(symbols, positions=positions)
        atoms.cell = [20, 20, 20]
        atoms.pbc = True
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)
        assert len(result) == 50  # 50 oxygen atoms
    
    def test_coordination_periodic_boundaries(self):
        """Test coordination across periodic boundaries."""
        atoms = Atoms('HO', positions=[[0, 0, 0], [9.9, 0, 0]])
        atoms.cell = [10, 10, 10]
        atoms.pbc = True
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H', mic=True)
        
        # With MIC, the H and O should be close
        assert isinstance(result, dict)
    
    def test_coordination_very_close_atoms(self):
        """Test with very close atoms."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0]])
        atoms.cell = [10, 10, 10]
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)


class TestCoordinationNumericalStability:
    """Test numerical stability of coordination calculations."""
    
    def test_coordination_with_identical_positions(self):
        """Test with atoms at identical positions."""
        atoms = Atoms('HO', positions=[[0, 0, 0], [0, 0, 0]])
        atoms.cell = [10, 10, 10]
        
        # Should handle gracefully
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        assert isinstance(result, dict)
    
    def test_coordination_with_extreme_cell_size(self):
        """Test with very large cell."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.cell = [10000, 10000, 10000]
        atoms.pbc = True
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H')
        
        assert isinstance(result, dict)
    
    def test_coordination_with_small_cell(self):
        """Test with very small cell (but valid)."""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
        atoms.cell = [1, 1, 1]
        atoms.pbc = True
        
        result = coordination_frame(atoms, central_atoms='O', target_atoms='H', mic=True)
        
        assert isinstance(result, dict)
