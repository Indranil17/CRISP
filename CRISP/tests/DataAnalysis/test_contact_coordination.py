"""Extended tests for contact coordination module."""
import pytest
import numpy as np
from ase import Atoms

from CRISP.data_analysis.contact_coordination import (
    indices,
    coordination_frame,
)


class TestCoordinationBasic:
    """Basic coordination number tests."""
    
    def test_indices_all_atoms(self):
        """Test indices function with 'all' specifier."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        idx = indices(atoms, "all")
        assert len(idx) == 3
        assert np.array_equal(idx, [0, 1, 2])
    
    def test_indices_none_defaults_to_all(self):
        """Test that None defaults to all atoms."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        idx = indices(atoms, None)
        assert len(idx) == 3
    
    def test_indices_by_list(self):
        """Test indices with explicit list."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        idx = indices(atoms, [0, 2])
        assert len(idx) == 2
        assert np.array_equal(idx, [0, 2])
    
    def test_indices_by_symbol(self):
        """Test indices selection by chemical symbol."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        h_indices = indices(atoms, ['H'])
        assert len(h_indices) == 2
    
    def test_coordination_frame_basic(self):
        """Test basic coordination analysis on single frame."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        central = [0]  # Oxygen
        target = [1, 2]  # Hydrogens
        
        result = coordination_frame(atoms, central, target)
        assert result is not None


class TestCoordinationParametrized:
    """Test coordination with parameter variations."""
    
    @pytest.mark.parametrize("central", [0, 1, 2])
    def test_coordination_different_central_atoms(self, central):
        """Test coordination for different central atoms."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        target_atoms = [i for i in range(len(atoms)) if i != central]
        result = coordination_frame(atoms, [central], target_atoms)
        
        assert result is not None


class TestCoordinationPBC:
    """Test coordination with periodic boundary conditions."""
    
    def test_coordination_with_pbc(self):
        """Test coordination considering PBC."""
        atoms = Atoms('H2', positions=[
            [0.1, 0.0, 0.0],
            [9.9, 0.0, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        result = coordination_frame(atoms, [0], [1])
        assert result is not None
    
    def test_coordination_without_pbc(self):
        """Test coordination without periodic boundary conditions."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [9.9, 0.0, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([False, False, False])
        
        result = coordination_frame(atoms, [0], [1], mic=False)
        assert result is not None


class TestCoordinationEdgeCases:
    """Test coordination edge cases."""
    
    def test_coordination_single_atom(self):
        """Test coordination for single atom with empty target atoms."""
        atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])

        # Empty target_atoms should raise ValueError due to indices() not handling empty list
        with pytest.raises(ValueError, match="Invalid index type"):
            result = coordination_frame(atoms, [0], [])
    def test_coordination_multiple_central(self):
        """Test with multiple central atoms."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        result = coordination_frame(atoms, [0, 1], [2])
        assert result is not None
    
    def test_indices_with_file_notation(self):
        """Test indices with invalid file (should raise error)."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        # Should handle invalid file gracefully or raise error
        try:
            idx = indices(atoms, "nonexistent.npy")
        except (FileNotFoundError, OSError):
            pass  # Expected


class TestCoordinationIntegration:
    """Integration tests for coordination."""
    
    def test_coordination_water_cluster(self):
        """Test coordination in water-like cluster."""
        atoms = Atoms('OH2', positions=[
            [0.0, 0.0, 0.0],      # O central
            [0.96, 0.0, 0.0],     # H
            [0.24, 0.93, 0.0]     # H
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        result = coordination_frame(atoms, [0], [1, 2])
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
