"""Comprehensive tests for hydrogen bond analysis module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from CRISP.data_analysis.h_bond import (
    count_hydrogen_bonds,
    hydrogen_bonds,
)

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestHydrogenBondsCore:
    """Test core hydrogen bond detection."""
    
    def test_count_hydrogen_bonds_basic(self):
        """Test basic hydrogen bond counting."""
        # Create water dimer with hydrogen bond
        atoms = Atoms('H2OH2O', positions=[
            [0.0, 0.0, 0.0],    # O1
            [0.96, 0.0, 0.0],   # H1
            [0.24, 0.93, 0.0],  # H2
            [2.8, 0.0, 0.0],    # O2 (H-bonded to O1)
            [3.76, 0.0, 0.0],   # H3
            [3.04, 0.93, 0.0]   # H4
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        
        assert isinstance(bond_dict, dict)
        assert isinstance(count, (int, np.integer))
        assert count >= 0
    
    def test_count_hydrogen_bonds_no_bonds(self):
        """Test structure with no hydrogen bonds."""
        # Two isolated atoms far apart
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0]
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['H'],
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )

        assert count == 0
    
    def test_count_hydrogen_bonds_angle_cutoff_effect(self):
        """Test hydrogen bond counting with different angle cutoffs."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        # Test with strict angle cutoff
        bond_dict_strict, count_strict = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=150,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        
        # Test with loose angle cutoff
        bond_dict_loose, count_loose = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=90,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        
        assert isinstance(count_strict, (int, np.integer))
        assert isinstance(count_loose, (int, np.integer))
        # Loose cutoff should find at least as many as strict
        assert count_loose >= count_strict
    
    def test_count_hydrogen_bonds_distance_cutoff_effect(self):
        """Test hydrogen bond counting with different distance cutoffs."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        # Test with smaller cutoff
        result_small = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=1.5,
            bond_cutoff=1.6
        )
        
        # Test with larger cutoff
        result_large = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=3.5,
            bond_cutoff=1.6
        )
        
        # Larger cutoff should find at least as many bonds
        assert result_large >= result_small
    
    def test_count_hydrogen_bonds_multiple_acceptor_types(self):
        """Test hydrogen bond counting with multiple acceptor types."""
        atoms = Atoms('H2OH2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
            [2.8, 0.0, 0.0],
            [3.76, 0.0, 0.0],
            [3.04, 0.93, 0.0]
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O', 'N'],  # Multiple acceptor types
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )

        assert count >= 0


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestHydrogenBondsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_acceptor_atoms(self):
        """Test with empty acceptor atoms list."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        with pytest.raises((ValueError, Exception)):
            count_hydrogen_bonds(
                atoms=atoms,
                acceptor_atoms=[],
                angle_cutoff=120,
                h_bond_cutoff=2.5,
                bond_cutoff=1.6
            )
    
    def test_invalid_angle_cutoff_too_high(self):
        """Test with angle cutoff > 180 (should still work, just illogical)."""  
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        # Function doesn't validate angle cutoff, so it just works
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=181,  # Illogical but accepted
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        assert count >= 0
    
    def test_invalid_negative_distance_cutoff(self):
        """Test with negative distance cutoff (should return 0 bonds)."""  
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        # Negative distance means no bonds will be found
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=-1,  # No matches
            bond_cutoff=1.6
        )
        assert count == 0
    
    def test_single_atom(self):
        """Test with single atom."""
        atoms = Atoms('O', positions=[[0.0, 0.0, 0.0]])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )

        # Should return 0 (no bonds possible with single atom)
        assert count == 0
    
    def test_nonexistent_acceptor_atom_type(self):
        """Test with acceptor atom type not in structure."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],  # No oxygen in structure
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )

        # Should return 0 (no acceptor atoms found)
        assert count == 0


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestHydrogenBondsIntegration:
    """Test integration with hydrogen_bonds wrapper function."""
    
    def test_hydrogen_bonds_module_import(self):
        """Test hydrogen_bonds module can be imported."""
        from CRISP.data_analysis import h_bond
        assert hasattr(h_bond, 'count_hydrogen_bonds')
        assert hasattr(h_bond, 'hydrogen_bonds')
    
    def test_count_consistency(self):
        """Test that hydrogen bond count is consistent."""
        atoms = Atoms('H2OH2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
            [2.8, 0.0, 0.0],
            [3.76, 0.0, 0.0],
            [3.04, 0.93, 0.0]
        ])
        
        # Run count_hydrogen_bonds twice with same parameters
        result1 = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        
        result2 = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )
        
        # Results should be identical
        assert result1 == result2


class TestHydrogenBondsParameters:
    """Test parameter variations."""
    
    @pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
    @pytest.mark.parametrize("angle_cutoff", [90, 120, 150])
    def test_angle_cutoff_variations(self, angle_cutoff):
        """Test with different angle cutoff values."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=angle_cutoff,
            h_bond_cutoff=2.5,
            bond_cutoff=1.6
        )

        assert isinstance(count, (int, np.integer))
        assert count >= 0
    
    @pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
    @pytest.mark.parametrize("h_bond_cutoff", [1.5, 2.0, 2.5, 3.0])
    def test_h_bond_distance_cutoff_variations(self, h_bond_cutoff):
        """Test with different H-bond distance cutoffs."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        bond_dict, count = count_hydrogen_bonds(
            atoms=atoms,
            acceptor_atoms=['O'],
            angle_cutoff=120,
            h_bond_cutoff=h_bond_cutoff,
            bond_cutoff=1.6
        )

        assert count >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
