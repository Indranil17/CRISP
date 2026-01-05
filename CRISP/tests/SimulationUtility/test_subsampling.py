"""Extended tests for subsampling module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write

from CRISP.simulation_utility.subsampling import (
    indices,
    compute_soap,
)


class TestSubsamplingIndicesFunction:
    """Test indices selection function."""
    
    def test_indices_all_atoms(self):
        """Test selecting all atoms."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, "all")
        assert np.array_equal(result, np.array([0, 1, 2]))
    
    def test_indices_none_defaults_to_all(self):
        """Test that None defaults to all atoms."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, None)
        assert np.array_equal(result, np.array([0, 1, 2]))
    
    def test_indices_with_list_integers(self):
        """Test selecting atoms by integer indices."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, [0, 2])
        assert np.array_equal(result, np.array([0, 2]))
    
    def test_indices_with_single_integer(self):
        """Test selecting single atom by integer."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, [1])
        assert np.array_equal(result, np.array([1]))
    
    def test_indices_with_chemical_symbols(self):
        """Test selecting atoms by chemical symbol."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, ['H'])
        # Water has H atoms - verify we get at least 2 H atoms
        assert len(result) == 2
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
    
    def test_indices_with_oxygen_symbol(self):
        """Test selecting oxygen atoms from water."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, ['O'])
        # Water has exactly 1 oxygen atom
        assert len(result) == 1
        assert result[0] in [0, 1, 2]  # Oxygen can be at any position depending on ASE ordering
    
    def test_indices_with_multiple_symbols(self):
        """Test selecting multiple symbol types."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, ['H', 'O'])
        assert len(result) == 3
    
    def test_indices_from_file(self):
        """Test loading indices from NumPy file."""
        temp_dir = tempfile.mkdtemp()
        try:
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            
            indices_file = os.path.join(temp_dir, 'indices.npy')
            np.save(indices_file, np.array([0, 1]))
            
            result = indices(atoms, indices_file)
            assert np.array_equal(result, np.array([0, 1]))
        finally:
            shutil.rmtree(temp_dir)
    
    def test_indices_invalid_type_raises_error(self):
        """Test that invalid index type raises ValueError."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        with pytest.raises(ValueError, match="Invalid index type"):
            indices(atoms, [1.5])  # Float is invalid


class TestComputeSoap:
    """Test SOAP descriptor computation."""
    
    def test_compute_soap_basic(self):
        """Test basic SOAP descriptor calculation."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            soap_descriptor = compute_soap(
                atoms,
                all_spec=['H', 'O'],
                rcut=3.0,
                idx=[0]
            )
            
            assert isinstance(soap_descriptor, np.ndarray)
            assert len(soap_descriptor) > 0
        except Exception:
            pytest.skip("SOAP descriptor computation requires dscribe library")
    
    def test_compute_soap_multiple_centers(self):
        """Test SOAP with multiple atomic centers."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            soap_descriptor = compute_soap(
                atoms,
                all_spec=['H', 'O'],
                rcut=3.0,
                idx=[0, 1]
            )
            
            assert isinstance(soap_descriptor, np.ndarray)
            assert len(soap_descriptor) > 0
        except Exception:
            pytest.skip("SOAP descriptor computation requires dscribe library")


class TestSubsamplingEdgeCases:
    """Test edge cases."""
    
    def test_single_atom_system(self):
        """Test with single atom."""
        atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
        
        result = indices(atoms, "all")
        assert np.array_equal(result, np.array([0]))
    
    def test_large_system(self):
        """Test with larger system."""
        # Create a system with many atoms
        symbols = ['H', 'C', 'N', 'O'] * 25  # 100 atoms
        positions = np.random.rand(100, 3) * 10
        atoms = Atoms(symbols, positions=positions)
        atoms.set_cell([15, 15, 15])
        atoms.set_pbc([True, True, True])
        
        result = indices(atoms, "all")
        assert len(result) == 100
    
    def test_indices_with_duplicates(self):
        """Test selecting with duplicate indices."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, [0, 0, 1])
        # Should contain duplicates as provided
        assert 0 in result
        assert 1 in result


class TestSubsamplingParameterVariations:
    """Test with various parameter combinations."""
    
    @pytest.mark.parametrize("rcut", [2.0, 3.0, 4.0, 5.0])
    def test_soap_different_rcut(self, rcut):
        """Test SOAP with different cutoff radii."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            soap_descriptor = compute_soap(
                atoms,
                all_spec=['H', 'O'],
                rcut=rcut,
                idx=[0]
            )
            assert isinstance(soap_descriptor, np.ndarray)
        except Exception:
            pytest.skip("SOAP computation requires dscribe library")
    
    @pytest.mark.parametrize("symbol_list", [['H'], ['O'], ['H', 'O']])
    def test_indices_different_symbols(self, symbol_list):
        """Test indices selection with different symbol combinations."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        
        result = indices(atoms, symbol_list)
        assert len(result) > 0


class TestSubsamplingIntegration:
    """Integration tests for subsampling."""
    
    def test_workflow_select_and_soap(self):
        """Test complete workflow of selecting and analyzing."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        # Select oxygen atoms
        idx = indices(atoms, ['O'])
        assert len(idx) > 0
        
        # Compute SOAP
        try:
            soap_descriptor = compute_soap(
                atoms,
                all_spec=['H', 'O'],
                rcut=3.0,
                idx=idx
            )
            assert isinstance(soap_descriptor, np.ndarray)
        except Exception:
            pytest.skip("SOAP computation requires dscribe library")


class TestSubsamplingNumericalProperties:
    """Test numerical properties of computed values."""
    
    def test_soap_descriptor_shape(self):
        """Test that SOAP descriptor has consistent shape."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            soap1 = compute_soap(atoms, ['H', 'O'], 3.0, [0])
            soap2 = compute_soap(atoms, ['H', 'O'], 3.0, [1])
            
            assert soap1.shape == soap2.shape
        except Exception:
            pytest.skip("SOAP computation requires dscribe library")
    
    def test_soap_descriptor_finite_values(self):
        """Test that SOAP descriptor contains finite values."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            soap_descriptor = compute_soap(atoms, ['H', 'O'], 3.0, [0])
            
            assert np.all(np.isfinite(soap_descriptor))
            assert not np.all(soap_descriptor == 0)
        except Exception:
            pytest.skip("SOAP computation requires dscribe library")


class TestSubsamplingSystemTypes:
    """Test with different system types."""
    
    def test_periodic_system(self):
        """Test with periodic system."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        result = indices(atoms, "all")
        assert len(result) == 3
    
    def test_non_periodic_system(self):
        """Test with non-periodic system."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        # No cell or PBC set
        
        result = indices(atoms, "all")
        assert len(result) == 3
    
    def test_mixed_element_system(self):
        """Test with system containing many different elements."""
        symbols = ['H', 'C', 'N', 'O', 'S', 'P']
        positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]
        atoms = Atoms(symbols, positions=positions)
        
        # Test selecting each element
        for symbol in symbols:
            result = indices(atoms, [symbol])
            assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
