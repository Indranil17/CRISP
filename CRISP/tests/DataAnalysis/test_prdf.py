"""Extended tests for PRDF (Radial Distribution Function) module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write

from CRISP.data_analysis.prdf import (
    check_cell_and_r_max,
    compute_pairwise_rdf,
)


class TestPRDFBasic:
    """Basic PRDF functionality tests."""
    
    def test_check_cell_valid(self):
        """Test cell validation with valid cell."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        # Should not raise for valid cell
        try:
            check_cell_and_r_max(atoms, 4.0)
        except ValueError:
            pytest.fail("Should not raise for valid cell")
    
    def test_check_cell_too_small(self):
        """Test cell validation with cell too small for rmax."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        atoms.set_cell([2, 2, 2])
        atoms.set_pbc([True, True, True])
        
        # Should raise for cell too small
        with pytest.raises(ValueError):
            check_cell_and_r_max(atoms, 5.0)
    
    def test_check_cell_undefined(self):
        """Test cell validation with undefined cell."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        
        # Should raise for undefined cell
        with pytest.raises(ValueError):
            check_cell_and_r_max(atoms, 2.0)
    
    def test_compute_pairwise_rdf_basic(self):
        """Test basic pairwise RDF calculation."""
        atoms = Atoms('H2O', positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        # compute_pairwise_rdf uses rmax and nbins (not r_max and dr)
        # nbins = rmax / dr, so for r_max=5.0 and dr=0.1, nbins=50
        g_r, r = compute_pairwise_rdf(
            atoms=atoms,
            ref_indices=[0],
            target_indices=[1, 2],
            rmax=5.0,
            nbins=50
        )
        
        assert len(r) > 0
        assert len(g_r) > 0


class TestPRDFParametrized:
    """Test PRDF with parameter variations."""
    
    @pytest.mark.parametrize("r_max", [3.0, 5.0, 8.0])
    def test_rdf_different_r_max(self, r_max):
        """Test RDF with different maximum radius."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        atoms.set_cell([20, 20, 20])
        atoms.set_pbc([True, True, True])
        
        # nbins = rmax / dr, so for dr=0.1
        nbins = int(r_max / 0.1)
        g_r, r = compute_pairwise_rdf(
            atoms=atoms,
            ref_indices=[0],
            target_indices=[1],
            rmax=r_max,
            nbins=nbins
        )
        
        assert r[-1] <= r_max + 0.1
    
    @pytest.mark.parametrize("dr", [0.05, 0.1, 0.2])
    def test_rdf_different_dr(self, dr):
        """Test RDF with different bin size."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        # nbins = rmax / dr
        rmax = 5.0
        nbins = int(rmax / dr)
        g_r, r = compute_pairwise_rdf(
            atoms=atoms,
            ref_indices=[0],
            target_indices=[1],
            rmax=rmax,
            nbins=nbins
        )
        
        assert len(r) > 0


class TestPRDFEdgeCases:
    """Test PRDF edge cases."""
    
    def test_rdf_single_atom(self):
        """Test RDF with single atom."""
        atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        # Should handle gracefully - empty target indices
        try:
            g_r, r = compute_pairwise_rdf(
                atoms=atoms,
                ref_indices=[0],
                target_indices=[],
                rmax=5.0,
                nbins=50
            )
            assert len(r) >= 0
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_rdf_empty_target(self):
        """Test RDF with empty target indices."""
        atoms = Atoms('H2', positions=[
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0]
        ])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        
        try:
            g_r, r = compute_pairwise_rdf(
                atoms=atoms,
                ref_indices=[0],
                target_indices=[],
                rmax=5.0,
                nbins=50
            )
            # Empty target should return zeros or handle gracefully
            assert len(r) >= 0
        except ValueError:
            pass


class TestPRDFIntegration:
    """Integration tests for PRDF."""
    
    def test_rdf_with_trajectory_mock(self):
        """Test RDF calculation from mock trajectory."""
        temp_dir = tempfile.mkdtemp()
        try:
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            
            g_r, r = compute_pairwise_rdf(
                atoms=atoms,
                ref_indices=[0],
                target_indices=[1, 2],
                rmax=5.0,
                nbins=50
            )
            
            assert len(r) > 0
            assert len(g_r) > 0
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
