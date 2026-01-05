"""Extended tests for clustering module to improve coverage."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write

try:
    from CRISP.data_analysis.clustering import (
        analyze_frame,
        analyze_trajectory,
    )
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestClusteringExtended:
    """Extended clustering tests for coverage."""
    
    def test_analyze_frame_basic(self):
        """Test frame clustering analysis."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            atoms = Atoms('H2OH2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0],
                [2.8, 0.0, 0.0],
                [3.76, 0.0, 0.0],
                [3.04, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            write(traj_file, atoms)
            
            atom_indices = np.array([0, 1, 2, 3, 4, 5])
            analyzer = analyze_frame(
                traj_path=traj_file,
                atom_indices=atom_indices,
                threshold=2.5,
                min_samples=2
            )
            assert analyzer is not None
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.parametrize("threshold", [1.5, 2.0, 2.5, 3.0])
    def test_analyze_frame_different_cutoffs(self, threshold):
        """Test with different distance cutoffs."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            atoms = Atoms('H2OH2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0],
                [2.8, 0.0, 0.0],
                [3.76, 0.0, 0.0],
                [3.04, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            write(traj_file, atoms)
            
            atom_indices = np.array([0, 1, 2])
            analyzer = analyze_frame(
                traj_path=traj_file,
                atom_indices=atom_indices,
                threshold=threshold,
                min_samples=1
            )
            assert analyzer is not None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_analyze_frame_calculate_distance_matrix(self):
        """Test distance matrix calculation."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            write(traj_file, atoms)
            
            atom_indices = np.array([0, 1, 2])
            analyzer = analyze_frame(
                traj_path=traj_file,
                atom_indices=atom_indices,
                threshold=2.5,
                min_samples=2
            )
            
            frame = analyzer.read_custom_frame()
            assert frame is not None
            dist_matrix, positions = analyzer.calculate_distance_matrix(frame)
            assert dist_matrix is not None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_analyze_trajectory_basic(self):
        """Test trajectory clustering."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0]
            ])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            write(traj_file, atoms)
            
            atom_indices = np.array([0, 1, 2])
            results = analyze_trajectory(
                traj_path=traj_file,
                indices_path=atom_indices,
                threshold=2.5,
                min_samples=2,
                frame_skip=1
            )
            assert isinstance(results, list)
        finally:
            shutil.rmtree(temp_dir)


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not available")
class TestClusteringEdgeCases:
    """Test edge cases for clustering."""
    
    def test_clustering_min_atoms_validation(self):
        """Test minimum atoms validation."""
        temp_dir = tempfile.mkdtemp()
        try:
            traj_file = os.path.join(temp_dir, 'test.traj')
            atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc([True, True, True])
            write(traj_file, atoms)
            
            atom_indices = np.array([0])
            analyzer = analyze_frame(
                traj_path=traj_file,
                atom_indices=atom_indices,
                threshold=2.5,
                min_samples=5
            )
            
            frame = analyzer.read_custom_frame()
            with pytest.raises(ValueError):
                analyzer.calculate_distance_matrix(frame)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_clustering_invalid_trajectory(self):
        """Test handling of invalid trajectory file."""
        temp_dir = tempfile.mkdtemp()
        try:
            nonexistent = os.path.join(temp_dir, 'nonexistent.traj')
            analyzer = analyze_frame(
                traj_path=nonexistent,
                atom_indices=np.array([0, 1, 2]),
                threshold=2.5,
                min_samples=2
            )
            frame = analyzer.read_custom_frame()
            assert frame is None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_clustering_indices_from_file(self):
        """Test loading indices from numpy file."""
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
            
            indices = np.array([0, 1, 2])
            np.save(indices_file, indices)
            
            analyzer = analyze_frame(
                traj_path=traj_file,
                atom_indices=indices_file,
                threshold=2.5,
                min_samples=2
            )
            assert analyzer is not None
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
