"""Extended tests for clustering module to improve coverage."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from ase import Atoms
from ase.io import write
import matplotlib
matplotlib.use("Agg")  # ensure plotting works headless

try:
    from CRISP.data_analysis.clustering import (
        analyze_frame,
        analyze_trajectory,
        create_html_visualization,
        calculate_silhouette_score,
        extract_cluster_info,
        print_cluster_summary,
        save_frame_info_to_file,
        save_analysis_results,
        plot_analysis_results,
        cluster_analysis,
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

    def test_find_clusters_without_distance_matrix_raises(self, tmp_path):
        """Calling find_clusters before distance matrix should fail."""
        traj_file = tmp_path / "test.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        write(traj_file, atoms)

        analyzer = analyze_frame(
            traj_path=str(traj_file),
            atom_indices=np.array([0, 1, 2]),
            threshold=2.5,
            min_samples=2,
        )
        with pytest.raises(ValueError):
            analyzer.find_clusters()

    def test_analyze_structure_creates_outputs(self, tmp_path):
        """Exercise analyze_structure, including HTML + pickle output."""
        traj_file = tmp_path / "test.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        write(traj_file, atoms)

        analyzer = analyze_frame(
            traj_path=str(traj_file),
            atom_indices=np.array([0, 1, 2]),
            threshold=2.5,
            min_samples=2,
        )
        html_path = tmp_path / "vis.html"
        result = analyzer.analyze_structure(save_html_path=str(html_path), output_dir=str(tmp_path))
        assert result is not None
        assert html_path.exists()
        assert (tmp_path / "single_frame_analysis.pkl").exists()

    def test_save_and_plot_analysis_results(self, tmp_path):
        """Test saving analysis results and generating plots."""
        mock_results = [
            [0, 2, 1, 0.75, 4.0],
            [10, 2, 0, 0.80, 4.5],
            [20, 3, 2, 0.65, 3.0],
        ]

        pickle_path = save_analysis_results(
            analysis_results=mock_results,
            output_dir=str(tmp_path),
            output_prefix="mock_results",
        )

        assert os.path.exists(os.path.join(tmp_path, "mock_results.csv"))
        assert os.path.exists(os.path.join(tmp_path, "mock_results.txt"))
        assert os.path.exists(pickle_path)

        from unittest.mock import patch
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_analysis_results(pickle_file=pickle_path, output_dir=str(tmp_path))
            mock_show.assert_called_once()

        assert os.path.exists(os.path.join(tmp_path, "mock_results_plot.png"))

    def test_cluster_analysis_wrapper_single_mode(self, tmp_path):
        """Test cluster_analysis in 'single' mode."""
        traj_file = tmp_path / "test_wrapper.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        write(traj_file, atoms)

        indices = np.array([0, 1, 2])

        result = cluster_analysis(
            traj_path=str(traj_file),
            indices_path=indices,
            threshold=2.5,
            min_samples=2,
            mode="single",
            output_dir=str(tmp_path),
        )

        assert result is not None
        assert "num_clusters" in result

    def test_cluster_analysis_wrapper_trajectory_mode(self, tmp_path):
        """Test cluster_analysis in 'trajectory' mode."""
        traj_file = tmp_path / "test_wrapper_traj.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, True])
        write(traj_file, atoms)  # single frame is enough to hit the code path

        indices = np.array([0, 1, 2])

        results = cluster_analysis(
            traj_path=str(traj_file),
            indices_path=indices,
            threshold=2.5,
            min_samples=2,
            mode="trajectory",
            output_dir=str(tmp_path),
            frame_skip=1,
        )

        assert results is None
        traj_dir = tmp_path / "trajectory"
        assert traj_dir.exists()
        # first‑frame HTML visualization should have been created
        html_files = list(traj_dir.glob("*first_frame_clusters.html"))
        assert html_files

    def test_calculate_silhouette_score_edge_cases(self):
        """Test silhouette score calculation edge cases."""
        dm = np.zeros((3, 3))

        labels_all_outliers = np.array([-1, -1, -1])
        score1 = calculate_silhouette_score(dm, labels_all_outliers)
        assert score1 == 0

        labels_one_valid = np.array([0, -1, -1])
        score2 = calculate_silhouette_score(dm, labels_one_valid)
        assert score2 == 0


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



