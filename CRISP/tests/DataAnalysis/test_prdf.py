"""Extended tests for PRDF (Radial Distribution Function) module."""
import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch
from ase import Atoms
from ase.io import write

from CRISP.data_analysis.prdf import (
    check_cell_and_r_max,
    compute_pairwise_rdf,
    Analysis,
    plot_rdf,
    animate_rdf,
    analyze_rdf,
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


def test_check_cell_negative_rmax_allowed():
    """Current implementation does not validate sign of rmax; just ensure no crash."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc([True, True, True])
    check_cell_and_r_max(atoms, -1.0)  # should not raise


def test_check_cell_non_periodic_allowed():
    """Non-periodic cells are currently accepted; this documents that behavior."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc([False, False, False])
    check_cell_and_r_max(atoms, 3.0)  # should not raise


def test_pairwise_rdf_invalid_nbins_raises_zero_division():
    """nbins=0 causes a ZeroDivisionError via rmax/nbins; test that path."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc([True, True, True])
    with pytest.raises(ZeroDivisionError):
        compute_pairwise_rdf(
            atoms=atoms,
            ref_indices=[0],
            target_indices=[1],
            rmax=5.0,
            nbins=0,
        )


class TestPRDFCoverageExpansion:
    """Tests targeting unexecuted high-level RDF functions."""

    @pytest.fixture
    def mock_traj(self):
        """Creates a temporary trajectory for integration tests."""
        temp_dir = tempfile.mkdtemp()
        traj_path = os.path.join(temp_dir, "test_rdf.traj")
        frames = []
        for _ in range(3):
            atoms = Atoms(
                "H2O",
                positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]],
            )
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc(True)
            frames.append(atoms)
        write(traj_path, frames)
        yield traj_path, temp_dir
        shutil.rmtree(temp_dir)

    def test_analysis_class_workflow(self, mock_traj):
        """Covers the Analysis class and get_rdf method logic."""
        traj_path, _ = mock_traj
        from ase.io import read

        images = read(traj_path, index=":")
        analyzer = Analysis(images)
        results = analyzer.get_rdf(
            rmax=4.0,
            nbins=50,
            atomic_indices=([0], [1, 2]),
            return_dists=True,
        )
        assert len(results) == 3
        assert isinstance(results[0], tuple)

    @patch("matplotlib.pyplot.show")
    def test_plotting_and_animation(self, mock_show):
        """Covers plot_rdf and animate_rdf functions."""
        x = np.linspace(0, 5, 50)
        y = [np.random.rand(50) for _ in range(2)]

        # static plot
        plot_rdf(x, y, title="Test Plot")

        # animation
        ani = animate_rdf(x, y)
        assert ani is not None
        mock_show.assert_called()

    def test_analyze_rdf_integration(self, mock_traj):
        """Covers the analyze_rdf wrapper, directory creation, and pickling."""
        traj_path, temp_dir = mock_traj
        out_dir = os.path.join(temp_dir, "results")

        data = analyze_rdf(
            use_prdf=True,
            rmax=4.0,
            traj_path=traj_path,
            nbins=40,
            frame_skip=1,
            output_dir=out_dir,
            atomic_indices=([0], [1, 2]),
            create_plots=False,
        )

        assert os.path.exists(out_dir)
        assert any(fname.endswith(".pkl") for fname in os.listdir(out_dir))
        assert "x_data" in data
        assert len(data["y_data_all"]) == 3


class TestPRDFCoverageFinalPush:
    """Targeted tests to resolve remaining gaps in prdf.py coverage."""

    def test_check_cell_volume_logic(self):
        """Covers volume-based cell validation for very small cells."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.set_cell([1, 1, 1])  # very small box
        atoms.set_pbc(True)

        # Expect ValueError from check_cell_and_r_max for too-large rmax
        with pytest.raises(ValueError, match="RDF Error"):
            check_cell_and_r_max(atoms, rmax=10.0)

    def test_total_rdf_workflow(self, tmp_path):
        """Covers total RDF (use_prdf=False) logic."""
        traj_path = tmp_path / "total_rdf.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        write(str(traj_path), [atoms, atoms])

        results = analyze_rdf(
            use_prdf=False,
            rmax=4.0,
            traj_path=str(traj_path),
            nbins=20,
            frame_skip=1,
            output_dir=str(tmp_path),
            create_plots=False,
        )
        assert "y_data_all" in results
        assert "x_data" in results

    @patch("matplotlib.pyplot.show")
    def test_full_visualization_and_animation_save(self, mock_show, tmp_path):
        """Covers plotting and animation save branches in analyze_rdf."""
        traj_path = tmp_path / "ani.traj"
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        write(str(traj_path), [atoms, atoms, atoms])

        analyze_rdf(
            use_prdf=True,
            rmax=4.0,
            traj_path=str(traj_path),
            atomic_indices=([0], [1, 2]),
            output_dir=str(tmp_path),
            create_plots=True,
            frame_skip=1,
        )

        # At least one PNG plot and one HTML animation should be created
        png_files = list(tmp_path.glob("*.png"))
        html_files = list(tmp_path.glob("*.html"))
        assert any("prdf" in f.name for f in png_files)
        assert any("prdf" in f.name and "animation" in f.name for f in html_files)
        mock_show.assert_called()

    def test_analyze_rdf_exceptions(self, tmp_path):
        """Covers error handling branches in analyze_rdf."""
        # 1) Non-readable / empty trajectory -> ASE read fails
        empty_traj = tmp_path / "empty.traj"
        empty_traj.write_text("")
        with pytest.raises(Exception):
            analyze_rdf(True, 4.0, str(empty_traj))

        # 2) use_prdf=True but atomic_indices is missing -> explicit ValueError
        valid_traj = tmp_path / "valid.traj"
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10], pbc=True)
        write(str(valid_traj), atoms)

        with pytest.raises(ValueError, match="atomic_indices must be provided"):
            analyze_rdf(use_prdf=True, rmax=4.0, traj_path=str(valid_traj))
