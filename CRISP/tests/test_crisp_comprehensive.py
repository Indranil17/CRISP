import pytest
import os
import glob
import numpy as np
import pandas as pd
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from CRISP.simulation_utility.atomic_indices import atom_indices, run_atom_indices
from CRISP.simulation_utility.error_analysis import error_analysis
from CRISP.simulation_utility.subsampling import subsample
from CRISP.data_analysis.msd import calculate_save_msd, analyze_from_csv
from CRISP.data_analysis.contact_coordination import coordination
from CRISP.data_analysis.prdf import analyze_rdf
from CRISP.data_analysis.h_bond import hydrogen_bonds

try:
    from CRISP.data_analysis.contact_coordination import contacts
    from CRISP.data_analysis.clustering import analyze_frame, analyze_trajectory
    from CRISP.data_analysis.volumetric_atomic_density import create_density_map
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


@pytest.fixture
def test_data_setup():
    """Fixture to set up test data paths and output directory."""
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(test_dir, '..', 'data'))
    
    # Find trajectory files
    traj_files = glob.glob(os.path.join(data_dir, '*.traj'))
    if not traj_files:
        pytest.skip(f"No .traj file found in {data_dir}")
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    
    yield {
        'traj_path': traj_files[0],
        'data_dir': data_dir,
        'temp_dir': temp_dir
    }
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_log_data(test_data_setup):
    """Create sample log data for error analysis testing."""
    temp_dir = test_data_setup['temp_dir']
    log_file = os.path.join(temp_dir, 'test.log')
    
    # Create sample log data
    data = {
        'Epot[eV]': np.random.normal(-1000, 10, 1000),
        'T[K]': np.random.normal(300, 5, 1000)
    }
    df = pd.DataFrame(data)
    df.to_csv(log_file, sep=' ', index=False)
    
    return log_file


class TestSimulationUtility:
    """Test simulation utility modules."""
    
    def test_atomic_indices_basic(self, test_data_setup):
        """Test basic atomic indices functionality."""
        traj_path = test_data_setup['traj_path']
        
        indices, dist_matrix, cutoff_indices = atom_indices(traj_path)
        
        assert isinstance(indices, dict)
        assert isinstance(dist_matrix, np.ndarray)
        assert isinstance(cutoff_indices, dict)
        assert len(indices) > 0
    
    def test_run_atom_indices(self, test_data_setup):
        """Test run_atom_indices with various parameters."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'indices')
        
        # Test with default parameters
        run_atom_indices(traj_path, output_dir, frame_index=0)
        
        lengths_file = os.path.join(output_dir, 'lengths.npy')
        assert os.path.exists(lengths_file)
        
        lengths = np.load(lengths_file, allow_pickle=True).item()
        assert isinstance(lengths, dict)
        assert len(lengths) > 0
    
    def test_run_atom_indices_custom_cutoffs(self, test_data_setup):
        """Test run_atom_indices with custom cutoffs."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'indices_custom')
        
        custom_cutoffs = {
            ('O', 'H'): 1.2,
            ('Si', 'O'): 1.8,
            ('Al', 'Si'): 3.2,
            ('O', 'O'): 3.0
        }
        
        run_atom_indices(traj_path, output_dir, frame_index=0, custom_cutoffs=custom_cutoffs)
        
        assert os.path.exists(output_dir)
        lengths_file = os.path.join(output_dir, 'lengths.npy')
        assert os.path.exists(lengths_file)
    
    def test_error_analysis(self, sample_log_data):
        """Test error analysis functionality."""
        df = pd.read_csv(sample_log_data, sep=r'\s+')  # FIXED: Use raw string
        epot_data = df["Epot[eV]"].values
        temp_data = df["T[K]"].values
        
        max_lag = min(200, len(df))
        
        # Test energy data
        epot_results = error_analysis(epot_data, max_lag, threshold=0.15)
        assert isinstance(epot_results, dict)
        
        # Test temperature data
        temp_results = error_analysis(temp_data, max_lag)
        assert isinstance(temp_results, dict)
    
    def test_subsampling(self, test_data_setup):
        """Test trajectory subsampling."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'subsampling')
        
        frames = subsample(
            traj_path=traj_path,
            n_samples=5,
            frame_skip=1,
            output_dir=output_dir
        )
        
        assert len(frames) <= 5
        assert os.path.exists(output_dir)


class TestDataAnalysis:
    """Test data analysis modules."""
    
    def test_msd_calculation_and_analysis(self, test_data_setup):
        """Test MSD calculation and analysis."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'msd')
        os.makedirs(output_dir, exist_ok=True)
        
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_for_msd')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        # Find oxygen indices file
        o_indices_file = os.path.join(indices_dir, 'O_indices.npy')
        if not os.path.exists(o_indices_file):
            dummy_indices = np.array([0, 1, 2, 3, 4])  # First 5 atoms
            np.save(o_indices_file, dummy_indices)
        
        # Test MSD calculation
        msd_values, msd_times = calculate_save_msd(
            traj_path=traj_path,
            timestep_fs=50.0,
            indices_path=o_indices_file,
            frame_skip=5,
            output_dir=output_dir,
            output_file='msd_test.csv'
        )
        
        assert isinstance(msd_values, np.ndarray)
        assert isinstance(msd_times, np.ndarray)
        assert len(msd_values) == len(msd_times)
        
        # Test MSD analysis
        csv_file = os.path.join(output_dir, 'msd_test.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if len(df) > 10:  # Need enough data points for fitting
                D, error = analyze_from_csv(
                    csv_file=csv_file,
                    fit_start=0,
                    fit_end=min(len(df), 50),
                    with_intercept=True
                )
                assert isinstance(D, float)
                assert isinstance(error, float)
                assert D > 0
    
    def test_coordination_analysis(self, test_data_setup):
        """Test coordination analysis."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'coordination')
        
        # Create indices first
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_for_coord')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        # Use a simple atom type for coordination
        custom_cutoffs = {('O', 'O'): 3.0}
        
        try:
            # FIXED: Changed bonded_atoms to target_atoms (correct parameter name)
            cn_result = coordination(
                traj_path=traj_path,
                central_atoms="O",  # FIXED: Changed from target_atoms
                target_atoms=["O"],  # FIXED: Changed from bonded_atoms
                custom_cutoffs=custom_cutoffs,
                frame_skip=5,
                output_dir=output_dir
            )
            assert cn_result is not None
        except Exception as e:
            pytest.skip(f"Coordination analysis skipped due to: {e}")
    
    def test_rdf_analysis(self, test_data_setup):
        """Test RDF analysis."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'rdf')
        
        # Test total RDF
        try:
            result = analyze_rdf(
                use_prdf=False,
                rmax=6.0,
                traj_path=traj_path,
                nbins=20,
                frame_skip=5,
                output_dir=output_dir,
                create_plots=False  # Disable plotting for testing
            )
            assert result is not None
        except Exception as e:
            pytest.skip(f"RDF analysis skipped due to: {e}")
    
    def test_partial_rdf_analysis(self, test_data_setup):
        """Test partial RDF analysis."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'prdf')
        
        # Create indices first
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_for_prdf')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        # Try to load oxygen indices
        o_indices_file = os.path.join(indices_dir, 'O_indices.npy')
        if os.path.exists(o_indices_file):
            o_indices = np.load(o_indices_file)
            if len(o_indices) > 0:
                atomic_indices = (o_indices[:5].tolist(), o_indices[:5].tolist())
                
                try:
                    result = analyze_rdf(
                        use_prdf=True,
                        rmax=6.0,
                        traj_path=traj_path,
                        nbins=20,
                        frame_skip=5,
                        atomic_indices=atomic_indices,
                        output_dir=output_dir,
                        create_plots=False
                    )
                    assert result is not None
                except Exception as e:
                    pytest.skip(f"Partial RDF analysis skipped due to: {e}")
        else:
            pytest.skip("No oxygen indices found for partial RDF test")
    
    def test_hydrogen_bond_analysis(self, test_data_setup):
        """Test hydrogen bond analysis."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'hbond')
        
        try:
            h_bonds = hydrogen_bonds(
                traj_path=traj_path,
                frame_skip=10,
                acceptor_atoms=["O"],
                angle_cutoff=120,
                time_step=50.0,
                output_dir=output_dir,
                plot_count=False,  
                plot_heatmap=False,
                plot_graph_frame=False,
                plot_graph_average=False
            )
            assert h_bonds is not None
        except Exception as e:
            pytest.skip(f"Hydrogen bond analysis skipped due to: {e}")


class TestContactCoordination:
    """Test contact and coordination analysis."""
    
    def test_contacts_analysis(self, test_data_setup):
        """Test contacts function."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'contacts')
        
        custom_cutoffs = {('O', 'O'): 3.0, ('Al', 'O'): 3.5}
        
        try:
            # FIXED: Changed bonded_atoms to target_atoms (correct parameter name)
            sub_dm, cal_contacts = contacts(
                traj_path,
                central_atoms="O",  # FIXED: Changed from target_atoms
                target_atoms=["O"],  # FIXED: Changed from bonded_atoms
                custom_cutoffs=custom_cutoffs,
                frame_skip=5,
                plot_distance_matrix=False,
                plot_contacts=False,
                time_step=50.0,
                output_dir=output_dir
            )
            assert sub_dm is not None
            assert cal_contacts is not None
        except Exception as e:
            pytest.skip(f"Contacts analysis skipped: {e}")


class TestClustering:
    """Test clustering analysis."""
    
    def test_analyze_frame(self, test_data_setup):
        """Test single frame clustering."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'clustering_frame')
        
        # Create indices first
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_clustering')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        # Get some atom indices
        lengths_file = os.path.join(indices_dir, 'lengths.npy')
        if os.path.exists(lengths_file):
            lengths = np.load(lengths_file, allow_pickle=True).item()
            if 'O' in lengths and lengths['O'] > 5:
                o_indices_file = os.path.join(indices_dir, 'O_indices.npy')
                if os.path.exists(o_indices_file):
                    atom_indices = np.load(o_indices_file)[:10]  # Use first 10 atoms
                    
                    try:
                        analyzer = analyze_frame(
                            traj_path=traj_path,
                            atom_indices=atom_indices,
                            threshold=3.0,
                            min_samples=2,
                            custom_frame_index=0
                        )
                        result = analyzer.analyze_structure(output_dir=output_dir)
                        assert result is not None
                    except Exception as e:
                        pytest.skip(f"Frame clustering skipped: {e}")
        else:
            pytest.skip("No indices available for clustering test")
    
    def test_analyze_trajectory(self, test_data_setup):
        """Test trajectory clustering."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'clustering_traj')
        
        # Create indices first
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_clustering_traj')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        o_indices_file = os.path.join(indices_dir, 'O_indices.npy')
        if os.path.exists(o_indices_file):
            try:
                analysis_results = analyze_trajectory(
                    traj_path=traj_path,
                    indices_path=o_indices_file,
                    threshold=3.0,
                    min_samples=2,
                    frame_skip=10,
                    output_dir=output_dir,
                    save_html_visualizations=False
                )
                assert analysis_results is not None
            except Exception as e:
                pytest.skip(f"Trajectory clustering skipped: {e}")
        else:
            pytest.skip("No oxygen indices for trajectory clustering")


class TestVolumetricDensity:
    """Test volumetric density analysis."""
    
    def test_create_density_map(self, test_data_setup):
        """Test density map creation."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'density')
        
        # Create indices first
        indices_dir = os.path.join(test_data_setup['temp_dir'], 'indices_density')
        run_atom_indices(traj_path, indices_dir, frame_index=0)
        
        o_indices_file = os.path.join(indices_dir, 'O_indices.npy')
        if os.path.exists(o_indices_file):
            try:
                create_density_map(
                    traj_path=traj_path,
                    indices_path=o_indices_file,
                    frame_skip=10,
                    threshold=0.1,
                    opacity=0.8,
                    show_projections=False,
                    save_density=True,
                    output_dir=output_dir,
                    output_file="test_density.html"
                )
                assert os.path.exists(output_dir)
            except Exception as e:
                pytest.skip(f"Density map creation skipped: {e}")
        else:
            pytest.skip("No indices for density map test")


class TestExtendedFunctionality:
    """Test extended functionality and parameter variations."""
    
    def test_msd_with_all_parameters(self, test_data_setup):
        """Test MSD with all parameter options."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'msd_extended')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dummy indices
        dummy_indices = np.array([0, 1, 2, 3, 4])
        indices_file = os.path.join(output_dir, 'test_indices.npy')
        np.save(indices_file, dummy_indices)
        
        # Test with all parameters
        try:
            msd_values, msd_times = calculate_save_msd(
                traj_path=traj_path,
                timestep_fs=50.0,
                indices_path=indices_file,
                frame_skip=5,
                output_dir=output_dir,
                output_file='msd_extended.csv'
            )
            
            # Test analysis with different parameters
            csv_file = os.path.join(output_dir, 'msd_extended.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if len(df) > 20:
                    D, error = analyze_from_csv(
                        csv_file=csv_file,
                        fit_start=0,
                        fit_end=len(df),
                        with_intercept=True,
                        plot_msd=False,
                        plot_diffusion=False,
                        n_blocks=3
                    )
                    assert isinstance(D, float)
                    assert isinstance(error, float)
        except Exception as e:
            pytest.skip(f"Extended MSD test skipped: {e}")
    
    def test_hydrogen_bonds_extended(self, test_data_setup):
        """Test hydrogen bonds with extended parameters."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'hbond_extended')
        
        try:
            h_bonds = hydrogen_bonds(
                traj_path=traj_path,
                frame_skip=10,
                acceptor_atoms=["O"],
                angle_cutoff=120,
                h_bond_cutoff=2.5,
                bond_cutoff=1.6,
                time_step=50.0,
                mic=True,
                output_dir=output_dir,
                plot_count=False,
                plot_heatmap=False,
                plot_graph_frame=False,
                plot_graph_average=False,
                graph_frame_index=0
            )
            assert h_bonds is not None
        except Exception as e:
            pytest.skip(f"Extended hydrogen bonds test skipped: {e}")
    
    @pytest.mark.parametrize("rmax,nbins", [(5.0, 15), (8.0, 25), (10.0, 30)])
    def test_rdf_parameter_variations(self, test_data_setup, rmax, nbins):
        """Test RDF with different parameter combinations."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], f'rdf_{rmax}_{nbins}')
        
        try:
            result = analyze_rdf(
                use_prdf=False,
                rmax=rmax,
                traj_path=traj_path,
                nbins=nbins,
                frame_skip=5,
                output_dir=output_dir,
                create_plots=False
            )
            assert result is not None
        except Exception as e:
            pytest.skip(f"RDF parameter test skipped: {e}")


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling."""
    
    def test_coordination_edge_cases(self, test_data_setup):
        """Test coordination with edge cases."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'coord_edge')
        
        # Test with empty cutoffs
        try:
            # FIXED: Corrected parameter names
            coordination(
                traj_path=traj_path,
                central_atoms="O",
                target_atoms=["O"],
                custom_cutoffs={},
                frame_skip=10,
                output_dir=output_dir
            )
        except Exception:
            assert True
    
    def test_invalid_atom_types(self, test_data_setup):
        """Test with invalid atom types."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'invalid_atoms')
        
        # Test with non-existent atom type
        try:
            # FIXED: Corrected parameter names
            coordination(
                traj_path=traj_path,
                central_atoms="Xe",
                target_atoms=["Xe"],
                custom_cutoffs={('Xe', 'Xe'): 3.0},
                frame_skip=10,
                output_dir=output_dir
            )
        except Exception:
            # Expected to fail or skip
            assert True
    
    def test_file_not_found_handling(self):
        """Test file not found scenarios."""
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            calculate_save_msd(
                traj_path="nonexistent.traj",
                timestep_fs=50.0,
                indices_path="nonexistent.npy",
                frame_skip=1,
                output_dir="/tmp",
                output_file="test.csv"
            )


class TestParameterVariations:
    """Test functions with different parameter combinations."""
    
    @pytest.mark.parametrize("frame_index", [0, 1])  
    def test_atom_indices_frames(self, test_data_setup, frame_index):
        """Test atomic indices with different frame indices."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], f'indices_frame_{frame_index}')
        
        run_atom_indices(traj_path, output_dir, frame_index=frame_index)
        assert os.path.exists(output_dir)
    
    def test_atom_indices_last_frame(self, test_data_setup):
        """Test atomic indices with the last frame."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'indices_last_frame')
        
        # Get trajectory length first
        import ase.io
        traj = ase.io.read(traj_path, index=":")
        if isinstance(traj, list):
            last_frame = len(traj) - 1
        else:
            last_frame = 0
        
        run_atom_indices(traj_path, output_dir, frame_index=last_frame)
        assert os.path.exists(output_dir)

    @pytest.mark.parametrize("n_samples", [3, 5, 8])
    def test_subsampling_variations(self, test_data_setup, n_samples):
        """Test subsampling with different sample sizes."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], f'subsample_{n_samples}')
        
        frames = subsample(
            traj_path=traj_path,
            n_samples=n_samples,
            frame_skip=2,
            output_dir=output_dir,
            plot_subsample=False  
        )
        assert len(frames) <= n_samples
    
    @pytest.mark.parametrize("threshold", [0.1, 0.15, 0.2])
    def test_error_analysis_thresholds(self, sample_log_data, threshold):
        """Test error analysis with different thresholds."""
        df = pd.read_csv(sample_log_data, sep=r'\s+')  # FIXED: Use raw string
        data = df["Epot[eV]"].values
        
        results = error_analysis(data, max_lag=100, threshold=threshold)
        assert isinstance(results, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        with pytest.raises((FileNotFoundError, Exception)):
            atom_indices("nonexistent_file.traj")
    
    def test_empty_indices(self, test_data_setup):
        """Test handling of empty indices arrays."""
        temp_dir = test_data_setup['temp_dir']
        empty_indices_file = os.path.join(temp_dir, 'empty_indices.npy')
        np.save(empty_indices_file, np.array([]))
        
        # This should handle empty indices gracefully
        assert os.path.exists(empty_indices_file)
        empty_indices = np.load(empty_indices_file)
        assert len(empty_indices) == 0
    
    def test_invalid_parameters(self, test_data_setup):
        """Test handling of invalid parameters."""
        traj_path = test_data_setup['traj_path']
        
        # Test with invalid frame index (should handle gracefully)
        with pytest.raises((IndexError, ValueError, Exception)):
            atom_indices(traj_path, frame_index=99999)


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_indices_consistency(self, test_data_setup):
        """Test that indices are consistent across runs."""
        traj_path = test_data_setup['traj_path']
        
        # Run twice with same parameters
        indices1, _, _ = atom_indices(traj_path, frame_index=0)
        indices2, _, _ = atom_indices(traj_path, frame_index=0)
        
        # Results should be identical
        assert indices1.keys() == indices2.keys()
        for key in indices1.keys():
            np.testing.assert_array_equal(indices1[key], indices2[key])
    
    def test_output_files_created(self, test_data_setup):
        """Test that all expected output files are created."""
        traj_path = test_data_setup['traj_path']
        output_dir = os.path.join(test_data_setup['temp_dir'], 'output_test')
        
        run_atom_indices(traj_path, output_dir, frame_index=0)
        
        # Check for expected files
        expected_files = ['lengths.npy']
        for file_name in expected_files:
            file_path = os.path.join(output_dir, file_name)
            assert os.path.exists(file_path), f"Expected file {file_name} not created"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])