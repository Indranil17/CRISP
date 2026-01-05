"""Comprehensive tests for Mean Square Displacement (MSD) module."""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from CRISP.data_analysis.msd import (
    calculate_diffusion_coefficient,
    calculate_save_msd,
    analyze_from_csv,
)


class TestMSDCalculations:
    """Test MSD calculation core functionality."""
    
    def test_calculate_diffusion_coefficient_basic(self):
        """Test basic diffusion coefficient calculation."""
        # Create linear MSD data (slope = 0.4)
        msd_times = np.array([0, 1, 2, 3, 4, 5])
        msd_values = np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0])
        
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=len(msd_values),
            with_intercept=True
        )
        
        assert isinstance(D, float)
        assert isinstance(error, float)
        assert D > 0
        assert error >= 0
    
    def test_calculate_diffusion_without_intercept(self):
        """Test diffusion coefficient calculation without intercept."""
        msd_times = np.array([0, 1, 2, 3, 4])
        msd_values = np.array([0, 0.2, 0.4, 0.6, 0.8])
        
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=len(msd_values),
            with_intercept=False
        )
        
        assert D > 0
        assert error >= 0
    
    def test_calculate_diffusion_subset(self):
        """Test diffusion coefficient with subset of data."""
        msd_times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        msd_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
        
        # Use only first 5 points
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=5,
            with_intercept=True
        )
        
        assert D > 0
        assert error >= 0
    
    def test_calculate_diffusion_different_start_end(self):
        """Test diffusion coefficient with different start/end indices."""
        msd_times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        msd_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # Use middle portion (indices 2-6)
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=2,
            end_index=7,
            with_intercept=True
        )
        
        assert D > 0
    
    def test_calculate_diffusion_single_point(self):
        """Test with insufficient data points - returns result but with nan R²."""  
        msd_times = np.array([0])
        msd_values = np.array([0])
        
        # Function doesn't raise error, just returns result with warnings
        result = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=1
        )
        # Should return a tuple
        assert result is not None
        assert len(result) == 2
    
    def test_calculate_diffusion_empty_data(self):
        """Test with empty data."""
        with pytest.raises((ValueError, IndexError)):
            calculate_diffusion_coefficient(
                msd_times=np.array([]),
                msd_values=np.array([]),
                start_index=0,
                end_index=0
            )


class TestMSDFileOperations:
    """Test MSD file reading and writing."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create sample MSD CSV file."""
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, 'msd_sample.csv')
        
        # Create realistic MSD data
        data = {
            'Time (fs)': np.arange(0, 100, 10),
            'MSD': np.array([0, 0.5, 1.2, 2.1, 3.2, 4.5, 5.8, 7.2, 8.8, 10.5])
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        yield csv_path
        shutil.rmtree(temp_dir)
    
    def test_analyze_from_csv_basic(self, sample_csv_file):
        """Test MSD analysis from CSV file."""
        D, error = analyze_from_csv(
            csv_file=sample_csv_file,
            fit_start=0,
            fit_end=5,
            with_intercept=True
        )
        
        assert isinstance(D, float)
        assert isinstance(error, float)
        assert D > 0
    
    def test_analyze_from_csv_different_fit_range(self, sample_csv_file):
        """Test MSD analysis with different fit ranges."""
        # Test with different ranges
        D1, _ = analyze_from_csv(
            csv_file=sample_csv_file,
            fit_start=0,
            fit_end=3,
            with_intercept=True
        )
        
        D2, _ = analyze_from_csv(
            csv_file=sample_csv_file,
            fit_start=3,
            fit_end=8,
            with_intercept=True
        )
        
        assert D1 > 0
        assert D2 > 0
        assert D1 != D2  # Different ranges should give different results
    
    def test_analyze_from_csv_invalid_file(self):
        """Test analysis with nonexistent file returns None."""
        D, error = analyze_from_csv(
            csv_file='nonexistent_file.csv',
            fit_start=0,
            fit_end=5
        )
        # Function catches error and returns (None, None)
        assert D is None
        assert error is None
    
    def test_analyze_from_csv_with_blocks(self, sample_csv_file):
        """Test MSD analysis with block averaging."""
        D, error = analyze_from_csv(
            csv_file=sample_csv_file,
            fit_start=0,
            fit_end=5,
            with_intercept=True,
            n_blocks=2
        )
        
        assert D > 0
        assert error >= 0


class TestMSDEdgeCases:
    """Test edge cases and error handling."""
    
    def test_negative_msd_values(self):
        """Test handling of negative MSD values (physically impossible)."""
        msd_times = np.array([0, 1, 2])
        msd_values = np.array([0, -0.1, -0.2])  # Negative values
        
        # Function should handle gracefully
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=3
        )
        
        assert isinstance(D, float)
    
    def test_constant_msd_values(self):
        """Test with constant MSD values (no motion)."""
        msd_times = np.array([0, 1, 2, 3, 4])
        msd_values = np.array([0, 0, 0, 0, 0])  # No motion
        
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=5
        )
        
        # D should be close to 0
        assert D >= 0
        assert D < 0.1
    
    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths - may work due to slicing."""  
        # Function uses slicing with start_index:end_index, so mismatch may work
        try:
            result = calculate_diffusion_coefficient(
                msd_times=np.array([0, 1, 2]),
                msd_values=np.array([0, 1]),  # Different length
                start_index=0,
                end_index=2
            )
            # If it works, result should be a tuple
            assert result is not None
        except (ValueError, IndexError):
            # Or it may raise an error, which is also acceptable
            pass
    
    def test_invalid_index_range(self):
        """Test with invalid start/end indices."""
        msd_times = np.array([0, 1, 2, 3])
        msd_values = np.array([0, 0.1, 0.2, 0.3])
        
        with pytest.raises((ValueError, IndexError)):
            calculate_diffusion_coefficient(
                msd_times=msd_times,
                msd_values=msd_values,
                start_index=5,  # Out of bounds
                end_index=10
            )
    
    def test_start_greater_than_end(self):
        """Test with start index greater than end index."""
        msd_times = np.array([0, 1, 2, 3, 4])
        msd_values = np.array([0, 0.1, 0.2, 0.3, 0.4])
        
        with pytest.raises((ValueError, Exception)):
            calculate_diffusion_coefficient(
                msd_times=msd_times,
                msd_values=msd_values,
                start_index=4,
                end_index=2  # Start > End
            )


class TestMSDIntegration:
    """Test MSD module integration."""
    
    def test_msd_module_imports(self):
        """Test that all MSD functions can be imported."""
        from CRISP.data_analysis.msd import (
            calculate_diffusion_coefficient,
            calculate_save_msd,
            analyze_from_csv,
        )
        
        assert callable(calculate_diffusion_coefficient)
        assert callable(calculate_save_msd)
        assert callable(analyze_from_csv)
    
    def test_diffusion_values_reasonable(self):
        """Test that diffusion values are physically reasonable."""
        # Create MSD data typical for molecular dynamics
        msd_times = np.linspace(0, 1000, 50)  # Time in fs
        # D ≈ 1e-5 cm²/s = 1e-9 m²/s, in Angstrom²/fs: 1e-9 * 1e20 * 1e-15 ≈ 0.001
        msd_values = 4 * 0.001 * msd_times  # MSD = 4*D*t
        
        D, error = calculate_diffusion_coefficient(
            msd_times=msd_times,
            msd_values=msd_values,
            start_index=0,
            end_index=len(msd_values)
        )
        
        # D should be positive and reasonable
        assert D > 0
        assert D < 10  # Not unreasonably large


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
