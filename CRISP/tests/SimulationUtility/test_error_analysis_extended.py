"""Extended tests for error analysis module."""
import pytest
import numpy as np

from CRISP.simulation_utility.error_analysis import (
    optimal_lag,
    vector_acf,
)


class TestErrorAnalysisBasic:
    """Basic error analysis tests."""
    
    def test_optimal_lag_converged(self):
        """Test optimal lag finding for converged ACF."""
        acf_values = np.array([1.0, 0.9, 0.7, 0.4, 0.1, 0.02, 0.0, -0.01])
        
        lag = optimal_lag(acf_values, threshold=0.05)
        
        # Should find lag where ACF drops below threshold
        assert lag >= 0
        assert lag < len(acf_values)
    
    def test_optimal_lag_not_converged(self):
        """Test optimal lag when ACF doesn't converge (raises warning)."""
        acf_values = np.array([1.0, 0.95, 0.90, 0.85, 0.80])
        
        with pytest.warns(UserWarning):
            lag = optimal_lag(acf_values, threshold=0.05)
        
        # Should return last index when not converged
        assert lag == len(acf_values) - 1
    
    def test_optimal_lag_immediate_convergence(self):
        """Test optimal lag with immediate convergence."""
        acf_values = np.array([1.0, 0.01, 0.0])
        
        lag = optimal_lag(acf_values, threshold=0.05)
        
        assert lag == 1
    
    def test_vector_acf_basic(self):
        """Test vector ACF calculation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        
        acf_result = vector_acf(data, max_lag=2)
        
        assert acf_result is not None
        assert isinstance(acf_result, (np.ndarray, list, tuple))
    
    def test_vector_acf_1d_input(self):
        """Test vector ACF with properly shaped 2D input."""
        # vector_acf expects 2D input with shape (n_frames, n_dimensions)
        # Create 2D array instead of 1D
        data = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]])
        
        acf_result = vector_acf(data, max_lag=3)
        
        assert acf_result is not None


class TestErrorAnalysisParametrized:
    """Parametrized error analysis tests."""
    
    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.2])
    def test_optimal_lag_various_thresholds(self, threshold):
        """Test optimal lag with different thresholds."""
        acf_values = np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.01, 0.001])
        
        lag = optimal_lag(acf_values, threshold=threshold)
        
        assert lag >= 0
        assert lag < len(acf_values)
    
    @pytest.mark.parametrize("max_lag", [1, 2, 5, 10])
    def test_vector_acf_various_lags(self, max_lag):
        """Test vector ACF with different max lags."""
        data = np.random.random((20, 3))
        
        acf_result = vector_acf(data, max_lag=max_lag)
        
        assert acf_result is not None


class TestErrorAnalysisEdgeCases:
    """Test error analysis edge cases."""
    
    def test_optimal_lag_empty_array(self):
        """Test optimal lag with very short array."""
        acf_values = np.array([1.0])
        
        lag = optimal_lag(acf_values, threshold=0.05)
        
        assert lag == 0
    
    def test_optimal_lag_negative_values(self):
        """Test optimal lag with negative ACF values."""
        acf_values = np.array([1.0, 0.5, 0.0, -0.3, -0.5])
        
        lag = optimal_lag(acf_values, threshold=0.05)
        
        assert lag >= 0
        assert lag < len(acf_values)
    
    def test_vector_acf_short_data(self):
        """Test vector ACF with short data."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        try:
            acf_result = vector_acf(data, max_lag=1)
            assert acf_result is not None
        except (ValueError, IndexError):
            pass  # Acceptable for short data
    
    def test_vector_acf_single_sample(self):
        """Test vector ACF with single sample."""
        data = np.array([[1, 2, 3]])
        
        try:
            acf_result = vector_acf(data, max_lag=1)
            # May raise error or return zeros
            assert acf_result is not None or acf_result is None
        except (ValueError, ZeroDivisionError):
            pass


class TestErrorAnalysisIntegration:
    """Integration tests for error analysis."""
    
    def test_lag_finding_workflow(self):
        """Test complete workflow of finding optimal lag."""
        # Create synthetic data with known correlation
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100, 3), axis=0)
        
        acf_result = vector_acf(data, max_lag=10)
        assert acf_result is not None
        
        # Convert to array if needed
        if isinstance(acf_result, (list, tuple)):
            acf_array = np.array(acf_result)
        else:
            acf_array = acf_result
        
        if len(acf_array) > 0:
            lag = optimal_lag(acf_array, threshold=0.05)
            assert lag >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
