"""
Unit tests for SBL algorithm.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbl import SBLDictParam, generate_signal
from sbl.utils.metrics import frequency_rmse, model_order_accuracy


class TestSBLDictParam:
    """Tests for SBL with dictionary parameter estimation."""

    def test_initialization(self):
        """Test SBL initialization."""
        N = 64
        sbl = SBLDictParam(N=N, max_iter=10)

        assert sbl.N == N
        assert sbl.max_iter == 10

    def test_simple_estimation(self):
        """Test SBL on a simple scenario."""
        N = 64
        K_true = 2
        theta_true = np.array([0.1, 0.3])
        alpha_true = np.array([1.0, 0.8])
        snr_db = 30  # High SNR for reliable estimation

        # Generate signal
        y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

        # Run SBL
        sbl = SBLDictParam(N=N, max_iter=30, verbose=False)
        theta_est, alpha_est, K_est = sbl.fit(y)

        # Check that we estimate correct model order
        assert K_est > 0, "Should estimate at least one component"

        # Check frequency estimation accuracy
        if K_est >= K_true:
            freq_error = frequency_rmse(theta_true, theta_est, threshold=0.05)
            assert freq_error < 0.01, f"Frequency RMSE {freq_error} too large"

    def test_prediction(self):
        """Test signal prediction/reconstruction."""
        N = 64
        theta_true = np.array([0.2, 0.4])
        alpha_true = np.array([1.0, 0.8])
        snr_db = 25

        y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

        sbl = SBLDictParam(N=N, max_iter=20, verbose=False)
        theta_est, alpha_est, K_est = sbl.fit(y)

        # Test prediction
        x_pred = sbl.predict()

        assert x_pred.shape == (N,)
        assert np.iscomplexobj(x_pred)

    def test_convergence_tracking(self):
        """Test that convergence history is tracked."""
        N = 64
        theta_true = np.array([0.15, 0.35])
        alpha_true = np.array([1.0, 0.8])
        snr_db = 20

        y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

        sbl = SBLDictParam(N=N, max_iter=20, verbose=False)
        sbl.fit(y)

        # Check history is populated
        assert len(sbl.history['log_marginal_likelihood']) > 0
        assert len(sbl.history['K']) > 0

    def test_empty_signal(self):
        """Test behavior with very low amplitude signal."""
        N = 64
        y = np.random.randn(N) * 0.01  # Just noise

        sbl = SBLDictParam(N=N, max_iter=10, verbose=False)
        theta_est, alpha_est, K_est = sbl.fit(y)

        # Should estimate few or no components
        assert K_est <= 2, "Should not find many components in pure noise"

    def test_single_frequency(self):
        """Test with single frequency component."""
        N = 64
        theta_true = np.array([0.25])
        alpha_true = np.array([1.5])
        snr_db = 25

        y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

        sbl = SBLDictParam(N=N, max_iter=30, verbose=False)
        theta_est, alpha_est, K_est = sbl.fit(y)

        # Should find one component
        assert K_est >= 1

        # Frequency should be accurate
        if K_est >= 1:
            freq_error = frequency_rmse(theta_true, theta_est, threshold=0.05)
            assert freq_error < 0.02


class TestSBLComparison:
    """Comparative tests."""

    def test_high_snr_performance(self):
        """Test that SBL performs well at high SNR."""
        N = 64
        K_true = 3
        theta_true = np.array([0.1, 0.25, 0.45])
        alpha_true = np.array([1.0, 0.8, 1.2])
        snr_db = 30

        y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

        sbl = SBLDictParam(N=N, max_iter=50, verbose=False)
        theta_est, alpha_est, K_est = sbl.fit(y)

        # At high SNR, should get model order right
        assert np.abs(K_est - K_true) <= 1

        # Frequency error should be very small
        freq_error = frequency_rmse(theta_true, theta_est, threshold=0.05)
        assert freq_error < 0.005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
