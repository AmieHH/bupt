"""
Unit tests for signal model and atom functions.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbl.models.atoms import fourier_atom, atom_derivative, fourier_dictionary
from sbl.models.signal_model import SignalModel, generate_signal


class TestFourierAtom:
    """Tests for Fourier atom functions."""

    def test_fourier_atom_shape(self):
        """Test that fourier_atom returns correct shape."""
        N = 32
        theta = 0.25
        atom = fourier_atom(theta, N)

        assert atom.shape == (N,)
        assert np.iscomplexobj(atom)

    def test_fourier_atom_normalization(self):
        """Test normalized Fourier atom."""
        N = 32
        theta = 0.25
        atom = fourier_atom(theta, N, normalize=True)

        # Check unit norm
        assert np.abs(np.linalg.norm(atom) - 1.0) < 1e-10

    def test_fourier_atom_values(self):
        """Test Fourier atom values."""
        N = 4
        theta = 0.25

        atom = fourier_atom(theta, N)

        # Manual computation
        expected = np.array([
            np.exp(1j * 2 * np.pi * 0.25 * 0),
            np.exp(1j * 2 * np.pi * 0.25 * 1),
            np.exp(1j * 2 * np.pi * 0.25 * 2),
            np.exp(1j * 2 * np.pi * 0.25 * 3),
        ])

        np.testing.assert_array_almost_equal(atom, expected)

    def test_atom_derivative_shape(self):
        """Test atom derivative shape."""
        N = 32
        theta = 0.25
        d_atom = atom_derivative(theta, N)

        assert d_atom.shape == (N,)
        assert np.iscomplexobj(d_atom)

    def test_fourier_dictionary_shape(self):
        """Test dictionary construction."""
        N = 32
        M = 64
        theta_grid = np.linspace(0, 1, M, endpoint=False)

        D = fourier_dictionary(theta_grid, N)

        assert D.shape == (N, M)
        assert np.iscomplexobj(D)


class TestSignalModel:
    """Tests for signal model."""

    def test_signal_generation(self):
        """Test basic signal generation."""
        N = 64
        theta = np.array([0.1, 0.3])
        alpha = np.array([1.0, 0.8])
        snr_db = 20

        y, x = generate_signal(N, theta, alpha, snr_db, seed=42)

        assert y.shape == (N,)
        assert x.shape == (N,)
        assert np.iscomplexobj(y)
        assert np.iscomplexobj(x)

    def test_signal_model_clean(self):
        """Test clean signal generation."""
        N = 32
        model = SignalModel(N)

        theta = np.array([0.2, 0.5])
        alpha = np.array([1.0 + 0.5j, 0.8 - 0.3j])

        x = model.generate_clean_signal(theta, alpha)

        assert x.shape == (N,)
        assert np.iscomplexobj(x)

    def test_noise_addition(self):
        """Test noise addition."""
        N = 64
        model = SignalModel(N)

        x_clean = np.random.randn(N) + 1j * np.random.randn(N)
        snr_db = 10

        y, noise_var = model.add_noise(x_clean, snr_db, seed=42)

        # Check that noise was added
        assert not np.allclose(y, x_clean)

        # Check SNR is approximately correct
        signal_power = np.mean(np.abs(x_clean) ** 2)
        noise_power = noise_var
        snr_actual = 10 * np.log10(signal_power / noise_power)

        assert np.abs(snr_actual - snr_db) < 0.1

    def test_reproducibility(self):
        """Test that same seed produces same signal."""
        N = 64
        theta = np.array([0.1, 0.3])
        alpha = np.array([1.0, 0.8])
        snr_db = 20

        y1, x1 = generate_signal(N, theta, alpha, snr_db, seed=42)
        y2, x2 = generate_signal(N, theta, alpha, snr_db, seed=42)

        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(x1, x2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
