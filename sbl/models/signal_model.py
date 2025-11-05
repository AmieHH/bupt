"""
Signal model for sparse representation and generation.

This module provides classes and functions for generating test signals
and managing the signal model.
"""

import numpy as np
from typing import Tuple, Optional, Union
from .atoms import fourier_atom, fourier_dictionary


class SignalModel:
    """
    Signal model for sparse representation.

    The signal model is:
        x = Σ ψ(θᵢ)αᵢ,  i = 1,...,K
        y = x + w

    where:
        - x: clean signal
        - y: noisy observation
        - ψ(θᵢ): atom function with parameter θᵢ
        - αᵢ: coefficient/amplitude
        - w: additive white Gaussian noise
        - K: model order (number of atoms)
    """

    def __init__(self, N: int, atom_func=None):
        """
        Initialize the signal model.

        Parameters
        ----------
        N : int
            Signal length
        atom_func : callable, optional
            Atom function ψ(θ, N) that returns an N-dimensional vector
            Default is Fourier atom
        """
        self.N = N
        self.atom_func = atom_func if atom_func is not None else fourier_atom

    def generate_clean_signal(self, theta: np.ndarray,
                              alpha: np.ndarray) -> np.ndarray:
        """
        Generate clean signal without noise.

        Parameters
        ----------
        theta : np.ndarray, shape (K,)
            Atom parameters
        alpha : np.ndarray, shape (K,)
            Coefficients (can be complex)

        Returns
        -------
        x : np.ndarray, shape (N,)
            Clean signal
        """
        K = len(theta)
        x = np.zeros(self.N, dtype=complex)

        for i in range(K):
            x += alpha[i] * self.atom_func(theta[i], self.N)

        return x

    def add_noise(self, x: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Add white Gaussian noise to signal.

        Parameters
        ----------
        x : np.ndarray
            Clean signal
        snr_db : float
            Signal-to-noise ratio in dB
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        y : np.ndarray
            Noisy signal
        noise_var : float
            Noise variance (λ^(-1) in the paper)
        """
        if seed is not None:
            np.random.seed(seed)

        # Compute signal power
        signal_power = np.mean(np.abs(x) ** 2)

        # Compute noise variance from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_var = signal_power / snr_linear

        # Generate complex Gaussian noise
        if np.iscomplexobj(x):
            # Complex noise: real and imaginary parts are independent
            noise = np.sqrt(noise_var / 2) * (
                np.random.randn(self.N) + 1j * np.random.randn(self.N)
            )
        else:
            # Real noise
            noise = np.sqrt(noise_var) * np.random.randn(self.N)

        y = x + noise

        return y, noise_var

    def generate_signal(self, theta: np.ndarray, alpha: np.ndarray,
                       snr_db: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate noisy signal observation.

        Parameters
        ----------
        theta : np.ndarray, shape (K,)
            Atom parameters
        alpha : np.ndarray, shape (K,)
            Coefficients
        snr_db : float
            Signal-to-noise ratio in dB
        seed : int, optional
            Random seed

        Returns
        -------
        y : np.ndarray, shape (N,)
            Noisy observation
        x : np.ndarray, shape (N,)
            Clean signal
        noise_var : float
            Noise variance
        """
        x = self.generate_clean_signal(theta, alpha)
        y, noise_var = self.add_noise(x, snr_db, seed)

        return y, x, noise_var


def generate_signal(N: int, theta: np.ndarray, alpha: np.ndarray,
                    snr_db: float, seed: Optional[int] = None,
                    atom_func=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate a noisy signal.

    Parameters
    ----------
    N : int
        Signal length
    theta : np.ndarray, shape (K,)
        Atom parameters (e.g., normalized frequencies)
    alpha : np.ndarray, shape (K,)
        Coefficients/amplitudes (can be complex)
    snr_db : float
        Signal-to-noise ratio in dB
    seed : int, optional
        Random seed for reproducibility
    atom_func : callable, optional
        Atom function (default: Fourier atom)

    Returns
    -------
    y : np.ndarray, shape (N,)
        Noisy observation
    x : np.ndarray, shape (N,)
        Clean signal

    Examples
    --------
    >>> N = 64
    >>> theta = np.array([0.1, 0.2, 0.3])
    >>> alpha = np.array([1.0, 0.8, 1.2])
    >>> y, x = generate_signal(N, theta, alpha, snr_db=20, seed=42)
    >>> print(y.shape, x.shape)
    (64,) (64,)
    """
    model = SignalModel(N, atom_func)
    y, x, _ = model.generate_signal(theta, alpha, snr_db, seed)

    return y, x


def compute_snr(x: np.ndarray, noise_var: float) -> float:
    """
    Compute signal-to-noise ratio in dB.

    Parameters
    ----------
    x : np.ndarray
        Clean signal
    noise_var : float
        Noise variance

    Returns
    -------
    snr_db : float
        SNR in decibels
    """
    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = signal_power / noise_var
    snr_db = 10 * np.log10(snr_linear)

    return snr_db


def generate_test_scenario(N: int, K: int, snr_db: float,
                           freq_separation: Optional[float] = None,
                           amplitude_range: Tuple[float, float] = (0.5, 1.5),
                           seed: Optional[int] = None) -> dict:
    """
    Generate a complete test scenario for algorithm evaluation.

    Parameters
    ----------
    N : int
        Signal length
    K : int
        Number of signal components
    snr_db : float
        Signal-to-noise ratio in dB
    freq_separation : float, optional
        Minimum frequency separation (default: 2/N for Rayleigh limit)
    amplitude_range : tuple of float
        (min, max) range for random amplitudes
    seed : int, optional
        Random seed

    Returns
    -------
    scenario : dict
        Dictionary containing:
        - 'y': noisy observation
        - 'x': clean signal
        - 'theta': true frequencies
        - 'alpha': true amplitudes
        - 'N': signal length
        - 'K': model order
        - 'snr_db': SNR
        - 'noise_var': noise variance

    Examples
    --------
    >>> scenario = generate_test_scenario(N=64, K=3, snr_db=20, seed=42)
    >>> print(scenario['theta'])
    """
    if seed is not None:
        np.random.seed(seed)

    # Set default frequency separation (Rayleigh limit)
    if freq_separation is None:
        freq_separation = 2.0 / N

    # Generate random frequencies with minimum separation
    from .atoms import generate_random_theta
    theta = generate_random_theta(K, min_separation=freq_separation, seed=seed)

    # Generate random amplitudes (complex with random phase)
    amp_magnitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], K)
    phases = np.random.uniform(0, 2 * np.pi, K)
    alpha = amp_magnitudes * np.exp(1j * phases)

    # Generate signal
    model = SignalModel(N)
    y, x, noise_var = model.generate_signal(theta, alpha, snr_db, seed)

    scenario = {
        'y': y,
        'x': x,
        'theta': theta,
        'alpha': alpha,
        'N': N,
        'K': K,
        'snr_db': snr_db,
        'noise_var': noise_var,
        'freq_separation': freq_separation,
    }

    return scenario


def generate_closely_spaced_scenario(N: int, K: int, snr_db: float,
                                      center_freq: float = 0.25,
                                      freq_span: float = None,
                                      seed: Optional[int] = None) -> dict:
    """
    Generate a challenging scenario with closely-spaced frequencies.

    This is particularly useful for testing super-resolution capabilities,
    as mentioned in the paper.

    Parameters
    ----------
    N : int
        Signal length
    K : int
        Number of closely-spaced components
    snr_db : float
        Signal-to-noise ratio in dB
    center_freq : float
        Center of the frequency cluster (default: 0.25)
    freq_span : float, optional
        Total span of frequencies (default: K/N, around Rayleigh limit)
    seed : int, optional
        Random seed

    Returns
    -------
    scenario : dict
        Test scenario dictionary

    Examples
    --------
    >>> # Generate 3 components within Rayleigh limit
    >>> scenario = generate_closely_spaced_scenario(N=64, K=3, snr_db=20, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    # Default span is around Rayleigh limit
    if freq_span is None:
        freq_span = K / N

    # Generate closely-spaced frequencies
    # Place them uniformly within the span
    theta = center_freq + freq_span * (np.arange(K) / K - 0.5)

    # Wrap to [0, 1) if necessary
    theta = np.mod(theta, 1.0)

    # Sort frequencies
    theta = np.sort(theta)

    # Generate random amplitudes
    amp_magnitudes = np.random.uniform(0.7, 1.3, K)
    phases = np.random.uniform(0, 2 * np.pi, K)
    alpha = amp_magnitudes * np.exp(1j * phases)

    # Generate signal
    model = SignalModel(N)
    y, x, noise_var = model.generate_signal(theta, alpha, snr_db, seed)

    scenario = {
        'y': y,
        'x': x,
        'theta': theta,
        'alpha': alpha,
        'N': N,
        'K': K,
        'snr_db': snr_db,
        'noise_var': noise_var,
        'center_freq': center_freq,
        'freq_span': freq_span,
    }

    return scenario
