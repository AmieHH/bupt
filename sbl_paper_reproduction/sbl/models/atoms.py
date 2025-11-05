"""
Atom functions for parameterized dictionaries.

This module implements various atom functions, particularly the Fourier atom
used for spectral estimation.
"""

import numpy as np
from typing import Union, Optional


def fourier_atom(theta: float, N: int, normalize: bool = False) -> np.ndarray:
    """
    Generate a Fourier atom (complex sinusoid) for spectral estimation.

    The Fourier atom is defined as:
        ψ(θ) = [e^(j2πθ·0), e^(j2πθ·1), ..., e^(j2πθ·(N-1))]^T

    Parameters
    ----------
    theta : float
        Normalized frequency parameter in [0, 1)
        (theta = f/fs where f is frequency and fs is sampling rate)
    N : int
        Length of the signal/atom vector
    normalize : bool, optional
        If True, normalize the atom to unit norm (default: False)

    Returns
    -------
    psi : np.ndarray, shape (N,)
        Complex-valued Fourier atom vector

    Examples
    --------
    >>> atom = fourier_atom(0.25, 8)
    >>> print(atom.shape)
    (8,)
    """
    n = np.arange(N)
    psi = np.exp(1j * 2 * np.pi * theta * n)

    if normalize:
        psi = psi / np.linalg.norm(psi)

    return psi


def atom_derivative(theta: float, N: int, normalize: bool = False) -> np.ndarray:
    """
    Compute the derivative of the Fourier atom with respect to theta.

    The derivative is:
        ∂ψ(θ)/∂θ = j2π * [0·e^(j2πθ·0), 1·e^(j2πθ·1), ..., (N-1)·e^(j2πθ·(N-1))]^T

    This is used in the gradient-based optimization of dictionary parameters.

    Parameters
    ----------
    theta : float
        Normalized frequency parameter in [0, 1)
    N : int
        Length of the signal/atom vector
    normalize : bool, optional
        If True, normalize consistently with the atom (default: False)

    Returns
    -------
    dpsi : np.ndarray, shape (N,)
        Derivative of the Fourier atom

    Examples
    --------
    >>> d_atom = atom_derivative(0.25, 8)
    >>> print(d_atom.shape)
    (8,)
    """
    n = np.arange(N)
    dpsi = 1j * 2 * np.pi * n * np.exp(1j * 2 * np.pi * theta * n)

    if normalize:
        # For normalized atoms, we need the derivative of ||ψ||^(-1) * ψ
        # But for Fourier atoms ||ψ|| = sqrt(N) is constant w.r.t. theta
        dpsi = dpsi / np.sqrt(N)

    return dpsi


def fourier_dictionary(theta_grid: np.ndarray, N: int,
                       normalize: bool = False) -> np.ndarray:
    """
    Construct a Fourier dictionary matrix from a grid of frequency parameters.

    This is the traditional gridded approach for comparison purposes.

    Parameters
    ----------
    theta_grid : np.ndarray, shape (M,)
        Array of frequency parameters
    N : int
        Length of each atom (signal length)
    normalize : bool, optional
        If True, normalize each atom to unit norm (default: False)

    Returns
    -------
    Psi : np.ndarray, shape (N, M)
        Dictionary matrix where each column is an atom

    Examples
    --------
    >>> theta_grid = np.linspace(0, 1, 64, endpoint=False)
    >>> D = fourier_dictionary(theta_grid, 32)
    >>> print(D.shape)
    (32, 64)
    """
    M = len(theta_grid)
    Psi = np.zeros((N, M), dtype=complex)

    for i, theta in enumerate(theta_grid):
        Psi[:, i] = fourier_atom(theta, N, normalize=normalize)

    return Psi


def compute_gram_matrix(Psi: np.ndarray) -> np.ndarray:
    """
    Compute the Gram matrix (dictionary coherence matrix).

    G = Ψ^H Ψ where Ψ is the dictionary matrix.

    Parameters
    ----------
    Psi : np.ndarray, shape (N, M)
        Dictionary matrix

    Returns
    -------
    G : np.ndarray, shape (M, M)
        Gram matrix
    """
    return Psi.conj().T @ Psi


def compute_mutual_coherence(Psi: np.ndarray) -> float:
    """
    Compute the mutual coherence of a dictionary.

    μ = max_{i≠j} |⟨ψ_i, ψ_j⟩| / (||ψ_i|| ||ψ_j||)

    High coherence indicates that dictionary atoms are similar,
    which can cause problems in sparse recovery.

    Parameters
    ----------
    Psi : np.ndarray, shape (N, M)
        Dictionary matrix

    Returns
    -------
    mu : float
        Mutual coherence value
    """
    # Normalize columns
    Psi_norm = Psi / np.linalg.norm(Psi, axis=0, keepdims=True)

    # Compute Gram matrix
    G = np.abs(Psi_norm.conj().T @ Psi_norm)

    # Set diagonal to zero (we want max off-diagonal element)
    np.fill_diagonal(G, 0)

    mu = np.max(G)

    return mu


def dirac_delta_representation(theta: np.ndarray, alpha: np.ndarray,
                                 theta_grid: np.ndarray) -> np.ndarray:
    """
    Represent a sparse signal as delta functions on a grid.

    Useful for visualization and comparison.

    Parameters
    ----------
    theta : np.ndarray, shape (K,)
        True frequency parameters
    alpha : np.ndarray, shape (K,)
        Coefficients/amplitudes
    theta_grid : np.ndarray, shape (M,)
        Grid for representation

    Returns
    -------
    spectrum : np.ndarray, shape (M,)
        Discrete representation of the spectrum
    """
    spectrum = np.zeros(len(theta_grid), dtype=complex)

    for t, a in zip(theta, alpha):
        # Find nearest grid point
        idx = np.argmin(np.abs(theta_grid - t))
        spectrum[idx] += a

    return spectrum


def generate_random_theta(K: int, min_separation: Optional[float] = None,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random frequency parameters with optional minimum separation.

    Parameters
    ----------
    K : int
        Number of frequencies to generate
    min_separation : float, optional
        Minimum separation between frequencies (default: None)
        If None, no separation constraint is enforced
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    theta : np.ndarray, shape (K,)
        Sorted array of frequency parameters in [0, 1)

    Examples
    --------
    >>> theta = generate_random_theta(3, min_separation=0.05, seed=42)
    >>> print(len(theta))
    3
    """
    if seed is not None:
        np.random.seed(seed)

    if min_separation is None:
        # Generate completely random frequencies
        theta = np.sort(np.random.rand(K))
    else:
        # Generate with minimum separation constraint
        if K * min_separation > 1.0:
            raise ValueError("Cannot generate K frequencies with specified minimum separation")

        theta = []
        attempts = 0
        max_attempts = 1000

        while len(theta) < K and attempts < max_attempts:
            # Generate candidate frequency
            candidate = np.random.rand()

            # Check separation from existing frequencies
            if len(theta) == 0:
                theta.append(candidate)
            else:
                # Check minimum distance considering wrap-around at 0 and 1
                min_dist = min([
                    min(abs(candidate - t), 1 - abs(candidate - t))
                    for t in theta
                ])

                if min_dist >= min_separation:
                    theta.append(candidate)

            attempts += 1

        if len(theta) < K:
            raise ValueError(f"Could not generate {K} frequencies with separation {min_separation}")

        theta = np.sort(theta)

    return theta
