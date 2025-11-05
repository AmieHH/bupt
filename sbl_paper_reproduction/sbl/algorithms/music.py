"""
MUSIC (MUltiple SIgnal Classification)

Classical subspace method for frequency estimation using pseudospectrum.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.signal import find_peaks


def music(y: np.ndarray, K: int, grid_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MUSIC algorithm for frequency estimation.

    Computes the MUSIC pseudospectrum and finds peaks corresponding to
    frequency estimates.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    K : int
        Number of frequency components (model order)
    grid_size : int
        Number of grid points for pseudospectrum evaluation

    Returns
    -------
    theta : np.ndarray, shape (K,)
        Estimated frequencies
    amplitudes : np.ndarray, shape (K,)
        Estimated amplitudes
    pseudospectrum : np.ndarray, shape (grid_size,)
        MUSIC pseudospectrum values

    References
    ----------
    Schmidt, R. (1986). Multiple emitter location and signal parameter estimation.
    IEEE transactions on antennas and propagation, 34(3), 276-280.
    """
    N = len(y)

    # Construct covariance matrix estimate
    M = min(N // 2, N - 1)
    Y = np.zeros((M, N - M + 1), dtype=complex)
    for i in range(N - M + 1):
        Y[:, i] = y[i:i + M]

    # Sample covariance matrix
    R = (Y @ Y.conj().T) / (N - M + 1)

    # Forward-backward averaging
    J = np.eye(M)[::-1]
    R = 0.5 * (R + J @ R.conj() @ J)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Noise subspace (M-K smallest eigenvectors)
    if K >= M:
        raise ValueError(f"Model order K={K} must be less than subspace dimension M={M}")

    E_n = eigenvectors[:, K:]

    # Compute MUSIC pseudospectrum on a grid
    freq_grid = np.linspace(0, 1, grid_size, endpoint=False)
    pseudospectrum = np.zeros(grid_size)

    from ..models.atoms import fourier_atom

    for i, f in enumerate(freq_grid):
        # Get steering vector at frequency f
        # Only use first M elements
        a = fourier_atom(f, N)[:M]

        # MUSIC pseudospectrum: P(f) = 1 / (a^H E_n E_n^H a)
        denominator = np.linalg.norm(E_n.conj().T @ a) ** 2
        pseudospectrum[i] = 1.0 / (denominator + 1e-12)

    # Normalize pseudospectrum
    pseudospectrum = pseudospectrum / np.max(pseudospectrum)

    # Find K largest peaks
    peaks, properties = find_peaks(pseudospectrum, distance=grid_size // (4 * K))

    if len(peaks) < K:
        # Not enough peaks found, take K largest values
        peaks = np.argsort(pseudospectrum)[-K:]

    # Sort by peak height and take top K
    peak_heights = pseudospectrum[peaks]
    top_k_idx = np.argsort(peak_heights)[-K:]
    theta = freq_grid[peaks[top_k_idx]]

    # Sort frequencies
    theta = np.sort(theta)

    # Estimate amplitudes using least squares
    from ..models.atoms import fourier_atom
    Psi = np.zeros((N, K), dtype=complex)
    for i in range(K):
        Psi[:, i] = fourier_atom(theta[i], N)

    amplitudes = np.linalg.lstsq(Psi, y, rcond=None)[0]

    return theta, amplitudes, pseudospectrum


class MUSIC:
    """
    MUSIC algorithm wrapper class for consistent interface.
    """

    def __init__(self, K: Optional[int] = None, grid_size: int = 512,
                 auto_order: bool = False, max_K: int = 10):
        """
        Initialize MUSIC estimator.

        Parameters
        ----------
        K : int, optional
            Model order (number of components)
        grid_size : int
            Grid size for pseudospectrum
        auto_order : bool
            Automatically estimate model order
        max_K : int
            Maximum model order for estimation
        """
        self.K = K
        self.grid_size = grid_size
        self.auto_order = auto_order
        self.max_K = max_K

        self.theta = None
        self.alpha = None
        self.pseudospectrum = None

    def fit(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fit MUSIC to the observed signal.

        Parameters
        ----------
        y : np.ndarray
            Observed signal

        Returns
        -------
        theta : np.ndarray
            Estimated frequencies
        alpha : np.ndarray
            Estimated amplitudes
        K : int
            Model order
        """
        # Estimate model order if needed
        if self.auto_order or self.K is None:
            from .esprit import estimate_model_order
            K = estimate_model_order(y, max_K=self.max_K, method='mdl')
        else:
            K = self.K

        # Run MUSIC
        self.theta, self.alpha, self.pseudospectrum = music(y, K, self.grid_size)

        return self.theta, self.alpha, K

    def predict(self) -> np.ndarray:
        """
        Reconstruct signal.

        Returns
        -------
        y_pred : np.ndarray
            Reconstructed signal
        """
        if self.theta is None:
            raise ValueError("Must call fit() first")

        from ..models.signal_model import SignalModel
        N = len(self.alpha)  # Assuming this corresponds to signal length
        model = SignalModel(N)
        x_pred = model.generate_clean_signal(self.theta, self.alpha)

        return x_pred

    def get_pseudospectrum(self, freq_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the MUSIC pseudospectrum.

        Parameters
        ----------
        freq_grid : np.ndarray, optional
            Frequency grid (if None, uses internal grid)

        Returns
        -------
        pseudospectrum : np.ndarray
            Pseudospectrum values
        """
        if self.pseudospectrum is None:
            raise ValueError("Must call fit() first")

        return self.pseudospectrum
