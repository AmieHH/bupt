"""
ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)

Classical subspace method for frequency estimation.
"""

import numpy as np
from typing import Tuple, Optional


def esprit(y: np.ndarray, K: int, method: str = 'ls') -> Tuple[np.ndarray, np.ndarray]:
    """
    ESPRIT algorithm for frequency estimation.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    K : int
        Number of frequency components (model order)
    method : str
        'ls': Least squares ESPRIT
        'tls': Total least squares ESPRIT

    Returns
    -------
    theta : np.ndarray, shape (K,)
        Estimated normalized frequencies in [0, 1)
    amplitudes : np.ndarray, shape (K,)
        Estimated amplitudes

    References
    ----------
    Roy, R., & Kailath, T. (1989). ESPRIT-estimation of signal parameters
    via rotational invariance techniques. IEEE Transactions on acoustics,
    speech, and signal processing, 37(7), 984-995.
    """
    N = len(y)

    # Step 1: Construct data matrix (Hankel-like structure)
    # For a single snapshot, we use forward-backward averaging
    M = N // 2  # Subarray size

    # Construct data matrix from signal
    Y = np.zeros((M, N - M + 1), dtype=complex)
    for i in range(N - M + 1):
        Y[:, i] = y[i:i + M]

    # Step 2: Compute covariance matrix estimate
    R = (Y @ Y.conj().T) / (N - M + 1)

    # Forward-backward averaging for improved performance
    J = np.eye(M)[::-1]  # Exchange matrix
    R_fb = 0.5 * (R + J @ R.conj() @ J)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R_fb)

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select signal subspace (K largest eigenvalues)
    E_s = eigenvectors[:, :K]

    # Step 4: ESPRIT frequency estimation
    # Split signal subspace into two subarrays
    E1 = E_s[:-1, :]  # First M-1 rows
    E2 = E_s[1:, :]   # Last M-1 rows

    # Compute the rotation matrix
    if method == 'ls':
        # Least Squares ESPRIT
        Phi = np.linalg.lstsq(E1, E2, rcond=None)[0]
    elif method == 'tls':
        # Total Least Squares ESPRIT
        C = np.vstack([E1, E2])
        _, _, V = np.linalg.svd(C, full_matrices=True)
        V12 = V[K:, :K].conj().T
        V22 = V[K:, K:].conj().T
        Phi = -V12 @ np.linalg.inv(V22)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 5: Extract frequencies from eigenvalues of Phi
    eigenvalues_phi = np.linalg.eigvals(Phi)

    # Convert to frequencies
    # e^(j2πθ) = eigenvalue => θ = angle / (2π)
    angles = np.angle(eigenvalues_phi)
    theta = angles / (2 * np.pi)

    # Wrap to [0, 1)
    theta = np.mod(theta, 1.0)

    # Sort frequencies
    theta = np.sort(theta)

    # Step 6: Estimate amplitudes using least squares
    # Construct dictionary at estimated frequencies
    from ..models.atoms import fourier_atom
    Psi = np.zeros((N, K), dtype=complex)
    for i in range(K):
        Psi[:, i] = fourier_atom(theta[i], N)

    # Least squares amplitude estimation
    amplitudes = np.linalg.lstsq(Psi, y, rcond=None)[0]

    return theta, amplitudes


def root_music(y: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Root-MUSIC algorithm for frequency estimation.

    Similar to ESPRIT but uses polynomial rooting instead of eigenvalue decomposition
    of the rotation matrix.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    K : int
        Number of frequency components

    Returns
    -------
    theta : np.ndarray, shape (K,)
        Estimated normalized frequencies
    amplitudes : np.ndarray, shape (K,)
        Estimated amplitudes
    """
    N = len(y)

    # Construct covariance matrix estimate
    # Using Toeplitz structure for single snapshot
    r = np.correlate(y, y, mode='full')
    r = r[N - 1:]  # Take positive lags

    # Construct Toeplitz covariance matrix
    from scipy.linalg import toeplitz
    R = toeplitz(r[:N])

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Noise subspace
    E_n = eigenvectors[:, K:]

    # Form the polynomial for root finding
    # P(z) = sum_k |sum_n E_n[k,n] z^(-n)|^2
    # This is equivalent to finding zeros of a polynomial

    # Simplified approach: use pseudospectrum peaks
    freq_grid = np.linspace(0, 1, 1024, endpoint=False)
    spectrum = np.zeros(len(freq_grid))

    from ..models.atoms import fourier_atom
    for i, f in enumerate(freq_grid):
        a = fourier_atom(f, N)
        # MUSIC pseudospectrum: 1 / (a^H E_n E_n^H a)
        spectrum[i] = 1.0 / (np.linalg.norm(E_n.conj().T @ a) ** 2 + 1e-12)

    # Find K largest peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, distance=5)

    if len(peaks) < K:
        # If not enough peaks found, take K largest values
        peaks = np.argsort(spectrum)[-K:]

    # Sort by spectrum value and take top K
    peak_values = spectrum[peaks]
    top_k_indices = np.argsort(peak_values)[-K:]
    theta = freq_grid[peaks[top_k_indices]]

    # Sort frequencies
    theta = np.sort(theta)

    # Estimate amplitudes
    Psi = np.zeros((N, K), dtype=complex)
    for i in range(K):
        Psi[:, i] = fourier_atom(theta[i], N)

    amplitudes = np.linalg.lstsq(Psi, y, rcond=None)[0]

    return theta, amplitudes


def estimate_model_order(y: np.ndarray, max_K: int = 10,
                         method: str = 'aic') -> int:
    """
    Estimate the model order (number of frequency components) using
    information theoretic criteria.

    Parameters
    ----------
    y : np.ndarray
        Observed signal
    max_K : int
        Maximum model order to consider
    method : str
        'aic': Akaike Information Criterion
        'mdl': Minimum Description Length
        'eft': Exponential Fitting Test

    Returns
    -------
    K_est : int
        Estimated model order
    """
    N = len(y)

    # Construct covariance matrix
    M = N // 2
    Y = np.zeros((M, N - M + 1), dtype=complex)
    for i in range(N - M + 1):
        Y[:, i] = y[i:i + M]

    R = (Y @ Y.conj().T) / (N - M + 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = np.maximum(eigenvalues, 1e-12)  # Avoid numerical issues

    # Compute criteria
    criteria = np.zeros(max_K + 1)

    for k in range(max_K + 1):
        # Noise eigenvalues
        noise_eigenvalues = eigenvalues[k:]
        L = len(noise_eigenvalues)

        if L == 0:
            criteria[k] = np.inf
            continue

        # Geometric and arithmetic means
        geometric_mean = np.exp(np.mean(np.log(noise_eigenvalues)))
        arithmetic_mean = np.mean(noise_eigenvalues)

        if method == 'aic':
            # AIC criterion
            criteria[k] = -2 * (N - M + 1) * L * np.log(geometric_mean / arithmetic_mean) + 2 * k * (2 * M - k)

        elif method == 'mdl':
            # MDL criterion
            criteria[k] = -(N - M + 1) * L * np.log(geometric_mean / arithmetic_mean) + 0.5 * k * (2 * M - k) * np.log(
                N - M + 1)

        else:
            raise ValueError(f"Unknown method: {method}")

    # Find minimum
    K_est = np.argmin(criteria)

    return K_est


class ESPRIT:
    """
    ESPRIT algorithm wrapper class for consistent interface.
    """

    def __init__(self, K: Optional[int] = None, method: str = 'ls',
                 auto_order: bool = False, max_K: int = 10):
        """
        Initialize ESPRIT estimator.

        Parameters
        ----------
        K : int, optional
            Model order (number of components)
            If None and auto_order=True, will be estimated
        method : str
            'ls' or 'tls'
        auto_order : bool
            If True, automatically estimate model order
        max_K : int
            Maximum model order for automatic estimation
        """
        self.K = K
        self.method = method
        self.auto_order = auto_order
        self.max_K = max_K

        self.theta = None
        self.alpha = None

    def fit(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fit ESPRIT to the observed signal.

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
            Model order used
        """
        # Estimate model order if needed
        if self.auto_order or self.K is None:
            K = estimate_model_order(y, max_K=self.max_K, method='mdl')
        else:
            K = self.K

        # Run ESPRIT
        self.theta, self.alpha = esprit(y, K, method=self.method)

        return self.theta, self.alpha, K

    def predict(self) -> np.ndarray:
        """
        Reconstruct signal using estimated parameters.

        Returns
        -------
        y_pred : np.ndarray
            Reconstructed signal
        """
        if self.theta is None:
            raise ValueError("Must call fit() first")

        from ..models.signal_model import SignalModel
        model = SignalModel(len(self.alpha))
        x_pred = model.generate_clean_signal(self.theta, self.alpha)

        return x_pred
