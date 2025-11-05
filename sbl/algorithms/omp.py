"""
OMP (Orthogonal Matching Pursuit)

Greedy algorithm for sparse signal recovery on a fixed dictionary.
This represents the traditional gridded approach for comparison.
"""

import numpy as np
from typing import Tuple, Optional


def omp(y: np.ndarray, Dictionary: np.ndarray, K: int,
        tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Matching Pursuit algorithm.

    Greedy algorithm that iteratively selects atoms from a fixed dictionary
    to represent the signal.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    Dictionary : np.ndarray, shape (N, M)
        Dictionary matrix (columns are atoms)
    K : int
        Number of atoms to select (sparsity level)
    tol : float
        Residual tolerance for early stopping

    Returns
    -------
    indices : np.ndarray, shape (K,)
        Indices of selected dictionary atoms
    coefficients : np.ndarray, shape (K,)
        Coefficients for selected atoms

    References
    ----------
    Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random measurements
    via orthogonal matching pursuit. IEEE Transactions on information theory, 53(12).
    """
    N, M = Dictionary.shape
    residual = y.copy()
    indices = []
    coefficients = []

    for k in range(K):
        # Compute correlations with all dictionary atoms
        correlations = np.abs(Dictionary.conj().T @ residual)

        # Find atom with maximum correlation
        max_idx = np.argmax(correlations)

        # Add to selected set
        indices.append(max_idx)

        # Update coefficients using least squares on selected atoms
        Psi_selected = Dictionary[:, indices]
        coeffs = np.linalg.lstsq(Psi_selected, y, rcond=None)[0]

        # Update residual
        residual = y - Psi_selected @ coeffs

        # Check stopping criterion
        if np.linalg.norm(residual) < tol:
            break

    # Final least squares fit
    Psi_selected = Dictionary[:, indices]
    coefficients = np.linalg.lstsq(Psi_selected, y, rcond=None)[0]

    return np.array(indices), coefficients


def omp_spectral(y: np.ndarray, K: int, grid_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    OMP for spectral estimation using a Fourier dictionary.

    This is the traditional gridded approach that suffers from basis mismatch.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    K : int
        Number of frequency components
    grid_size : int
        Number of grid points (dictionary size)

    Returns
    -------
    theta : np.ndarray, shape (K,)
        Estimated frequencies (from grid)
    amplitudes : np.ndarray, shape (K,)
        Estimated amplitudes
    """
    N = len(y)

    # Construct Fourier dictionary on a uniform grid
    from ..models.atoms import fourier_dictionary
    freq_grid = np.linspace(0, 1, grid_size, endpoint=False)
    Dictionary = fourier_dictionary(freq_grid, N)

    # Run OMP
    indices, coefficients = omp(y, Dictionary, K)

    # Extract frequencies from selected indices
    theta = freq_grid[indices]

    # Sort by frequency
    sort_idx = np.argsort(theta)
    theta = theta[sort_idx]
    amplitudes = coefficients[sort_idx]

    return theta, amplitudes


class OMP:
    """
    OMP algorithm wrapper class for consistent interface.
    """

    def __init__(self, K: int, grid_size: int = 256, tol: float = 1e-6):
        """
        Initialize OMP estimator.

        Parameters
        ----------
        K : int
            Sparsity level (number of components)
        grid_size : int
            Dictionary grid size
        tol : float
            Residual tolerance
        """
        self.K = K
        self.grid_size = grid_size
        self.tol = tol

        self.theta = None
        self.alpha = None

    def fit(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fit OMP to the observed signal.

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
            Number of components
        """
        self.theta, self.alpha = omp_spectral(y, self.K, self.grid_size)

        return self.theta, self.alpha, self.K

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
        # Infer signal length from alpha
        # This is a workaround; ideally we'd store N
        N = len(self.alpha) * 4  # Rough estimate
        model = SignalModel(N)
        x_pred = model.generate_clean_signal(self.theta, self.alpha)

        return x_pred


def basis_pursuit(y: np.ndarray, Dictionary: np.ndarray,
                  lambda_reg: float = 0.1) -> np.ndarray:
    """
    Basis Pursuit Denoising (BPDN) using LASSO.

    Solves: min_x ||y - Dx||^2 + λ||x||_1

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observed signal
    Dictionary : np.ndarray, shape (N, M)
        Dictionary matrix
    lambda_reg : float
        Regularization parameter

    Returns
    -------
    x : np.ndarray, shape (M,)
        Sparse coefficient vector
    """
    try:
        from sklearn.linear_model import Lasso

        # Use LASSO from scikit-learn
        # Note: For complex signals, we need to handle real/imaginary parts separately
        if np.iscomplexobj(y):
            # Real part
            lasso_real = Lasso(alpha=lambda_reg, max_iter=5000, tol=1e-4)
            lasso_real.fit(Dictionary.real, y.real)
            x_real = lasso_real.coef_

            # Imaginary part
            lasso_imag = Lasso(alpha=lambda_reg, max_iter=5000, tol=1e-4)
            lasso_imag.fit(Dictionary.imag, y.imag)
            x_imag = lasso_imag.coef_

            x = x_real + 1j * x_imag
        else:
            lasso = Lasso(alpha=lambda_reg, max_iter=5000, tol=1e-4)
            lasso.fit(Dictionary, y)
            x = lasso.coef_

    except ImportError:
        # Fallback: Use simple iterative soft thresholding
        print("Warning: scikit-learn not available, using simple ISTA")
        x = iterative_soft_thresholding(y, Dictionary, lambda_reg)

    return x


def iterative_soft_thresholding(y: np.ndarray, Dictionary: np.ndarray,
                                 lambda_reg: float, max_iter: int = 1000,
                                 tol: float = 1e-6) -> np.ndarray:
    """
    Iterative Soft Thresholding Algorithm (ISTA) for LASSO.

    Parameters
    ----------
    y : np.ndarray
        Observed signal
    Dictionary : np.ndarray
        Dictionary matrix
    lambda_reg : float
        Regularization parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    x : np.ndarray
        Sparse coefficient vector
    """
    N, M = Dictionary.shape

    # Initialize
    x = np.zeros(M, dtype=complex if np.iscomplexobj(y) else float)

    # Compute step size (inverse of Lipschitz constant)
    L = np.linalg.norm(Dictionary.conj().T @ Dictionary, 2)
    step_size = 1.0 / (L + 1e-8)

    # Soft thresholding function
    def soft_threshold(z, threshold):
        if np.iscomplexobj(z):
            magnitude = np.abs(z)
            scale = np.maximum(magnitude - threshold, 0) / (magnitude + 1e-12)
            return z * scale
        else:
            return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    # ISTA iterations
    for iter_idx in range(max_iter):
        x_old = x.copy()

        # Gradient step
        gradient = Dictionary.conj().T @ (Dictionary @ x - y)
        x = x - step_size * gradient

        # Proximal step (soft thresholding)
        x = soft_threshold(x, step_size * lambda_reg)

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x


class BasisPursuit:
    """
    Basis Pursuit wrapper for consistent interface.
    """

    def __init__(self, grid_size: int = 256, lambda_reg: float = 0.1):
        """
        Initialize Basis Pursuit estimator.

        Parameters
        ----------
        grid_size : int
            Dictionary grid size
        lambda_reg : float
            Regularization parameter
        """
        self.grid_size = grid_size
        self.lambda_reg = lambda_reg

        self.theta = None
        self.alpha = None

    def fit(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fit using Basis Pursuit.

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
            Number of components
        """
        N = len(y)

        # Construct dictionary
        from ..models.atoms import fourier_dictionary
        freq_grid = np.linspace(0, 1, self.grid_size, endpoint=False)
        Dictionary = fourier_dictionary(freq_grid, N)

        # Solve LASSO
        x = basis_pursuit(y, Dictionary, self.lambda_reg)

        # Extract non-zero coefficients
        threshold = np.max(np.abs(x)) * 0.01  # 1% of maximum
        significant = np.abs(x) > threshold

        self.theta = freq_grid[significant]
        self.alpha = x[significant]

        # Sort by frequency
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.alpha = self.alpha[sort_idx]

        return self.theta, self.alpha, len(self.theta)

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
        N = len(self.alpha) * 4
        model = SignalModel(N)
        x_pred = model.generate_clean_signal(self.theta, self.alpha)

        return x_pred
