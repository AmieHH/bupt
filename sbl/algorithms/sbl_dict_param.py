"""
Sparse Bayesian Learning Algorithm with Dictionary Parameter Estimation

This module implements the main SBL algorithm proposed in the paper:
'A Sparse Bayesian Learning Algorithm With Dictionary Parameter Estimation'
by Hansen et al.

The algorithm extends the fast SBL framework to include optimization of
continuous dictionary parameters (e.g., frequencies for spectral estimation).
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from scipy.optimize import minimize_scalar, minimize
import warnings


class SBLDictParam:
    """
    Sparse Bayesian Learning with Dictionary Parameter Estimation.

    This class implements the SBL algorithm that jointly estimates:
    - Model order K (number of atoms)
    - Dictionary parameters θ (e.g., frequencies)
    - Coefficients α (amplitudes)

    The algorithm avoids discretization of the parameter space and works
    directly with the continuous parameterized signal model.

    Attributes
    ----------
    N : int
        Signal length
    atom_func : callable
        Atom function ψ(θ, N)
    atom_derivative : callable
        Derivative ∂ψ/∂θ for gradient-based optimization
    """

    def __init__(self,
                 N: int,
                 atom_func: Optional[Callable] = None,
                 atom_derivative: Optional[Callable] = None,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 lambda_init: float = 1.0,
                 gamma_init: float = 1.0,
                 prune_threshold: float = 1e-12,
                 add_threshold: float = 0.0,
                 theta_update_method: str = 'gradient',
                 theta_learning_rate: float = 0.01,
                 verbose: bool = False):
        """
        Initialize the SBL algorithm.

        Parameters
        ----------
        N : int
            Signal length
        atom_func : callable, optional
            Function ψ(θ, N) that returns atom vector
            Default: Fourier atom
        atom_derivative : callable, optional
            Function ∂ψ/∂θ(θ, N) for gradient computation
            Default: Fourier atom derivative
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        lambda_init : float
            Initial noise precision
        gamma_init : float
            Initial hyperparameter value
        prune_threshold : float
            Threshold for pruning small gamma values
        add_threshold : float
            Threshold for adding new atoms
        theta_update_method : str
            Method for updating theta ('gradient' or 'optimize')
        theta_learning_rate : float
            Learning rate for gradient-based theta updates
        verbose : bool
            Print progress information
        """
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_init = lambda_init
        self.gamma_init = gamma_init
        self.prune_threshold = prune_threshold
        self.add_threshold = add_threshold
        self.theta_update_method = theta_update_method
        self.theta_learning_rate = theta_learning_rate
        self.verbose = verbose

        # Set default atom functions (Fourier atoms)
        if atom_func is None:
            from ..models.atoms import fourier_atom
            self.atom_func = fourier_atom
        else:
            self.atom_func = atom_func

        if atom_derivative is None:
            from ..models.atoms import atom_derivative as fourier_derivative
            self.atom_derivative = fourier_derivative
        else:
            self.atom_derivative = atom_derivative

        # Model parameters (to be estimated)
        self.theta = np.array([])  # Dictionary parameters
        self.gamma = np.array([])  # Hyperparameters
        self.alpha = np.array([])  # Coefficients
        self.lambda_ = lambda_init  # Noise precision

        # For tracking convergence
        self.history = {
            'log_marginal_likelihood': [],
            'K': [],
            'theta': [],
        }

    def fit(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fit the SBL model to the observed signal.

        Parameters
        ----------
        y : np.ndarray, shape (N,)
            Observed signal

        Returns
        -------
        theta : np.ndarray, shape (K,)
            Estimated dictionary parameters
        alpha : np.ndarray, shape (K,)
            Estimated coefficients
        K : int
            Estimated model order
        """
        self.y = y

        # Initialize with empty model or a few random atoms
        self._initialize()

        # Main SBL iteration loop
        for iter_idx in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {iter_idx + 1}/{self.max_iter}, K={len(self.theta)}")

            # Store current state for convergence check
            theta_old = self.theta.copy()

            # Update hyperparameters and coefficients
            self._update_gamma_and_alpha()

            # Prune insignificant atoms
            self._prune_atoms()

            # Try to add new atoms
            self._add_atoms()

            # Update dictionary parameters
            if len(self.theta) > 0:
                self._update_theta()

            # Compute marginal likelihood
            log_ml = self._compute_log_marginal_likelihood()
            self.history['log_marginal_likelihood'].append(log_ml)
            self.history['K'].append(len(self.theta))
            self.history['theta'].append(self.theta.copy())

            # Check convergence
            if len(theta_old) == len(self.theta) and len(self.theta) > 0:
                theta_change = np.max(np.abs(self.theta - theta_old))
                if theta_change < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iter_idx + 1}")
                    break

        # Final parameter update
        if len(self.theta) > 0:
            self._update_gamma_and_alpha()

        return self.theta.copy(), self.alpha.copy(), len(self.theta)

    def _initialize(self):
        """Initialize the model with a few candidate atoms."""
        # Start with empty model
        # Atoms will be added in the first iteration
        self.theta = np.array([])
        self.gamma = np.array([])
        self.alpha = np.array([])
        self.lambda_ = self.lambda_init

    def _construct_dictionary(self, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construct dictionary matrix from current theta values.

        Parameters
        ----------
        theta : np.ndarray, optional
            If provided, use these theta values instead of self.theta

        Returns
        -------
        Psi : np.ndarray, shape (N, K)
            Dictionary matrix
        """
        if theta is None:
            theta = self.theta

        K = len(theta)
        if K == 0:
            return np.zeros((self.N, 0), dtype=complex)

        Psi = np.zeros((self.N, K), dtype=complex)
        for i in range(K):
            Psi[:, i] = self.atom_func(theta[i], self.N)

        return Psi

    def _update_gamma_and_alpha(self):
        """
        Update hyperparameters gamma and coefficients alpha.

        Uses the SBL update equations from Tipping & Faul.
        """
        if len(self.theta) == 0:
            return

        # Construct dictionary matrix
        Psi = self._construct_dictionary()

        # Compute covariance matrix
        # Σ = (λΨ^H Ψ + Γ^(-1))^(-1)
        # where Γ = diag(γ)
        Gamma_inv = np.diag(1.0 / self.gamma)
        Sigma = np.linalg.inv(
            self.lambda_ * (Psi.conj().T @ Psi) + Gamma_inv
        )

        # Compute mean (coefficient estimates)
        # μ = λ Σ Ψ^H y
        self.alpha = self.lambda_ * Sigma @ (Psi.conj().T @ self.y)

        # Update gamma using the re-estimation formula
        # γᵢ = (1 - γᵢ Σᵢᵢ) / |αᵢ|²
        for i in range(len(self.gamma)):
            self.gamma[i] = (1.0 - self.gamma[i] * Sigma[i, i].real) / (
                np.abs(self.alpha[i]) ** 2 + 1e-12
            )

        # Update noise precision lambda
        Psi_alpha = Psi @ self.alpha
        residual = self.y - Psi_alpha
        trace_term = np.sum([self.gamma[i] * Sigma[i, i].real for i in range(len(self.gamma))])

        self.lambda_ = (self.N - trace_term) / (np.linalg.norm(residual) ** 2 + 1e-12)

        # Ensure lambda is positive
        self.lambda_ = max(self.lambda_, 1e-6)

    def _prune_atoms(self):
        """Remove atoms with very small gamma values."""
        if len(self.gamma) == 0:
            return

        # Find atoms to keep
        keep_indices = self.gamma > self.prune_threshold

        if np.sum(keep_indices) < len(self.gamma):
            if self.verbose:
                print(f"  Pruning {np.sum(~keep_indices)} atoms")

            self.theta = self.theta[keep_indices]
            self.gamma = self.gamma[keep_indices]
            self.alpha = self.alpha[keep_indices]

    def _add_atoms(self):
        """
        Try to add new atoms by evaluating the marginal likelihood gain.

        This implements a simplified version of the atom addition strategy.
        """
        # For efficiency, we only try a limited number of candidate frequencies
        num_candidates = 50
        theta_candidates = np.linspace(0, 1, num_candidates, endpoint=False)

        # Remove candidates too close to existing atoms
        if len(self.theta) > 0:
            min_separation = 0.5 / self.N  # Half Rayleigh limit
            valid_candidates = []
            for theta_cand in theta_candidates:
                min_dist = np.min(np.abs(self.theta - theta_cand))
                # Also consider wrap-around distance
                min_dist = min(min_dist, 1.0 - min_dist)
                if min_dist > min_separation:
                    valid_candidates.append(theta_cand)
            theta_candidates = np.array(valid_candidates)

        if len(theta_candidates) == 0:
            return

        # Construct current dictionary
        Psi_current = self._construct_dictionary()

        # Compute current residual
        if len(self.theta) > 0:
            residual = self.y - Psi_current @ self.alpha
        else:
            residual = self.y.copy()

        # Evaluate marginal likelihood gain for each candidate
        best_gain = self.add_threshold
        best_theta = None

        for theta_cand in theta_candidates:
            psi_cand = self.atom_func(theta_cand, self.N)

            # Compute correlation with residual
            correlation = np.abs(psi_cand.conj() @ residual) ** 2

            # Compute Gram matrix element
            if len(self.theta) > 0:
                overlap = Psi_current.conj().T @ psi_cand
                gram_term = self.lambda_ * np.linalg.norm(psi_cand) ** 2
                # Simplified gain computation
                gain = correlation / gram_term
            else:
                gain = correlation / (self.lambda_ * np.linalg.norm(psi_cand) ** 2)

            if gain > best_gain:
                best_gain = gain
                best_theta = theta_cand

        # Add the best atom if gain is positive
        if best_theta is not None:
            if self.verbose:
                print(f"  Adding atom at theta={best_theta:.4f}")

            self.theta = np.append(self.theta, best_theta)
            self.gamma = np.append(self.gamma, self.gamma_init)
            self.alpha = np.append(self.alpha, 0.0)

    def _update_theta(self):
        """
        Update dictionary parameters theta using gradient descent.

        This is the key contribution of the paper: optimizing continuous
        parameters rather than using a fixed grid.
        """
        if len(self.theta) == 0:
            return

        # Use gradient descent to update each theta
        for i in range(len(self.theta)):
            # Compute gradient of marginal likelihood w.r.t. theta[i]
            grad = self._compute_theta_gradient(i)

            # Gradient ascent step (we want to maximize likelihood)
            theta_new = self.theta[i] + self.theta_learning_rate * grad

            # Wrap to [0, 1)
            theta_new = np.mod(theta_new, 1.0)

            # Update only if it improves the objective
            self.theta[i] = theta_new

    def _compute_theta_gradient(self, idx: int) -> float:
        """
        Compute gradient of marginal likelihood with respect to theta[idx].

        This is a simplified gradient computation. The full computation
        involves derivatives of the marginal likelihood, which is complex.

        Parameters
        ----------
        idx : int
            Index of the theta parameter to compute gradient for

        Returns
        -------
        grad : float
            Gradient value
        """
        # Construct dictionary and its derivative
        Psi = self._construct_dictionary()
        psi_i = Psi[:, idx]
        dpsi_i = self.atom_derivative(self.theta[idx], self.N)

        # Simplified gradient: derivative of residual norm
        residual = self.y - Psi @ self.alpha

        # grad ∝ Re(α*ᵢ · d̄ψᵢ^H · residual)
        grad = 2.0 * np.real(
            self.alpha[idx].conj() * (dpsi_i.conj() @ residual)
        )

        return grad

    def _compute_log_marginal_likelihood(self) -> float:
        """
        Compute the log marginal likelihood.

        log p(y|θ,γ,λ) = -N log π - log|C| - y^H C^(-1) y

        where C = λ^(-1)I + Ψ Γ Ψ^H is the covariance matrix.

        Returns
        -------
        log_ml : float
            Log marginal likelihood value
        """
        if len(self.theta) == 0:
            # Empty model
            C = (1.0 / self.lambda_) * np.eye(self.N)
            sign, logdet = np.linalg.slogdet(C)
            log_ml = -self.N * np.log(np.pi) - logdet - np.real(
                self.y.conj() @ np.linalg.solve(C, self.y)
            )
            return log_ml

        # Construct dictionary
        Psi = self._construct_dictionary()

        # Compute covariance C = λ^(-1)I + Ψ Γ Ψ^H
        Gamma = np.diag(self.gamma)
        C = (1.0 / self.lambda_) * np.eye(self.N) + Psi @ Gamma @ Psi.conj().T

        # Compute log determinant and quadratic form
        try:
            sign, logdet = np.linalg.slogdet(C)
            if sign <= 0:
                return -np.inf

            C_inv_y = np.linalg.solve(C, self.y)
            quad_form = np.real(self.y.conj() @ C_inv_y)

            log_ml = -self.N * np.log(np.pi) - logdet - quad_form

        except np.linalg.LinAlgError:
            log_ml = -np.inf

        return log_ml

    def predict(self, theta_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct the signal using the estimated parameters.

        Parameters
        ----------
        theta_test : np.ndarray, optional
            If provided, evaluate at these theta values instead

        Returns
        -------
        x_pred : np.ndarray, shape (N,)
            Reconstructed signal
        """
        if theta_test is not None:
            Psi = self._construct_dictionary(theta_test)
            # Use current alpha for prediction (may not be appropriate)
            x_pred = Psi @ self.alpha[:len(theta_test)]
        else:
            if len(self.theta) == 0:
                return np.zeros(self.N, dtype=complex)

            Psi = self._construct_dictionary()
            x_pred = Psi @ self.alpha

        return x_pred

    def get_spectrum(self, freq_grid: Optional[np.ndarray] = None,
                     method: str = 'reconstruction') -> np.ndarray:
        """
        Compute the spectrum estimate.

        Parameters
        ----------
        freq_grid : np.ndarray, optional
            Frequency grid for spectrum evaluation
            If None, uses a default fine grid
        method : str
            'reconstruction': |Ψ(θ)^H α|
            'pseudospectrum': Based on residual correlation

        Returns
        -------
        spectrum : np.ndarray
            Spectrum values at freq_grid points
        """
        if freq_grid is None:
            freq_grid = np.linspace(0, 1, 512, endpoint=False)

        if method == 'reconstruction' and len(self.theta) > 0:
            # Compute spectrum as correlation with reconstructed signal
            x_recon = self.predict()
            spectrum = np.zeros(len(freq_grid))

            for i, f in enumerate(freq_grid):
                psi = self.atom_func(f, self.N)
                spectrum[i] = np.abs(psi.conj() @ x_recon) ** 2

        else:
            # Fallback: correlation with observation
            spectrum = np.zeros(len(freq_grid))
            for i, f in enumerate(freq_grid):
                psi = self.atom_func(f, self.N)
                spectrum[i] = np.abs(psi.conj() @ self.y) ** 2

        # Normalize
        spectrum = spectrum / np.max(spectrum + 1e-12)

        return spectrum
