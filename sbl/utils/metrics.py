"""
Evaluation metrics for frequency estimation algorithms.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment


def match_frequencies(theta_true: np.ndarray, theta_est: np.ndarray,
                      alpha_true: Optional[np.ndarray] = None,
                      alpha_est: Optional[np.ndarray] = None,
                      threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match estimated frequencies to true frequencies using Hungarian algorithm.

    This handles the permutation ambiguity in frequency estimation.

    Parameters
    ----------
    theta_true : np.ndarray, shape (K_true,)
        True frequencies
    theta_est : np.ndarray, shape (K_est,)
        Estimated frequencies
    alpha_true : np.ndarray, optional
        True amplitudes
    alpha_est : np.ndarray, optional
        Estimated amplitudes
    threshold : float
        Maximum distance for a valid match

    Returns
    -------
    matched_true : np.ndarray
        Indices of matched true frequencies
    matched_est : np.ndarray
        Indices of matched estimated frequencies
    cost_matrix : np.ndarray
        Distance matrix used for matching

    Examples
    --------
    >>> theta_true = np.array([0.1, 0.2, 0.3])
    >>> theta_est = np.array([0.102, 0.198, 0.305])
    >>> idx_true, idx_est, _ = match_frequencies(theta_true, theta_est)
    """
    K_true = len(theta_true)
    K_est = len(theta_est)

    # Compute pairwise distance matrix
    # Consider wrap-around at 0 and 1
    cost_matrix = np.zeros((K_true, K_est))

    for i in range(K_true):
        for j in range(K_est):
            # Minimum distance considering periodicity
            diff = np.abs(theta_true[i] - theta_est[j])
            dist = min(diff, 1.0 - diff)
            cost_matrix[i, j] = dist

    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter out matches that exceed threshold
    valid_matches = cost_matrix[row_ind, col_ind] < threshold

    matched_true = row_ind[valid_matches]
    matched_est = col_ind[valid_matches]

    return matched_true, matched_est, cost_matrix


def frequency_rmse(theta_true: np.ndarray, theta_est: np.ndarray,
                   threshold: float = 0.01) -> float:
    """
    Compute Root Mean Square Error (RMSE) for frequency estimation.

    Only matched frequencies (within threshold) are included.

    Parameters
    ----------
    theta_true : np.ndarray
        True frequencies
    theta_est : np.ndarray
        Estimated frequencies
    threshold : float
        Matching threshold

    Returns
    -------
    rmse : float
        Frequency RMSE
    """
    if len(theta_true) == 0 or len(theta_est) == 0:
        return np.inf

    # Match frequencies
    idx_true, idx_est, _ = match_frequencies(theta_true, theta_est, threshold=threshold)

    if len(idx_true) == 0:
        # No successful matches
        return np.inf

    # Compute distances for matched pairs
    distances = []
    for i, j in zip(idx_true, idx_est):
        diff = np.abs(theta_true[i] - theta_est[j])
        dist = min(diff, 1.0 - diff)
        distances.append(dist)

    rmse = np.sqrt(np.mean(np.array(distances) ** 2))

    return rmse


def amplitude_rmse(alpha_true: np.ndarray, alpha_est: np.ndarray,
                   theta_true: np.ndarray, theta_est: np.ndarray,
                   threshold: float = 0.01) -> float:
    """
    Compute RMSE for amplitude estimation.

    Amplitudes are matched based on their corresponding frequencies.

    Parameters
    ----------
    alpha_true : np.ndarray
        True amplitudes
    alpha_est : np.ndarray
        Estimated amplitudes
    theta_true : np.ndarray
        True frequencies (for matching)
    theta_est : np.ndarray
        Estimated frequencies (for matching)
    threshold : float
        Matching threshold

    Returns
    -------
    rmse : float
        Amplitude RMSE
    """
    if len(theta_true) == 0 or len(theta_est) == 0:
        return np.inf

    # Match frequencies
    idx_true, idx_est, _ = match_frequencies(theta_true, theta_est, threshold=threshold)

    if len(idx_true) == 0:
        return np.inf

    # Compute amplitude errors for matched pairs
    errors = []
    for i, j in zip(idx_true, idx_est):
        error = np.abs(alpha_true[i] - alpha_est[j])
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    return rmse


def model_order_accuracy(K_true: int, K_est: int) -> float:
    """
    Compute model order estimation accuracy.

    Returns 1.0 if correct, 0.0 otherwise.

    Parameters
    ----------
    K_true : int
        True model order
    K_est : int
        Estimated model order

    Returns
    -------
    accuracy : float
        1.0 if K_est == K_true, else 0.0
    """
    return 1.0 if K_true == K_est else 0.0


def detection_rate(theta_true: np.ndarray, theta_est: np.ndarray,
                   threshold: float = 0.01) -> float:
    """
    Compute the detection rate (recall).

    Detection rate = (# correctly detected frequencies) / (# true frequencies)

    Parameters
    ----------
    theta_true : np.ndarray
        True frequencies
    theta_est : np.ndarray
        Estimated frequencies
    threshold : float
        Matching threshold

    Returns
    -------
    rate : float
        Detection rate in [0, 1]
    """
    if len(theta_true) == 0:
        return 1.0 if len(theta_est) == 0 else 0.0

    idx_true, _, _ = match_frequencies(theta_true, theta_est, threshold=threshold)

    rate = len(idx_true) / len(theta_true)

    return rate


def false_alarm_rate(theta_true: np.ndarray, theta_est: np.ndarray,
                     threshold: float = 0.01) -> float:
    """
    Compute the false alarm rate.

    False alarm rate = (# falsely detected frequencies) / (# estimated frequencies)

    Parameters
    ----------
    theta_true : np.ndarray
        True frequencies
    theta_est : np.ndarray
        Estimated frequencies
    threshold : float
        Matching threshold

    Returns
    -------
    rate : float
        False alarm rate in [0, 1]
    """
    if len(theta_est) == 0:
        return 0.0

    _, idx_est, _ = match_frequencies(theta_true, theta_est, threshold=threshold)

    num_false_alarms = len(theta_est) - len(idx_est)
    rate = num_false_alarms / len(theta_est)

    return rate


def normalized_mse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """
    Compute normalized mean squared error for signal reconstruction.

    NMSE = ||x_true - x_est||^2 / ||x_true||^2

    Parameters
    ----------
    x_true : np.ndarray
        True signal
    x_est : np.ndarray
        Estimated signal

    Returns
    -------
    nmse : float
        Normalized MSE
    """
    error = np.linalg.norm(x_true - x_est) ** 2
    signal_power = np.linalg.norm(x_true) ** 2

    if signal_power < 1e-12:
        return np.inf

    nmse = error / signal_power

    return nmse


def resolution_probability(theta_true: np.ndarray, theta_est: np.ndarray,
                           min_separation: float, threshold: float = 0.01) -> float:
    """
    Compute the probability of resolving closely-spaced frequencies.

    Checks if all true frequencies separated by at least min_separation
    are correctly detected.

    Parameters
    ----------
    theta_true : np.ndarray
        True frequencies
    theta_est : np.ndarray
        Estimated frequencies
    min_separation : float
        Minimum separation to consider "close"
    threshold : float
        Matching threshold

    Returns
    -------
    prob : float
        1.0 if all close frequencies are resolved, else 0.0
    """
    K = len(theta_true)

    # Find pairs of close frequencies
    close_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = np.abs(theta_true[i] - theta_true[j])
            sep = min(diff, 1.0 - diff)
            if sep <= min_separation:
                close_pairs.append((i, j))

    if len(close_pairs) == 0:
        # No close pairs to resolve
        return 1.0

    # Check if all frequencies in close pairs are detected
    idx_true, _, _ = match_frequencies(theta_true, theta_est, threshold=threshold)

    for i, j in close_pairs:
        if i not in idx_true or j not in idx_true:
            return 0.0

    return 1.0


def compute_crlb_frequency(N: int, snr: float, K: int = 1) -> float:
    """
    Compute Cramér-Rao Lower Bound (CRLB) for frequency estimation.

    Approximate CRLB for a single sinusoid in white Gaussian noise.

    Parameters
    ----------
    N : int
        Signal length
    snr : float
        Signal-to-noise ratio (linear, not dB)
    K : int
        Number of frequency components

    Returns
    -------
    crlb : float
        CRLB for frequency estimation variance
    """
    # Simplified CRLB for frequency of a complex sinusoid
    # var(θ_hat) >= 6 / (π^2 * N * (N^2 - 1) * SNR)

    crlb = 6.0 / (np.pi ** 2 * N * (N ** 2 - 1) * snr)

    # For multiple components, CRLB increases (simplified approximation)
    crlb = crlb * K

    return crlb


def evaluate_estimator(estimator, scenario: dict, threshold: float = 0.01) -> dict:
    """
    Comprehensive evaluation of an estimator on a test scenario.

    Parameters
    ----------
    estimator : object
        Estimator with fit() method
    scenario : dict
        Test scenario from generate_test_scenario()
    threshold : float
        Matching threshold

    Returns
    -------
    results : dict
        Dictionary containing all evaluation metrics
    """
    # Extract ground truth
    y = scenario['y']
    x_true = scenario['x']
    theta_true = scenario['theta']
    alpha_true = scenario['alpha']
    K_true = scenario['K']

    # Run estimator
    try:
        theta_est, alpha_est, K_est = estimator.fit(y)
    except Exception as e:
        print(f"Estimator failed: {e}")
        return {
            'success': False,
            'error': str(e),
        }

    # Compute metrics
    results = {
        'success': True,
        'K_true': K_true,
        'K_est': K_est,
        'model_order_correct': model_order_accuracy(K_true, K_est),
        'freq_rmse': frequency_rmse(theta_true, theta_est, threshold),
        'amp_rmse': amplitude_rmse(alpha_true, alpha_est, theta_true, theta_est, threshold),
        'detection_rate': detection_rate(theta_true, theta_est, threshold),
        'false_alarm_rate': false_alarm_rate(theta_true, theta_est, threshold),
        'theta_est': theta_est,
        'alpha_est': alpha_est,
    }

    # Signal reconstruction error (if possible)
    try:
        x_est = estimator.predict()
        if len(x_est) == len(x_true):
            results['nmse'] = normalized_mse(x_true, x_est)
    except:
        pass

    return results
