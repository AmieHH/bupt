"""
Visualization utilities for signal and spectrum analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
import warnings


def plot_signal(t: np.ndarray, signals: Dict[str, np.ndarray],
                title: str = "Signal", figsize: Tuple[int, int] = (12, 4),
                show: bool = True) -> plt.Figure:
    """
    Plot time-domain signals.

    Parameters
    ----------
    t : np.ndarray
        Time indices
    signals : dict
        Dictionary of {label: signal_array}
    title : str
        Plot title
    figsize : tuple
        Figure size
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    for label, signal in signals.items():
        # Real part
        axes[0].plot(t, np.real(signal), label=label, alpha=0.7)
        # Imaginary part
        if np.iscomplexobj(signal):
            axes[1].plot(t, np.imag(signal), label=label, alpha=0.7)

    axes[0].set_ylabel('Real Part')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if np.iscomplexobj(list(signals.values())[0]):
        axes[1].set_ylabel('Imaginary Part')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    axes[-1].set_xlabel('Sample Index')
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_spectrum(freq_grid: np.ndarray, spectra: Dict[str, np.ndarray],
                  theta_true: Optional[np.ndarray] = None,
                  theta_est: Optional[np.ndarray] = None,
                  title: str = "Spectrum", figsize: Tuple[int, int] = (12, 6),
                  db_scale: bool = True, show: bool = True) -> plt.Figure:
    """
    Plot frequency spectrum.

    Parameters
    ----------
    freq_grid : np.ndarray
        Frequency grid
    spectra : dict
        Dictionary of {label: spectrum_array}
    theta_true : np.ndarray, optional
        True frequencies (shown as vertical lines)
    theta_est : np.ndarray, optional
        Estimated frequencies (shown as markers)
    title : str
        Plot title
    figsize : tuple
        Figure size
    db_scale : bool
        If True, use dB scale
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectra
    for label, spectrum in spectra.items():
        spectrum_plot = np.abs(spectrum)

        if db_scale:
            spectrum_plot = 10 * np.log10(spectrum_plot + 1e-12)

        ax.plot(freq_grid, spectrum_plot, label=label, alpha=0.7, linewidth=1.5)

    # Mark true frequencies
    if theta_true is not None:
        ymin, ymax = ax.get_ylim()
        for theta in theta_true:
            ax.axvline(theta, color='green', linestyle='--', alpha=0.5, linewidth=2)
        # Add dummy line for legend
        ax.axvline(-1, color='green', linestyle='--', alpha=0.5,
                   linewidth=2, label='True Frequencies')

    # Mark estimated frequencies
    if theta_est is not None:
        ymin, ymax = ax.get_ylim()
        for theta in theta_est:
            ax.plot(theta, ymin + 0.1 * (ymax - ymin), 'rv', markersize=10, alpha=0.7)
        # Add dummy marker for legend
        ax.plot(-1, ymin, 'rv', markersize=10, alpha=0.7, label='Estimated Frequencies')

    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)' if db_scale else 'Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([freq_grid[0], freq_grid[-1]])

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_estimation_results(y: np.ndarray, theta_true: np.ndarray,
                            alpha_true: np.ndarray, theta_est: np.ndarray,
                            alpha_est: np.ndarray, algorithm_name: str = "Algorithm",
                            figsize: Tuple[int, int] = (14, 10),
                            show: bool = True) -> plt.Figure:
    """
    Comprehensive visualization of estimation results.

    Parameters
    ----------
    y : np.ndarray
        Observed signal
    theta_true : np.ndarray
        True frequencies
    alpha_true : np.ndarray
        True amplitudes
    theta_est : np.ndarray
        Estimated frequencies
    alpha_est : np.ndarray
        Estimated amplitudes
    algorithm_name : str
        Name of the algorithm
    figsize : tuple
        Figure size
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    from ..models.signal_model import SignalModel

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    N = len(y)
    model = SignalModel(N)

    # 1. Observed signal
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(N)
    ax1.plot(t, np.real(y), 'b-', alpha=0.7, label='Observed (Real)')
    if np.iscomplexobj(y):
        ax1.plot(t, np.imag(y), 'r-', alpha=0.7, label='Observed (Imag)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Observed Signal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Reconstructed vs True signal
    ax2 = fig.add_subplot(gs[1, :])
    x_true = model.generate_clean_signal(theta_true, alpha_true)
    if len(theta_est) > 0:
        x_est = model.generate_clean_signal(theta_est, alpha_est)
    else:
        x_est = np.zeros_like(x_true)

    ax2.plot(t, np.real(x_true), 'g-', linewidth=2, label='True (Real)', alpha=0.7)
    ax2.plot(t, np.real(x_est), 'r--', linewidth=2, label='Estimated (Real)', alpha=0.7)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Signal Reconstruction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Frequency comparison (stem plot)
    ax3 = fig.add_subplot(gs[2, 0])
    if len(theta_true) > 0:
        ax3.stem(theta_true, np.abs(alpha_true), linefmt='g-',
                markerfmt='go', basefmt=' ', label='True')
    if len(theta_est) > 0:
        ax3.stem(theta_est, np.abs(alpha_est), linefmt='r--',
                markerfmt='r^', basefmt=' ', label='Estimated')
    ax3.set_xlabel('Normalized Frequency')
    ax3.set_ylabel('|Amplitude|')
    ax3.set_title('Frequency-Amplitude Pairs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])

    # 4. Estimation errors
    ax4 = fig.add_subplot(gs[2, 1])

    from ..utils.metrics import match_frequencies, frequency_rmse, amplitude_rmse

    freq_rmse_val = frequency_rmse(theta_true, theta_est)
    amp_rmse_val = amplitude_rmse(alpha_true, alpha_est, theta_true, theta_est)

    error_text = f"Frequency RMSE: {freq_rmse_val:.6f}\n"
    error_text += f"Amplitude RMSE: {amp_rmse_val:.4f}\n"
    error_text += f"True K: {len(theta_true)}\n"
    error_text += f"Estimated K: {len(theta_est)}"

    ax4.text(0.1, 0.5, error_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title(f'{algorithm_name} Performance')

    fig.suptitle(f'{algorithm_name} - Estimation Results', fontsize=14, fontweight='bold')

    if show:
        plt.show()

    return fig


def plot_comparison(y: np.ndarray, theta_true: np.ndarray, alpha_true: np.ndarray,
                   results: Dict[str, Dict], figsize: Tuple[int, int] = (16, 10),
                   show: bool = True) -> plt.Figure:
    """
    Compare multiple algorithms on the same data.

    Parameters
    ----------
    y : np.ndarray
        Observed signal
    theta_true : np.ndarray
        True frequencies
    alpha_true : np.ndarray
        True amplitudes
    results : dict
        Dictionary of {algorithm_name: results_dict}
        Each results_dict should have 'theta_est', 'alpha_est', 'freq_rmse', etc.
    figsize : tuple
        Figure size
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    num_algs = len(results)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, num_algs, hspace=0.3, wspace=0.3)

    N = len(y)

    # Top row: Spectrum comparison for each algorithm
    freq_grid = np.linspace(0, 1, 512, endpoint=False)

    from ..models.atoms import fourier_atom

    for idx, (alg_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, idx])

        theta_est = result.get('theta_est', np.array([]))
        alpha_est = result.get('alpha_est', np.array([]))

        # Compute spectrum from estimated parameters
        if len(theta_est) > 0:
            spectrum = np.zeros(len(freq_grid))
            for f_idx, f in enumerate(freq_grid):
                atom = fourier_atom(f, N)
                # Reconstruct spectrum
                x_est = sum(alpha_est[i] * fourier_atom(theta_est[i], N)
                           for i in range(len(theta_est)))
                spectrum[f_idx] = np.abs(atom.conj() @ x_est)

            spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
            ax.plot(freq_grid, spectrum_db, 'b-', linewidth=1.5)

        # Mark true frequencies
        ymin, ymax = ax.get_ylim()
        for theta in theta_true:
            ax.axvline(theta, color='green', linestyle='--', alpha=0.5, linewidth=2)

        # Mark estimated frequencies
        if len(theta_est) > 0:
            for theta in theta_est:
                ax.plot(theta, ymin + 0.1 * (ymax - ymin), 'rv', markersize=8)

        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'{alg_name}\nFreq RMSE: {result.get("freq_rmse", np.inf):.6f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    # Bottom row: Performance metrics comparison
    ax_metrics = fig.add_subplot(gs[1, :])

    alg_names = list(results.keys())
    metrics = ['freq_rmse', 'amp_rmse', 'detection_rate']
    metric_labels = ['Frequency RMSE', 'Amplitude RMSE', 'Detection Rate']

    x = np.arange(len(metrics))
    width = 0.8 / len(alg_names)

    for idx, alg_name in enumerate(alg_names):
        result = results[alg_name]
        values = [
            result.get('freq_rmse', 0),
            result.get('amp_rmse', 0),
            result.get('detection_rate', 0),
        ]
        ax_metrics.bar(x + idx * width, values, width, label=alg_name, alpha=0.7)

    ax_metrics.set_xlabel('Metrics')
    ax_metrics.set_ylabel('Value')
    ax_metrics.set_title('Performance Comparison')
    ax_metrics.set_xticks(x + width * (len(alg_names) - 1) / 2)
    ax_metrics.set_xticklabels(metric_labels)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Algorithm Comparison', fontsize=14, fontweight='bold')

    if show:
        plt.show()

    return fig


def plot_convergence(history: Dict, figsize: Tuple[int, int] = (12, 4),
                    show: bool = True) -> plt.Figure:
    """
    Plot convergence history of an iterative algorithm.

    Parameters
    ----------
    history : dict
        Dictionary with keys like 'log_marginal_likelihood', 'K', 'theta'
    figsize : tuple
        Figure size
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot log marginal likelihood
    if 'log_marginal_likelihood' in history:
        ax = axes[0]
        lml = history['log_marginal_likelihood']
        ax.plot(lml, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Marginal Likelihood')
        ax.set_title('Convergence: Objective Function')
        ax.grid(True, alpha=0.3)

    # Plot model order evolution
    if 'K' in history:
        ax = axes[1]
        K = history['K']
        ax.plot(K, 'r-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Model Order K')
        ax.set_title('Model Order Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('Algorithm Convergence', fontsize=14)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_monte_carlo_results(snr_range: np.ndarray, results: Dict[str, List],
                             metric_name: str = 'freq_rmse',
                             ylabel: str = 'Frequency RMSE',
                             figsize: Tuple[int, int] = (10, 6),
                             show: bool = True) -> plt.Figure:
    """
    Plot Monte Carlo simulation results.

    Parameters
    ----------
    snr_range : np.ndarray
        SNR values (in dB)
    results : dict
        Dictionary of {algorithm_name: [metric_values]}
    metric_name : str
        Name of the metric to plot
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    show : bool
        If True, display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for alg_name, metric_values in results.items():
        ax.plot(snr_range, metric_values, marker='o', linewidth=2,
               markersize=6, label=alg_name, alpha=0.7)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.tight_layout()

    if show:
        plt.show()

    return fig
