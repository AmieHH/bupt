"""
Basic usage example of the SBL algorithm with dictionary parameter estimation.

This script demonstrates how to use the SBL algorithm for frequency estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbl import SBLDictParam, generate_signal
from sbl.utils.metrics import frequency_rmse, amplitude_rmse
from sbl.utils.visualization import plot_estimation_results


def main():
    """Run basic SBL example."""

    print("=" * 60)
    print("Basic SBL Algorithm Example")
    print("=" * 60)

    # Signal parameters
    N = 64  # Signal length
    K_true = 3  # Number of frequency components
    snr_db = 20  # Signal-to-noise ratio in dB

    # Define true frequencies and amplitudes
    theta_true = np.array([0.15, 0.25, 0.45])
    alpha_true = np.array([1.0 + 0.2j, 0.8 - 0.1j, 1.2 + 0.3j])

    print(f"\nSignal Configuration:")
    print(f"  Signal length N: {N}")
    print(f"  Number of components K: {K_true}")
    print(f"  SNR: {snr_db} dB")
    print(f"\nTrue Parameters:")
    print(f"  Frequencies: {theta_true}")
    print(f"  Amplitudes: {alpha_true}")

    # Generate noisy signal
    print("\nGenerating noisy signal...")
    y, x_clean = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

    # Initialize and run SBL algorithm
    print("\nRunning SBL algorithm...")
    sbl = SBLDictParam(
        N=N,
        max_iter=50,
        tol=1e-6,
        prune_threshold=1e-10,
        theta_learning_rate=0.01,
        verbose=True
    )

    theta_est, alpha_est, K_est = sbl.fit(y)

    # Display results
    print("\n" + "=" * 60)
    print("Estimation Results:")
    print("=" * 60)
    print(f"Estimated model order K: {K_est} (True: {K_true})")
    print(f"\nEstimated frequencies:")
    for i, (theta, alpha) in enumerate(zip(theta_est, alpha_est)):
        print(f"  Component {i + 1}: θ = {theta:.6f}, α = {alpha:.4f}")

    # Compute error metrics
    freq_rmse_val = frequency_rmse(theta_true, theta_est)
    amp_rmse_val = amplitude_rmse(alpha_true, alpha_est, theta_true, theta_est)

    print(f"\nPerformance Metrics:")
    print(f"  Frequency RMSE: {freq_rmse_val:.8f}")
    print(f"  Amplitude RMSE: {amp_rmse_val:.6f}")
    print(f"  Model order correct: {K_est == K_true}")

    # Visualize results
    print("\nGenerating plots...")
    fig = plot_estimation_results(
        y, theta_true, alpha_true, theta_est, alpha_est,
        algorithm_name="SBL-DictParam",
        show=False
    )

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'basic_usage_results.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")

    # Plot convergence
    from sbl.utils.visualization import plot_convergence
    fig_conv = plot_convergence(sbl.history, show=False)
    conv_path = os.path.join(output_dir, 'basic_usage_convergence.png')
    fig_conv.savefig(conv_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to: {conv_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
