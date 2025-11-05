"""
Single-trial comparison of different algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbl import SBLDictParam
from sbl.algorithms.esprit import ESPRIT
from sbl.algorithms.music import MUSIC
from sbl.algorithms.omp import OMP
from sbl.models.signal_model import generate_closely_spaced_scenario
from sbl.utils.metrics import evaluate_estimator
from sbl.utils.visualization import plot_comparison


def main():
    """Run a single comparison experiment."""

    print("=" * 70)
    print("Algorithm Comparison on a Single Test Case")
    print("=" * 70)

    # Test configuration
    N = 64
    K = 3
    snr_db = 20

    # Generate test scenario with closely-spaced frequencies
    print(f"\nTest Configuration:")
    print(f"  Signal length N: {N}")
    print(f"  Number of components K: {K}")
    print(f"  SNR: {snr_db} dB")
    print(f"  Scenario: Closely-spaced frequencies")

    scenario = generate_closely_spaced_scenario(
        N=N, K=K, snr_db=snr_db,
        center_freq=0.25,
        freq_span=3.0 / N,  # About 3 bins spacing
        seed=42
    )

    print(f"\nTrue frequencies: {scenario['theta']}")
    print(f"True amplitudes: {np.abs(scenario['alpha'])}")

    # Initialize algorithms
    algorithms = {
        'SBL-DictParam': SBLDictParam(N=N, max_iter=50, verbose=False),
        'ESPRIT': ESPRIT(K=K, method='ls'),
        'MUSIC': MUSIC(K=K, grid_size=512),
        'OMP-Grid256': OMP(K=K, grid_size=256),
    }

    # Run each algorithm and collect results
    print("\nRunning algorithms...")
    results = {}

    for alg_name, estimator in algorithms.items():
        print(f"  {alg_name}...", end=' ')
        try:
            eval_results = evaluate_estimator(estimator, scenario, threshold=0.02)

            if eval_results['success']:
                results[alg_name] = eval_results
                print(f"✓ (K_est={eval_results['K_est']}, RMSE={eval_results['freq_rmse']:.6f})")
            else:
                print(f"✗ Failed: {eval_results.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"✗ Exception: {e}")

    # Print detailed results
    print("\n" + "=" * 70)
    print("Detailed Results")
    print("=" * 70)

    print(f"\n{'Algorithm':<15} {'K_est':<6} {'Freq RMSE':<12} {'Amp RMSE':<12} {'Detect Rate':<12}")
    print("-" * 70)

    for alg_name, result in results.items():
        print(f"{alg_name:<15} {result['K_est']:<6} "
              f"{result['freq_rmse']:<12.8f} {result['amp_rmse']:<12.6f} "
              f"{result['detection_rate']:<12.3f}")

    # Visualize comparison
    print("\nGenerating comparison plots...")

    fig = plot_comparison(
        scenario['y'],
        scenario['theta'],
        scenario['alpha'],
        results,
        show=False
    )

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'algorithm_comparison.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("Comparison completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
