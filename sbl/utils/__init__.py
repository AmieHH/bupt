"""
Utility functions for evaluation and visualization.
"""

from .metrics import (
    frequency_rmse,
    amplitude_rmse,
    model_order_accuracy,
    match_frequencies,
    normalized_mse,
)

from .visualization import (
    plot_signal,
    plot_spectrum,
    plot_estimation_results,
    plot_comparison,
    plot_convergence,
)

__all__ = [
    'frequency_rmse',
    'amplitude_rmse',
    'model_order_accuracy',
    'match_frequencies',
    'normalized_mse',
    'plot_signal',
    'plot_spectrum',
    'plot_estimation_results',
    'plot_comparison',
    'plot_convergence',
]
