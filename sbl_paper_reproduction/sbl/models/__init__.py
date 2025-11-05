"""
Signal models and atom functions for sparse representation.
"""

from .atoms import fourier_atom, fourier_dictionary, atom_derivative
from .signal_model import SignalModel, generate_signal

__all__ = [
    'fourier_atom',
    'fourier_dictionary',
    'atom_derivative',
    'SignalModel',
    'generate_signal',
]
