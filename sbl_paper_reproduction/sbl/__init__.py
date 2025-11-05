"""
Sparse Bayesian Learning with Dictionary Parameter Estimation

This package implements the SBL algorithm proposed in:
'A Sparse Bayesian Learning Algorithm With Dictionary Parameter Estimation'
by Hansen et al.
"""

__version__ = "1.0.0"
__author__ = "Paper Reproduction Team"

from .algorithms.sbl_dict_param import SBLDictParam
from .models.signal_model import generate_signal, SignalModel
from .models.atoms import fourier_atom, fourier_dictionary

__all__ = [
    'SBLDictParam',
    'generate_signal',
    'SignalModel',
    'fourier_atom',
    'fourier_dictionary',
]
