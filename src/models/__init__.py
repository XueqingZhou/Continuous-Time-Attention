"""
Models for Continuous-Time Attention

This package provides PDE-enhanced Transformer models for various tasks.
"""

from .pde_layers import (
    DiffusionPDELayer,
    WavePDELayer,
    ReactionDiffusionPDELayer,
    AdvectionDiffusionPDELayer,
    create_pde_layer,
)
from .transformers import (
    PDETransformerClassifier,
    StandardTransformerClassifier,
    PDETransformerLM,
    StandardTransformerLM,
)

__all__ = [
    # PDE Layers
    'DiffusionPDELayer',
    'WavePDELayer',
    'ReactionDiffusionPDELayer',
    'AdvectionDiffusionPDELayer',
    'create_pde_layer',
    # Transformer Models
    'PDETransformerClassifier',
    'StandardTransformerClassifier',
    'PDETransformerLM',
    'StandardTransformerLM',
]

