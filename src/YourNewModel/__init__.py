"""
YourNewModel - A transformer implementation from scratch.

This package provides a clean, educational implementation of a dense
transformer-based language model using PyTorch.
"""

from .modelling_YourNewModel import YourNewModel, YourNewModelConfig, create_model
from .training.YourNewModel.train_YourNewModel import Trainer, TextDataset

ModelConfig = YourNewModelConfig

__version__ = "1.0.0"
__all__ = [
    "YourNewModel",
    "YourNewModelConfig", 
    "create_model",
    "Trainer",
    "TextDataset",
    "ModelConfig",
]
