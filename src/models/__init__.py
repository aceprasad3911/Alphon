# src/models/__init__.py
# This file makes the 'models' directory a Python package.

# Import base model and specific model types for easier access
from .base_model import BaseModel
from .baseline_models import LinearRegressionModel, RandomForestModel
from .gnn_models import GATModel, GCNModel
from .time_series_models import CNNLSTMHybrid, TransformerEncoderModel
from .fusion_models import FusionModel
from .disentanglement import BetaVAE

# TODO: Add any shared model utilities or configurations here.
