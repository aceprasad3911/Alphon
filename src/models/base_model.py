# src/models/base_model.py

# Abstract base class for all model types

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all machine learning models.
    Defines the common interface for model initialization, training, prediction, saving, and loading.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base model with common configuration.
        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        logger.info(f"Initializing {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method for the forward pass of the model.
        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
        Returns:
            torch.Tensor: Model output.
        """
        pass

    @abstractmethod
    def train_model(self, train_loader: Any, val_loader: Any, **kwargs):
        """
        Abstract method to train the model.
        Args:
            train_loader (Any): DataLoader for training data.
            val_loader (Any): DataLoader for validation data.
            **kwargs: Additional training parameters (e.g., epochs, optimizer, loss_fn).
        """
        pass

    @abstractmethod
    def predict(self, data_loader: Any) -> torch.Tensor:
        """
        Abstract method to make predictions using the trained model.
        Args:
            data_loader (Any): DataLoader for prediction data.
        Returns:
            torch.Tensor: Model predictions.
        """
        pass

    def save(self, path: str):
        """
        Saves the model's state dictionary to a specified path.
        Args:
            path (str): File path to save the model.
        """
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model {self.__class__.__name__} saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model {self.__class__.__name__} to {path}: {e}")
            raise

    def load(self, path: str, map_location: Optional[str] = None):
        """
        Loads the model's state dictionary from a specified path.
        Args:
            path (str): File path to load the model from.
            map_location (Optional[str]): Device to load the model to (e.g., 'cpu', 'cuda').
        """
        try:
            self.load_state_dict(torch.load(path, map_location=map_location))
            self.to(self.device) # Move model to its configured device
            logger.info(f"Model {self.__class__.__name__} loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model {self.__class__.__name__} from {path}: {e}")
            raise

# TODO: Add common utility methods like `_get_optimizer`, `_get_loss_fn`.
# TODO: Integrate with experiment tracking (e.g., MLflow, Weights & Biases) here.
