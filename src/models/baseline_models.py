# src/models/baseline_models.py

# Implementations of traditional and simple ML baseline models

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .base_model import BaseModel # Although not a PyTorch model, keep for interface consistency

logger = logging.getLogger(__name__)

class LinearRegressionModel:
    """
    A wrapper for Scikit-learn's Linear Regression model.
    Implements a simplified interface consistent with BaseModel for training and prediction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = LinearRegression(fit_intercept=config.get("fit_intercept", True))
        logger.info(f"Initialized Linear Regression Model with config: {config}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Trains the Linear Regression model.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (Optional[pd.DataFrame]): Validation features (not used by LR, but for interface consistency).
            y_val (Optional[pd.Series]): Validation targets (not used by LR).
        """
        logger.info("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_r2 = r2_score(y_train, train_preds)
        logger.info(f"Linear Regression Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_r2 = r2_score(y_val, val_preds)
            logger.info(f"Linear Regression Validation RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained Linear Regression model.
        Args:
            X (pd.DataFrame): Features to predict on.
        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model first.")
        return self.model.predict(X)

    def save(self, path: str):
        """
        Saves the trained model using joblib.
        Args:
            path (str): File path to save the model.
        """
        import joblib
        try:
            joblib.dump(self.model, path)
            logger.info(f"Linear Regression model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Linear Regression model to {path}: {e}")
            raise

    def load(self, path: str):
        """
        Loads a trained model using joblib.
        Args:
            path (str): File path to load the model from.
        """
        import joblib
        try:
            self.model = joblib.load(path)
            logger.info(f"Linear Regression model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load Linear Regression model from {path}: {e}")
            raise

class RandomForestModel:
    """
    A wrapper for Scikit-learn's RandomForestRegressor model.
    Implements a simplified interface consistent with BaseModel for training and prediction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=config.get("random_state", 42),
            n_jobs=-1  # Use all available cores
        )
        logger.info(f"Initialized Random Forest Model with config: {config}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Trains the Random Forest model.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (Optional[pd.DataFrame]): Validation features.
            y_val (Optional[pd.Series]): Validation targets.
        """
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_r2 = r2_score(y_train, train_preds)
        logger.info(f"Random Forest Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_r2 = r2_score(y_val, val_preds)
            logger.info(f"Random Forest Validation RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained Random Forest model.
        Args:
            X (pd.DataFrame): Features to predict on.
        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model first.")
        return self.model.predict(X)

    def save(self, path: str):
        """
        Saves the trained model using joblib.
        Args:
            path (str): File path to save the model.
        """
        import joblib
        try:
            joblib.dump(self.model, path)
            logger.info(f"Random Forest model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Random Forest model to {path}: {e}")
            raise

    def load(self, path: str):
        """
        Loads a trained model using joblib.
        Args:
            path (str): File path to load the model from.
        """
        import joblib
        try:
            self.model = joblib.load(path)
            logger.info(f"Random Forest model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load Random Forest model from {path}: {e}")
            raise

# TODO: Add other baseline models (e.g., XGBoost, LightGBM).
# TODO: Implement a common interface for all baseline models if they are to be used interchangeably.
# Note: These models don't inherit from BaseModel (nn.Module) as they are not PyTorch models.
# If you want strict inheritance, you'd need to create a separate BaseSklearnModel or adapt BaseModel.
