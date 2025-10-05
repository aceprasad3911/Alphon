# src/models/time_series_models.py

# Implementations of advanced time-series models (e.g., CNN-LSTM, Transformers)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class CNNLSTMHybrid(BaseModel):
    """
    A hybrid CNN-LSTM model for time series forecasting or feature extraction.
    CNN layers extract local features, LSTM layers capture temporal dependencies.
    """

    def __init__(self, config: Dict[str, Any], input_channels: int, sequence_length: int):
        super().__init__(config)
        self.input_channels = input_channels  # Number of features per time step
        self.sequence_length = sequence_length  # Length of input sequence
        self.cnn_filters = config.get("cnn_filters", 64)
        self.cnn_kernel_size = config.get("cnn_kernel_size", 3)
        self.lstm_hidden_size = config.get("lstm_hidden_size", 128)
        self.lstm_num_layers = config.get("lstm_num_layers", 2)
        self.dropout = config.get("dropout", 0.3)

        # CNN layers to extract features from each time step
        # Input: (batch_size, input_channels, sequence_length)
        self.conv1 = nn.Conv1d(input_channels, self.cnn_filters, kernel_size=self.cnn_kernel_size, padding='same')
        self.conv2 = nn.Conv1d(self.cnn_filters, self.cnn_filters * 2, kernel_size=self.cnn_kernel_size, padding='same')
        # TODO: Add more CNN layers if needed

        # Calculate the output size of CNN before passing to LSTM
        # Assuming padding='same', sequence_length remains the same
        cnn_output_channels = self.cnn_filters * 2

        # LSTM layers
        # Input to LSTM: (batch_size, sequence_length, cnn_output_channels)
        self.lstm = nn.LSTM(cnn_output_channels, self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, batch_first=True, dropout=self.dropout)

        # Output layer (e.g., for a single prediction or a time series embedding)
        self.fc = nn.Linear(self.lstm_hidden_size, 1)  # Example: predicting a single value

        self.to(self.device)
        logger.info(
            f"Initialized CNNLSTMHybrid with input_channels={input_channels}, sequence_length={sequence_length}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN-LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_channels).
        Returns:
            torch.Tensor: Model output (e.g., predictions or embeddings).
        """
        # Permute to (batch_size, input_channels, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # TODO: Add pooling layers if desired (e.g., MaxPool1d)

        # Permute back to (batch_size, sequence_length, cnn_output_channels) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM expects (batch_size, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state for prediction or as an embedding
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # Take the last layer's hidden state
        last_hidden_state = h_n[-1, :, :]

        output = self.fc(last_hidden_state)  # Example: single prediction
        return output

    def train_model(self, train_loader: Any, val_loader: Any,
                    epochs: int = None, optimizer_name: str = None, loss_fn_name: str = None):
        """
        Trains the CNN-LSTM model.
        Args:
            train_loader (Any): DataLoader for training time series data.
            val_loader (Any): DataLoader for validation time series data.
            epochs (int): Number of training epochs. Defaults to config.
            optimizer_name (str): Name of the optimizer (e.g., "Adam"). Defaults to config.
            loss_fn_name (str): Name of the loss function (e.g., "MSELoss"). Defaults to config.
        """
        epochs = epochs if epochs is not None else self.config.get("epochs", 100)
        optimizer_name = optimizer_name if optimizer_name is not None else self.config.get("optimizer", "Adam")
        loss_fn_name = loss_fn_name if loss_fn_name is not None else self.config.get("loss_function", "MSELoss")

        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), lr=self.config.get("learning_rate", 0.001))
        loss_fn = getattr(nn, loss_fn_name)()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

            # TODO: Implement validation step and early stopping
            # self.eval()
            # with torch.no_grad():
            #     val_loss = 0
            #     for X_val, y_val in val_loader:
            #         X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            #         outputs_val = self(X_val)
            #         val_loss += loss_fn(outputs_val, y_val).item()
            #     avg_val_loss = val_loss / len(val_loader)
            #     logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
            # self.train()

        logger.info("CNNLSTMHybrid training complete.")

    def predict(self, data_loader: Any) -> torch.Tensor:
        """
        Makes predictions using the trained CNN-LSTM model.
        Args:
            data_loader (Any): DataLoader for time series data.
        Returns:
            torch.Tensor: Concatenated predictions.
        """
        self.eval()
        all_predictions = []
        with torch.no_grad():
            for X_batch, _ in data_loader:  # Assuming DataLoader yields (features, targets)
                X_batch = X_batch.to(self.device)
                predictions = self(X_batch)
                all_predictions.append(predictions.cpu())
        return torch.cat(all_predictions, dim=0)


class TransformerEncoderModel(BaseModel):
    """
    A Transformer Encoder model for time series data.
    Leverages self-attention to capture long-range dependencies.
    """

    def __init__(self, config: Dict[str, Any], input_dim: int, sequence_length: int):
        super().__init__(config)
        self.input_dim = input_dim  # Number of features per time step
        self.sequence_length = sequence_length  # Length of input sequence
        self.d_model = config.get("d_model", 128)  # Embedding dimension
        self.nhead = config.get("nhead", 8)  # Number of attention heads
        self.num_encoder_layers = config.get("num_encoder_layers", 3)
        self.dim_feedforward = config.get("dim_feedforward", 512)
        self.dropout = config.get("dropout", 0.1)

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, self.d_model)
        # Positional encoding (crucial for transformers with sequential data)
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True  # Input shape (batch_size, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Output layer
        self.fc = nn.Linear(self.d_model, 1)  # Example: single prediction

        self.to(self.device)
        logger.info(
            f"Initialized TransformerEncoderModel with input_dim={input_dim}, sequence_length={sequence_length}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            torch.Tensor: Model output (e.g., predictions or embeddings).
        """
        # Apply input embedding
        x = self.embedding(x)  # (batch_size, sequence_length, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]  # Ensure positional encoding matches sequence length

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)

        # Use the output of the last time step for prediction, or average, or use a pooling layer
        # For forecasting, often the last time step's output is used
        last_time_step_output = transformer_output[:, -1, :]  # (batch_size, d_model)

        output = self.fc(last_time_step_output)  # Example: single prediction
        return output

    def train_model(self, train_loader: Any, val_loader: Any,
                    epochs: int = None, optimizer_name: str = None, loss_fn_name: str = None):
        """
        Trains the Transformer Encoder model.
        (Similar training loop as CNNLSTMHybrid)
        """
        epochs = epochs if epochs is not None else self.config.get("epochs", 100)
        optimizer_name = optimizer_name if optimizer_name is not None else self.config.get("optimizer", "Adam")
        loss_fn_name = loss_fn_name if loss_fn_name is not None else self.config.get("loss_function", "MSELoss")

        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), lr=self.config.get("learning_rate", 0.001))
        loss_fn = getattr(nn, loss_fn_name)()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")
            # TODO: Implement validation step and early stopping

        logger.info("TransformerEncoderModel training complete.")

    def predict(self, data_loader: Any) -> torch.Tensor:
        """
        Makes predictions using the trained Transformer Encoder model.
        (Similar prediction logic as CNNLSTMHybrid)
        """
        self.eval()
        all_predictions = []
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                predictions = self(X_batch)
                all_predictions.append(predictions.cpu())
        return torch.cat(all_predictions, dim=0)

# TODO: Add other advanced time series models (e.g., Temporal Convolutional Networks (TCNs), DeepAR).
# TODO: Implement custom DataLoader for time series data (e.g., sliding window).
