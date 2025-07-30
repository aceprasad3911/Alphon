# src/visualization/model_viz.py

# Visualizations for model training and embeddings

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import numpy as np
from typing import List, Optional
from sklearn.manifold import TSNE # For t-SNE visualization
from sklearn.decomposition import PCA # For PCA visualization

logger = logging.getLogger(__name__)

def plot_training_loss(train_losses: List[float],
                       val_losses: Optional[List[float]] = None,
                       title: str = "Training and Validation Loss",
                       save_path: Optional[str] = None):
    """
    Plots the training and optional validation loss over epochs.
    Args:
        train_losses (List[float]): List of training loss values per epoch.
        val_losses (Optional[List[float]]): List of validation loss values per epoch.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot.
    """
    if not train_losses:
        logger.warning("No training loss data provided. Cannot plot training loss.")
        return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', marker='x', markersize=4)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_embeddings(embeddings: np.ndarray,
                    labels: Optional[np.ndarray] = None,
                    method: str = 'tsne',
                    title: str = "2D Embedding Visualization",
                    save_path: Optional[str] = None):
    """
    Visualizes high-dimensional embeddings in 2D using t-SNE or PCA.
    Args:
        embeddings (np.ndarray): High-dimensional embeddings (N_samples, N_features).
        labels (Optional[np.ndarray]): Optional labels for coloring points (e.g., asset sectors, clusters).
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot.
    """
    if embeddings.shape[0] == 0 or embeddings.shape[1] < 2:
        logger.warning("Embeddings array is empty or has insufficient dimensions for visualization.")
        return

    if embeddings.shape[1] == 2:
        reduced_embeddings = embeddings
        logger.info("Embeddings are already 2D. Skipping dimensionality reduction.")
    else:
        if method == 'tsne':
            logger.info("Applying t-SNE for dimensionality reduction...")
            # TODO: Adjust t-SNE parameters (perplexity, n_iter) based on dataset size
            reduced_embeddings = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1)).fit_transform(embeddings)
        elif method == 'pca':
            logger.info("Applying PCA for dimensionality reduction...")
            reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        else:
            logger.error(f"Unsupported dimensionality reduction method: {method}. Use 'tsne' or 'pca'.")
            return

    plt.figure(figsize=(10, 8))
    if labels is not None and len(labels) == reduced_embeddings.shape[0]:
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis', legend='full')
    else:
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], color='blue')

    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Embedding plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# TODO: Add functions for plotting attention weights from Transformer models.
# TODO: Add functions for visualizing graph structures (e.g., using networkx drawing).
