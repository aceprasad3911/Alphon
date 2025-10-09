# src/utils/graph_utils.py

# Helper functions for graph data manipulation

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def convert_networkx_to_pyg_data(
        graph: nx.Graph,
        node_features: Optional[pd.DataFrame] = None,
        edge_features: Optional[pd.DataFrame] = None,
        target_node_feature: Optional[str] = None
) -> Data:
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    Args:
        graph (nx.Graph): The NetworkX graph.
        node_features (Optional[pd.DataFrame]): DataFrame of node features. Index should be node IDs.
                                                Columns are feature names.
        edge_features (Optional[pd.DataFrame]): DataFrame of edge features. Index should be (u, v) tuples.
        target_node_feature (Optional[str]): Name of a column in node_features to be used as the target (y).
    Returns:
        torch_geometric.data.Data: The PyTorch Geometric Data object.
    """
    if graph.number_of_nodes() == 0:
        logger.warning("Empty NetworkX graph provided. Returning empty PyG Data object.")
        return Data(x=torch.empty(0, 0), edge_index=torch.empty(2, 0, dtype=torch.long))

    # 1. Node Features (x)
    x = None
    if node_features is not None:
        # Ensure node_features index matches graph nodes
        node_ids = list(graph.nodes())
        if not all(node in node_features.index for node in node_ids):
            logger.warning(
                "Node features index does not fully match graph nodes. Missing nodes will have NaN features.")
            # Reindex to ensure all graph nodes are present, filling missing with NaN
            node_features = node_features.reindex(node_ids)

        # Convert to tensor, handling potential NaNs
        x = torch.tensor(node_features.values, dtype=torch.float)
        if torch.isnan(x).any():
            logger.warning("NaNs found in node features. Consider imputation before conversion.")
            x = torch.nan_to_num(x, nan=0.0)  # Simple imputation for now

    # 2. Edge Index (edge_index)
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()

    # 3. Edge Attributes (edge_attr)
    edge_attr = None
    if edge_features is not None:
        # Ensure edge_features index matches graph edges
        graph_edges = list(graph.edges())
        # Convert graph_edges to a consistent format for indexing (e.g., frozenset for undirected)
        # Or ensure edge_features index is tuples (u,v)

        # TODO: More robust matching of edge_features to graph.edges()
        # For now, assume edge_features are ordered correctly or indexed by (u,v) tuples
        edge_attr = torch.tensor(edge_features.values, dtype=torch.float)
        if torch.isnan(edge_attr).any():
            logger.warning("NaNs found in edge features. Consider imputation before conversion.")
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

    # 4. Target (y)
    y = None
    if target_node_feature and node_features is not None and target_node_feature in node_features.columns:
        y = torch.tensor(node_features[target_node_feature].values, dtype=torch.float)
        if torch.isnan(y).any():
            logger.warning("NaNs found in target node feature. Consider imputation or filtering.")
            y = torch.nan_to_num(y, nan=0.0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    logger.debug(f"Converted NetworkX graph to PyG Data object: {data}")
    return data


def create_dynamic_graph_dataset(
        graph_features_dict: Dict[pd.Timestamp, Dict[str, Any]],
        node_feature_cols: Optional[List[str]] = None,
        target_node_feature: Optional[str] = None
) -> List[Data]:
    """
    Converts a dictionary of dynamic graph features (from featurizer) into a list of PyG Data objects.
    Args:
        graph_features_dict (Dict[pd.Timestamp, Dict[str, Any]]): Output from generate_graph_features.
        node_feature_cols (Optional[List[str]]): List of columns from 'node_features' DataFrame to use as 'x'.
                                                 If None, all numeric columns are used.
        target_node_feature (Optional[str]): Name of a column in 'node_features' to be used as the target (y).
    Returns:
        List[torch_geometric.data.Data]: A list of PyTorch Geometric Data objects, one for each timestamp.
    """
    pyg_dataset = []
    for timestamp, graph_data in sorted(graph_features_dict.items()):
        graph = graph_data['graph']
        node_features_df = graph_data['node_features']

        if node_feature_cols:
            selected_node_features = node_features_df[node_feature_cols]
        else:
            selected_node_features = node_features_df.select_dtypes(include=np.number)  # Use all numeric

        if selected_node_features.empty:
            logger.warning(f"No numeric node features found for graph at {timestamp}. Skipping.")
            continue

        pyg_data = convert_networkx_to_pyg_data(
            graph,
            node_features=selected_node_features,
            target_node_feature=target_node_feature
            # TODO: Add edge features if available in graph_data
        )
        # Add timestamp as an attribute to the PyG Data object for later use
        pyg_data.date = timestamp
        pyg_dataset.append(pyg_data)

    logger.info(f"Created PyG dataset with {len(pyg_dataset)} graph snapshots.")
    return pyg_dataset

# TODO: Implement functions for graph pooling (e.g., for global graph-level embeddings).
# TODO: Add functions for graph normalization (e.g., symmetric normalization for GCN).
