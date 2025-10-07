"""
graph_construction.py
Create firm relationship graphs (co-coverage, supply chain, sector similarity).
Outputs adjacency lists or edge tables used by GNNs.
"""
from pathlib import Path
import pandas as pd
import networkx as nx
from src.utils.logging_utils import setup_logger

logger = setup_logger("transform.graph", "reports/pipeline.log")
DATA_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

def build_sample_graph(symbols):
    G = nx.Graph()
    for s in symbols:
        G.add_node(s)
    # add synthetic edges (replace with real co-coverage / supply links)
    for i, a in enumerate(symbols):
        for b in symbols[i+1:i+4]:
            G.add_edge(a, b, weight=0.5)
    return G

def run():
    # placeholder: read list of assets from file or DB
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    G = build_sample_graph(symbols)
    out = DATA_PROCESSED / "firm_graph.edgelist"
    nx.write_edgelist(G, out)
    logger.info(f"Wrote firm graph to {out}")
