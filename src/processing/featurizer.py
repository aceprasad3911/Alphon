# src/processing/featurizer.py

# Contains functions for computing all types of features (technical, fundamental, graph-based, time-series specific)

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx
from scipy.stats import skew, kurtosis
from PyWavelets import wavedec # For wavelet transforms
from statsmodels.tsa.stattools import acf, pacf # For auto/partial correlation
from arch import arch_model # For GARCH-related features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters # For automated feature extraction

logger = logging.getLogger(__name__)

def generate_technical_indicators(df: pd.DataFrame,
                                  price_col: str = 'close',
                                  volume_col: str = 'volume') -> pd.DataFrame:
    """
    Generates a set of common technical indicators.
    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        price_col (str): Name of the column to use for price calculations (e.g., 'close', 'adj_close').
        volume_col (str): Name of the column to use for volume calculations.
    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns.
    """
    if df.empty or price_col not in df.columns or volume_col not in df.columns:
        logger.warning("Input DataFrame is empty or missing required columns for technical indicators.")
        return df

    ti_df = df.copy()

    # Ensure numeric types
    for col in [price_col, volume_col]:
        if col in ti_df.columns:
            ti_df[col] = pd.to_numeric(ti_df[col], errors='coerce')

    # Simple Moving Averages (SMA)
    ti_df['sma_10'] = ti_df[price_col].rolling(window=10).mean()
    ti_df['sma_20'] = ti_df[price_col].rolling(window=20).mean()
    ti_df['sma_50'] = ti_df[price_col].rolling(window=50).mean()

    # Exponential Moving Averages (EMA)
    ti_df['ema_10'] = ti_df[price_col].ewm(span=10, adjust=False).mean()
    ti_df['ema_20'] = ti_df[price_col].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = ti_df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ti_df['rsi'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = ti_df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = ti_df[price_col].ewm(span=26, adjust=False).mean()
    ti_df['macd'] = exp1 - exp2
    ti_df['macd_signal'] = ti_df['macd'].ewm(span=9, adjust=False).mean()
    ti_df['macd_hist'] = ti_df['macd'] - ti_df['macd_signal']

    # Bollinger Bands
    ti_df['bollinger_mid'] = ti_df[price_col].rolling(window=20).mean()
    ti_df['bollinger_std'] = ti_df[price_col].rolling(window=20).std()
    ti_df['bollinger_upper'] = ti_df['bollinger_mid'] + (ti_df['bollinger_std'] * 2)
    ti_df['bollinger_lower'] = ti_df['bollinger_mid'] - (ti_df['bollinger_std'] * 2)

    # Volatility (Standard Deviation of Returns)
    ti_df['returns'] = ti_df[price_col].pct_change()
    ti_df['volatility_20d'] = ti_df['returns'].rolling(window=20).std() * np.sqrt(252) # Annualized

    # On-Balance Volume (OBV)
    # TODO: Ensure 'close' and 'volume' are correctly handled for OBV
    # obv_direction = np.sign(ti_df[price_col].diff())
    # ti_df['obv'] = (obv_direction * ti_df[volume_col]).cumsum()

    # TODO: Add more indicators: Stochastic Oscillator, ADX, Ichimoku Cloud, etc.
    # TODO: Consider using a dedicated TA library like `ta` for more robust indicator generation.

    logger.info("Technical indicators generated.")
    return ti_df.drop(columns=['returns'], errors='ignore') # Drop intermediate columns

def generate_fundamental_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates fundamental ratios from raw fundamental data.
    Assumes input DataFrame contains columns like 'price', 'eps', 'ebitda', 'equity', 'debt', 'cash_flow'.
    Args:
        df (pd.DataFrame): DataFrame with fundamental data.
    Returns:
        pd.DataFrame: DataFrame with added fundamental ratio columns.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty for fundamental ratio generation.")
        return df

    fr_df = df.copy()

    # Ensure numeric types
    for col in ['price', 'eps', 'ebitda', 'equity', 'debt', 'cash_flow']:
        if col in fr_df.columns:
            fr_df[col] = pd.to_numeric(fr_df[col], errors='coerce')

    # P/E Ratio
    if 'price' in fr_df.columns and 'eps' in fr_df.columns:
        fr_df['pe_ratio'] = fr_df['price'] / fr_df['eps']
        fr_df['pe_ratio'] = fr_df['pe_ratio'].replace([np.inf, -np.inf], np.nan) # Handle division by zero

    # EV/EBITDA
    # TODO: Requires Market Cap, Debt, Cash, EBITDA. Assume 'market_cap', 'total_debt', 'cash_and_equivalents'
    if 'market_cap' in fr_df.columns and 'total_debt' in fr_df.columns and \
       'cash_and_equivalents' in fr_df.columns and 'ebitda' in fr_df.columns:
        fr_df['enterprise_value'] = fr_df['market_cap'] + fr_df['total_debt'] - fr_df['cash_and_equivalents']
        fr_df['ev_ebitda'] = fr_df['enterprise_value'] / fr_df['ebitda']
        fr_df['ev_ebitda'] = fr_df['ev_ebitda'].replace([np.inf, -np.inf], np.nan)

    # ROE (Return on Equity)
    # TODO: Requires Net Income and Shareholder Equity. Assume 'net_income', 'shareholder_equity'
    if 'net_income' in fr_df.columns and 'shareholder_equity' in fr_df.columns:
        fr_df['roe'] = fr_df['net_income'] / fr_df['shareholder_equity']
        fr_df['roe'] = fr_df['roe'].replace([np.inf, -np.inf], np.nan)

    # Debt/Equity Ratio
    if 'total_debt' in fr_df.columns and 'shareholder_equity' in fr_df.columns:
        fr_df['debt_equity_ratio'] = fr_df['total_debt'] / fr_df['shareholder_equity']
        fr_df['debt_equity_ratio'] = fr_df['debt_equity_ratio'].replace([np.inf, -np.inf], np.nan)

    # Cash Flow Ratios (e.g., Operating Cash Flow Ratio)
    # TODO: Requires Operating Cash Flow and Sales/Revenue. Assume 'operating_cash_flow', 'revenue'
    if 'operating_cash_flow' in fr_df.columns and 'revenue' in fr_df.columns:
        fr_df['operating_cash_flow_ratio'] = fr_df['operating_cash_flow'] / fr_df['revenue']
        fr_df['operating_cash_flow_ratio'] = fr_df['operating_cash_flow_ratio'].replace([np.inf, -np.inf], np.nan)

    logger.info("Fundamental ratios generated.")
    return fr_df

def generate_time_series_features(series: pd.Series) -> pd.Series:
    """
    Generates advanced time-series specific features for a single series.
    Args:
        series (pd.Series): A pandas Series with a DatetimeIndex.
    Returns:
        pd.Series: A Series containing the generated features.
    """
    if series.empty:
        logger.warning("Input Series is empty for time series feature generation.")
        return pd.Series()

    features = {}
    series = pd.to_numeric(series, errors='coerce').dropna() # Ensure numeric and no NaNs

    if series.empty:
        return pd.Series()

    # Partial/Auto correlation
    try:
        features['acf_lag1'] = acf(series, nlags=1, fft=True)[1] # Autocorrelation at lag 1
        features['pacf_lag1'] = pacf(series, nlags=1)[1] # Partial autocorrelation at lag 1
    except Exception as e:
        logger.warning(f"Could not compute ACF/PACF: {e}")
        features['acf_lag1'] = np.nan
        features['pacf_lag1'] = np.nan

    # Hurst Exponent (using a simple R/S analysis approximation)
    # TODO: Use a more robust Hurst exponent calculation library if available.
    try:
        lags = range(2, 20)
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        features['hurst_exponent'] = poly[0]
    except Exception as e:
        logger.warning(f"Could not compute Hurst Exponent: {e}")
        features['hurst_exponent'] = np.nan

    # Wavelet Coefficients (example using Daubechies 4 wavelet)
    try:
        coeffs = wavedec(series.values, 'db4', level=2)
        # Use energy of approximation and detail coefficients as features
        features['wavelet_approx_energy'] = np.sum(np.array(coeffs[0])**2)
        features['wavelet_detail1_energy'] = np.sum(np.array(coeffs[1])**2)
        features['wavelet_detail2_energy'] = np.sum(np.array(coeffs[2])**2)
    except Exception as e:
        logger.warning(f"Could not compute Wavelet Coefficients: {e}")
        features['wavelet_approx_energy'] = np.nan
        features['wavelet_detail1_energy'] = np.nan
        features['wavelet_detail2_energy'] = np.nan

    # Statistical Moments of Returns (Skewness, Kurtosis)
    returns = series.pct_change().dropna()
    if not returns.empty:
        features['returns_skewness'] = skew(returns)
        features['returns_kurtosis'] = kurtosis(returns)
    else:
        features['returns_skewness'] = np.nan
        features['returns_kurtosis'] = np.nan

    # TODO: Add fractional differencing (requires more complex implementation).
    # TODO: Add features from GARCH models (e.g., conditional volatility).
    # TODO: Consider using tsfresh for automated feature extraction from time series.

    logger.debug("Time series features generated.")
    return pd.Series(features)

def generate_graph_features(
    price_df: pd.DataFrame,
    gics_mapping: Optional[Dict[str, str]] = None,
    correlation_window: int = 60,
    correlation_threshold: float = 0.7
) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    Generates dynamic graph features based on asset relationships.
    Args:
        price_df (pd.DataFrame): DataFrame with asset prices, columns are symbols, index is DatetimeIndex.
        gics_mapping (Optional[Dict[str, str]]): Dictionary mapping ticker symbols to GICS sectors.
        correlation_window (int): Rolling window for correlation calculation.
        correlation_threshold (float): Minimum correlation to form an edge.
    Returns:
        Dict[pd.Timestamp, Dict[str, Any]]: A dictionary where keys are timestamps, and values are
                                            dictionaries containing graph-based features for that time.
                                            Each inner dict might contain:
                                            - 'graph': networkx graph object
                                            - 'node_features': DataFrame of node-level features
                                            - 'global_features': Series of graph-level features
    """
    if price_df.empty:
        logger.warning("Input price DataFrame is empty for graph feature generation.")
        return {}

    graph_features_over_time = {}
    symbols = price_df.columns.tolist()

    # Ensure prices are numeric
    price_df = price_df.apply(pd.to_numeric, errors='coerce')

    # Calculate daily returns for correlation
    returns_df = price_df.pct_change().dropna()

    # Iterate through time to create dynamic graphs
    for i in range(correlation_window, len(returns_df)):
        window_returns = returns_df.iloc[i - correlation_window : i]
        current_date = returns_df.index[i]

        if window_returns.shape[0] < correlation_window:
            continue # Not enough data for the window

        # 1. Correlation Graph
        correlation_matrix = window_returns.corr()
        G_corr = nx.Graph()
        G_corr.add_nodes_from(symbols)

        for s1 in symbols:
            for s2 in symbols:
                if s1 != s2 and s1 in correlation_matrix.index and s2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[s1, s2]
                    if pd.notna(corr) and abs(corr) >= correlation_threshold:
                        G_corr.add_edge(s1, s2, weight=corr)

        # 2. Sector Graph (if GICS mapping is provided)
        G_sector = nx.Graph()
        if gics_mapping:
            sectors = list(set(gics_mapping.values()))
            for sector in sectors:
                G_sector.add_node(sector)
            for s1 in symbols:
                for s2 in symbols:
                    if s1 != s2 and gics_mapping.get(s1) == gics_mapping.get(s2) and gics_mapping.get(s1) is not None:
                        G_sector.add_edge(s1, s2, type='same_sector')
            # TODO: Add edges between sectors based on industry relationships if available.

        # Combine graphs or keep separate for different GNN inputs
        # For simplicity, let's use G_corr for now.
        current_graph = G_corr

        # Node-level features
        node_features = pd.DataFrame(index=current_graph.nodes)
        if current_graph.nodes:
            # Centrality Measures
            node_features['degree_centrality'] = pd.Series(nx.degree_centrality(current_graph))
            node_features['betweenness_centrality'] = pd.Series(nx.betweenness_centrality(current_graph))
            node_features['eigenvector_centrality'] = pd.Series(nx.eigenvector_centrality(current_graph, max_iter=1000))
            # TODO: Add PageRank, Closeness Centrality, etc.

            # Community Membership (requires a community detection algorithm)
            # TODO: Implement community detection (e.g., Louvain, Leiden) and add as node feature.

            # Node Embeddings (will be generated by GNNs later, but can be initialized here)
            # node_features['initial_embedding'] = ...

        # Global graph-level features
        global_features = pd.Series()
        if current_graph.nodes:
            global_features['num_nodes'] = current_graph.number_of_nodes()
            global_features['num_edges'] = current_graph.number_of_edges()
            global_features['graph_density'] = nx.density(current_graph)
            global_features['avg_clustering_coefficient'] = nx.average_clustering(current_graph)
            # TODO: Add more global features: connected components, diameter, etc.

        graph_features_over_time[current_date] = {
            'graph': current_graph,
            'node_features': node_features,
            'global_features': global_features
        }

    logger.info(f"Generated graph features for {len(graph_features_over_time)} time steps.")
    return graph_features_over_time

def generate_regime_indicators(df: pd.DataFrame,
                               vix_col: str = 'vix_close',
                               yield_curve_col: str = 'yield_curve_spread') -> pd.DataFrame:
    """
    Generates market regime indicators.
    Args:
        df (pd.DataFrame): DataFrame containing relevant columns like VIX, yield curve spread.
        vix_col (str): Column name for VIX data.
        yield_curve_col (str): Column name for yield curve spread data.
    Returns:
        pd.DataFrame: DataFrame with added regime indicator columns.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty for regime indicator generation.")
        return df

    ri_df = df.copy()

    # VIX-based regime (e.g., high vs. low volatility)
    if vix_col in ri_df.columns:
        ri_df['vix_regime_high'] = (ri_df[vix_col] > ri_df[vix_col].rolling(window=60).mean() * 1.2).astype(int)
        ri_df['vix_regime_low'] = (ri_df[vix_col] < ri_df[vix_col].rolling(window=60).mean() * 0.8).astype(int)
        # TODO: Define more sophisticated VIX regimes (e.g., using thresholds, percentile ranks).

    # Drawdown Markers
    # Requires a portfolio equity curve or benchmark index
    # TODO: Implement drawdown calculation and marking based on a 'portfolio_value' or 'benchmark_price' column.
    # Example:
    # if 'portfolio_value' in ri_df.columns:
    #     rolling_max = ri_df['portfolio_value'].cummax()
    #     drawdown = (ri_df['portfolio_value'] / rolling_max) - 1
    #     ri_df['in_drawdown'] = (drawdown < -0.10).astype(int) # Example: 10% drawdown

    # Yield Curve Slope (e.g., 10Y-2Y spread)
    if yield_curve_col in ri_df.columns:
        ri_df['yield_curve_inverted'] = (ri_df[yield_curve_col] < 0).astype(int)
        # TODO: Define more nuanced yield curve regimes (e.g., steepening, flattening).

    # Credit Spreads
    # TODO: Integrate credit spread data (e.g., BAA-AAA corporate bond spread) and define regimes.

    # Macroeconomic Cycles (e.g., NBER recession indicators)
    # TODO: Integrate NBER recession data or other macro cycle indicators.

    logger.info("Market regime indicators generated.")
    return ri_df

def generate_features(
    df: pd.DataFrame,
    gics_mapping: Optional[Dict[str, str]] = None,
    price_col: str = 'close',
    volume_col: str = 'volume',
    vix_col: str = 'vix_close',
    yield_curve_col: str = 'yield_curve_spread'
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, Dict[str, Any]]]:
    """
    Orchestrates the generation of all feature types.
    Args:
        df (pd.DataFrame): The main DataFrame containing cleaned and aligned data.
                           Expected to have a DatetimeIndex and columns for prices, volume, fundamentals, etc.
        gics_mapping (Optional[Dict[str, str]]): Mapping for graph features.
        price_col (str): Column name for price data.
        volume_col (str): Column name for volume data.
        vix_col (str): Column name for VIX data.
        yield_curve_col (str): Column name for yield curve spread data.
    Returns:
        Tuple[pd.DataFrame, Dict[pd.Timestamp, Dict[str, Any]]]:
            - pd.DataFrame: DataFrame with all generated numerical features.
            - Dict[pd.Timestamp, Dict[str, Any]]: Dictionary of dynamic graph features.
    """
    logger.info("Starting feature generation process.")

    # 1. Generate Technical Indicators
    features_df = generate_technical_indicators(df, price_col=price_col, volume_col=volume_col)

    # 2. Generate Fundamental Ratios (assuming fundamental data is part of df)
    features_df = generate_fundamental_ratios(features_df) # Pass the df with technicals

    # 3. Generate Time-Series Specific Features for key series
    # This needs to be applied per asset or per relevant time series
    # Example: Apply to 'returns' or 'close' price for each asset
    ts_features_list = []
    # Assuming df has a MultiIndex (Date, Symbol) or symbols are columns
    if isinstance(features_df.columns, pd.MultiIndex): # If data is multi-indexed by symbol
        for symbol in features_df.columns.levels[1]:
            symbol_df = features_df.xs(symbol, level=1, axis=1)
            if price_col in symbol_df.columns:
                ts_feats = generate_time_series_features(symbol_df[price_col])
                ts_feats = ts_feats.add_prefix(f'ts_{symbol}_')
                ts_features_list.append(ts_feats)
        # TODO: How to merge these back into the main DataFrame? This is complex for multi-asset.
        # For now, let's assume a single asset or simplify the merge.
        # A common approach is to create a separate TS feature DF and merge later.
        # For simplicity, let's assume a single asset for now or that these features are global.
        logger.warning("Time-series specific features for multi-asset data needs careful merging strategy.")
    else: # Single asset or features applied globally
        if price_col in features_df.columns:
            ts_feats = generate_time_series_features(features_df[price_col])
            # Add these as new columns, broadcasting across rows if needed (e.g., for global features)
            for col, val in ts_feats.items():
                features_df[col] = val # This will broadcast the single value to all rows
            logger.warning("Time-series specific features are currently broadcasted. Adjust if per-asset needed.")


    # 4. Generate Regime Indicators
    features_df = generate_regime_indicators(features_df, vix_col=vix_col, yield_curve_col=yield_curve_col)

    # 5. Generate Graph-Based Features (returns a separate structure)
    # This assumes price_df is structured with symbols as columns and date as index
    graph_features_dict = generate_graph_features(df.filter(like=price_col), gics_mapping,
                                                  correlation_window=60, correlation_threshold=0.7)

    logger.info("Feature generation process completed.")
    return features_df, graph_features_dict

# TODO: Implement feature scaling/normalization (e.g., StandardScaler, MinMaxScaler).
# TODO: Implement feature selection techniques (e.g., PCA, mutual information, tree-based importance).
# TODO: Add more sophisticated graph construction methods (e.g., based on supply chains, ownership).
# TODO: Ensure features are generated in a way that avoids look-ahead bias.
