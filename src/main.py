# src/main.py
import logging
import pandas as pd
import os
from datetime import datetime

# Import modules from your project structure
from src.utils.config_loader import load_all_configs
from src.data_sourcing import DATA_SOURCES
from src.processing.cleaner import clean_data, align_data
from src.processing.featurizer import generate_features
from src.models import BaseModel, LinearRegressionModel, RandomForestModel, GATModel, CNNLSTMHybrid, FusionModel, \
    BetaVAE
from src.backtesting.engine import BacktestingEngine
from src.backtesting.strategy_manager import TopNLongOnlyStrategy, SignalProportionalStrategy
from src.backtesting.slippage_commissions import FixedCommission, PercentageSlippage
from src.backtesting.rolling_window_validation import RollingWindowValidator
from src.backtesting.stress_testing import MarketRegimeTester
from src.backtesting.sensitivity_analysis import SensitivityAnalyzer
from src.reporting.report_generator import generate_backtest_report
from src.reporting.interpretability_analyzer import analyze_disentanglement, interpret_signals
from src.visualization.data_viz import plot_data_distribution, plot_time_series
from src.visualization.model_viz import plot_training_loss, plot_embeddings
from src.visualization.backtest_viz import plot_equity_curve, plot_drawdown

logger = logging.getLogger(__name__)


def setup_directories():
    """Ensures necessary data and output directories exist."""
    dirs = [
        "data/raw/yahoo", "data/raw/alpha_vantage", "data/raw/quandl",
        "data/raw/wrds", "data/raw/fred", "data/raw/edgar",
        "data/processed", "data/features", "models", "reports"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Project directories ensured.")


def run_data_engineering(configs: dict) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the data acquisition, cleaning, and feature engineering phases.
    Args:
        configs (dict): Loaded configuration dictionary.
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of processed dataframes (e.g., 'features_df', 'prices_df').
    """
    logger.info("--- Starting Data Engineering Phase ---")

    # 1. Data Acquisition
    acquired_data = {}
    data_sources_config = configs.get("data_sourcing", {})

    # Example: Fetching Yahoo Finance data
    yahoo_config = data_sources_config.get("yahoo_finance", {})
    yahoo_source = DATA_SOURCES["yahoo"](yahoo_config)
    # TODO: Define symbols and date ranges dynamically or from config
    try:
        aapl_prices = yahoo_source.get_data(symbol="AAPL", start_date="2010-01-01", end_date="2023-12-31")
        acquired_data['aapl_prices'] = aapl_prices
        # Save raw data
        aapl_prices.to_parquet("data/raw/yahoo/AAPL_prices.parquet")
        logger.info(f"Acquired AAPL prices: {aapl_prices.shape}")
    except Exception as e:
        logger.error(f"Failed to acquire AAPL prices: {e}")

    # Example: Fetching FRED data
    fred_config = data_sources_config.get("fred", {})
    fred_source = DATA_SOURCES["fred"](fred_config)
    try:
        macro_data = fred_source.fetch_multiple_series(series_ids=["GDP", "CPIAUCSL", "UNRATE"],
                                                       start_date="2010-01-01", end_date="2023-12-31")
        acquired_data['macro_data'] = macro_data
        macro_data.to_parquet("data/raw/fred/macro_data.parquet")
        logger.info(f"Acquired Macro data: {macro_data.shape}")
    except Exception as e:
        logger.error(f"Failed to acquire Macro data: {e}")

    # TODO: Acquire data from Alpha Vantage, Quandl, WRDS, EDGAR as per project needs.
    # Remember to handle their specific parameters and rate limits.

    # 2. Data Cleaning and Alignment
    logger.info("Starting data cleaning and alignment.")

    # Example: Clean and align AAPL prices and macro data
    if 'aapl_prices' in acquired_data and 'macro_data' in acquired_data:
        cleaned_aapl = clean_data(acquired_data['aapl_prices'])
        cleaned_macro = clean_data(acquired_data['macro_data'])

        # Align to daily frequency
        aligned_data = align_data({
            'prices': cleaned_aapl[['close', 'volume']],  # Select relevant columns
            'macro': cleaned_macro
        }, freq='D', how='outer', fill_method='ffill')

        aligned_data.to_parquet("data/processed/aligned_financial_macro.parquet")
        logger.info(f"Aligned data shape: {aligned_data.shape}")

        # For backtesting, we need the original prices separately
        prices_for_backtest = cleaned_aapl[['close']].rename(columns={'close': 'AAPL'})  # Example for single asset
        prices_for_backtest.to_parquet("data/processed/prices_for_backtest.parquet")
    else:
        aligned_data = pd.DataFrame()
        prices_for_backtest = pd.DataFrame()
        logger.warning("Insufficient data for alignment. Skipping this step.")

    # 3. Feature Engineering
    logger.info("Starting feature engineering.")
    features_df = pd.DataFrame()
    graph_features_dict = {}

    if not aligned_data.empty:
        # TODO: Pass correct column names for price, volume, VIX, yield curve etc.
        # These columns should be present in `aligned_data` after alignment.
        # Example: if 'prices_close' and 'prices_volume' are columns after alignment
        features_df, graph_features_dict = generate_features(
            aligned_data,
            price_col='prices_close',  # Example column name after alignment
            volume_col='prices_volume',
            # vix_col='macro_vix', # If VIX is in macro_data
            # yield_curve_col='macro_yield_spread' # If yield spread is in macro_data
        )
        features_df.to_parquet("data/features/all_features.parquet")
        logger.info(f"Generated features DataFrame shape: {features_df.shape}")
        # TODO: Save graph_features_dict (e.g., using pickle or a custom format)

    logger.info("--- Data Engineering Phase Complete ---")
    return {
        "features_df": features_df,
        "graph_features_dict": graph_features_dict,
        "prices_for_backtest": prices_for_backtest  # Important for backtesting
    }


def run_model_training(data: Dict[str, pd.DataFrame], configs: dict):
    """
    Orchestrates the model training phase.
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of processed dataframes from data engineering.
        configs (dict): Loaded configuration dictionary.
    """
    logger.info("--- Starting Model Training Phase ---")

    features_df = data.get("features_df")
    graph_features_dict = data.get("graph_features_dict")
    model_configs = configs.get("model_configs", {})

    if features_df.empty:
        logger.error("Features DataFrame is empty. Cannot proceed with model training.")
        return

    # TODO: Define your target variable (e.g., next day's return)
    # For demonstration, let's create a dummy target
    target_col = 'next_day_return'
    if target_col not in features_df.columns:
        features_df[target_col] = features_df['prices_close'].pct_change().shift(-1)  # Example: next day's return
        features_df = features_df.dropna()  # Drop rows with NaN targets

    if features_df.empty:
        logger.error("Features DataFrame is empty after target creation/dropna. Cannot train models.")
        return

    # Split data into train/validation sets
    # TODO: Implement robust time-series split (e.g., walk-forward validation)
    train_size = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Example: Train a Baseline Model (Random Forest)
    rf_config = model_configs.get("random_forest", {})
    rf_model = RandomForestModel(rf_config)
    rf_model.train_model(X_train, y_train, X_val, y_val)
    rf_model.save("models/baseline_model_v1/random_forest_model.joblib")
    logger.info("Random Forest Model trained and saved.")

    # Example: Train a GNN Model (GAT)
    # TODO: Prepare PyTorch Geometric Data objects from graph_features_dict
    # This requires converting NetworkX graphs to PyG Data objects.
    # from src.utils.graph_utils import create_dynamic_graph_dataset
    # pyg_dataset = create_dynamic_graph_dataset(graph_features_dict, node_feature_cols=X_train.columns.tolist())
    # For now, assume a dummy input_dim
    gnn_config = model_configs.get("gnn_gat", {})
    # gnn_model = GATModel(gnn_config, input_dim=X_train.shape[1]) # Input dim for node features
    # TODO: Create PyTorch DataLoaders for GNNs
    # gnn_model.train_model(train_loader=None, val_loader=None) # Placeholder
    # gnn_model.save("models/gnn_model_alpha_v2/gat_model.pt")
    logger.info("GNN Model (GAT) training placeholder executed.")

    # Example: Train a Time Series Model (CNNLSTMHybrid)
    # TODO: Prepare time series data for CNN-LSTM (e.g., sliding windows)
    # For now, assume a dummy input_channels and sequence_length
    ts_config = model_configs.get("cnn_lstm_hybrid", {})
    # ts_model = CNNLSTMHybrid(ts_config, input_channels=X_train.shape[1], sequence_length=20) # Example sequence length
    # TODO: Create PyTorch DataLoaders for time series models
    # ts_model.train_model(train_loader=None, val_loader=None) # Placeholder
    # ts_model.save("models/time_series_model_v1/cnn_lstm_model.pt")
    logger.info("Time Series Model (CNNLSTMHybrid) training placeholder executed.")

    # Example: Train a Fusion Model
    # TODO: This requires embeddings from GNN and TS models.
    # fusion_config = model_configs.get("fusion_model", {})
    # fusion_model = FusionModel(fusion_config, gnn_embedding_dim=32, ts_embedding_dim=1) # Example dims
    # fusion_model.train_model(train_loader=None, val_loader=None) # Placeholder
    # fusion_model.save("models/fusion_model_v1/fusion_model.pt")
    logger.info("Fusion Model training placeholder executed.")

    # Example: Train a Disentanglement Model (BetaVAE)
    # vae_config = model_configs.get("beta_vae", {})
    # vae_model = BetaVAE(vae_config, input_dim=X_train.shape[1])
    # vae_model.train_model(train_loader=None, val_loader=None) # Placeholder
    # vae_model.save("models/disentanglement_model_v1/beta_vae_model.pt")
    logger.info("Disentanglement Model (BetaVAE) training placeholder executed.")

    logger.info("--- Model Training Phase Complete ---")


def run_backtesting_and_evaluation(data: Dict[str, pd.DataFrame], configs: dict):
    """
    Orchestrates the backtesting and evaluation phases.
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of processed dataframes.
        configs (dict): Loaded configuration dictionary.
    """
    logger.info("--- Starting Backtesting and Evaluation Phase ---")

    features_df = data.get("features_df")
    prices_for_backtest = data.get("prices_for_backtest")
    backtest_configs = configs.get("backtest_configs", {})
    model_configs = configs.get("model_configs", {})  # To load model for signal generation

    if features_df.empty or prices_for_backtest.empty:
        logger.error("Features or prices data is empty. Cannot proceed with backtesting.")
        return

    # 1. Generate Alpha Signals (using a trained model)
    logger.info("Generating alpha signals for backtesting.")
    # TODO: Load your best trained model (e.g., FusionModel)
    # For demonstration, let's use a dummy signal generation
    # In a real scenario, you'd load the model, prepare test data, and call model.predict()
    # Example:
    # fusion_model = FusionModel(model_configs.get("fusion_model", {}), gnn_embedding_dim=32, ts_embedding_dim=1)
    # fusion_model.load("models/fusion_model_v1/fusion_model.pt")
    # alpha_signals_df = fusion_model.predict(test_data_loader) # This would be a tensor, convert to DataFrame

    # Dummy alpha signals for all assets in prices_for_backtest
    alpha_signals_df = pd.DataFrame(
        np.random.rand(len(features_df), len(prices_for_backtest.columns)),
        index=features_df.index,
        columns=prices_for_backtest.columns
    )
    logger.info(f"Generated dummy alpha signals shape: {alpha_signals_df.shape}")

    # 2. Initialize Backtesting Engine
    initial_capital = backtest_configs.get("initial_capital", 1000000)
    commission_rate = backtest_configs.get("commission_rate", 0.0005)
    slippage_bps = backtest_configs.get("slippage_bps", 1)

    commission_model = FixedCommission(commission_rate)
    slippage_model = PercentageSlippage(slippage_bps)

    engine = BacktestingEngine(
        initial_capital=initial_capital,
        commission_model=commission_model,
        slippage_model=slippage_model
    )

    # 3. Define Strategy
    strategy_config = {"top_n": 10, "min_signal_threshold": 0.5}  # Example config
    strategy = TopNLongOnlyStrategy(strategy_config)

    # 4. Run Rolling Window Validation
    logger.info("Running rolling window validation.")
    rolling_validator = RollingWindowValidator(
        config=backtest_configs.get("rolling_window", {}),
        backtesting_engine=engine,
        model_class=RandomForestModel,  # Use the model you trained
        strategy_class=TopNLongOnlyStrategy
    )
    rolling_results = rolling_validator.validate(
        full_data=features_df,  # This should contain features and targets
        model_config=model_configs.get("random_forest", {}),
        strategy_config=strategy_config,
        price_data_for_backtest=prices_for_backtest
    )

    if "full_portfolio_history" in rolling_results:
        full_portfolio_history = rolling_results["full_portfolio_history"]
        full_trade_log = rolling_results["full_trade_log"]
        logger.info(f"Overall Rolling Backtest Final Value: ${full_portfolio_history['total_value'].iloc[-1]:,.2f}")

        # Generate overall report
        generate_backtest_report(
            portfolio_history=full_portfolio_history,
            trade_log=full_trade_log,
            output_dir="reports/",
            report_name="overall_rolling_backtest_report",
            strategy_name="Rolling Alpha Strategy"
        )
    else:
        logger.warning("Rolling window validation did not produce full portfolio history.")

    # 5. Run Market Regime Stress Testing
    logger.info("Running market regime stress testing.")
    regime_tester = MarketRegimeTester(
        config=backtest_configs,  # Uses market_regimes from backtest_configs
        backtesting_engine=engine
    )
    # For stress testing, use the full period signals and prices
    regime_results = regime_tester.run_regime_tests(
        alpha_signals=alpha_signals_df,
        prices=prices_for_backtest,
        strategy=strategy
    )
    logger.info(f"Market Regime Test Results: {regime_results}")

    # 6. Run Sensitivity Analysis
    logger.info("Running sensitivity analysis.")
    sensitivity_analyzer = SensitivityAnalyzer(
        config=backtest_configs,  # Uses sensitivity_params from backtest_configs
        backtesting_engine=engine
    )
    sensitivity_results = sensitivity_analyzer.run_sensitivity_tests(
        alpha_signals=alpha_signals_df,
        prices=prices_for_backtest,
        base_strategy_config=strategy_config,
        strategy_class=TopNLongOnlyStrategy
    )
    logger.info(f"Sensitivity Analysis Results: {sensitivity_results}")

    # 7. Interpret Signals (Example)
    logger.info("Interpreting alpha signals.")
    # TODO: Load the actual trained model for interpretation
    # For now, use a dummy model or a simple sklearn model if trained
    # rf_model = RandomForestModel(model_configs.get("random_forest", {}))
    # rf_model.load("models/baseline_model_v1/random_forest_model.joblib")
    # interpretation_results = interpret_signals(
    #     model=rf_model.model, # Pass the sklearn model object
    #     features=features_df.drop(columns=[target_col]),
    #     signals=alpha_signals_df.iloc[:, 0], # Assuming signals is a single series
    #     method='feature_importance'
    # )
    # logger.info(f"Signal Interpretation Results: {interpretation_results}")

    # 8. Analyze Disentanglement (Example)
    logger.info("Analyzing disentangled factors.")
    # TODO: Load the BetaVAE model and get latent factors
    # vae_model = BetaVAE(model_configs.get("beta_vae", {}), input_dim=features_df.shape[1]-1)
    # vae_model.load("models/disentanglement_model_v1/beta_vae_model.pt")
    # latent_factors_tensor = vae_model.predict(features_df.drop(columns=[target_col])) # Needs DataLoader
    # latent_factors_df = pd.DataFrame(latent_factors_tensor.numpy(), index=features_df.index)
    # disentanglement_results = analyze_disentanglement(
    #     latent_factors=latent_factors_df,
    #     original_features=features_df.drop(columns=[target_col]),
    #     method='correlation'
    # )
    # logger.info(f"Disentanglement Analysis Results: {disentanglement_results}")

    logger.info("--- Backtesting and Evaluation Phase Complete ---")


def main():
    """Main function to run the entire quantitative project pipeline."""
    setup_directories()

    # Load all configurations
    configs = load_all_configs(config_dir="config/")
    if not configs:
        logger.error("No configurations loaded. Exiting.")
        return

    # Run Data Engineering
    processed_data = run_data_engineering(configs)

    # Run Model Training
    run_model_training(processed_data, configs)

    # Run Backtesting and Evaluation
    run_backtesting_and_evaluation(processed_data, configs)

    logger.info("--- Project Pipeline Execution Complete ---")


if __name__ == "__main__":
    # Configure logging for main script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
