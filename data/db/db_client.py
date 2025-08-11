# db_client.py

# Initializes PostgreSQL database with the Alpha Signal Discovery schema.

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, BigInteger, String, Date,
    Numeric, Boolean, Text, JSON, ForeignKey, TIMESTAMP
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# --------------------
# CONFIGURE DB CONNECTION
# --------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password_here")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "alpha_signals")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --------------------
# SQLAlchemy Base & Engine
# --------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# --------------------
# 1. Asset & Universe Management
# --------------------
class Asset(Base):
    __tablename__ = "assets"
    asset_id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(50))
    currency = Column(String(3))
    listing_exchange = Column(String(50))
    inception_date = Column(Date)
    isin = Column(String(20))
    sedol = Column(String(20))
    active_flag = Column(Boolean, default=True)

class AssetUniverseVersion(Base):
    __tablename__ = "asset_universe_versions"
    universe_id = Column(Integer, primary_key=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP)
    effective_start_date = Column(Date)
    effective_end_date = Column(Date)

class AssetUniverseMember(Base):
    __tablename__ = "asset_universe_members"
    universe_id = Column(Integer, ForeignKey("asset_universe_versions.universe_id"), primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"), primary_key=True)
    weight = Column(Numeric(12,6))

# --------------------
# 2. Raw Data Storage
# --------------------
class PriceData(Base):
    __tablename__ = "price_data"
    price_id = Column(BigInteger, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date, nullable=False)
    open_price = Column(Numeric(18,6))
    high_price = Column(Numeric(18,6))
    low_price = Column(Numeric(18,6))
    close_price = Column(Numeric(18,6))
    adj_close_price = Column(Numeric(18,6))
    volume = Column(BigInteger)
    adjustment_factor = Column(Numeric(12,6))
    data_source = Column(String(100))

class Fundamentals(Base):
    __tablename__ = "fundamentals"
    fund_id = Column(BigInteger, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    report_date = Column(Date, nullable=False)
    fiscal_quarter = Column(String(10))
    pe_ratio = Column(Numeric(12,6))
    ev_ebitda = Column(Numeric(12,6))
    roe = Column(Numeric(12,6))
    debt_equity = Column(Numeric(12,6))
    cash_flow_ratio = Column(Numeric(12,6))
    data_source = Column(String(100))

class MacroIndicator(Base):
    __tablename__ = "macro_indicators"
    macro_id = Column(BigInteger, primary_key=True)
    indicator_code = Column(String(50))
    indicator_name = Column(String(255))
    date = Column(Date, nullable=False)
    value = Column(Numeric(18,6))
    frequency = Column(String(20))
    data_source = Column(String(100))

class RegimeIndicator(Base):
    __tablename__ = "regime_indicators"
    regime_id = Column(BigInteger, primary_key=True)
    date = Column(Date, nullable=False)
    vix = Column(Numeric(12,6))
    drawdown_marker = Column(Numeric(12,6))
    yield_curve_slope = Column(Numeric(12,6))
    credit_spread = Column(Numeric(12,6))
    macro_cycle = Column(String(50))
    data_source = Column(String(100))

# --------------------
# 3. Feature Engineering
# --------------------
class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"
    tech_id = Column(BigInteger, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date, nullable=False)
    momentum = Column(Numeric(18,6))
    volatility = Column(Numeric(18,6))
    obv = Column(Numeric(18,6))
    mean_reversion = Column(Numeric(18,6))
    bollinger_band = Column(Numeric(18,6))
    ema_fast = Column(Numeric(18,6))
    ema_slow = Column(Numeric(18,6))

class GraphFeature(Base):
    __tablename__ = "graph_features"
    graph_feat_id = Column(BigInteger, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date, nullable=False)
    node_embedding = Column(JSONB)
    degree_centrality = Column(Numeric(18,6))
    betweenness_central = Column(Numeric(18,6))
    eigenvector_central = Column(Numeric(18,6))
    community_id = Column(Integer)
    graph_density = Column(Numeric(18,6))
    clustering_coeff = Column(Numeric(18,6))

class TimeSeriesFeature(Base):
    __tablename__ = "time_series_features"
    ts_feat_id = Column(BigInteger, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date, nullable=False)
    pacf = Column(Numeric(18,6))
    hurst_exp = Column(Numeric(18,6))
    wavelet_coeffs = Column(JSONB)
    skewness = Column(Numeric(18,6))
    kurtosis = Column(Numeric(18,6))

# --------------------
# 4. Model Lifecycle & Experiment Tracking
# --------------------
class ModelRun(Base):
    __tablename__ = "model_runs"
    run_id = Column(BigInteger, primary_key=True)
    model_name = Column(String(255))
    model_type = Column(String(100))
    start_date = Column(Date)
    end_date = Column(Date)
    parameters = Column(JSONB)
    training_metrics = Column(JSONB)
    version_tag = Column(String(50))

class ExperimentTag(Base):
    __tablename__ = "experiment_tags"
    experiment_id = Column(BigInteger, primary_key=True)
    experiment_name = Column(String(255))
    description = Column(Text)
    created_at = Column(TIMESTAMP)

class ModelExperimentLink(Base):
    __tablename__ = "model_experiment_link"
    run_id = Column(BigInteger, ForeignKey("model_runs.run_id"), primary_key=True)
    experiment_id = Column(BigInteger, ForeignKey("experiment_tags.experiment_id"), primary_key=True)

class ValidationFold(Base):
    __tablename__ = "validation_folds"
    fold_id = Column(BigInteger, primary_key=True)
    run_id = Column(BigInteger, ForeignKey("model_runs.run_id"))
    fold_number = Column(Integer)
    start_date = Column(Date)
    end_date = Column(Date)
    metrics = Column(JSONB)

class ModelExplanation(Base):
    __tablename__ = "model_explanations"
    explain_id = Column(BigInteger, primary_key=True)
    run_id = Column(BigInteger, ForeignKey("model_runs.run_id"))
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date)
    method = Column(String(50))
    explanation = Column(JSONB)

class AlphaSignal(Base):
    __tablename__ = "alpha_signals"
    signal_id = Column(BigInteger, primary_key=True)
    run_id = Column(BigInteger, ForeignKey("model_runs.run_id"))
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    date = Column(Date)
    signal_value = Column(Numeric(18,6))
    confidence = Column(Numeric(12,6))

# --------------------
# 5. Backtesting & Portfolio Tracking
# --------------------
class BacktestResult(Base):
    __tablename__ = "backtest_results"
    backtest_id = Column(BigInteger, primary_key=True)
    run_id = Column(BigInteger, ForeignKey("model_runs.run_id"))
    sharpe_ratio = Column(Numeric(12,6))
    sortino_ratio = Column(Numeric(12,6))
    max_drawdown = Column(Numeric(12,6))
    annualized_return = Column(Numeric(12,6))
    turnover = Column(Numeric(12,6))
    validation_type = Column(String(50))

class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    holding_id = Column(BigInteger, primary_key=True)
    backtest_id = Column(BigInteger, ForeignKey("backtest_results.backtest_id"))
    date = Column(Date)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    weight = Column(Numeric(12,6))
    position_size = Column(Numeric(18,6))

class TradeLog(Base):
    __tablename__ = "trade_log"
    trade_id = Column(BigInteger, primary_key=True)
    backtest_id = Column(BigInteger, ForeignKey("backtest_results.backtest_id"))
    date = Column(Date)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))
    action = Column(String(10))
    quantity = Column(Numeric(18,6))
    price = Column(Numeric(18,6))

# --------------------
# 6. Data Pipeline Management
# --------------------
class DataSourceLog(Base):
    __tablename__ = "data_source_log"
    log_id = Column(BigInteger, primary_key=True)
    source_name = Column(String(100))
    endpoint = Column(String(255))
    request_time = Column(TIMESTAMP)
    status_code = Column(Integer)
    records_fetched = Column(Integer)
    error_message = Column(Text)

class PreprocessingStep(Base):
    __tablename__ = "preprocessing_steps"
    step_id = Column(BigInteger, primary_key=True)
    dataset_name = Column(String(100))
    step_order = Column(Integer)
    description = Column(Text)
    parameters = Column(JSONB)
    executed_at = Column(TIMESTAMP)

class RawDataCache(Base):
    __tablename__ = "raw_data_cache"
    cache_id = Column(BigInteger, primary_key=True)
    source_name = Column(String(100))
    endpoint = Column(String(255))
    request_time = Column(TIMESTAMP)
    raw_payload = Column(JSONB)
    asset_id = Column(Integer, ForeignKey("assets.asset_id"))

# --------------------
# MAIN EXECUTION
# --------------------
if __name__ == "__main__":
    print("Creating all tables in the database...")
    Base.metadata.create_all(engine)
    print("✅ Database initialized successfully!")
