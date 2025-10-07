-- create_indexes.sql
CREATE INDEX IF NOT EXISTS idx_price_asset_date ON price_data (asset_id, date);
CREATE INDEX IF NOT EXISTS idx_features_asset_date ON time_series_features (asset_id, date);
