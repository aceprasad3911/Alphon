-- upsert_features.sql
-- Example upsert statement for a feature table (Postgres)
INSERT INTO time_series_features (asset_id, date, feature_name, value)
VALUES (:asset_id, :date, :feature_name, :value)
ON CONFLICT (asset_id, date, feature_name)
DO UPDATE SET value = EXCLUDED.value;
