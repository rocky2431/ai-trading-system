-- IQFMP Database Initialization Script
-- Creates necessary extensions and base tables

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Market data table (hypertable)
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(30, 8),
    open_interest DECIMAL(30, 8),
    funding_rate DECIMAL(20, 10),
    PRIMARY KEY (time, symbol, exchange)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Create index for symbol queries
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, time DESC);

-- Factor values table (hypertable)
CREATE TABLE IF NOT EXISTS factor_values (
    time TIMESTAMPTZ NOT NULL,
    factor_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    value DECIMAL(30, 10),
    PRIMARY KEY (time, factor_id, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('factor_values', 'time', if_not_exists => TRUE);

-- Create index for factor queries
CREATE INDEX IF NOT EXISTS idx_factor_values_factor ON factor_values (factor_id, time DESC);

-- Factors metadata table
CREATE TABLE IF NOT EXISTS factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    family TEXT[] NOT NULL DEFAULT '{}',
    code TEXT NOT NULL,
    code_hash VARCHAR(64) NOT NULL UNIQUE,
    target_task VARCHAR(100),
    metrics JSONB,
    stability JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'candidate',
    cluster_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    experiment_number INTEGER NOT NULL DEFAULT 0
);

-- Create index for factor queries
CREATE INDEX IF NOT EXISTS idx_factors_status ON factors (status);
CREATE INDEX IF NOT EXISTS idx_factors_family ON factors USING GIN (family);

-- Research ledger table
CREATE TABLE IF NOT EXISTS research_ledger (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    factor_id UUID REFERENCES factors(id),
    code_hash VARCHAR(64) NOT NULL,
    prompt TEXT,
    config JSONB,
    metrics JSONB NOT NULL,
    passed BOOLEAN NOT NULL,
    rejection_reason TEXT,
    experiment_number SERIAL
);

-- Create index for ledger queries
CREATE INDEX IF NOT EXISTS idx_research_ledger_timestamp ON research_ledger (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_research_ledger_factor ON research_ledger (factor_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for factors table
CREATE TRIGGER update_factors_updated_at
    BEFORE UPDATE ON factors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO iqfmp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO iqfmp;
