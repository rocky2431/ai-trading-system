-- Migration: Add new columns to research_trials table
-- Date: 2025-12-10
-- Description: Add trial_id, factor_family, win_rate columns for enhanced trial tracking

-- Add trial_id column (unique identifier for trial records)
ALTER TABLE research_trials ADD COLUMN IF NOT EXISTS trial_id VARCHAR(36);

-- Create a unique trial_id for existing rows
UPDATE research_trials SET trial_id = CONCAT('trial_', id) WHERE trial_id IS NULL;

-- Make trial_id NOT NULL after populating
ALTER TABLE research_trials ALTER COLUMN trial_id SET NOT NULL;

-- Create unique index on trial_id
CREATE UNIQUE INDEX IF NOT EXISTS ix_research_trials_trial_id ON research_trials(trial_id);

-- Add factor_family column
ALTER TABLE research_trials ADD COLUMN IF NOT EXISTS factor_family VARCHAR(50) DEFAULT 'unknown';

-- Update factor_family to NOT NULL
ALTER TABLE research_trials ALTER COLUMN factor_family SET NOT NULL;

-- Create index on factor_family
CREATE INDEX IF NOT EXISTS ix_research_trials_factor_family ON research_trials(factor_family);

-- Add win_rate column
ALTER TABLE research_trials ADD COLUMN IF NOT EXISTS win_rate FLOAT;

-- Verify columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'research_trials'
ORDER BY ordinal_position;
