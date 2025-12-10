-- Migration: Add pipeline_runs table
-- Date: 2025-12-10
-- Description: Add table for persistent pipeline execution tracking

-- Create pipeline_runs table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id VARCHAR(36) PRIMARY KEY,
    pipeline_type VARCHAR(50) NOT NULL,
    config JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress FLOAT NOT NULL DEFAULT 0.0,
    current_step VARCHAR(100),
    result JSONB,
    error_message TEXT,
    celery_task_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_pipeline_type ON pipeline_runs(pipeline_type);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_celery_task_id ON pipeline_runs(celery_task_id);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_status_created ON pipeline_runs(status, created_at);

-- Verify table
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'pipeline_runs'
ORDER BY ordinal_position;
