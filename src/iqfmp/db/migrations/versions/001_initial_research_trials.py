"""Initial research_trials table migration.

P3.2 FIX: Add Alembic migration for research_trials table.

Revision ID: 001_initial_research_trials
Revises:
Create Date: 2025-12-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_research_trials"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create research_trials table with TimescaleDB hypertable support."""

    # Create the research_trials table
    op.create_table(
        "research_trials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trial_id", sa.String(length=36), nullable=False),
        sa.Column("trial_number", sa.Integer(), nullable=False),
        sa.Column("factor_id", sa.String(length=36), nullable=True),
        sa.Column("factor_name", sa.String(length=100), nullable=False),
        sa.Column("factor_family", sa.String(length=50), nullable=False, server_default="unknown"),
        # Metrics
        sa.Column("sharpe_ratio", sa.Float(), nullable=False),
        sa.Column("ic_mean", sa.Float(), nullable=True),
        sa.Column("ir", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        # Dynamic threshold (DSR)
        sa.Column("threshold_used", sa.Float(), nullable=False),
        sa.Column("passed_threshold", sa.Boolean(), nullable=False, server_default="false"),
        # Metadata
        sa.Column("evaluation_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Timestamp
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trial_id"),
    )

    # Create indexes
    op.create_index("ix_research_trials_trial_id", "research_trials", ["trial_id"])
    op.create_index("ix_research_trials_trial_number", "research_trials", ["trial_number"])
    op.create_index("ix_research_trials_created_at", "research_trials", ["created_at"])
    op.create_index("ix_research_trials_factor_family", "research_trials", ["factor_family"])

    # Add foreign key constraint to factors table (if exists)
    # Note: This is conditional since factors table may not exist yet
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'factors') THEN
                ALTER TABLE research_trials
                ADD CONSTRAINT fk_research_trials_factor_id
                FOREIGN KEY (factor_id) REFERENCES factors(id) ON DELETE SET NULL;
            END IF;
        END $$;
        """
    )

    # Convert to TimescaleDB hypertable (if extension is available)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('research_trials', 'created_at', if_not_exists => TRUE);
                -- Add compression policy (compress chunks older than 30 days)
                PERFORM add_compression_policy('research_trials', INTERVAL '30 days', if_not_exists => TRUE);
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Drop research_trials table."""

    # Drop indexes first
    op.drop_index("ix_research_trials_factor_family", table_name="research_trials")
    op.drop_index("ix_research_trials_created_at", table_name="research_trials")
    op.drop_index("ix_research_trials_trial_number", table_name="research_trials")
    op.drop_index("ix_research_trials_trial_id", table_name="research_trials")

    # Drop table
    op.drop_table("research_trials")
