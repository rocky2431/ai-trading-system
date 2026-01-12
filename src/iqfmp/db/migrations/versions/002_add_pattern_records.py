"""Add pattern_records table for closed-loop factor mining.

This table stores success/failure patterns from factor evaluation,
enabling similarity-based retrieval for LLM feedback injection.

Revision ID: 002_add_pattern_records
Revises: 001_initial_research_trials
Create Date: 2026-01-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_add_pattern_records"
down_revision: Union[str, None] = "001_initial_research_trials"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create pattern_records table for closed-loop factor mining."""

    # Create the pattern_records table
    op.create_table(
        "pattern_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pattern_id", sa.String(length=36), nullable=False),
        sa.Column("pattern_type", sa.String(length=20), nullable=False),  # "success" or "failure"
        # Factor information
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("factor_code", sa.Text(), nullable=False),
        sa.Column("factor_family", sa.String(length=50), nullable=False),
        # Metrics (JSONB for flexibility)
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        # Feedback (for failure patterns)
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("failure_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Reference to evaluation trial
        sa.Column("trial_id", sa.String(length=36), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pattern_id"),
    )

    # Create indexes
    op.create_index("ix_pattern_records_pattern_id", "pattern_records", ["pattern_id"])
    op.create_index("ix_pattern_records_pattern_type", "pattern_records", ["pattern_type"])
    op.create_index("ix_pattern_records_factor_family", "pattern_records", ["factor_family"])
    op.create_index("ix_pattern_records_trial_id", "pattern_records", ["trial_id"])
    op.create_index("ix_pattern_records_type_family", "pattern_records", ["pattern_type", "factor_family"])
    op.create_index("ix_pattern_records_created_at", "pattern_records", ["created_at"])


def downgrade() -> None:
    """Drop pattern_records table."""

    # Drop indexes first
    op.drop_index("ix_pattern_records_created_at", table_name="pattern_records")
    op.drop_index("ix_pattern_records_type_family", table_name="pattern_records")
    op.drop_index("ix_pattern_records_trial_id", table_name="pattern_records")
    op.drop_index("ix_pattern_records_factor_family", table_name="pattern_records")
    op.drop_index("ix_pattern_records_pattern_type", table_name="pattern_records")
    op.drop_index("ix_pattern_records_pattern_id", table_name="pattern_records")

    # Drop table
    op.drop_table("pattern_records")
