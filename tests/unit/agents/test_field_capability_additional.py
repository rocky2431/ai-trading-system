"""Additional tests for field capability utilities."""

from __future__ import annotations

from iqfmp.agents.field_capability import (
    DataSourceType,
    create_default_capability,
    create_capability_for_sources,
    create_full_capability,
    validate_expression_fields,
    generate_field_error_feedback,
)


def test_default_capability_fields() -> None:
    capability = create_default_capability()
    fields = capability.field_registry.get_available_fields()
    assert "$close" in fields
    assert "$volume" in fields


def test_validate_expression_fields_and_feedback() -> None:
    capability = create_capability_for_sources([DataSourceType.OHLCV])
    is_valid, invalid = validate_expression_fields("$close + $foo", capability)

    assert is_valid is False
    assert "$foo" in invalid

    feedback = generate_field_error_feedback(invalid, capability)
    assert "invalid field" in feedback.lower()


def test_enable_disable_sources() -> None:
    capability = create_full_capability()
    registry = capability.field_registry

    registry.disable_source(DataSourceType.ORDERBOOK)
    assert DataSourceType.ORDERBOOK not in registry.get_enabled_sources()

    registry.enable_source(DataSourceType.ORDERBOOK)
    assert DataSourceType.ORDERBOOK in registry.get_enabled_sources()
