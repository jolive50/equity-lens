"""Tests for smart money service behaviour when optional adapters are missing."""
from pipelines.realtime.smart_money import SmartMoneyService


def test_congressional_summary_unconfigured_adapter():
    """SmartMoneyService should emit a guidance message when no congressional adapter is set."""
    service = SmartMoneyService(adapters={}, congressional_adapter=None)
    summary = service.get_congressional_summary("AAPL")
    # What: Validate that the response clearly communicates the missing integration.
    # Why: Ensures the API never fabricates disclosure data.
    # How: Call the summary method and inspect the returned dictionary fields.
    # Data: With no adapter, the service should produce zero trades and a descriptive summary.
    assert summary["total_trades"] == 0
    assert "Congressional trading analytics are unavailable" in summary["summary"]
