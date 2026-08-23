# tests/test_database.py — database.py, with psycopg2.connect mocked out
# (no real Postgres needed). This exists specifically to check the thing the
# repo's old audit scripts never did: that the SQL placeholder count matches
# the values tuple, and that the right values land in the right columns --
# not just that some INSERT statement is present in the source text.

from unittest.mock import MagicMock

import pytest

import database


@pytest.fixture
def mock_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    monkeypatch.setattr(database.psycopg2, "connect", MagicMock(return_value=mock_conn))
    return mock_cursor


def _last_insert_trades_call(mock_cursor):
    for call in mock_cursor.execute.call_args_list:
        sql = call.args[0]
        if "INSERT INTO trades" in sql:
            return call.args[0], call.args[1]
    raise AssertionError("no INSERT INTO trades call found")


def test_record_trade_placeholder_count_matches_values_tuple(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 100.0,
        order_id="oid-1", fee=0.5, fill_price=101.0,
        realized_pnl=19.5, realized_pnl_pct=0.1,
    )
    sql, values = _last_insert_trades_call(mock_db)
    # one literal 'Alpaca' for exchange, one NOW() for timestamp -- everything
    # else in the column list must be a %s with a matching value.
    assert sql.count("%s") == len(values)


def test_record_trade_stores_realized_pnl_in_the_right_position(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 100.0,
        order_id="oid-2", fee=0.5, fill_price=101.0,
        realized_pnl=19.5, realized_pnl_pct=0.1,
    )
    sql, values = _last_insert_trades_call(mock_db)
    columns = sql.split("(", 1)[1].split(")", 1)[0]
    columns = [c.strip() for c in columns.split(",")]
    # columns list includes the literal 'Alpaca' position (exchange) and
    # NOW() (timestamp), which don't have a corresponding %s -- build the
    # %s-only column list to zip against `values`.
    literal_cols = {"exchange", "timestamp"}
    placeholder_cols = [c for c in columns if c not in literal_cols]
    row = dict(zip(placeholder_cols, values))
    assert row["realized_pnl"] == pytest.approx(19.5)
    assert row["realized_pnl_pct"] == pytest.approx(0.1)


def test_record_trade_buy_has_null_realized_pnl(mock_db):
    database.record_trade("bot", "BTC/USD", "buy", 1.0, 100.0, order_id="oid-3", fee=0.1, fill_price=100.1)
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["realized_pnl"] is None
    assert row["realized_pnl_pct"] is None


def test_record_trade_is_a_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mock_connect = MagicMock()
    monkeypatch.setattr(database.psycopg2, "connect", mock_connect)
    database.record_trade("bot", "BTC/USD", "sell", 1.0, 100.0)
    assert not mock_connect.called


def test_get_realized_pnl_summary_returns_none_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.get_realized_pnl_summary("bot") is None


def test_get_realized_pnl_summary_returns_summed_values(mock_db):
    mock_db.fetchone.return_value = (123.45, 6.78, 9)
    result = database.get_realized_pnl_summary("bot")
    assert result == {"total_realized_pnl": 123.45, "total_fees": 6.78, "closed_trades": 9}


def test_get_realized_pnl_summary_filters_by_bot_name_and_sell_side(mock_db):
    mock_db.fetchone.return_value = (0.0, 0.0, 0)
    database.get_realized_pnl_summary("my-bot")
    sql, params = mock_db.execute.call_args.args
    assert "side = 'sell'" in sql
    assert params == ("my-bot",)


def test_report_equity_is_a_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.report_equity("bot", 1000.0) is False
