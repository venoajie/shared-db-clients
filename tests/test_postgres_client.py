# tests/test_postgres_client.py

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from shared_config.config import PostgresSettings

from shared_db_clients.postgres_client import PostgresClient

# --- Helpers ---


class MockAsyncContextManager:
    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def mock_pg_config():
    return PostgresSettings(
        user="test", password="pwd", host="localhost", port=5432, db="test_db"
    )


@pytest.fixture
def mock_pool_and_conn():
    # mock_conn represents the connection object yielded by pool.acquire()
    # NOTE: execute/fetch are async methods
    mock_conn = AsyncMock(name="mock_connection")

    # CRITICAL FIX: transaction() is a SYNC method returning an async context manager.
    # We must overwrite the auto-generated AsyncMock.
    mock_conn.transaction = MagicMock(name="mock_transaction")
    mock_conn.transaction.return_value = MockAsyncContextManager()

    # mock_pool represents the pool
    mock_pool = MagicMock(name="mock_pool")
    mock_pool._closed = False

    # CRITICAL FIX: close() is an ASYNC method
    mock_pool.close = AsyncMock(name="mock_pool_close")

    # acquire() is a SYNC method returning an async context manager
    mock_pool.acquire.return_value = MockAsyncContextManager(mock_conn)

    return mock_pool, mock_conn


@pytest.fixture
def client(mock_pg_config, mock_pool_and_conn):
    mock_pool, _ = mock_pool_and_conn
    client = PostgresClient(config=mock_pg_config)
    client._pool = mock_pool
    client.start_pool = AsyncMock(return_value=mock_pool)
    return client


# --- Connection & Setup Tests ---


@pytest.mark.asyncio
async def test_start_pool_success(mock_pg_config):
    client = PostgresClient(config=mock_pg_config)

    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_pool = AsyncMock()
        mock_pool._closed = False
        mock_create.return_value = mock_pool

        pool = await client.start_pool()
        assert pool is mock_pool
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_start_pool_failure_retries(mock_pg_config):
    client = PostgresClient(config=mock_pg_config)

    with patch(
        "asyncpg.create_pool", side_effect=Exception("Connection Refused")
    ) as mock_create:
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip sleep delays
            with pytest.raises(ConnectionError) as exc:
                await client.start_pool()

            assert "Fatal: Could not create PostgreSQL pool" in str(exc.value)
            assert mock_create.call_count == 5


@pytest.mark.asyncio
async def test_close_pool(client, mock_pool_and_conn):
    mock_pool, _ = mock_pool_and_conn
    await client.close_pool()
    mock_pool.close.assert_awaited_once()  # Now correctly awaits
    assert client._pool is None


# --- Upsert Tests ---


@pytest.mark.asyncio
async def test_bulk_upsert_tickers(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    data = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "last_price": 50000.0,
            "exchange_timestamp": 1600000000000,
        }
    ]
    # Add missing timestamp case for coverage
    data_bad = [{"instrument_name": "BAD", "last_price": 1}]

    await client.bulk_upsert_tickers(data + data_bad)

    mock_conn.executemany.assert_called_once()
    args = mock_conn.executemany.call_args[0]
    assert len(args[1]) == 1  # Only the valid record


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc_success(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    candles = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "resolution": "1",
            "tick": 1600000000000,
            "open": 100,
            "high": 110,
            "low": 90,
            "close": 105,
            "volume": 10,
        }
    ]

    await client.bulk_upsert_ohlc(candles)
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc_retry_logic(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    candles = [
        {
            "exchange": "ex",
            "instrument_name": "i",
            "resolution": "1",
            "tick": 1000,
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
        }
    ]

    # Simulate "relation does not exist" error twice, then success
    db_err = asyncpg.PostgresError("relation 'ohlc_upsert_type' does not exist")
    mock_conn.execute.side_effect = [db_err, db_err, None]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await client.bulk_upsert_ohlc(candles)

    assert mock_conn.execute.call_count == 3


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc_fatal_error(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    candles = [
        {
            "exchange": "ex",
            "instrument_name": "i",
            "resolution": "1",
            "tick": 1000,
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
        }
    ]

    # Simulate unknown error
    mock_conn.execute.side_effect = Exception("Fatal DB Error")

    with pytest.raises(Exception) as exc:
        await client.bulk_upsert_ohlc(candles)
    assert "Fatal DB Error" in str(exc.value)


@pytest.mark.asyncio
async def test_bulk_upsert_instruments(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    instruments = [
        {
            "exchange": "binance",
            "instrument_name": "BTCUSDT",
            "expiration_timestamp": datetime.now(UTC).isoformat(),
            "data": {},
        }
    ]

    await client.bulk_upsert_instruments(instruments, "binance")
    mock_conn.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_upsert_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    orders = [{"order_id": "123", "price": 500}]
    await client.bulk_upsert_orders(orders)
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_bulk_insert_public_trades(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    trades = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "market_type": "future",
            "trade_id": "t1",
            "price": 1000.0,
            "quantity": 1.0,
            "is_buyer_maker": False,
            "was_best_price_match": True,
            "trade_timestamp": datetime.now(UTC),
        }
    ]

    await client.bulk_insert_public_trades(trades)
    mock_conn.execute.assert_awaited_once()

    # Test missing key handling (coverage)
    bad_trade = [{"trade_id": "t2"}]  # Missing fields
    await client.bulk_insert_public_trades(
        bad_trade
    )  # Should log error but not crash/insert
    # Call count remains 1 from previous call (or we reset mock)


@pytest.mark.asyncio
async def test_delete_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    to_delete = [("binance", "BTCUSDT", "oid_1")]
    await client.delete_orders(to_delete)
    mock_conn.executemany.assert_called_once()


# --- Fetch Tests ---


@pytest.mark.asyncio
async def test_fetch_active_trades(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_active_trades("user123")
    mock_conn.fetch.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_ohlc_for_instrument(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_ohlc_for_instrument("deribit", "BTC-PERP", "1", 100)
    mock_conn.fetch.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_latest_timestamps(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    # OHLC timestamp
    await client.fetch_latest_ohlc_timestamp("ex", "inst", timedelta(minutes=1))
    mock_conn.fetchval.assert_called()

    mock_conn.reset_mock()
    # Public trade timestamp
    await client.fetch_latest_public_trade_timestamp("ex", "inst")
    mock_conn.fetchval.assert_called()


@pytest.mark.asyncio
async def test_utilities_and_misc(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn

    # Resolution Parsing
    assert client._parse_resolution_to_timedelta("15 minutes") == timedelta(minutes=15)
    assert client._parse_resolution_to_timedelta("1D") == timedelta(days=1)
    with pytest.raises(ValueError):
        client._parse_resolution_to_timedelta("BAD")

    # Account Info
    await client.insert_account_information("u1", {})
    mock_conn.execute.assert_awaited()

    # Futures Summary
    await client.fetch_futures_summary_for_exchange("deribit")
    mock_conn.fetchrow.assert_awaited()

    # Bulk update is_open
    await client.bulk_update_is_open_status(["t1"], False)
    mock_conn.execute.assert_awaited()


@pytest.mark.asyncio
async def test_check_bootstrap_status(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    mock_conn.fetchval.return_value = "complete"
    assert await client.check_bootstrap_status("ex") is True

    await client.set_bootstrap_status(False, "ex")
    mock_conn.execute.assert_awaited()
