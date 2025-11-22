# tests/test_postgres_client.py

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared_config.config import PostgresSettings

from shared_db_clients.postgres_client import PostgresClient

# --- Helpers ---


class MockAsyncContextManager:
    """
    A helper to mock 'async with'.
    Usage:
       mock_conn = AsyncMock()
       pool.acquire.return_value = MockAsyncContextManager(mock_conn)
    """

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
    """
    Creates a MagicMock for the pool and an AsyncMock for the connection.
    Correctly wires up the pool.acquire() -> context manager -> connection flow.
    """
    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = MockAsyncContextManager()

    mock_pool = MagicMock()  # Pool methods (acquire) are synchronous
    mock_pool._closed = False
    mock_pool.acquire.return_value = MockAsyncContextManager(mock_conn)

    return mock_pool, mock_conn


@pytest.fixture
def client(mock_pg_config, mock_pool_and_conn):
    """
    Returns a client with the pool already 'started' (injected).
    """
    mock_pool, _ = mock_pool_and_conn
    client = PostgresClient(config=mock_pg_config)

    # We bypass start_pool logic by injecting directly for method tests
    client._pool = mock_pool
    # We also patch start_pool to return our mock_pool if called explicitly
    client.start_pool = AsyncMock(return_value=mock_pool)

    return client


# --- Connection & Setup Tests ---


@pytest.mark.asyncio
async def test_start_pool_success(mock_pg_config):
    client = PostgresClient(config=mock_pg_config)

    # We test the logic by patching asyncpg.create_pool
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_pool = AsyncMock()
        mock_pool._closed = False
        mock_create.return_value = mock_pool

        pool = await client.start_pool()
        assert pool is mock_pool
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_close_pool(client, mock_pool_and_conn):
    mock_pool, _ = mock_pool_and_conn
    await client.close_pool()
    mock_pool.close.assert_called_once()
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

    await client.bulk_upsert_tickers(data)

    mock_conn.executemany.assert_called_once()
    args = mock_conn.executemany.call_args[0]
    assert "INSERT INTO tickers" in args[0]


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc(client, mock_pool_and_conn):
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
    mock_conn.execute.assert_called_once()
    assert "bulk_upsert_ohlc" in mock_conn.execute.call_args[0][0]


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
    assert "INSERT INTO instruments" in mock_conn.executemany.call_args[0][0]


@pytest.mark.asyncio
async def test_bulk_upsert_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    orders = [{"order_id": "123", "price": 500}]

    await client.bulk_upsert_orders(orders)
    mock_conn.execute.assert_called_once()
    assert "bulk_upsert_orders_from_json" in mock_conn.execute.call_args[0][0]


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
    mock_conn.execute.assert_called_once()
    assert "bulk_insert_public_trades" in mock_conn.execute.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    to_delete = [("binance", "BTCUSDT", "oid_1")]

    await client.delete_orders(to_delete)
    mock_conn.executemany.assert_called_once()
    assert "delete_order" in mock_conn.executemany.call_args[0][0]


# --- Fetch / Read Tests ---


@pytest.mark.asyncio
async def test_fetch_active_trades(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_active_trades("user123")

    mock_conn.fetch.assert_called_once()
    assert "WHERE user_id = $1" in mock_conn.fetch.call_args[0][0]


@pytest.mark.asyncio
async def test_fetch_open_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_open_orders("user123")
    mock_conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_all_user_records(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    # Mock return with one open order and one filled trade
    mock_conn.fetch.return_value = [
        {"trade_id": None, "order_id": "1"},
        {"trade_id": "t1", "order_id": "2"},
    ]

    open_orders, filled_trades = await client.fetch_all_user_records("u1")
    assert len(open_orders) == 1
    assert len(filled_trades) == 1


@pytest.mark.asyncio
async def test_bootstrap_status(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn

    # Test Check
    mock_conn.fetchval.return_value = "complete"
    assert await client.check_bootstrap_status("deribit") is True

    # Test Set
    await client.set_bootstrap_status(True, "deribit")
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_ohlc_for_instrument(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_ohlc_for_instrument("deribit", "BTC-PERP", "1", 100)

    mock_conn.fetch.assert_called_once()
    # Check argument parsing
    args = mock_conn.fetch.call_args[0]
    assert args[3] == timedelta(minutes=1)


@pytest.mark.asyncio
async def test_fetch_latest_timestamps(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn

    await client.fetch_latest_ohlc_timestamp("ex", "inst", timedelta(minutes=1))
    mock_conn.fetchval.assert_called()

    mock_conn.reset_mock()

    await client.fetch_latest_public_trade_timestamp("ex", "inst")
    mock_conn.fetchval.assert_called()


@pytest.mark.asyncio
async def test_utilities_and_misc(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn

    # Resolution Parsing
    assert client._parse_resolution_to_timedelta("15 minutes") == timedelta(minutes=15)

    # Account Info
    await client.insert_account_information("u1", {})
    mock_conn.execute.assert_called()

    # Futures Summary
    await client.fetch_futures_summary_for_exchange("deribit")
    mock_conn.fetchrow.assert_called()

    # Bulk update is_open
    await client.bulk_update_is_open_status(["t1"], False)
    mock_conn.execute.assert_called()
