import asyncio
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
    mock_conn = AsyncMock(name="mock_connection")
    mock_conn.transaction = MagicMock(name="mock_transaction")
    mock_conn.transaction.return_value = MockAsyncContextManager()

    mock_pool = MagicMock(name="mock_pool")
    mock_pool._closed = False
    mock_pool.close = AsyncMock(name="mock_pool_close")
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
async def test_start_pool_no_config():
    client = PostgresClient(config=None)
    client.postgres_settings = None
    client.dsn = None

    with pytest.raises(ValueError, match="No PostgreSQL configuration"):
        await client.start_pool()


@pytest.mark.asyncio
async def test_start_pool_failure_retries(mock_pg_config):
    client = PostgresClient(config=mock_pg_config)
    with patch(
        "asyncpg.create_pool", side_effect=Exception("Connection Refused")
    ) as mock_create:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError, match="Fatal: Could not create"):
                await client.start_pool()
            assert mock_create.call_count == 5


@pytest.mark.asyncio
async def test_start_pool_recovery(mock_pg_config):
    """Test the loop logic: Fail once, then succeed."""
    client = PostgresClient(config=mock_pg_config)

    mock_pool = AsyncMock()
    mock_pool._closed = False

    # FIX: Wrap the success result in a Future so 'await' works
    f = asyncio.Future()
    f.set_result(mock_pool)

    with patch(
        "asyncpg.create_pool", side_effect=[Exception("Temp Fail"), f]
    ) as mock_create:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            pool = await client.start_pool()

    assert pool is mock_pool
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_close_pool(client, mock_pool_and_conn):
    mock_pool, _ = mock_pool_and_conn
    await client.close_pool()
    mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_json_codec(client):
    mock_conn = AsyncMock()
    await client._setup_json_codec(mock_conn)
    assert mock_conn.set_type_codec.call_count == 2


@pytest.mark.asyncio
async def test_json_codec_lambdas(client):
    """Explicitly execute the lambdas to verify code inside them."""
    mock_conn = AsyncMock()
    await client._setup_json_codec(mock_conn)
    calls = mock_conn.set_type_codec.await_args_list
    _, kwargs = calls[0]
    encoder = kwargs.get("encoder")
    decoder = kwargs.get("decoder")

    test_dict = {"key": "value"}
    encoded = encoder(test_dict)
    assert '"key":"value"' in encoded.replace(" ", "")
    decoded = decoder('{"key": "value"}')
    assert decoded == test_dict


# --- Upsert Tests ---


@pytest.mark.asyncio
async def test_bulk_upsert_tickers(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_upsert_tickers([])
    mock_conn.executemany.assert_not_called()

    data = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "last_price": 50000.0,
            "exchange_timestamp": 1600000000000,
        },
        {"instrument_name": "BAD_TICKER"},
    ]
    await client.bulk_upsert_tickers(data)
    mock_conn.executemany.assert_called_once()
    args = mock_conn.executemany.call_args[0]
    assert len(args[1]) == 1

    mock_conn.executemany.side_effect = Exception("DB Fail")
    with pytest.raises(Exception, match="DB Fail"):
        await client.bulk_upsert_tickers(data)


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_upsert_ohlc([])
    mock_conn.execute.assert_not_called()

    candles = [
        {
            "exchange": "e",
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
    db_err = asyncpg.PostgresError("relation 'ohlc_upsert_type' does not exist")
    mock_conn.execute.side_effect = [db_err, None]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await client.bulk_upsert_ohlc(candles)
    assert mock_conn.execute.call_count == 2

    mock_conn.execute.side_effect = Exception("Fatal")
    with pytest.raises(Exception, match="Fatal"):
        await client.bulk_upsert_ohlc(candles)


@pytest.mark.asyncio
async def test_bulk_upsert_instruments(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_upsert_instruments([], "binance")
    mock_conn.executemany.assert_not_called()

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

    mock_conn.executemany.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client.bulk_upsert_instruments(instruments, "binance")


@pytest.mark.asyncio
async def test_bulk_upsert_instruments_timestamps(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    inst_valid = {
        "exchange": "binance",
        "instrument_name": "A",
        "expiration_timestamp": "2025-01-01T00:00:00+00:00",
    }
    inst_none = {
        "exchange": "binance",
        "instrument_name": "B",
        "expiration_timestamp": None,
    }

    await client.bulk_upsert_instruments([inst_valid, inst_none], "binance")
    args = mock_conn.executemany.call_args[0]
    records = args[1]
    assert isinstance(records[0][10], datetime)
    assert records[1][10] is None


@pytest.mark.asyncio
async def test_bulk_upsert_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_upsert_orders([])
    mock_conn.execute.assert_not_called()

    orders = [{"order_id": "123"}]
    await client.bulk_upsert_orders(orders)
    mock_conn.execute.assert_awaited_once()

    mock_conn.execute.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client.bulk_upsert_orders(orders)


@pytest.mark.asyncio
async def test_bulk_insert_public_trades(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_insert_public_trades([])
    mock_conn.execute.assert_not_called()

    await client.bulk_insert_public_trades([{"bad": "data"}])
    mock_conn.execute.assert_not_called()

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

    mock_conn.execute.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client.bulk_insert_public_trades(trades)


@pytest.mark.asyncio
async def test_delete_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.delete_orders([])
    mock_conn.executemany.assert_not_called()

    await client.delete_orders([("ex", "inst", "oid")])
    mock_conn.executemany.assert_called_once()

    mock_conn.executemany.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client.delete_orders([("ex", "inst", "oid")])


# --- Fetch & Read Tests ---


@pytest.mark.asyncio
async def test_fetch_all_instruments(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_all_instruments()
    mock_conn.fetch.assert_called_with("SELECT * FROM v_instruments")


@pytest.mark.asyncio
async def test_fetch_active_trades(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_active_trades()
    assert "WHERE user_id" not in mock_conn.fetch.call_args[0][0]
    await client.fetch_active_trades("u1")
    assert "WHERE user_id = $1" in mock_conn.fetch.call_args[0][0]


@pytest.mark.asyncio
async def test_fetch_open_orders(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_open_orders()
    await client.fetch_open_orders("u1")
    assert "AND user_id = $1" in mock_conn.fetch.call_args[0][0]


@pytest.mark.asyncio
async def test_fetch_all_user_records(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    mock_conn.fetch.return_value = []
    await client.fetch_all_user_records("u1")
    mock_conn.fetch.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_trades_by_timestamp(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_trades_by_timestamp(1000, 2000, "deribit")
    mock_conn.fetch.assert_called()
    args = mock_conn.fetch.call_args[0]
    assert isinstance(args[1], datetime)
    assert args[3] == "deribit"


@pytest.mark.asyncio
async def test_fetch_all_trades_for_instrument(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_all_trades_for_instrument("BTC-PERP")
    mock_conn.fetch.assert_called()


@pytest.mark.asyncio
async def test_fetch_all_trades_for_user(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_all_trades_for_user("u1")
    mock_conn.fetch.assert_called()


@pytest.mark.asyncio
async def test_check_bootstrap_status(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    mock_conn.fetchval.return_value = "complete"
    assert await client.check_bootstrap_status("ex") is True
    await client.set_bootstrap_status(True, "ex")
    mock_conn.execute.assert_called()


@pytest.mark.asyncio
async def test_bulk_update_is_open_status(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.bulk_update_is_open_status([], True)
    mock_conn.execute.assert_not_called()
    await client.bulk_update_is_open_status(["t1"], True)
    mock_conn.execute.assert_called()

    mock_conn.execute.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client.bulk_update_is_open_status(["t1"], True)


@pytest.mark.asyncio
async def test_fetch_ohlc_methods(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.fetch_ohlc_for_instrument("ex", "inst", "1", 100)
    mock_conn.fetch.assert_called()

    mock_conn.fetchval.side_effect = Exception("DB Fail")
    res = await client.fetch_latest_ohlc_timestamp("ex", "inst", timedelta(minutes=1))
    assert res is None
    mock_conn.fetchval.side_effect = None

    mock_conn.fetchval.side_effect = Exception("DB Fail")
    res = await client.fetch_latest_public_trade_timestamp("ex", "inst")
    assert res is None


@pytest.mark.asyncio
async def test_futures_summary_error(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    mock_conn.fetchrow.side_effect = Exception("Fail")
    res = await client.fetch_futures_summary_for_exchange("deribit")
    assert res is None


@pytest.mark.asyncio
async def test_utilities(client, mock_pool_and_conn):
    _, mock_conn = mock_pool_and_conn
    await client.insert_account_information("u1", {})
    mock_conn.execute.assert_called()
    assert client._parse_resolution_to_timedelta("1 minute") == timedelta(minutes=1)


@pytest.mark.asyncio
async def test_start_pool_loop_coverage(mock_pg_config):
    """
    Forces execution of the retry loop and exception logging (Lines 97-115).
    Simulates: Fail, Fail, Success.
    """
    client = PostgresClient(config=mock_pg_config)

    # Success Future
    mock_pool = AsyncMock()
    mock_pool._closed = False
    f_success = asyncio.Future()
    f_success.set_result(mock_pool)

    # We need side_effect to raise exception twice, then return the future
    with patch(
        "asyncpg.create_pool",
        side_effect=[Exception("Fail 1"), Exception("Fail 2"), f_success],
    ) as mock_create:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            pool = await client.start_pool()

    assert pool is mock_pool
    assert mock_create.call_count == 3
    assert mock_sleep.call_count == 2  # Slept twice


@pytest.mark.asyncio
async def test_bulk_upsert_instruments_timestamps_coverage(client, mock_pool_and_conn):
    """
    Directly targets Lines 228-229 (ISO parsing) by calling the method
    with data that triggers the parsing logic.
    """
    _, mock_conn = mock_pool_and_conn

    # Data with ISO string
    inst_iso = {
        "exchange": "binance",
        "instrument_name": "ISO",
        "expiration_timestamp": "2025-01-01T12:00:00",
    }
    # Data with None
    inst_none = {
        "exchange": "binance",
        "instrument_name": "NONE",
        "expiration_timestamp": None,
    }

    await client.bulk_upsert_instruments([inst_iso, inst_none], "binance")

    # Extract arguments passed to DB
    call_args = mock_conn.executemany.call_args[0][1]

    # Verify ISO parsed to datetime
    assert isinstance(call_args[0][10], datetime)
    # Verify None stays None
    assert call_args[1][10] is None


@pytest.mark.asyncio
async def test_fetch_latest_public_trade_timestamp_exception(
    client, mock_pool_and_conn
):
    """Targets Lines 541-543 (Exception handling)."""
    _, mock_conn = mock_pool_and_conn
    mock_conn.fetchval.side_effect = Exception("DB Error")

    res = await client.fetch_latest_public_trade_timestamp("ex", "inst")
    assert res is None
