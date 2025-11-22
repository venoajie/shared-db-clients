from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from shared_config.config import PostgresSettings

from shared_db_clients.postgres_client import PostgresClient


# --- Helper for mocking async context managers (pool.acquire) ---
class AsyncContextManagerMock(AsyncMock):
    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def mock_pg_config():
    return PostgresSettings(
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        db="test_db",
    )


@pytest.fixture
def client(mock_pg_config):
    return PostgresClient(config=mock_pg_config)


@pytest.mark.asyncio
async def test_start_pool_success(client):
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
        mock_pool = AsyncMock()
        mock_pool._closed = False
        mock_create_pool.return_value = mock_pool

        pool = await client.start_pool()

        assert pool is mock_pool
        mock_create_pool.assert_called_once()
        assert client._pool is mock_pool


@pytest.mark.asyncio
async def test_start_pool_failure_retries(client):
    # We patch asyncio.sleep to skip waiting during retries
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # We simulate 5 failures to trigger the final ConnectionError
        with patch("asyncpg.create_pool", side_effect=Exception("Fail")) as mock_create:
            with pytest.raises(ConnectionError):
                await client.start_pool()

            # Should have tried 5 times
            assert mock_create.call_count == 5


@pytest.mark.asyncio
async def test_bulk_upsert_tickers(client):
    # Setup Connection Mock
    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = AsyncContextManagerMock()

    # Setup Pool Mock with proper Async Context Manager for acquire()
    mock_pool = AsyncMock()
    mock_pool._closed = False

    # When pool.acquire() is called, it returns an Async Context Manager
    # which yields mock_conn on __aenter__
    acquire_mock = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.acquire = acquire_mock

    # Inject mock pool directly to avoid start_pool logic in this test
    client._pool = mock_pool
    # Bypass start_pool call inside the method by patching it
    client.start_pool = AsyncMock(return_value=mock_pool)

    data = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "last_price": 50000.0,
            "exchange_timestamp": 1600000000000,  # ms
        }
    ]

    await client.bulk_upsert_tickers(data)

    mock_conn.executemany.assert_called_once()
    args = mock_conn.executemany.call_args[0]
    assert "INSERT INTO tickers" in args[0]
    assert len(args[1]) == 1


@pytest.mark.asyncio
async def test_bulk_upsert_ohlc(client):
    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = AsyncContextManagerMock()

    mock_pool = AsyncMock()
    mock_pool._closed = False
    acquire_mock = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.acquire = acquire_mock

    client._pool = mock_pool
    client.start_pool = AsyncMock(return_value=mock_pool)

    candles = [
        {
            "exchange": "deribit",
            "instrument_name": "BTC-PERP",
            "resolution": "1",  # 1 minute
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


def test_parse_resolution(client):
    td = client._parse_resolution_to_timedelta("1")
    assert td == timedelta(minutes=1)

    td = client._parse_resolution_to_timedelta("1H")
    assert td == timedelta(hours=1)

    with pytest.raises(ValueError):
        client._parse_resolution_to_timedelta("INVALID")


@pytest.mark.asyncio
async def test_close_pool(client):
    mock_pool = AsyncMock()
    mock_pool._closed = False
    client._pool = mock_pool

    await client.close_pool()
    mock_pool.close.assert_called_once()
    assert client._pool is None


@pytest.mark.asyncio
async def test_misc_fetch_methods(client):
    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = AsyncContextManagerMock()

    mock_pool = AsyncMock()
    mock_pool._closed = False
    acquire_mock = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.acquire = acquire_mock

    client._pool = mock_pool
    client.start_pool = AsyncMock(return_value=mock_pool)

    # Test fetch_active_trades
    await client.fetch_active_trades("user123")
    mock_conn.fetch.assert_called()

    # Test insert_account_info
    await client.insert_account_information("user1", {"data": 1})
    mock_conn.execute.assert_called()
