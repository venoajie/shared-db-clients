import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta, timezone
from shared_db_clients.postgres_client import PostgresClient
from shared_config.config import PostgresSettings

@pytest.fixture
def mock_pg_config():
    return PostgresSettings(
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        db="test_db"
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
    with patch("asyncpg.create_pool", side_effect=[Exception("Fail"), Exception("Fail"), AsyncMock()]) as mock_create:
        pool = await client.start_pool()
        assert mock_create.call_count == 3
        assert pool is not None

@pytest.mark.asyncio
async def test_bulk_upsert_tickers(client):
    # Setup Mock Pool/Connection
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool._closed = False
    
    # Inject mock pool
    client._pool = mock_pool
    
    data = [{
        "exchange": "deribit",
        "instrument_name": "BTC-PERP",
        "last_price": 50000.0,
        "exchange_timestamp": 1600000000000 # ms
    }]
    
    await client.bulk_upsert_tickers(data)
    
    mock_conn.transaction.assert_called()
    mock_conn.executemany.assert_called_once()
    args = mock_conn.executemany.call_args[0]
    assert "INSERT INTO tickers" in args[0]
    assert len(args[1]) == 1

@pytest.mark.asyncio
async def test_bulk_upsert_ohlc(client):
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool._closed = False
    client._pool = mock_pool
    
    candles = [{
        "exchange": "deribit",
        "instrument_name": "BTC-PERP",
        "resolution": "1", # 1 minute
        "tick": 1600000000000,
        "open": 100, "high": 110, "low": 90, "close": 105, "volume": 10
    }]
    
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
    # Setup generic mock
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool._closed = False
    client._pool = mock_pool
    
    # Test fetch_active_trades
    await client.fetch_active_trades("user123")
    mock_conn.fetch.assert_called()
    
    # Test insert_account_info
    await client.insert_account_information("user1", {"data": 1})
    mock_conn.execute.assert_called()
