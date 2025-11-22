from unittest.mock import AsyncMock, patch

import orjson
import pytest
from redis import exceptions as redis_exceptions

from shared_db_clients.redis_client import CustomRedisClient


@pytest.fixture
def client():
    return CustomRedisClient()


@pytest.mark.asyncio
async def test_get_pool_success(client):
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis

        pool = await client.get_pool()

        assert pool is mock_redis
        mock_redis.ping.assert_awaited()


@pytest.mark.asyncio
async def test_parse_stream_message():
    # Test 1: JSON encoded data
    payload = {"price": 100, "vol": 1}
    msg_data = {b"data": orjson.dumps(payload)}

    parsed = CustomRedisClient.parse_stream_message(msg_data)
    assert parsed["data"] == payload

    # Test 2: Plain string
    msg_data_str = {b"status": b"ok"}
    parsed = CustomRedisClient.parse_stream_message(msg_data_str)
    assert parsed["status"] == "ok"


@pytest.mark.asyncio
async def test_xadd_bulk_success(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool
    # Mock locking and semaphore
    client._lock = AsyncMock()

    # Setup pipeline mock
    mock_pipe = AsyncMock()
    mock_pool.pipeline.return_value = mock_pipe

    messages = [{"id": 1}, {"id": 2}]
    await client.xadd_bulk("test_stream", messages)

    assert mock_pipe.xadd.call_count == 2
    mock_pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_xadd_dlq_fallback(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool

    # Mock xadd_bulk failing 3 times
    with patch.object(client, "get_pool", side_effect=redis_exceptions.ConnectionError):
        # Also need to mock xadd_to_dlq to verify it's called
        with patch.object(client, "xadd_to_dlq", new_callable=AsyncMock) as mock_dlq:
            with pytest.raises(ConnectionError):
                await client.xadd_bulk("test_stream", [{"a": 1}])

            mock_dlq.assert_awaited()


@pytest.mark.asyncio
async def test_read_stream_messages(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool

    # Mock XREADGROUP response
    # Structure: [[stream_name, [[msg_id, {data}]]]]
    mock_response = [[b"test_stream", [[b"1-0", {b"data": b"{}"}]]]]
    mock_pool.xreadgroup.return_value = mock_response

    msgs = await client.read_stream_messages("test_stream", "g1", "c1")
    assert len(msgs) == 1
    assert msgs[0][0] == b"1-0"


@pytest.mark.asyncio
async def test_ensure_consumer_group(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool

    await client.ensure_consumer_group("s1", "g1")
    mock_pool.xgroup_create.assert_awaited()

    # Test ignore busygroup error
    mock_pool.xgroup_create.side_effect = redis_exceptions.ResponseError("BUSYGROUP")
    await client.ensure_consumer_group("s1", "g1")  # Should not raise


@pytest.mark.asyncio
async def test_system_state(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool

    mock_pool.get.return_value = b"ACTIVE"
    state = await client.get_system_state()
    assert state == "ACTIVE"

    await client.set_system_state("LOCKED", "Panic")
    mock_pool.hset.assert_awaited()
