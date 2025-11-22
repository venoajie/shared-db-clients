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
    # Mock the pipeline object
    mock_pipe = AsyncMock()

    # Mock the pool to return the pipeline
    mock_pool = AsyncMock()
    mock_pool.pipeline.return_value = mock_pipe

    client.pool = mock_pool
    # Mock get_pool to return our mock pool
    client.get_pool = AsyncMock(return_value=mock_pool)

    client._lock = AsyncMock()

    messages = [{"id": 1}, {"id": 2}]
    await client.xadd_bulk("test_stream", messages)

    # xadd is a synchronous method on the pipeline object in redis-py (it just queues commands)
    # but execute() is async.
    # However, in our code we do `pipe.xadd(...)`.
    # If using redis.asyncio, pipeline methods are usually not awaitable, only execute() is.
    assert mock_pipe.xadd.call_count == 2
    mock_pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_xadd_dlq_fallback(client):
    # Mock xadd_bulk raising ConnectionError
    # We need to simulate the internal failure of the try/except block in xadd_bulk

    # Strategy: Mock get_pool to fail
    client.get_pool = AsyncMock(side_effect=redis_exceptions.ConnectionError)

    # Mock xadd_to_dlq
    client.xadd_to_dlq = AsyncMock()

    # Mock sleep to speed up test
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError):
            await client.xadd_bulk("test_stream", [{"a": 1}])

        # Should have tried to send to DLQ before raising
        client.xadd_to_dlq.assert_awaited()


@pytest.mark.asyncio
async def test_read_stream_messages(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Mock XREADGROUP response
    mock_response = [[b"test_stream", [[b"1-0", {b"data": b"{}"}]]]]
    mock_pool.xreadgroup.return_value = mock_response

    msgs = await client.read_stream_messages("test_stream", "g1", "c1")
    assert len(msgs) == 1
    assert msgs[0][0] == b"1-0"


@pytest.mark.asyncio
async def test_ensure_consumer_group(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool
    client.get_pool = AsyncMock(return_value=mock_pool)

    await client.ensure_consumer_group("s1", "g1")
    mock_pool.xgroup_create.assert_awaited()

    # Test ignore busygroup error
    mock_pool.xgroup_create.side_effect = redis_exceptions.ResponseError("BUSYGROUP")
    await client.ensure_consumer_group("s1", "g1")  # Should not raise


@pytest.mark.asyncio
async def test_system_state(client):
    mock_pool = AsyncMock()
    client.pool = mock_pool
    client.get_pool = AsyncMock(return_value=mock_pool)

    mock_pool.get.return_value = b"ACTIVE"
    state = await client.get_system_state()
    assert state == "ACTIVE"

    await client.set_system_state("LOCKED", "Panic")
    mock_pool.hset.assert_awaited()
