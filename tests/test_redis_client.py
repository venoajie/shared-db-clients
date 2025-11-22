# tests/test_redis_client.py

import time
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from redis import exceptions as redis_exceptions

from shared_db_clients.redis_client import CustomRedisClient


@pytest.fixture
def client():
    return CustomRedisClient()


@pytest.mark.asyncio
async def test_get_pool_circuit_breaker(client):
    # Mock redis to fail consistently
    with patch(
        "redis.asyncio.from_url", side_effect=redis_exceptions.ConnectionError("Fail")
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # First call triggers retries and opens circuit
            with pytest.raises(ConnectionError):
                await client.get_pool()

            assert client._circuit_open is True

            # Immediate second call should fail fast (circuit open)
            with pytest.raises(ConnectionError) as exc:
                await client.get_pool()
            assert "circuit breaker open" in str(exc.value)

            # Simulate time passing to reset circuit
            client._last_failure = time.time() - 100

            # Next call attempts reconnect (fails again in this mock, but passes circuit check)
            with pytest.raises(ConnectionError) as exc:
                await client.get_pool()
            assert "circuit breaker open" not in str(exc.value)


@pytest.mark.asyncio
async def test_xadd_bulk_dlq_fallback(client):
    mock_pool = MagicMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Pipeline returns a helper, execution fails
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(
        side_effect=redis_exceptions.ConnectionError("Redis Down")
    )
    mock_pool.pipeline.return_value = mock_pipe

    client.xadd_to_dlq = AsyncMock()

    msgs = [{"id": 1}]
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError):
            await client.xadd_bulk("stream", msgs)

    # Verify fallback to DLQ was called
    client.xadd_to_dlq.assert_awaited_once()


@pytest.mark.asyncio
async def test_xadd_to_dlq_failure(client):
    # Verify that if DLQ fails, we log critical but don't crash the app entirely (catch Exception)
    client.get_pool = AsyncMock(side_effect=Exception("Total Fail"))

    # Should not raise
    await client.xadd_to_dlq("stream", [{"id": 1}])


@pytest.mark.asyncio
async def test_parse_stream_message():
    # Happy path
    msg = {b"data": orjson.dumps({"price": 100})}
    res = CustomRedisClient.parse_stream_message(msg)
    assert res["data"]["price"] == 100

    # Broken JSON fallback
    msg_bad = {b"data": b"{bad_json"}
    res_bad = CustomRedisClient.parse_stream_message(msg_bad)
    assert res_bad["data"] == "{bad_json"  # Returns raw string


@pytest.mark.asyncio
async def test_read_stream_messages_nogroup(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Simulate NOGROUP error
    mock_pool.xreadgroup.side_effect = redis_exceptions.ResponseError(
        "NOGROUP No such key"
    )
    client.ensure_consumer_group = AsyncMock()

    res = await client.read_stream_messages("s", "g", "c")

    assert res == []
    client.ensure_consumer_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_ohlc_queue_ops(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Enqueue
    await client.enqueue_ohlc_work({"task": 1})
    mock_pool.lpush.assert_awaited()

    # Dequeue success
    mock_pool.brpop.return_value = [b"key", orjson.dumps({"task": 1})]
    res = await client.dequeue_ohlc_work()
    assert res["task"] == 1

    # Dequeue None
    mock_pool.brpop.return_value = None
    assert await client.dequeue_ohlc_work() is None


@pytest.mark.asyncio
async def test_get_ticker_data(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    mock_pool.hget.return_value = orjson.dumps({"price": 50000})

    res = await client.get_ticker_data("BTC-PERP")
    assert res["price"] == 50000


@pytest.mark.asyncio
async def test_system_state(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Set
    await client.set_system_state("ACTIVE")
    mock_pool.hset.assert_awaited()

    # Get
    mock_pool.get.return_value = b"ACTIVE"
    assert await client.get_system_state() == "ACTIVE"
