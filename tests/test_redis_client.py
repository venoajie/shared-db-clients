# tests/test_redis_client.py

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
    with patch(
        "redis.asyncio.from_url", side_effect=redis_exceptions.ConnectionError("Fail")
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError, match="after 5 attempts"):
                await client.get_pool()
            assert client._circuit_open is True

            # Fast fail
            with pytest.raises(ConnectionError, match="circuit breaker open"):
                await client.get_pool()


@pytest.mark.asyncio
async def test_xadd_bulk_logic(client):
    # Empty input
    await client.xadd_bulk("stream", [])

    # Valid input
    mock_pool = MagicMock()
    client.get_pool = AsyncMock(return_value=mock_pool)
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pool.pipeline.return_value = mock_pipe

    await client.xadd_bulk("stream", [{"id": 1}])
    mock_pipe.xadd.assert_called()
    mock_pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_xadd_bulk_dlq_fallback(client):
    mock_pool = MagicMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(
        side_effect=redis_exceptions.ConnectionError("Redis Down")
    )
    mock_pool.pipeline.return_value = mock_pipe
    client.xadd_to_dlq = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError, match="Failed to write to Redis"):
            await client.xadd_bulk("stream", [{"id": 1}])
    client.xadd_to_dlq.assert_awaited()


@pytest.mark.asyncio
async def test_xadd_to_dlq(client):
    # Empty
    await client.xadd_to_dlq("s", [])

    # Exception handling
    client.get_pool = AsyncMock(side_effect=Exception("Total Fail"))
    await client.xadd_to_dlq("s", [{"id": 1}])


@pytest.mark.asyncio
async def test_ensure_consumer_group(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Normal
    await client.ensure_consumer_group("s", "g")

    # Busy Group (Exists)
    mock_pool.xgroup_create.side_effect = redis_exceptions.ResponseError("BUSYGROUP")
    await client.ensure_consumer_group("s", "g")  # Should just log and return


@pytest.mark.asyncio
async def test_read_stream_messages(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Normal
    mock_pool.xreadgroup.return_value = [[b"s", [[b"1", {b"d": b"{}"}]]]]
    res = await client.read_stream_messages("s", "g", "c")
    assert len(res) == 1

    # No Group Error
    mock_pool.xreadgroup.side_effect = redis_exceptions.ResponseError("NOGROUP")
    client.ensure_consumer_group = AsyncMock()
    res = await client.read_stream_messages("s", "g", "c")
    assert res == []
    client.ensure_consumer_group.assert_awaited()

    # Connection Error
    mock_pool.xreadgroup.side_effect = redis_exceptions.ConnectionError("Fail")
    with pytest.raises(ConnectionError, match="Redis connection failed"):
        await client.read_stream_messages("s", "g", "c")


@pytest.mark.asyncio
async def test_acknowledge_message(client):
    # Empty
    await client.acknowledge_message("s", "g")

    # Valid
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)
    await client.acknowledge_message("s", "g", "1")
    mock_pool.xack.assert_awaited()


@pytest.mark.asyncio
async def test_xautoclaim_stale_messages(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Success
    mock_pool.xautoclaim.return_value = (b"0-0", [])
    await client.xautoclaim_stale_messages("s", "g", "c", 1000)
    mock_pool.xautoclaim.assert_awaited()

    # Response Error
    mock_pool.xautoclaim.side_effect = redis_exceptions.ResponseError("Fail")
    res_id, res_list = await client.xautoclaim_stale_messages("s", "g", "c", 1000)
    assert res_list == []

    # Generic Error
    mock_pool.xautoclaim.side_effect = Exception("Crash")
    with pytest.raises(Exception, match="Crash"):
        await client.xautoclaim_stale_messages("s", "g", "c", 1000)


@pytest.mark.asyncio
async def test_ohlc_queue_ops(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # Enqueue
    await client.enqueue_ohlc_work({"t": 1})

    # Clear
    await client.clear_ohlc_work_queue()
    mock_pool.delete.assert_awaited()

    # Enqueue Failed (DLQ)
    await client.enqueue_failed_ohlc_work({"t": 1})
    mock_pool.lpush.assert_awaited()

    # DLQ Exception
    mock_pool.lpush.side_effect = Exception("Fail")
    await client.enqueue_failed_ohlc_work({"t": 1})  # Should catch log critical

    # Get Size
    client.get_pool.side_effect = None  # Reset
    mock_pool.lpush.side_effect = None
    mock_pool.llen.return_value = 5
    assert await client.get_ohlc_work_queue_size() == 5

    # Get Size Exception
    mock_pool.llen.side_effect = Exception("Fail")
    assert await client.get_ohlc_work_queue_size() == 0

    # Dequeue
    mock_pool.brpop.return_value = [b"k", orjson.dumps({"t": 1})]
    res = await client.dequeue_ohlc_work()
    assert res["t"] == 1

    # Dequeue Timeout
    mock_pool.brpop.side_effect = TimeoutError()
    assert await client.dequeue_ohlc_work() is None


@pytest.mark.asyncio
async def test_ticker_data(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    mock_pool.hget.return_value = orjson.dumps({"p": 1})
    res = await client.get_ticker_data("i")
    assert res["p"] == 1

    # Connection Error
    mock_pool.hget.side_effect = redis_exceptions.ConnectionError("Fail")
    with pytest.raises(ConnectionError, match="Redis connection failed"):
        await client.get_ticker_data("i")

    # Generic Error
    mock_pool.hget.side_effect = Exception("Fail")
    assert await client.get_ticker_data("i") is None


@pytest.mark.asyncio
async def test_system_state(client):
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    await client.set_system_state("A", "Reason")

    # Set Error
    mock_pool.hset.side_effect = ConnectionError("Fail")
    await client.set_system_state("A")  # Should catch

    # Get Default
    client.get_pool.side_effect = None
    mock_pool.get.return_value = None
    assert await client.get_system_state() == "LOCKED"

    # Get Error
    mock_pool.get.side_effect = ConnectionError("Fail")
    assert await client.get_system_state() == "LOCKED"


@pytest.mark.asyncio
async def test_parse_message(client):
    # Good
    res = CustomRedisClient.parse_stream_message({b"data": orjson.dumps({"a": 1})})
    assert res["data"]["a"] == 1

    # Bad JSON
    res = CustomRedisClient.parse_stream_message({b"data": b"{bad"})
    assert res["data"] == "{bad"

    # Bad Decode
    res = CustomRedisClient.parse_stream_message({b"data": b"\xff"})
    assert res["data"] == b"\xff"


@pytest.mark.asyncio
async def test_get_pool_stale_reconnect(client):
    """Test get_pool when existing pool ping fails (Lines 32-37)."""
    # 1. Setup an existing "stale" pool
    mock_stale_pool = MagicMock()
    mock_stale_pool.ping = AsyncMock(side_effect=TimeoutError("Stale"))
    mock_stale_pool.close = AsyncMock()
    client.pool = mock_stale_pool

    # 2. Setup the "new" pool that works
    mock_new_pool = MagicMock()
    mock_new_pool.ping = AsyncMock(return_value=True)

    with patch("redis.asyncio.from_url", return_value=mock_new_pool):
        # This should trigger _safe_close_pool, then create new
        pool = await client.get_pool()

    assert pool is mock_new_pool
    mock_stale_pool.close.assert_awaited()


@pytest.mark.asyncio
async def test_deep_exception_coverage(client):
    """Target specific logging exception blocks."""
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # 1. Dequeue OHLC generic exception (Line 359)
    mock_pool.brpop.side_effect = Exception("Boom")
    assert await client.dequeue_ohlc_work() is None

    # 2. Queue Size generic exception (Line 371)
    mock_pool.llen.side_effect = Exception("Boom")
    assert await client.get_ohlc_work_queue_size() == 0

    # 3. Get Ticker generic exception (Line 403)
    mock_pool.hget.side_effect = Exception("Boom")
    assert await client.get_ticker_data("inst") is None


@pytest.mark.asyncio
async def test_safe_close_pool_error(client):
    """Test that exception during pool.close() is logged and ignored."""
    mock_pool = MagicMock()
    # close is async
    mock_pool.close = AsyncMock(side_effect=Exception("Close Error"))
    client.pool = mock_pool

    # Should not raise
    await client._safe_close_pool()
    assert client.pool is None


@pytest.mark.asyncio
async def test_get_pool_stale_and_fail_coverage(client):
    """
    Targets:
    - Line 34: Stale pool (ping fails) -> _safe_close_pool
    - Line 43: Connect failure -> _safe_close_pool
    """
    # 1. Setup Stale Pool logic
    mock_stale = MagicMock()
    # ping raises Timeout (triggering Line 32-34)
    mock_stale.ping = AsyncMock(side_effect=TimeoutError("Timeout"))
    mock_stale.close = AsyncMock()
    client.pool = mock_stale
    client._lock = AsyncMock()  # Mock the lock

    # 2. Setup Connect Failure logic (triggering Line 43)
    # When we try to recreate, let's fail once then succeed
    mock_good = MagicMock()
    mock_good.ping = AsyncMock(return_value=True)

    with patch(
        "redis.asyncio.from_url",
        side_effect=[
            Exception("Connect Fail"),  # Attempt 1 fails (Hits Line 43)
            mock_good,  # Attempt 2 succeeds
        ],
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            pool = await client.get_pool()

    # Verify Stale pool was closed
    mock_stale.close.assert_awaited()
    # Verify we got the good pool eventually
    assert pool is mock_good


@pytest.mark.asyncio
async def test_deep_exceptions_redis(client):
    """
    Targets specific exception blocks in:
    - xadd_bulk (Lines 180-184)
    - dequeue_ohlc_work (Lines 359-361)
    - get_ohlc_work_queue_size (Lines 371-373)
    """
    mock_pool = AsyncMock()
    client.get_pool = AsyncMock(return_value=mock_pool)

    # 1. xadd_bulk exception (Chunk failure)
    # We need to fail xadd, but not fatally enough to hit DLQ immediately if we want to hit the log warning
    # Actually, xadd_bulk retries 3 times. We want to hit the "except" block inside the loop.
    client._write_sem = AsyncMock()
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(side_effect=redis_exceptions.ConnectionError("Fail"))
    mock_pool.pipeline.return_value = mock_pipe

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # This will raise eventually, but we just want to ensure the lines ran
        with pytest.raises(ConnectionError):
            await client.xadd_bulk("s", [{"id": 1}])

    # 2. dequeue_ohlc_work exception
    mock_pool.brpop.side_effect = Exception("Generic Fail")
    assert await client.dequeue_ohlc_work() is None

    # 3. get_ohlc_work_queue_size exception
    mock_pool.llen.side_effect = Exception("Generic Fail")
    assert await client.get_ohlc_work_queue_size() == 0
