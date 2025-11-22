from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared_db_clients.health import check_time_sanity, health_check


@pytest.mark.asyncio
async def test_check_time_sanity_success():
    # Mock a successful response from worldtimeapi
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"utc_datetime": datetime.now(UTC).isoformat()}
    mock_resp.raise_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_resp

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await check_time_sanity()  # Should not raise


@pytest.mark.asyncio
async def test_check_time_sanity_failure():
    # Mock all sources failing
    mock_session = AsyncMock()
    mock_session.get.side_effect = Exception("Network Down")

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(SystemExit):
            await check_time_sanity()


@pytest.mark.asyncio
async def test_health_check():
    mock_pg = MagicMock()
    mock_pg._pool.get_size.return_value = 5
    mock_pg._pool._closed = False

    mock_redis = MagicMock()
    mock_redis.get_pool = AsyncMock()
    mock_redis.get_pool.return_value.connection_pool._available_connections = [1, 2]
    mock_redis.get_pool.return_value.xlen.return_value = 100

    stats = await health_check(mock_pg, mock_redis)

    assert stats["postgres"]["connections"] == 5
    assert stats["redis"]["pool_connections"] == 2
    assert stats["stream_backlog"] == 100
