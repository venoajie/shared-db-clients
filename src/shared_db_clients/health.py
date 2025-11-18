# core/health.py
import asyncio
import aiohttp
import psutil
from datetime import datetime, timezone
from loguru import logger as log

from core.db.postgres import PostgresClient
from core.db.redis import CustomRedisClient

# --- Configuration for the new time sanity check ---
TIME_SOURCES = [
    "http://worldtimeapi.org/api/timezone/Etc/UTC",
    "https://www.google.com",
    "https://www.cloudflare.com/cdn-cgi/trace",  # This is a text-based endpoint
]
# The maximum acceptable difference between system time and real time (in seconds).
MAX_TIME_SKEW_SECONDS = 60


async def check_time_sanity():
    """
    Checks the local system time against a reliable external time source.
    Tries multiple sources for robustness. This is a critical startup check.
    Raises SystemExit if the time skew is too large or if all sources fail.
    """
    log.info("Performing system time sanity check...")

    last_error = None
    for source_url in TIME_SOURCES:
        try:
            log.debug(f"Attempting to get real time from: {source_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url, timeout=10) as response:
                    response.raise_for_status()

                    real_utc_time = None
                    # --- Logic to parse time from different source types ---
                    if "worldtimeapi.org" in source_url:
                        data = await response.json()
                        real_utc_time_str = data.get("utc_datetime")
                        real_utc_time = datetime.fromisoformat(real_utc_time_str)
                    elif "google.com" in source_url:
                        # Google provides a standard 'Date' header in GMT (equivalent to UTC)
                        date_header = response.headers.get("Date")
                        if not date_header:
                            raise ValueError(
                                "Google response did not contain a 'Date' header."
                            )
                        # Format: 'Wed, 16 Jul 2025 05:47:41 GMT'
                        real_utc_time = datetime.strptime(
                            date_header, "%a, %d %b %Y %H:%M:%S %Z"
                        ).replace(tzinfo=timezone.utc)
                    elif "cloudflare.com" in source_url:
                        # Cloudflare's trace endpoint provides a 'ts=' line with a Unix timestamp
                        text_data = await response.text()
                        for line in text_data.splitlines():
                            if line.startswith("ts="):
                                ts_str = line.split("=")[1]
                                real_utc_time = datetime.fromtimestamp(
                                    float(ts_str), tz=timezone.utc
                                )
                                break
                        if not real_utc_time:
                            raise ValueError(
                                "Cloudflare trace did not contain a 'ts=' line."
                            )

                    if not real_utc_time:
                        continue  # Should not happen if parsing logic is correct

                    system_utc_time = datetime.now(timezone.utc)
                    time_skew = abs((system_utc_time - real_utc_time).total_seconds())

                    log.info(f"Successfully fetched time from {source_url}")
                    log.info(f"Real UTC time: {real_utc_time.isoformat()}")
                    log.info(f"System UTC time: {system_utc_time.isoformat()}")
                    log.info(f"Calculated time skew: {time_skew:.2f} seconds.")

                    if time_skew > MAX_TIME_SKEW_SECONDS:
                        error_message = (
                            f"CRITICAL TIME SKEW DETECTED! System clock is off by {time_skew:.2f} seconds, "
                            f"which is more than the allowed maximum of {MAX_TIME_SKEW_SECONDS}s. Aborting startup."
                        )
                        log.critical(error_message)
                        raise SystemExit(error_message)

                    log.success("System time sanity check passed.")
                    return  # Exit the function on the first successful check

        except Exception as e:
            log.warning(f"Could not get time from {source_url}: {e}")
            last_error = e
            await asyncio.sleep(1)  # Wait a second before trying the next source

    # This block is only reached if the loop completes without a successful check
    error_message = f"Failed to verify system time from all reliable sources. Aborting for safety. Last error: {last_error}"
    log.critical(error_message)
    raise SystemExit(error_message)


async def health_check(
    postgres_client: PostgresClient,
    redis_client: CustomRedisClient,
):
    """
    Performs a runtime health check, gathering statistics on connections,
    memory usage, and stream backlogs.
    """
    # PostgreSQL connections
    pg_stats = {}
    if postgres_client._pool and not postgres_client._pool._closed:
        try:
            pg_stats = {
                "connections": postgres_client._pool.get_size(),
                "idle": postgres_client._pool.get_idle_size(),
            }
        except Exception as e:
            log.warning(f"Could not get PostgreSQL pool stats: {e}")
            pg_stats = {"error": str(e)}

    # Redis connections
    redis_stats = {}
    try:
        pool = await redis_client.get_pool()
        # This is an approximation; aioredis doesn't expose a simple connection count.
        # We can check the number of connections in the pool's internal list.
        redis_stats = {
            "pool_connections": len(pool.connection_pool._available_connections)
        }
    except Exception as e:
        log.warning(f"Could not get Redis pool stats: {e}")
        redis_stats = {"error": str(e)}

    # Memory diagnostics
    process = psutil.Process()
    mem_info = process.memory_full_info()

    # Stream backlog monitoring
    stream_backlog = 0
    try:
        pool = await redis_client.get_pool()
        stream_backlog = await pool.xlen("stream:market_data")
    except Exception as e:
        log.warning(f"Could not get stream backlog length: {e}")
        stream_backlog = -1  # Indicate an error

    return {
        "postgres": pg_stats,
        "redis": redis_stats,
        "memory": {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
            "uss_mb": round(mem_info.uss / 1024 / 1024, 2),
            "swap_mb": round(mem_info.swap / 1024 / 1024, 2),
        },
        "stream_backlog": stream_backlog,
        "process": {
            "open_files": len(process.open_files()),
            "threads": process.num_threads(),
        },
    }
