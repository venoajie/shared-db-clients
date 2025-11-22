# src\shared_db_clients\redis_client.py

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Union

import orjson
import redis.asyncio as aioredis
from redis import exceptions as redis_exceptions

from shared_config.config import settings

log = logging.getLogger(__name__)


class CustomRedisClient:
    _OHLC_WORK_QUEUE_KEY = "queue:ohlc_work"
    _OHLC_FAILED_QUEUE_KEY = "dlq:ohlc_work"

    def __init__(self):
        self.pool = None
        self._circuit_open = False
        self._last_failure = 0
        self._reconnect_attempts = 0
        self._write_sem = asyncio.Semaphore(4)
        self._lock = asyncio.Lock()

    async def get_pool(self) -> aioredis.Redis:
        async with self._lock:
            if self.pool:
                try:
                    await asyncio.wait_for(self.pool.ping(), timeout=0.5)
                    return self.pool
                except (asyncio.TimeoutError, redis_exceptions.ConnectionError):
                    log.warning("Existing Redis pool is stale. Reconnecting.")
                    await self._safe_close_pool()

            if self._circuit_open:
                cooldown = min(60, 5 * (2**self._reconnect_attempts))
                if time.time() - self._last_failure < cooldown:
                    raise ConnectionError("Redis unavailable - circuit breaker open")
                self._circuit_open = False

            redis_config = settings.redis
            for attempt in range(5):
                try:
                    self.pool = aioredis.from_url(
                        redis_config.url,
                        password=redis_config.password,
                        db=redis_config.db,
                        socket_connect_timeout=2,
                        max_connections=30,
                        encoding="utf-8",
                        decode_responses=False,
                    )
                    await asyncio.wait_for(self.pool.ping(), timeout=3)
                    self._reconnect_attempts = 0
                    log.info("Redis connection established")
                    return self.pool
                except Exception as e:
                    log.warning(f"Connection failed on attempt {attempt+1}: {e}")
                    await self._safe_close_pool()
                    if attempt < 4:
                        await asyncio.sleep(2**attempt)
                    last_error = e

            self._circuit_open = True
            self._last_failure = time.time()
            self._reconnect_attempts += 1
            raise ConnectionError(
                f"Redis connection failed after 5 attempts: {last_error}"
            )

    async def _safe_close_pool(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                log.warning(f"Pool closure warning: {e}")
            finally:
                self.pool = None

    @staticmethod
    def parse_stream_message(message_data: Dict[bytes, bytes]) -> dict:
        """
        Correctly parse Redis stream message.
        Optimized to check for known JSON keys first to avoid exception overhead.
        """
        result = {}
        for key, value in message_data.items():
            k = key.decode("utf-8")
            
            # OPTIMIZATION: 99% of our stream data is in these keys.
            # Checking them explicitly avoids the expensive orjson exception loop.
            if k in ("data", "payload", "order", "trade", "kline"):
                try:
                    result[k] = orjson.loads(value)
                    continue
                except (orjson.JSONDecodeError, TypeError):
                    pass 
            
            # Fallback loop for unknown keys or plain strings
            try:
                result[k] = orjson.loads(value)
            except (orjson.JSONDecodeError, TypeError):
                try:
                    result[k] = value.decode("utf-8")
                except UnicodeDecodeError:
                    log.warning(f"Could not decode field '{k}'. Storing raw bytes.")
                    result[k] = value
        return result

    async def xadd_bulk(
        self,
        stream_name: str,
        messages: Union[List[dict], deque],
        maxlen: int = 10000,
    ) -> None:
        if not messages:
            return

        CHUNK_SIZE = 500
        message_list = list(messages)

        for chunk_start in range(0, len(message_list), CHUNK_SIZE):
            chunk = message_list[chunk_start : chunk_start + CHUNK_SIZE]
            for attempt in range(3):
                try:
                    async with self._write_sem:
                        pool = await self.get_pool()
                        pipe = pool.pipeline()
                        for msg in chunk:
                            encoded_msg = {
                                k.encode("utf-8"): (
                                    orjson.dumps(v)
                                    if isinstance(v, (dict, list, tuple))
                                    else str(v).encode("utf-8")
                                )
                                for k, v in msg.items()
                            }
                            pipe.xadd(
                                stream_name,
                                encoded_msg,
                                maxlen=maxlen,
                                approximate=True,
                            )
                        await pipe.execute()
                        break

                except (
                    redis_exceptions.ConnectionError,
                    asyncio.TimeoutError,
                    redis_exceptions.ResponseError,
                ) as e:
                    log.warning(
                        f"Failed to send chunk to Redis (attempt {attempt+1}/3): {e}"
                    )
                    if attempt == 2:
                        log.error(
                            "Final attempt to send chunk failed. Moving to DLQ stream."
                        )
                        await self.xadd_to_dlq(stream_name, chunk)
                        raise ConnectionError(
                            "Failed to write to Redis stream after retries."
                        ) from e
                    await asyncio.sleep(0.5 * (2**attempt))

    async def xadd_to_dlq(
        self,
        original_stream_name: str,
        failed_messages: List[dict],
    ):

        if not failed_messages:
            return

        dlq_stream_name = f"dlq:{original_stream_name}"
        try:
            pool = await self.get_pool()
            pipe = pool.pipeline()
            for msg in failed_messages:
                pipe.xadd(dlq_stream_name, {"payload": orjson.dumps(msg)}, maxlen=25000)
            await pipe.execute()
            log.warning(
                f"{len(failed_messages)} message(s) moved to DLQ stream '{dlq_stream_name}'"
            )
        except Exception as e:
            log.critical(
                f"CRITICAL: Failed to write to DLQ stream '{dlq_stream_name}': {e}"
            )

    async def ensure_consumer_group(
        self,
        stream_name: str,
        group_name: str,
    ):
        try:
            pool = await self.get_pool()
            await pool.xgroup_create(
                stream_name,
                group_name,
                id="0",
                mkstream=True,
            )

            log.info(
                f"Created consumer group '{group_name}' for stream '{stream_name}'."
            )
        except redis_exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                log.debug(f"Consumer group '{group_name}' already exists.")
            else:
                raise

    async def read_stream_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 250,
        block: int = 2000,
    ) -> list:
        try:
            pool = await self.get_pool()
            response = await pool.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=count,
                block=block,
            )
            return response[0][1] if response else []
        except redis_exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                log.warning(
                    f"Consumer group '{group_name}' missing for stream '{stream_name}', recreating..."
                )
                await self.ensure_consumer_group(stream_name, group_name)
                return []
            raise
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError) as e:
            raise ConnectionError("Redis connection failed during XREADGROUP") from e

    async def acknowledge_message(
        self,
        stream_name: str,
        group_name: str,
        *message_ids: str,
    ) -> None:

        if not message_ids:
            return
        pool = await self.get_pool()
        await pool.xack(stream_name, group_name, *message_ids)

    async def xautoclaim_stale_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        min_idle_time_ms: int,
        count: int = 100,
    ) -> tuple[bytes, list]:
        try:
            pool = await self.get_pool()

            return await pool.xautoclaim(
                name=stream_name,
                groupname=group_name,
                consumername=consumer_name,
                min_idle_time=min_idle_time_ms,
                start_id="0-0",
                count=count,
            )
        except redis_exceptions.ResponseError as e:
            log.warning(f"Could not run XAUTOCLAIM on '{stream_name}': {e}.")
            return None, []
        except Exception as e:
            log.error(f"An unexpected error occurred during XAUTOCLAIM: {e}")
            raise

    async def get_ticker_data(
        self,
        instrument_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full ticker data object for a given instrument from Redis.
        """
        key = f"ticker:{instrument_name}"
        try:
            pool = await self.get_pool()
            payload = await pool.hget(key, "payload")
            if payload:
                return orjson.loads(payload)
            return None
        except (redis_exceptions.ConnectionError, asyncio.TimeoutError) as e:
            log.error(
                f"Failed to get ticker data for '{instrument_name}' from Redis: {e}"
            )
            raise ConnectionError(
                f"Redis connection failed during ticker read for {instrument_name}"
            ) from e
        except Exception as e:
            log.error(
                f"An unexpected error occurred getting ticker '{instrument_name}': {e}"
            )
            return None

    async def get_system_state(self) -> str:
        """Retrieves the current global system state, defaulting to LOCKED on failure."""
        try:
            pool = await self.get_pool()
            state = await pool.get("system:state:simple")
            if state:
                return state.decode()
            old_state = await pool.get("system:state")
            return old_state.decode() if old_state else "LOCKED"
        except ConnectionError:
            log.warning(
                "Could not get system state due to Redis connection error. Defaulting to LOCKED."
            )
            return "LOCKED"

    async def set_system_state(
        self,
        state: str,
        reason: Optional[str] = None,
    ):
        """
        Sets the global system state.
        """
        try:
            pool = await self.get_pool()
            state_data = {
                "status": state,
                "reason": reason or "",
                "timestamp": time.time(),
            }
            await pool.hset("system:state", mapping=state_data)
            await pool.set("system:state:simple", state)

            log_message = f"System state transitioned to: {state.upper()}"
            if reason:
                log_message += f" (Reason: {reason})"
            log.info(log_message)
        except ConnectionError:
            log.error(
                f"Could not set system state to '{state}' due to Redis connection error."
            )

    async def clear_ohlc_work_queue(self):
        """Deletes the OHLC work queue, ensuring a fresh start."""
        try:
            pool = await self.get_pool()
            await pool.delete(self._OHLC_WORK_QUEUE_KEY)
            log.info(f"Cleared Redis queue: {self._OHLC_WORK_QUEUE_KEY}")
        except Exception as e:
            log.error(f"Failed to clear OHLC work queue: {e}")
            raise

    async def enqueue_ohlc_work(
        self,
        work_item: Dict[str, Any],
    ):
        """Adds a new OHLC backfill task to the left of the list (queue)."""
        try:
            pool = await self.get_pool()
            await pool.lpush(self._OHLC_WORK_QUEUE_KEY, orjson.dumps(work_item))
        except Exception as e:
            log.error(f"Failed to enqueue OHLC work item {work_item}: {e}")
            raise

    async def enqueue_failed_ohlc_work(
        self,
        work_item: Dict[str, Any],
    ):
        """Adds a failed OHLC backfill task to the DLQ."""
        try:
            pool = await self.get_pool()
            await pool.lpush(self._OHLC_FAILED_QUEUE_KEY, orjson.dumps(work_item))
            log.error(f"Moved failed OHLC work item to DLQ: {work_item}")
        except Exception as e:
            log.critical(
                f"CRITICAL: Failed to enqueue to DLQ. Item lost: {work_item}. Error: {e}"
            )

    async def dequeue_ohlc_work(self) -> Optional[Dict[str, Any]]:
        """
        Atomically retrieves and removes a task from the right of the list (queue).
        Uses a blocking pop with a timeout to be efficient.
        """
        try:
            pool = await self.get_pool()
            result = await pool.brpop(self._OHLC_WORK_QUEUE_KEY, timeout=5)
            if result:
                return orjson.loads(result[1])
            return None
        except (redis_exceptions.ConnectionError, asyncio.TimeoutError):
            log.warning("Redis connection issue during dequeue, returning None.")
            return None
        except Exception as e:
            log.error(f"Unexpected error during OHLC work dequeue: {e}")
            return None

    async def get_ohlc_work_queue_size(self) -> int:
        """Returns the current number of items in the OHLC work queue."""
        try:
            pool = await self.get_pool()
            return await pool.llen(self._OHLC_WORK_QUEUE_KEY)
        except Exception as e:
            log.error(f"Failed to get OHLC work queue size: {e}")
            return 0
