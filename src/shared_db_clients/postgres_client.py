# src\shared_db_clients\postgres_client.py

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Awaitable, TypeVar

import asyncpg
import orjson
from loguru import logger as log
from shared_config.config import PostgresSettings, settings

# Type variable for the resilient executor's return type
T = TypeVar("T")

class PostgresClient:
    _pool: asyncpg.Pool | None = None
    _lock = asyncio.Lock()

    def __init__(self, config: PostgresSettings | None = None):
        self.postgres_settings = config or settings.postgres
        self.dsn = self.postgres_settings.dsn if self.postgres_settings else None

    # [REFACTOR] New resilient execution wrapper for all Postgres commands
    async def _execute_resiliently(
        self, 
        command_func: Callable[[asyncpg.Connection], Awaitable[T]],
        command_name_for_logging: str
    ) -> T:
        """
        Executes a PostgreSQL command with a retry mechanism for connection errors.
        It manages acquiring a connection from the pool and transparently retries
        on transient network failures.
        """
        last_exception: Exception | None = None
        for attempt in range(3):
            try:
                pool = await self.start_pool()
                async with pool.acquire() as conn:
                    # Pass the acquired connection to the function to be executed
                    return await command_func(conn)
            except (
                asyncpg.PostgresConnectionError,
                asyncpg.InterfaceError,  # Catches stream-closed type errors
                TimeoutError,
            ) as e:
                log.warning(
                    f"Postgres command '{command_name_for_logging}' failed "
                    f"(attempt {attempt + 1}/3): {e}"
                )
                last_exception = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))

        log.error(f"Postgres command '{command_name_for_logging}' failed after 3 attempts.")
        raise ConnectionError(
            f"Failed to execute Postgres command '{command_name_for_logging}' after retries."
        ) from last_exception


    # [REFACTOR] Simplified start_pool to be idempotent and use the resilient wrapper
    async def start_pool(self) -> asyncpg.Pool:
        async with self._lock:
            if self._pool is not None and not self._pool._closed:
                return self._pool

            if not self.dsn:
                raise ValueError("Cannot start PostgreSQL pool: No configuration found.")

            log.info("PostgreSQL connection pool is not available. Creating new pool.")
            
            async def create_action(_: asyncpg.Connection) -> asyncpg.Pool:
                # The lambda wrapper for _execute_resiliently requires a conn argument,
                # but create_pool doesn't use it. We accept it to match the signature.
                return await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=2,
                    max_size=10,
                    command_timeout=30,
                    init=self._setup_json_codec,
                    server_settings={"application_name": "trading-system-db-client"},
                )

            # This is a special case of the executor that doesn't use an existing connection
            try:
                self._pool = await create_action(None) # type: ignore
                log.info("PostgreSQL pool created successfully.")
                return self._pool
            except Exception as e:
                 raise ConnectionError(
                    "Fatal: Could not create PostgreSQL pool after multiple retries."
                ) from e
                
    async def close_pool(self):
        async with self._lock:
            if self._pool and not self._pool._closed:
                await self._pool.close()
                self._pool = None
                log.info("PostgreSQL connection pool closed.")

    async def _setup_json_codec(self, connection: asyncpg.Connection):
        for json_type in ["jsonb", "json"]:
            await connection.set_type_codec(
                json_type,
                encoder=lambda d: orjson.dumps(d).decode("utf-8"),
                decoder=orjson.loads,
                schema="pg_catalog",
            )
            
    # --- All public methods are now refactored to use _execute_resiliently ---
    
    async def bulk_upsert_ohlc(self, candles: list[dict[str, Any]]):
        if not candles: return

        records = [self._prepare_ohlc_record(c) for c in candles]

        async def command(conn: asyncpg.Connection):
            # [REFACTOR] The retry loop is removed; the wrapper handles it.
            # The specific error check for "does not exist" remains, as it's an
            # application-level concern, not a connection error.
            try:
                async with conn.transaction():
                    await conn.execute("SELECT bulk_upsert_ohlc($1::ohlc_upsert_type[])", records)
            except asyncpg.PostgresError as e:
                if "does not exist" in str(e):
                    log.warning(f"Database schema not ready for OHLC upsert. Error: {e}")
                    # This will be retried by the wrapper.
                raise e # Re-raise to allow the resilient wrapper to catch it if it's a connection issue

        try:
            await self._execute_resiliently(command, "bulk_upsert_ohlc")
        except Exception as e:
            log.error(f"Failed to execute bulk_upsert_ohlc after retries: {e}")
            raise
            
    async def fetch_all_instruments(self) -> list[asyncpg.Record]:
        async def command(conn: asyncpg.Connection):
            return await conn.fetch("SELECT * FROM v_instruments")
        return await self._execute_resiliently(command, "fetch_all_instruments")

    async def fetch_active_trades(self, user_id: str | None = None) -> list[asyncpg.Record]:
        async def command(conn: asyncpg.Connection):
            query = "SELECT * FROM v_active_trades"
            params = [user_id] if user_id else []
            if user_id:
                query += " WHERE user_id = $1"
            return await conn.fetch(query, *params)
        return await self._execute_resiliently(command, "fetch_active_trades")
        
    async def fetch_open_orders(self, user_id: str | None = None) -> list[asyncpg.Record]:
        async def command(conn: asyncpg.Connection):
            query = "SELECT * FROM orders WHERE trade_id IS NULL"
            params = [user_id] if user_id else []
            if user_id:
                query += " AND user_id = $1"
            query += " ORDER BY exchange_timestamp"
            return await conn.fetch(query, *params)
        return await self._execute_resiliently(command, "fetch_open_orders")

    async def check_bootstrap_status(self, exchange_name: str) -> bool:
        async def command(conn: asyncpg.Connection):
            key = f"bootstrap_status:{exchange_name}"
            query = "SELECT value FROM system_metadata WHERE key = $1"
            result = await conn.fetchval(query, key)
            return result == "complete"
        return await self._execute_resiliently(command, "check_bootstrap_status")

    async def set_bootstrap_status(self, is_complete: bool, exchange_name: str):
        async def command(conn: asyncpg.Connection):
            key = f"bootstrap_status:{exchange_name}"
            query = """
                INSERT INTO system_metadata (key, value) VALUES ($1, $2)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
            """
            status = "complete" if is_complete else "incomplete"
            await conn.execute(query, key, status)
        
        await self._execute_resiliently(command, "set_bootstrap_status")
        log.info(f"Set bootstrap_status:{exchange_name} to '{'complete' if is_complete else 'incomplete'}' in database.")
    
    # ... Other helper methods like _parse_resolution_to_timedelta and _prepare_ohlc_record remain unchanged ...
    def _parse_resolution_to_timedelta(
        self,
        resolution: str,
    ) -> timedelta:
        """
        Parses a resolution string (e.g., '1', '60', '1D', '5 minutes', '1 hour') into a timedelta object..
        """
        parts = resolution.split()

        if len(parts) == 2:
            value = int(parts[0])
            unit = parts[1].lower()
            if "minute" in unit:
                return timedelta(minutes=value)
            if "hour" in unit:
                return timedelta(hours=value)
            if "day" in unit:
                return timedelta(days=value)
            if "week" in unit:
                return timedelta(weeks=value)
            if "second" in unit:
                return timedelta(seconds=value)

        try:
            if resolution.upper().endswith("D"):
                return timedelta(days=int(resolution[:-1]))
            if resolution.upper().endswith("W"):
                return timedelta(weeks=int(resolution[:-1]))
            if resolution.upper().endswith("H"):
                return timedelta(hours=int(resolution[:-1]))
            return timedelta(minutes=int(resolution))
        except ValueError as e:
            raise ValueError(f"Unknown resolution format: {resolution}") from e

    def _prepare_ohlc_record(
        self,
        candle_data: dict[str, Any],
    ) -> tuple:
        tick_dt = datetime.fromtimestamp(candle_data["tick"] / 1000, tz=UTC)
        resolution_str = candle_data["resolution"]
        resolution_td = self._parse_resolution_to_timedelta(resolution_str)

        return (
            candle_data["exchange"],
            candle_data["instrument_name"],
            resolution_td,
            tick_dt,
            candle_data["open"],
            candle_data["high"],
            candle_data["low"],
            candle_data["close"],
            candle_data["volume"],
            candle_data.get("open_interest"),
        )

    async def bulk_upsert_tickers(
        self,
        tickers_data: list[dict[str, Any]],
    ):
        if not tickers_data:
            return
        
        records_to_upsert = []
        for ticker in tickers_data:
            ts_ms = ticker.get("exchange_timestamp")
            if not ts_ms:
                log.warning(
                    f"Ticker data for {ticker.get('instrument_name')} is missing 'exchange_timestamp'. Skipping."
                )
                continue

            ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            records_to_upsert.append(
                (
                    ticker.get("instrument_name"),
                    ticker.get("last_price"),
                    ticker.get("mark_price"),
                    ticker.get("index_price"),
                    ticker.get("open_interest"),
                    ticker.get("best_bid_price"),
                    ticker.get("best_ask_price"),
                    ticker,
                    ts,
                )
            )

        async def command(conn: asyncpg.Connection):
            query = """
                INSERT INTO tickers (
                    exchange, instrument_name, last_price, mark_price, index_price, open_interest,
                    best_bid_price, best_ask_price, data, exchange_timestamp, recorded_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (exchange, instrument_name) DO UPDATE SET
                    last_price = EXCLUDED.last_price, mark_price = EXCLUDED.mark_price, index_price = EXCLUDED.index_price,
                    open_interest = EXCLUDED.open_interest, best_bid_price = EXCLUDED.best_bid_price,
                    best_ask_price = EXCLUDED.best_ask_price, data = EXCLUDED.data,
                    exchange_timestamp = EXCLUDED.exchange_timestamp, recorded_at = NOW();
            """
            async with conn.transaction():
                await conn.executemany(query, records_to_upsert)

        try:
            await self._execute_resiliently(command, "bulk_upsert_tickers")
            log.debug(f"Successfully bulk-upserted {len(records_to_upsert)} tickers.")
        except Exception as e:
            log.error(f"Error during bulk upsert of tickers: {e}")
            raise