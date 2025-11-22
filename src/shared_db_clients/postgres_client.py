# src\shared_db_clients\postgres_client.py

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import orjson
from loguru import logger as log
from shared_config.config import PostgresSettings, settings


class PostgresClient:
    _pool: asyncpg.Pool = None

    def __init__(self, config: PostgresSettings | None = None):
        """
        Initializes the PostgresClient.
        
        Args:
            config: Optional PostgresSettings object. If None, falls back to the global 'settings.postgres'.
                    This allows for Dependency Injection during testing.
        """
        self.postgres_settings = config or settings.postgres

        # Lazy validation: We don't raise an error immediately if config is missing,
        # to allow partial usage of shared-db-clients (e.g. Redis-only consumers).
        # We only check self.dsn when start_pool() is called.
        if self.postgres_settings:
            self.dsn = self.postgres_settings.dsn
        else:
            self.dsn = None

    async def start_pool(self) -> asyncpg.Pool:
        if not self.dsn:
            raise ValueError("Cannot start PostgreSQL pool: No PostgreSQL configuration found.")

        if self._pool is None or self._pool._closed:
            log.info("PostgreSQL connection pool is not available. Creating new pool.")
            for attempt in range(5):
                try:
                    self._pool = await asyncpg.create_pool(
                        dsn=self.dsn,
                        min_size=2,
                        max_size=10,
                        command_timeout=30,
                        init=self._setup_json_codec,
                        server_settings={
                            "application_name": "trading-system-db-client",
                        },
                    )
                    log.info(
                        "PostgreSQL pool created successfully."
                    )
                    return self._pool
                except Exception as e:
                    log.error(
                        f"Failed to create PostgreSQL pool (attempt {attempt+1}/5): {e}"
                    )
                    await asyncio.sleep(2**attempt)
            raise ConnectionError(
                "Fatal: Could not create PostgreSQL pool after multiple retries."
            )
        return self._pool

    async def close_pool(self):
        if self._pool and not self._pool._closed:
            await self._pool.close()
            self._pool = None
            log.info("PostgreSQL connection pool closed.")

    async def _setup_json_codec(
        self,
        connection: asyncpg.Connection,
    ):

        for json_type in ["jsonb", "json"]:
            await connection.set_type_codec(
                json_type,
                encoder=lambda d: orjson.dumps(d).decode("utf-8"),
                decoder=orjson.loads,
                schema="pg_catalog",
            )

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
        query = """
            INSERT INTO tickers (
                exchange,
                instrument_name,
                last_price,
                mark_price,
                index_price,
                open_interest,
                best_bid_price,
                best_ask_price,
                data,
                exchange_timestamp,
                recorded_at
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (exchange, instrument_name) DO UPDATE SET
                last_price = EXCLUDED.last_price, mark_price = EXCLUDED.mark_price, index_price = EXCLUDED.index_price, open_interest = EXCLUDED.open_interest,
                best_bid_price = EXCLUDED.best_bid_price, best_ask_price = EXCLUDED.best_ask_price, data = EXCLUDED.data, exchange_timestamp = EXCLUDED.exchange_timestamp, recorded_at = NOW();
        """
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
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(query, records_to_upsert)
            log.debug(
                f"Successfully bulk-upserted {len(records_to_upsert)} tickers to cold storage."
            )
        except Exception as e:
            log.error("Error during bulk upsert of tickers: {}", e)
            raise

    async def bulk_upsert_ohlc(
        self,
        candles: list[dict[str, Any]],
    ):

        if not candles:
            return

        records_to_upsert = [self._prepare_ohlc_record(c) for c in candles]
        pool = await self.start_pool()
        for attempt in range(5):
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT bulk_upsert_ohlc($1::ohlc_upsert_type[])",
                            records_to_upsert,
                        )
                if attempt > 0:
                    log.success(
                        f"DB connection recovered on attempt {attempt + 1} for OHLC upsert."
                    )
                return
            except asyncpg.PostgresError as e:
                if "does not exist" in str(e):
                    log.warning(
                        f"Database schema not ready (attempt {attempt + 1}/5). Retrying OHLC upsert in {2 ** attempt}s. Error: {e}"
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    log.error("Unhandled PostgreSQL error during OHLC upsert: {}", e)
                    raise
            except Exception as e:
                log.error("Unexpected error during bulk upsert of OHLC candles: {}", e)
                if candles:
                    log.debug(
                        f"First failing candle record (potential cause): {candles[0]}"
                    )
                raise
        raise ConnectionError(
            "Failed to execute bulk_upsert_ohlc after multiple retries. Database may be unavailable or misconfigured."
        )

    async def bulk_upsert_instruments(
        self,
        instruments: list[dict[str, Any]],
        exchange: str,
    ):
        if not instruments:
            return
        query = """
            INSERT INTO instruments (exchange, instrument_name, market_type, instrument_kind, base_asset, quote_asset, settlement_asset, settlement_period, tick_size, contract_size, expiration_timestamp, data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (exchange, instrument_name) DO UPDATE
            SET market_type = EXCLUDED.market_type, instrument_kind = EXCLUDED.instrument_kind, base_asset = EXCLUDED.base_asset,
                quote_asset = EXCLUDED.quote_asset, settlement_asset = EXCLUDED.settlement_asset, settlement_period = EXCLUDED.settlement_period,
                tick_size = EXCLUDED.tick_size, contract_size = EXCLUDED.contract_size, expiration_timestamp = EXCLUDED.expiration_timestamp,
                data = EXCLUDED.data, recorded_at = CURRENT_TIMESTAMP;
        """
        try:
            records_to_upsert = []
            for inst_data in instruments:
                exp_ts_iso = inst_data.get("expiration_timestamp")
                exp_ts = datetime.fromisoformat(exp_ts_iso) if exp_ts_iso else None

                records_to_upsert.append(
                    (
                        inst_data.get("exchange"),
                        inst_data.get("instrument_name"),
                        inst_data.get("market_type"),
                        inst_data.get("instrument_kind"),
                        inst_data.get("base_asset"),
                        inst_data.get("quote_asset"),
                        inst_data.get("settlement_asset"),
                        inst_data.get("settlement_period"),
                        inst_data.get("tick_size"),
                        inst_data.get("contract_size"),
                        exp_ts,
                        inst_data.get("data"),
                    )
                )

            pool = await self.start_pool()
            async with pool.acquire() as conn:
                await conn.executemany(query, records_to_upsert)

            log.info(
                f"Successfully bulk-upserted {len(records_to_upsert)} instruments for exchange '{exchange}'."
            )
        except Exception as e:
            log.error("Error during bulk upsert of instruments: {}", e, exc_info=True)
            raise

    async def bulk_upsert_orders(
        self,
        records: list[dict[str, Any]],
    ):
        """
        Upserts a batch of order records into the orders table.
        """
        if not records:
            return
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT bulk_upsert_orders_from_json($1::jsonb[])", records
                    )
            log.info(f"Successfully bulk-upserted {len(records)} order/trade records.")
        except Exception as e:
            log.error("Error during bulk upsert of orders/trades: {}", e)
            raise

    async def bulk_insert_public_trades(
        self,
        records: list[dict[str, Any]],
    ):
        """Inserts public trades by converting dicts to DB tuples"""
        if not records:
            return

        # Convert dicts to tuples in DB function order
        records_to_insert = []
        for rec in records:
            try:
                records_to_insert.append(
                    (
                        rec["exchange"],
                        rec["instrument_name"],
                        rec["market_type"],
                        rec["trade_id"],
                        rec["price"],
                        rec["quantity"],
                        rec["is_buyer_maker"],
                        rec["was_best_price_match"],
                        rec["trade_timestamp"],
                    )
                )
            except KeyError as e:
                log.error(f"Missing key in trade record: {e}. Record: {rec}")
                continue

        if not records_to_insert:
            return

        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT bulk_insert_public_trades($1::public_trade_insert_type[])",
                        records_to_insert,
                    )
            log.info(f"Inserted {len(records_to_insert)} public trades")
        except Exception as e:
            log.error(
                f"Public trade bulk insert failed: {e}. "
                f"First record: {records_to_insert[0] if records_to_insert else 'None'}",
                exc_info=True,
            )
            raise

    async def delete_orders(
        self,
        orders_to_delete: list[tuple[str, str, str]],
    ):
        if not orders_to_delete:
            return
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                await conn.executemany(
                    "SELECT delete_order($1, $2, $3)", orders_to_delete
                )
            log.info(f"Successfully deleted {len(orders_to_delete)} order records.")
        except Exception as e:
            log.error(f"Error during order deletion: {e}")
            raise

    async def insert_account_information(
        self,
        user_id: str,
        data: dict[str, Any],
    ) -> None:
        query = """
            INSERT INTO account_information (user_id, data) VALUES ($1, $2)
            ON CONFLICT (user_id, type) DO UPDATE SET data = EXCLUDED.data, recorded_at = CURRENT_TIMESTAMP
        """
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, user_id, data)

    async def fetch_all_instruments(self) -> list[asyncpg.Record]:
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM v_instruments")

    async def fetch_active_trades(
        self,
        user_id: str = None,
    ) -> list[asyncpg.Record]:

        query = "SELECT * FROM v_active_trades"
        params = []
        if user_id:
            query += " WHERE user_id = $1"
            params.append(user_id)
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *params)

    async def fetch_open_orders(
        self,
        user_id: str = None,
    ) -> list[asyncpg.Record]:
        query = """
            SELECT * FROM orders
            WHERE trade_id IS NULL
        """

        params = []
        if user_id:
            query += " AND user_id = $1"
            params.append(user_id)

        query += " ORDER BY exchange_timestamp"
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *params)

    async def fetch_all_user_records(
        self,
        user_id: str,
    ) -> tuple[list[asyncpg.Record], list[asyncpg.Record]]:
        query = "SELECT * FROM orders WHERE user_id = $1 ORDER BY exchange_timestamp"
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            all_records = await conn.fetch(query, user_id)

        open_orders = [rec for rec in all_records if rec["trade_id"] is None]
        filled_trades = [rec for rec in all_records if rec["trade_id"] is not None]

        return open_orders, filled_trades

    async def fetch_trades_by_timestamp(
        self,
        start_ts_ms: int,
        end_ts_ms: int,
        exchange_name: str,
    ) -> list[asyncpg.Record]:
        query = """
                SELECT trade_id FROM orders
                WHERE exchange = $3
                AND trade_id IS NOT NULL
                AND exchange_timestamp >= $1 AND exchange_timestamp <= $2
            """
        start_dt = datetime.fromtimestamp(start_ts_ms / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ts_ms / 1000, tz=UTC)
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, start_dt, end_dt, exchange_name)

    async def check_bootstrap_status(
        self,
        exchange_name: str,
    ) -> bool:
        key = f"bootstrap_status:{exchange_name}"
        query = "SELECT value FROM system_metadata WHERE key = $1"
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(query, key)
            return result == "complete"

    async def set_bootstrap_status(
        self,
        is_complete: bool,
        exchange_name: str,
    ):
        key = f"bootstrap_status:{exchange_name}"
        query = """
            INSERT INTO system_metadata (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """
        status = "complete" if is_complete else "incomplete"
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, key, status)
        log.info(f"Set {key} to '{status}' in database.")

    async def fetch_all_trades_for_instrument(
        self,
        instrument_name: str,
    ) -> list[asyncpg.Record]:
        query = """
            SELECT trade_id, label, side, amount,
                CASE WHEN side = 'sell' THEN -amount ELSE amount END AS net_amount
            FROM orders
            WHERE instrument_name = $1 AND trade_id IS NOT NULL
        """
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, instrument_name)

    async def bulk_update_is_open_status(
        self,
        trade_ids: list[str],
        is_open: bool,
    ):
        if not trade_ids:
            return
        query = """
            UPDATE orders SET is_open = $1 WHERE trade_id = ANY($2::text[])
        """
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                await conn.execute(query, is_open, trade_ids)
            log.info(f"Updated is_open={is_open} for {len(trade_ids)} trades.")
        except Exception as e:
            log.error(f"Failed to bulk update is_open status: {e}")
            raise

    async def fetch_ohlc_for_instrument(
        self,
        exchange_name: str,
        instrument_name: str,
        resolution: str,
        limit: int,
    ) -> list[asyncpg.Record]:
        query = """
            SELECT "open", high, low, "close", tick, volume
            FROM ohlc
            WHERE exchange = $1 AND instrument_name = $2 AND resolution = $3
            ORDER BY tick DESC
            LIMIT $4;
        """
        try:
            resolution_td = self._parse_resolution_to_timedelta(resolution)
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(
                    query, exchange_name, instrument_name, resolution_td, limit
                )
        except ValueError as e:
            log.error("Could not parse resolution string '{}': {}", resolution, e)
            return []

    async def fetch_latest_ohlc_timestamp(
        self,
        exchange_name: str,
        instrument_name: str,
        resolution_td: timedelta,
    ) -> datetime | None:
        query = "SELECT MAX(tick) FROM ohlc WHERE exchange = $1 AND instrument_name = $2 AND resolution = $3"
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                return await conn.fetchval(
                    query,
                    exchange_name,
                    instrument_name,
                    resolution_td,
                )

        except Exception as e:
            log.error(
                f"Failed to fetch latest OHLC timestamp for {exchange_name}:{instrument_name} ({resolution_td}): {e}"
            )
            return None

    async def fetch_latest_public_trade_timestamp(
        self,
        exchange_name: str,
        instrument_name: str,
    ) -> datetime | None:
        """
        [NEW] Fetches the timestamp of the most recent public trade for a given
        instrument from the database.
        """
        query = "SELECT MAX(trade_timestamp) FROM public_trades WHERE exchange = $1 AND instrument_name = $2"
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                return await conn.fetchval(query, exchange_name, instrument_name)
        except Exception as e:
            log.error(
                f"Failed to fetch latest public trade timestamp for {exchange_name}:{instrument_name}: {e}"
            )
            return None

    async def fetch_futures_summary_for_exchange(
        self, exchange: str
    ) -> asyncpg.Record | None:
        """
        Fetches the aggregated futures summary for a specific exchange.
        """

        query = """
            WITH active_futures AS (
                SELECT instrument_name
                FROM instruments
                WHERE exchange = $1 AND instrument_kind IN ('future', 'perpetual')
                AND (expiration_timestamp IS NULL OR expiration_timestamp > NOW())
            )
            SELECT
                (SELECT json_agg(instrument_name) FROM active_futures) AS instruments_name
            FROM (SELECT 1) AS dummy;
        """
        try:
            pool = await self.start_pool()
            async with pool.acquire() as conn:
                return await conn.fetchrow(query, exchange)
        except Exception as e:
            # Use positional formatting for safe logging
            log.error("Failed to fetch futures summary for '{}': {}", exchange, e)
            return None

    async def fetch_all_trades_for_user(
        self,
        user_id: str,
    ) -> list[asyncpg.Record]:
        query = """
            SELECT trade_id, label, side, amount, instrument_name,
                CASE WHEN side = 'sell' THEN -amount ELSE amount END AS net_amount
            FROM orders
            WHERE user_id = $1 AND trade_id IS NOT NULL
        """
        pool = await self.start_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, user_id)
