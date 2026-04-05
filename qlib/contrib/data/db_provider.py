# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Database-backed data providers for Qlib.

Reads metadata from PostgreSQL and time-series data from ClickHouse,
implementing qlib's provider interfaces so they can be used as drop-in
replacements for the default file-based providers.

Usage::

    import qlib
    from qlib.contrib.data.db_provider import (
        DBCalendarProvider,
        DBInstrumentProvider,
        DBFeatureProvider,
        DBExpressionProvider,
        DBDatasetProvider,
        DBProvider,
        init_qlib_with_db,
    )

    # Quick init
    init_qlib_with_db()

    # Or fine-grained control
    qlib.init(
        calendar_provider="qlib.contrib.data.db_provider.DBCalendarProvider",
        instrument_provider="qlib.contrib.data.db_provider.DBInstrumentProvider",
        feature_provider="qlib.contrib.data.db_provider.DBFeatureProvider",
        expression_provider="qlib.contrib.data.db_provider.DBExpressionProvider",
        dataset_provider="qlib.contrib.data.db_provider.DBDatasetProvider",
        provider="qlib.contrib.data.db_provider.DBProvider",
    )
"""

from __future__ import annotations

import os
import logging
import threading
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from qlib.data.data import (
    CalendarProvider,
    DatasetProvider,
    ExpressionProvider,
    FeatureProvider,
    InstrumentProvider,
    PITProvider,
    BaseProvider,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: ts_code  <->  qlib instrument id
#   ts_code  = "600000.SH"   (Tushare convention)
#   qlib id  = "SH600000"    (qlib convention)
# ---------------------------------------------------------------------------

def ts_code_to_qlib(ts_code: str) -> str:
    """600000.SH -> SH600000"""
    parts = ts_code.split(".")
    if len(parts) == 2:
        return parts[1] + parts[0]
    return ts_code


def qlib_to_ts_code(inst: str) -> str:
    """SH600000 -> 600000.SH"""
    for prefix in ("SH", "SZ", "BJ"):
        if inst.startswith(prefix):
            return inst[2:] + "." + prefix
    return inst


# ---------------------------------------------------------------------------
# Database connection helpers
# ---------------------------------------------------------------------------

_pg_engine = None
_ch_local = threading.local()


def _get_pg_engine():
    """Lazily create a SQLAlchemy engine for PostgreSQL (thread-safe via connection pool)."""
    global _pg_engine
    if _pg_engine is not None:
        return _pg_engine
    from sqlalchemy import create_engine

    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db = os.environ.get("PGDATABASE", "stock_meta")
    user = os.environ.get("PGUSER", "stock_user")
    # Support both PGPASSWORD (standard) and PG_PASSWORD (docker-compose .env)
    password = os.environ.get("PGPASSWORD", "") or os.environ.get("PG_PASSWORD", "")
    from urllib.parse import quote_plus
    url = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{db}"
    _pg_engine = create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    return _pg_engine


def _get_ch_client():
    """Return a thread-local ClickHouse client (one per thread to avoid concurrency issues)."""
    client = getattr(_ch_local, "client", None)
    if client is not None:
        return client
    try:
        from clickhouse_connect import get_client
    except ImportError:
        raise ImportError(
            "clickhouse-connect is required. Install it with: pip install clickhouse-connect"
        )

    host = os.environ.get("CHHOST", "localhost")
    port = int(os.environ.get("CHPORT_HTTP", os.environ.get("CHPORT", "8123")))
    database = os.environ.get("CHDATABASE", "stock")
    username = os.environ.get("CHUSER", "default")
    # Support both CHPASSWORD and CH_PASSWORD (docker-compose .env)
    password = os.environ.get("CHPASSWORD", "") or os.environ.get("CH_PASSWORD", "")
    client = get_client(
        host=host, port=port, database=database, username=username, password=password
    )
    _ch_local.client = client
    return client


def _pg_query(sql: str, params=None) -> pd.DataFrame:
    """Execute a SQL query on PostgreSQL and return a DataFrame."""
    engine = _get_pg_engine()
    return pd.read_sql(sql, engine, params=params)


def _ch_query(sql: str, params=None) -> pd.DataFrame:
    """Execute a SQL query on ClickHouse and return a DataFrame."""
    client = _get_ch_client()
    result = client.query_df(sql, parameters=params)
    return result


# ---------------------------------------------------------------------------
# Feature field mapping:  qlib field name  ->  ClickHouse column
# ---------------------------------------------------------------------------

# Fields stored in stock_daily_prices
PRICE_FIELDS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "pre_close": "pre_close",
    "change": "change",
    "pct_chg": "pct_chg",
    "volume": "vol",
    "amount": "amount",
    "vwap": None,  # computed: amount / (vol * 100) * 10  (amount is in 千元, vol in 手)
}

# Fields stored in stock_adj_factor
ADJ_FIELDS = {
    "adj_factor": "adj_factor",
}

# Fields stored in stock_daily_basic
BASIC_FIELDS = {
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio",
    "pe": "pe",
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps": "ps",
    "ps_ttm": "ps_ttm",
    "dv_ratio": "dv_ratio",
    "dv_ttm": "dv_ttm",
    "total_share": "total_share",
    "float_share": "float_share",
    "free_share": "free_share",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
}

# All known fields -> (table, column)
ALL_FIELDS: dict[str, tuple[str, str]] = {}
for _col, _ch_col in PRICE_FIELDS.items():
    if _ch_col is not None:
        ALL_FIELDS[_col] = ("stock_daily_prices", _ch_col)
for _col, _ch_col in ADJ_FIELDS.items():
    ALL_FIELDS[_col] = ("stock_adj_factor", _ch_col)
for _col, _ch_col in BASIC_FIELDS.items():
    ALL_FIELDS[_col] = ("stock_daily_basic", _ch_col)

# Factor fields (stored in factor_values) - all Float32 columns
FACTOR_TABLE = "factor_values"


def _resolve_field(field: str):
    """
    Given a qlib field name (without the leading '$'), return
    (table_name, column_name) or None if not recognised.

    The field is stripped of the leading '$' before lookup.
    """
    field = str(field)
    if field.startswith("$"):
        field = field[1:]
    field = field.lower()

    if field in ALL_FIELDS:
        return ALL_FIELDS[field]
    # Check factor_values table columns
    # These are stored as-is (lowercase)
    return None


# ---------------------------------------------------------------------------
# Calendar Provider
# ---------------------------------------------------------------------------

class DBCalendarProvider(CalendarProvider):
    """
    Load trading calendar from PostgreSQL ``dim_trade_cal`` table.
    """

    def load_calendar(self, freq, future=False):
        sql = """
            SELECT cal_date
            FROM dim_trade_cal
            WHERE exchange = 'SSE'
              AND is_open = TRUE
            ORDER BY cal_date
        """
        df = _pg_query(sql)
        return pd.to_datetime(df["cal_date"]).tolist()


# ---------------------------------------------------------------------------
# Instrument Provider
# ---------------------------------------------------------------------------

class DBInstrumentProvider(InstrumentProvider):
    """
    List instruments from PostgreSQL ``dim_stock_basic`` and ClickHouse
    ``stock_daily_prices`` to build the (instrument -> time_spans) map.
    """

    # Cache for the instrument dict
    _cache = {}

    def _load_instruments(self, market: str, freq: str = "day"):
        cache_key = f"{market}_{freq}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if market == "all":
            where = "list_status = 'L'"
        elif market == "csi300":
            where = (
                "ts_code IN (SELECT DISTINCT con_code FROM stock_index_weight "
                "WHERE index_code = '000300.SH') AND list_status = 'L'"
            )
        elif market == "csi500":
            where = (
                "ts_code IN (SELECT DISTINCT con_code FROM stock_index_weight "
                "WHERE index_code = '000905.SH') AND list_status = 'L'"
            )
        elif market == "sse50":
            where = (
                "ts_code IN (SELECT DISTINCT con_code FROM stock_index_weight "
                "WHERE index_code = '000016.SH') AND list_status = 'L'"
            )
        elif market == "sse":
            where = "exchange = 'SSE' AND list_status = 'L'"
        elif market == "szse":
            where = "exchange = 'SZSE' AND list_status = 'L'"
        else:
            where = "list_status = 'L'"

        # Get listing dates from PG
        pg_sql = f"SELECT ts_code, list_date, COALESCE(delist_date, '2099-12-31') AS delist_date FROM dim_stock_basic WHERE {where}"
        df_meta = _pg_query(pg_sql)

        # Get actual trading range from CH to narrow down
        ts_codes = df_meta["ts_code"].tolist()
        if not ts_codes:
            self._cache[cache_key] = {}
            return {}

        # Build instrument dict: qlib_id -> [(start, end), ...]
        instruments = {}
        for _, row in df_meta.iterrows():
            qlib_id = ts_code_to_qlib(row["ts_code"])
            start = pd.Timestamp(row["list_date"])
            end = pd.Timestamp(row["delist_date"])
            # Cap end date to today
            if end > pd.Timestamp.now():
                end = pd.Timestamp.now()
            instruments[qlib_id] = [(start, end)]

        self._cache[cache_key] = instruments
        return instruments

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        market = instruments.get("market", "all") if isinstance(instruments, dict) else "all"
        _instruments = self._load_instruments(market, freq)

        # Time filtering
        from qlib.data.data import Cal

        cal = Cal.calendar(freq=freq)
        start_time = pd.Timestamp(start_time or cal[0])
        end_time = pd.Timestamp(end_time or cal[-1])

        _instruments_filtered = {}
        for inst, spans in _instruments.items():
            filtered = []
            for s, e in spans:
                ns = max(start_time, pd.Timestamp(s))
                ne = min(end_time, pd.Timestamp(e))
                if ns <= ne:
                    filtered.append((ns, ne))
            if filtered:
                _instruments_filtered[inst] = filtered

        # Apply filter_pipe
        filter_pipe = instruments.get("filter_pipe", []) if isinstance(instruments, dict) else []
        for filter_config in filter_pipe:
            from qlib.data import filter as F

            filter_t = getattr(F, filter_config["filter_type"]).from_config(filter_config)
            _instruments_filtered = filter_t(_instruments_filtered, start_time, end_time, freq)

        if as_list:
            return list(_instruments_filtered)
        return _instruments_filtered


# ---------------------------------------------------------------------------
# Feature Provider
# ---------------------------------------------------------------------------

class DBFeatureProvider(FeatureProvider):
    """
    Load feature data for a single instrument from ClickHouse.

    Supports price fields (stock_daily_prices), adjustment factors
    (stock_adj_factor), daily basic indicators (stock_daily_basic),
    and computed factors (factor_values).
    """

    # Column cache for factor_values table
    _factor_columns: Optional[list[str]] = None

    @classmethod
    def _get_factor_columns(cls) -> list[str]:
        if cls._factor_columns is not None:
            return cls._factor_columns
        result = _ch_query("SELECT name FROM system.columns WHERE table = 'factor_values' AND database = currentDatabase()")
        cols = result["name"].tolist()
        # Exclude metadata columns
        cls._factor_columns = [c for c in cols if c not in ("ts_code", "trade_date", "updated_at")]
        return cls._factor_columns

    def feature(self, instrument, field, start_index, end_index, freq):
        field_str = str(field)
        if field_str.startswith("$"):
            field_str = field_str[1:]
        field_lower = field_str.lower()

        ts_code = qlib_to_ts_code(instrument)

        # Resolve which table + column to query
        mapping = _resolve_field(field_str)
        if mapping is not None:
            table, col = mapping
            sql_template = f"""
                SELECT trade_date, {{col}}
                FROM {{table}}
                WHERE ts_code = %(ts_code)s
                  AND trade_date >= %(start)s
                  AND trade_date <= %(end)s
                ORDER BY trade_date
            """
            sql = sql_template.format(col=col, table=table)
        elif field_lower == "vwap":
            sql = """
                SELECT trade_date,
                       CASE WHEN vol > 0 THEN amount / (vol * 100) * 1000 ELSE NaN END AS vwap
                FROM stock_daily_prices
                WHERE ts_code = %(ts_code)s
                  AND trade_date >= %(start)s
                  AND trade_date <= %(end)s
                ORDER BY trade_date
            """
            col = "vwap"
        else:
            # Try factor_values table
            factor_cols = self._get_factor_columns()
            if field_lower in factor_cols:
                col = field_lower
                sql = f"""
                    SELECT trade_date, {col}
                    FROM factor_values
                    WHERE ts_code = %(ts_code)s
                      AND trade_date >= %(start)s
                      AND trade_date <= %(end)s
                    ORDER BY trade_date
                """
            else:
                logger.debug(f"Field '{field_str}' not found in any table")
                return pd.Series(dtype=np.float32)

        # start_index/end_index are integer offsets into the calendar.
        # Convert to actual dates for the SQL query.
        from qlib.data.data import Cal

        calendar = Cal.calendar(freq=freq)
        if len(calendar) == 0:
            return pd.Series(dtype=np.float32)

        if isinstance(start_index, (int, np.integer)):
            start_idx = int(max(0, start_index))
            end_idx = int(min(end_index, len(calendar) - 1))
            start_date = calendar[start_idx]
            end_date = calendar[end_idx]
        else:
            start_date = pd.Timestamp(start_index)
            end_date = pd.Timestamp(end_index)
            start_idx = int(np.searchsorted(calendar, start_date))
            end_idx = int(np.searchsorted(calendar, end_date))

        params = {
            "ts_code": ts_code,
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }

        df = _ch_query(sql, params=params)
        if df.empty:
            # Return an integer-indexed Series matching the qlib convention
            return pd.Series(
                np.nan, index=np.arange(start_idx, end_idx + 1), dtype=np.float32
            )

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        s = df[col].astype(np.float32)

        # Build a mapping: date -> integer index in the calendar
        cal_sub = calendar[start_idx : end_idx + 1]
        # Reindex to the calendar slice, then replace DatetimeIndex with int index
        s = s.reindex(cal_sub)
        s.index = np.arange(start_idx, start_idx + len(s))
        return s


# ---------------------------------------------------------------------------
# Expression Provider  (wraps FeatureProvider through the calendar)
# ---------------------------------------------------------------------------

class DBExpressionProvider(ExpressionProvider):
    """
    Resolve qlib expressions by delegating to DBFeatureProvider for
    leaf variables ($open, $close, ...) and evaluating operators via
    qlib's expression engine.
    """

    def __init__(self, time2idx=True):
        super().__init__()
        self.time2idx = time2idx

    def expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        from qlib.utils import time_to_slc_point
        from qlib.data.data import Cal

        expression = self.get_expression_instance(field)
        start_time = time_to_slc_point(start_time)
        end_time = time_to_slc_point(end_time)

        if self.time2idx:
            _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq=freq, future=False)
            lft_etd, rght_etd = expression.get_extended_window_size()
            query_start = max(0, start_index - lft_etd)
            query_end = end_index + rght_etd
        else:
            start_index = end_index = query_start = start_time
            query_end = end_time

        try:
            series = expression.load(instrument, query_start, query_end, freq)
        except Exception as e:
            logger.debug(
                f"Loading expression error: instrument={instrument}, field=({field}), "
                f"start_time={start_time}, end_time={end_time}, freq={freq}. error: {e}"
            )
            raise

        try:
            series = series.astype(np.float32)
        except (ValueError, TypeError):
            pass

        if not series.empty:
            series = series.loc[start_index:end_index]
        return series


# ---------------------------------------------------------------------------
# Dataset Provider
# ---------------------------------------------------------------------------

class DBDatasetProvider(DatasetProvider):
    """
    Build a multi-index DataFrame (instrument, datetime) by querying
    ClickHouse in bulk for all requested instruments and fields.
    """

    def __init__(self, align_time: bool = True):
        super().__init__()
        self.align_time = align_time

    def dataset(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        inst_processors=[],
    ):
        from qlib.data.data import Cal

        instruments_d = self.get_instruments_d(instruments, freq)
        column_names = self.get_column_names(fields)

        if self.align_time:
            cal = Cal.calendar(start_time, end_time, freq)
            if len(cal) == 0:
                return pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                    columns=column_names,
                )
            start_time = cal[0]
            end_time = cal[-1]

        data = self.dataset_processor(
            instruments_d, column_names, start_time, end_time, freq, inst_processors=inst_processors
        )
        return data


# ---------------------------------------------------------------------------
# PIT Provider  (stub - returns empty data)
# ---------------------------------------------------------------------------

class DBPITProvider(PITProvider):
    """Point-in-time data provider (stub). Not yet implemented for DB backend."""

    def period_feature(self, instrument, field, start_index, end_index, cur_time, period=None):
        return pd.Series(dtype=np.float64)


# ---------------------------------------------------------------------------
# Top-level provider (the one registered as "provider" in qlib config)
# ---------------------------------------------------------------------------

class DBProvider(BaseProvider):
    """DB-backed drop-in replacement for LocalProvider."""

    pass


# ---------------------------------------------------------------------------
# High-level convenience: DBDataLoader
# ---------------------------------------------------------------------------

class DBDataLoader:
    """
    A standalone data loader that reads directly from PostgreSQL and
    ClickHouse without depending on qlib's expression engine.

    This is useful when you want a simple DataFrame for analysis,
    factor research, or custom model training without the full qlib
    provider stack.

    Parameters
    ----------
    pg_conn : dict, optional
        PostgreSQL connection kwargs. Defaults to env vars.
    ch_conn : dict, optional
        ClickHouse connection kwargs. Defaults to env vars.
    """

    def __init__(self, pg_conn: dict | None = None, ch_conn: dict | None = None):
        self._pg_conn = pg_conn
        self._ch_conn = ch_conn
        if pg_conn:
            global _pg_engine
            from sqlalchemy import create_engine

            _pg_engine = create_engine(
                f"postgresql+psycopg2://{pg_conn.get('user', 'stock_user')}:"
                f"{pg_conn.get('password', '')}@{pg_conn.get('host', 'localhost')}:"
                f"{pg_conn.get('port', 5432)}/{pg_conn.get('database', 'stock_meta')}",
                pool_size=5, max_overflow=10, pool_pre_ping=True,
            )
        if ch_conn:
            # Set env vars so that _get_ch_client() picks them up for each thread
            os.environ.setdefault("CHHOST", ch_conn.get("host", "localhost"))
            os.environ.setdefault("CHPORT_HTTP", str(ch_conn.get("port", 8123)))
            os.environ.setdefault("CHDATABASE", ch_conn.get("database", "stock"))
            os.environ.setdefault("CHUSER", ch_conn.get("username", "default"))
            os.environ.setdefault("CHPASSWORD", ch_conn.get("password", ""))
            # Pre-create client for current thread
            _get_ch_client()

    # -- Trading Calendar --------------------------------------------------

    def get_trading_calendar(
        self, exchange: str = "SSE", start_date: str | None = None, end_date: str | None = None
    ) -> pd.DatetimeIndex:
        """Return trading calendar as a DatetimeIndex."""
        sql = f"""
            SELECT cal_date FROM dim_trade_cal
            WHERE exchange = '{exchange}' AND is_open = TRUE
        """
        conditions = []
        if start_date:
            conditions.append(f"cal_date >= '{start_date}'")
        if end_date:
            conditions.append(f"cal_date <= '{end_date}'")
        if conditions:
            sql += " AND " + " AND ".join(conditions)
        sql += " ORDER BY cal_date"

        df = _pg_query(sql)
        return pd.DatetimeIndex(pd.to_datetime(df["cal_date"]))

    # -- Stock List ---------------------------------------------------------

    def get_stock_list(
        self,
        list_status: str = "L",
        exchange: str | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        """Return stock basic info from PostgreSQL."""
        conditions = [f"list_status = '{list_status}'"]
        if exchange:
            conditions.append(f"exchange = '{exchange}'")
        if market:
            conditions.append(f"market = '{market}'")
        where = " AND ".join(conditions)
        sql = f"SELECT * FROM dim_stock_basic WHERE {where} ORDER BY ts_code"
        return _pg_query(sql)

    # -- Daily Price Data ---------------------------------------------------

    def get_daily_prices(
        self,
        instruments: str | list[str],
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
        adjusted: bool = False,
    ) -> pd.DataFrame:
        """
        Load daily OHLCV data from ClickHouse.

        Parameters
        ----------
        instruments : str or list
            Stock code(s) in Tushare format (e.g. "600000.SH") or qlib format
            (e.g. "SH600000").
        start_date, end_date : str
            Date range in "YYYY-MM-DD" format.
        fields : list, optional
            Columns to return. Default: all price columns.
        adjusted : bool
            If True, join with adj_factor and return adjusted prices.

        Returns
        -------
        pd.DataFrame with MultiIndex (ts_code, trade_date)
        """
        if isinstance(instruments, str):
            instruments = [instruments]
        # Convert qlib format to ts_code
        ts_codes = [qlib_to_ts_code(i) if not "." in i else i for i in instruments]
        codes_str = ", ".join(f"'{c}'" for c in ts_codes)

        if adjusted:
            sql = f"""
                SELECT p.trade_date, p.ts_code,
                       p.open * a.adj_factor / first_value(a.adj_factor) OVER (PARTITION BY p.ts_code ORDER BY p.trade_date) AS open,
                       p.high * a.adj_factor / first_value(a.adj_factor) OVER (PARTITION BY p.ts_code ORDER BY p.trade_date) AS high,
                       p.low  * a.adj_factor / first_value(a.adj_factor) OVER (PARTITION BY p.ts_code ORDER BY p.trade_date) AS low,
                       p.close * a.adj_factor / first_value(a.adj_factor) OVER (PARTITION BY p.ts_code ORDER BY p.trade_date) AS close,
                       p.vol, p.amount
                FROM stock_daily_prices p
                JOIN stock_adj_factor a ON p.ts_code = a.ts_code AND p.trade_date = a.trade_date
                WHERE p.ts_code IN ({codes_str})
                  AND p.trade_date >= '{start_date}'
                  AND p.trade_date <= '{end_date}'
                ORDER BY p.ts_code, p.trade_date
            """
        else:
            select_cols = ", ".join(fields) if fields else "trade_date, ts_code, open, high, low, close, pre_close, change, pct_chg, vol, amount"
            sql = f"""
                SELECT {select_cols}
                FROM stock_daily_prices
                WHERE ts_code IN ({codes_str})
                  AND trade_date >= '{start_date}'
                  AND trade_date <= '{end_date}'
                ORDER BY ts_code, trade_date
            """

        df = _ch_query(sql)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index(["ts_code", "trade_date"])
        return df

    # -- Daily Basic Indicators ---------------------------------------------

    def get_daily_basic(
        self,
        instruments: str | list[str],
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load daily basic indicators (PE, PB, turnover, market cap, etc.)."""
        if isinstance(instruments, str):
            instruments = [instruments]
        ts_codes = [qlib_to_ts_code(i) if "." not in i else i for i in instruments]
        codes_str = ", ".join(f"'{c}'" for c in ts_codes)

        select_cols = ", ".join(fields) if fields else "*"
        sql = f"""
            SELECT {select_cols}
            FROM stock_daily_basic
            WHERE ts_code IN ({codes_str})
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY ts_code, trade_date
        """
        df = _ch_query(sql)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index(["ts_code", "trade_date"])

    # -- Factor Values ------------------------------------------------------

    def get_factor_values(
        self,
        instruments: str | list[str],
        start_date: str,
        end_date: str,
        factors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load pre-computed factor values from ClickHouse.

        Parameters
        ----------
        instruments : str or list
            Stock code(s).
        start_date, end_date : str
            Date range.
        factors : list, optional
            Factor column names. If None, returns all factors.
        """
        if isinstance(instruments, str):
            instruments = [instruments]
        ts_codes = [qlib_to_ts_code(i) if "." not in i else i for i in instruments]
        codes_str = ", ".join(f"'{c}'" for c in ts_codes)

        if factors:
            select_cols = "ts_code, trade_date, " + ", ".join(factors)
        else:
            select_cols = "*"

        sql = f"""
            SELECT {select_cols}
            FROM factor_values
            WHERE ts_code IN ({codes_str})
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY ts_code, trade_date
        """
        df = _ch_query(sql)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index(["ts_code", "trade_date"])

    # -- Index Data ---------------------------------------------------------

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load index daily data."""
        sql = f"""
            SELECT * FROM stock_index_daily
            WHERE ts_code = '{index_code}'
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        df = _ch_query(sql)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")

    def get_index_components(self, index_code: str, date: str | None = None) -> list[str]:
        """Get constituent stocks of an index."""
        if date:
            sql = f"""
                SELECT DISTINCT con_code FROM stock_index_weight
                WHERE index_code = '{index_code}' AND trade_date = '{date}'
            """
        else:
            sql = f"""
                SELECT DISTINCT con_code FROM stock_index_weight
                WHERE index_code = '{index_code}'
                ORDER BY con_code
            """
        df = _ch_query(sql)
        return df["con_code"].tolist()

    # -- Bulk Load for qlib-style DataFrame ---------------------------------

    def load_features(
        self,
        instruments: list[str],
        fields: list[str],
        start_time: str,
        end_time: str,
        freq: str = "day",
    ) -> pd.DataFrame:
        """
        Load a qlib-style DataFrame with MultiIndex (datetime, instrument)
        and columns matching the requested fields.

        This method does a single bulk query per table, which is far more
        efficient than querying per-instrument.

        Parameters
        ----------
        instruments : list
            Instrument IDs in qlib format (e.g. ["SH600000", "SZ000001"])
            or Tushare format (e.g. ["600000.SH", "000001.SZ"]).
        fields : list
            Field names without '$' prefix (e.g. ["open", "close", "volume"]).
        start_time, end_time : str
            Date range.
        freq : str
            Frequency, currently only "day" is supported.

        Returns
        -------
        pd.DataFrame with index (datetime, instrument), columns = fields
        """
        # Convert all to ts_code for DB queries
        ts_codes = []
        qlib_ids = []
        for inst in instruments:
            if "." in inst:
                ts_codes.append(inst)
                qlib_ids.append(ts_code_to_qlib(inst))
            else:
                ts_codes.append(qlib_to_ts_code(inst))
                qlib_ids.append(inst)

        id_map = dict(zip(ts_codes, qlib_ids))
        codes_str = ", ".join(f"'{c}'" for c in ts_codes)

        # Group fields by table
        table_fields: dict[str, list[str]] = {}
        for f in fields:
            mapping = _resolve_field(f)
            if mapping is not None:
                table, col = mapping
                table_fields.setdefault(table, []).append((f, col))
            elif f.lower() == "vwap":
                table_fields.setdefault("stock_daily_prices", []).append(("vwap", "vwap"))
            else:
                # Try factor_values
                table_fields.setdefault("factor_values", []).append((f, f.lower()))

        all_dfs = []

        for table, field_pairs in table_fields.items():
            ch_cols = [pair[1] for pair in field_pairs]
            qlib_names = [pair[0] for pair in field_pairs]

            if table == "factor_values" and any(pair[0] == f and pair[1] == f.lower() for pair in field_pairs):
                # Verify factor columns exist
                available = DBFeatureProvider._get_factor_columns()
                verified_pairs = []
                for name, col in field_pairs:
                    if col in available or name.lower() in available:
                        verified_pairs.append((name, col if col in available else name.lower()))
                field_pairs = verified_pairs
                if not field_pairs:
                    continue
                ch_cols = [pair[1] for pair in field_pairs]
                qlib_names = [pair[0] for pair in field_pairs]

            select_exprs = ["ts_code", "trade_date"] + ch_cols

            if table == "stock_daily_prices" and "vwap" in qlib_names:
                idx = qlib_names.index("vwap")
                select_exprs[idx + 2] = (
                    "CASE WHEN vol > 0 THEN amount / (vol * 100) * 1000 ELSE NaN END AS vwap"
                )

            select_str = ", ".join(select_exprs)
            sql = f"""
                SELECT {select_str}
                FROM {table}
                WHERE ts_code IN ({codes_str})
                  AND trade_date >= '{start_time}'
                  AND trade_date <= '{end_time}'
                ORDER BY ts_code, trade_date
            """
            df = _ch_query(sql)
            if df.empty:
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["instrument"] = df["ts_code"].map(id_map)
            df = df.drop(columns=["ts_code"])
            df = df.rename(columns=dict(zip(ch_cols, qlib_names)))
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=("datetime", "instrument")),
                columns=fields,
            )

        # Merge all table results
        from functools import reduce

        merged = reduce(
            lambda left, right: pd.merge(left, right, on=["trade_date", "instrument"], how="outer"),
            all_dfs,
        )
        merged = merged.set_index(["trade_date", "instrument"])
        merged.index.names = ["datetime", "instrument"]
        merged = merged.sort_index()

        # Ensure all requested columns exist
        for f in fields:
            if f not in merged.columns:
                merged[f] = np.nan

        merged = merged[fields].astype(np.float32)
        return merged


# ---------------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------------

def init_qlib_with_db(
    provider_uri: str = "",
    region: str = "cn",
    **kwargs,
):
    """
    Initialize qlib with database-backed providers.

    This is a convenience function that sets up all the provider
    configuration to use PostgreSQL + ClickHouse instead of local files.

    Environment variables used::

        # PostgreSQL
        PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD (or PG_PASSWORD)

        # ClickHouse
        CHHOST, CHPORT_HTTP, CHDATABASE, CHUSER, CHPASSWORD (or CH_PASSWORD)
    """
    import qlib

    # Inject DBProvider into qlib.data so that register_wrapper can find it
    # by name ("DBProvider") when qlib.init() calls register_all_wrappers().
    import qlib.data as _data_pkg
    _data_pkg.DBProvider = DBProvider

    qlib.init(
        provider_uri=provider_uri,
        region=region,
        calendar_provider={
            "class": "DBCalendarProvider",
            "module_path": "qlib.contrib.data.db_provider",
        },
        instrument_provider={
            "class": "DBInstrumentProvider",
            "module_path": "qlib.contrib.data.db_provider",
        },
        feature_provider={
            "class": "DBFeatureProvider",
            "module_path": "qlib.contrib.data.db_provider",
        },
        expression_provider={
            "class": "DBExpressionProvider",
            "module_path": "qlib.contrib.data.db_provider",
        },
        dataset_provider={
            "class": "DBDatasetProvider",
            "module_path": "qlib.contrib.data.db_provider",
        },
        provider="DBProvider",
        **kwargs,
    )
