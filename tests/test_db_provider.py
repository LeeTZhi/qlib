"""
Unit tests for qlib.contrib.data.db_provider.

These tests require:
- PostgreSQL running with stock_meta database (dim_trade_cal, dim_stock_basic tables)
- ClickHouse running with stock database (stock_daily_prices etc.)
- Environment variables set (or .env loaded)

Run with:
    python -m pytest tests/test_db_provider.py -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv("/data/StockData/code/stock_strategy_platform/.env")
except ImportError:
    pass

os.environ.setdefault("PGUSER", "stock_user")
os.environ.setdefault("PGDATABASE", "stock_meta")
os.environ.setdefault("CHDATABASE", "stock")


class TestCodeConversion(unittest.TestCase):
    """Test ts_code <-> qlib instrument ID conversions."""

    def test_ts_code_to_qlib_sh(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib

        self.assertEqual(ts_code_to_qlib("600000.SH"), "SH600000")

    def test_ts_code_to_qlib_sz(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib

        self.assertEqual(ts_code_to_qlib("000001.SZ"), "SZ000001")

    def test_ts_code_to_qlib_bj(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib

        self.assertEqual(ts_code_to_qlib("430047.BJ"), "BJ430047")

    def test_ts_code_to_qlib_passthrough(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib

        # No dot -> passthrough
        self.assertEqual(ts_code_to_qlib("SH600000"), "SH600000")

    def test_qlib_to_ts_code_sh(self):
        from qlib.contrib.data.db_provider import qlib_to_ts_code

        self.assertEqual(qlib_to_ts_code("SH600000"), "600000.SH")

    def test_qlib_to_ts_code_sz(self):
        from qlib.contrib.data.db_provider import qlib_to_ts_code

        self.assertEqual(qlib_to_ts_code("SZ000001"), "000001.SZ")

    def test_qlib_to_ts_code_bj(self):
        from qlib.contrib.data.db_provider import qlib_to_ts_code

        self.assertEqual(qlib_to_ts_code("BJ430047"), "430047.BJ")

    def test_roundtrip_sh(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib, qlib_to_ts_code

        self.assertEqual(qlib_to_ts_code(ts_code_to_qlib("600000.SH")), "600000.SH")

    def test_roundtrip_sz(self):
        from qlib.contrib.data.db_provider import ts_code_to_qlib, qlib_to_ts_code

        self.assertEqual(ts_code_to_qlib(qlib_to_ts_code("SZ000001")), "SZ000001")


class TestFieldResolution(unittest.TestCase):
    """Test field name -> table/column resolution."""

    def test_price_fields(self):
        from qlib.contrib.data.db_provider import _resolve_field

        self.assertEqual(_resolve_field("open"), ("stock_daily_prices", "open"))
        self.assertEqual(_resolve_field("close"), ("stock_daily_prices", "close"))
        self.assertEqual(_resolve_field("high"), ("stock_daily_prices", "high"))
        self.assertEqual(_resolve_field("low"), ("stock_daily_prices", "low"))
        self.assertEqual(_resolve_field("volume"), ("stock_daily_prices", "vol"))
        self.assertEqual(_resolve_field("amount"), ("stock_daily_prices", "amount"))

    def test_adj_fields(self):
        from qlib.contrib.data.db_provider import _resolve_field

        self.assertEqual(_resolve_field("adj_factor"), ("stock_adj_factor", "adj_factor"))

    def test_basic_fields(self):
        from qlib.contrib.data.db_provider import _resolve_field

        self.assertEqual(_resolve_field("pe"), ("stock_daily_basic", "pe"))
        self.assertEqual(_resolve_field("pb"), ("stock_daily_basic", "pb"))
        self.assertEqual(_resolve_field("total_mv"), ("stock_daily_basic", "total_mv"))

    def test_dollar_prefix(self):
        from qlib.contrib.data.db_provider import _resolve_field

        self.assertEqual(_resolve_field("$close"), ("stock_daily_prices", "close"))

    def test_unknown_field(self):
        from qlib.contrib.data.db_provider import _resolve_field

        self.assertIsNone(_resolve_field("nonexistent_field_xyz"))


# ===========================================================================
# Integration tests (require running databases)
# ===========================================================================

def _db_available():
    """Check if both databases are reachable."""
    try:
        from qlib.contrib.data.db_provider import _get_pg_engine, _get_ch_client

        _get_pg_engine().connect().close()
        _get_ch_client().command("SELECT 1")
        return True
    except Exception:
        return False


@unittest.skipUnless(_db_available(), "Database not available")
class TestDBDataLoaderStandalone(unittest.TestCase):
    """Test DBDataLoader standalone mode."""

    @classmethod
    def setUpClass(cls):
        from qlib.contrib.data.db_provider import DBDataLoader

        cls.loader = DBDataLoader()

    def test_trading_calendar(self):
        cal = self.loader.get_trading_calendar(start_date="2024-01-01", end_date="2024-01-31")
        self.assertIsInstance(cal, pd.DatetimeIndex)
        self.assertGreater(len(cal), 15)  # At least 15 trading days in Jan
        self.assertLessEqual(len(cal), 23)

    def test_trading_calendar_dates(self):
        cal = self.loader.get_trading_calendar(start_date="2024-01-01", end_date="2024-01-31")
        # Should not include weekends
        for dt in cal:
            self.assertLess(dt.dayofweek, 5, f"{dt} is a weekend")

    def test_stock_list(self):
        stocks = self.loader.get_stock_list()
        self.assertIsInstance(stocks, pd.DataFrame)
        self.assertIn("ts_code", stocks.columns)
        self.assertIn("name", stocks.columns)
        self.assertGreater(len(stocks), 100)

    def test_stock_list_filter_exchange(self):
        sse = self.loader.get_stock_list(exchange="SSE")
        self.assertTrue(all(sse["exchange"] == "SSE"))

    def test_daily_prices(self):
        prices = self.loader.get_daily_prices(
            instruments=["600000.SH"],
            start_date="2024-01-02",
            end_date="2024-01-31",
        )
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertIn("close", prices.columns)
        self.assertGreater(len(prices), 10)

    def test_daily_prices_qlib_format(self):
        """Accept qlib-format instrument IDs."""
        prices = self.loader.get_daily_prices(
            instruments=["SH600000"],
            start_date="2024-01-02",
            end_date="2024-01-10",
        )
        self.assertGreater(len(prices), 0)

    def test_daily_prices_multiple(self):
        prices = self.loader.get_daily_prices(
            instruments=["600000.SH", "000001.SZ"],
            start_date="2024-01-02",
            end_date="2024-01-10",
        )
        self.assertEqual(len(prices.index.get_level_values(0).unique()), 2)

    def test_daily_basic(self):
        basic = self.loader.get_daily_basic(
            instruments=["600000.SH"],
            start_date="2024-01-02",
            end_date="2024-01-10",
        )
        self.assertIsInstance(basic, pd.DataFrame)
        self.assertIn("pe", basic.columns)

    def test_factor_values(self):
        factors = self.loader.get_factor_values(
            instruments=["600000.SH"],
            start_date="2024-01-02",
            end_date="2024-01-10",
            factors=["roc_5d", "rsi_14"],
        )
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertIn("roc_5d", factors.columns)
        self.assertIn("rsi_14", factors.columns)

    def test_load_features(self):
        df = self.loader.load_features(
            instruments=["SH600000", "SZ000001"],
            fields=["open", "close", "volume"],
            start_time="2024-01-02",
            end_time="2024-01-10",
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.names, ["datetime", "instrument"])
        self.assertGreater(len(df), 0)
        for col in ["open", "close", "volume"]:
            self.assertIn(col, df.columns)

    def test_index_daily(self):
        """Test index daily data."""
        idx = self.loader.get_index_daily("000300.SH", start_date="2024-01-02", end_date="2024-01-10")
        self.assertIsInstance(idx, pd.DataFrame)
        self.assertGreater(len(idx), 0)

    def test_index_components(self):
        """Test index components."""
        components = self.loader.get_index_components("000300.SH")
        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 100)


@unittest.skipUnless(_db_available(), "Database not available")
class TestQlibIntegration(unittest.TestCase):
    """Test full qlib provider integration."""

    @classmethod
    def setUpClass(cls):
        from qlib.contrib.data.db_provider import init_qlib_with_db
        from qlib.data import D
        import qlib

        init_qlib_with_db()
        qlib.config.C["joblib_backend"] = "threading"
        cls.D = D

    def test_calendar(self):
        cal = self.D.calendar(start_time="2024-01-01", end_time="2024-03-31")
        self.assertIsInstance(cal, np.ndarray)
        self.assertGreater(len(cal), 50)

    def test_instruments_all(self):
        inst = self.D.instruments("all")
        inst_list = self.D.list_instruments(inst, start_time="2024-01-02", end_time="2024-01-10", as_list=True)
        self.assertIsInstance(inst_list, list)
        self.assertGreater(len(inst_list), 100)

    def test_instruments_csi300(self):
        """Test CSI300 instruments (queries stock_index_weight from ClickHouse)."""
        inst = self.D.instruments("csi300")
        inst_list = self.D.list_instruments(inst, start_time="2024-01-02", end_time="2024-01-10", as_list=True)
        self.assertGreater(len(inst_list), 100)

    def test_features_basic(self):
        df = self.D.features(
            ["SZ000001"],
            ["$open", "$close", "$volume"],
            start_time="2024-01-02",
            end_time="2024-01-10",
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 3)
        self.assertGreater(len(df), 0)

    def test_features_multiple_instruments(self):
        df = self.D.features(
            ["SZ000001", "SH600000"],
            ["$close"],
            start_time="2024-01-02",
            end_time="2024-01-10",
        )
        self.assertGreater(len(df), 0)
        # Should have data for both instruments
        self.assertEqual(len(df.index.get_level_values(0).unique()), 2)

    def test_features_expression(self):
        """Computed expressions should work."""
        df = self.D.features(
            ["SZ000001"],
            ["Ref($close, 1)/$close - 1"],
            start_time="2024-01-02",
            end_time="2024-01-15",
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
