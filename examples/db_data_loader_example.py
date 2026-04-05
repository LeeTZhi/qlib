"""
Example: Using qlib with PostgreSQL + ClickHouse data sources.

This script demonstrates two usage modes:

1. **Quick mode** — Use `init_qlib_with_db()` to replace all qlib file-based
   providers with database-backed providers.  After that, every call to
   `D.features()`, `D.calendar()`, `D.instruments()` will query the databases.

2. **Standalone mode** — Use `DBDataLoader` directly to fetch DataFrames
   without initialising the full qlib provider stack.  Great for ad-hoc
   analysis, factor research, or feeding into your own ML pipeline.

Environment variables (set these or create a .env)::

    # PostgreSQL
    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=stock_meta
    PGUSER=stock_user
    PGPASSWORD=your_password  (or PG_PASSWORD)

    # ClickHouse
    CHHOST=localhost
    CHPORT_HTTP=8123
    CHDATABASE=stock
    CHUSER=default
    CHPASSWORD=your_password  (or CH_PASSWORD)
"""

import os
import sys

# ---------------------------------------------------------------------------
# Load .env if python-dotenv is available
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv("/data/StockData/code/stock_strategy_platform/.env")
except ImportError:
    pass

# Ensure qlib is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Mode 1: Full qlib integration
# ===========================================================================
def demo_full_qlib():
    """Use database-backed providers with the full qlib framework."""
    from qlib.contrib.data.db_provider import init_qlib_with_db
    from qlib.data import D
    import qlib

    # One-line initialisation — replaces all file-based providers
    init_qlib_with_db()

    # Use threading backend (ClickHouse client is thread-local)
    qlib.config.C["joblib_backend"] = "threading"

    # Trading calendar
    calendar = D.calendar(start_time="2024-01-01", end_time="2024-03-31")
    print(f"Trading days in 2024 Q1: {len(calendar)}")
    print(f"First 5 days: {calendar[:5]}")

    # Instrument list
    instruments = D.instruments("all")
    inst_list = D.list_instruments(
        instruments, start_time="2024-01-02", end_time="2024-01-10", as_list=True
    )
    print(f"\nTotal instruments: {len(inst_list)} (first 5: {inst_list[:5]})")

    # Features — qlib expression engine works as usual
    df = D.features(
        inst_list[:3],
        ["$open", "$close", "$volume"],
        start_time="2024-01-02",
        end_time="2024-01-10",
    )
    print(f"\nFeature data shape: {df.shape}")
    print(df)

    # Computed expressions also work
    df_ret = D.features(
        inst_list[:3],
        ["Ref($close, 1)/$close - 1"],  # daily return
        start_time="2024-01-02",
        end_time="2024-01-15",
    )
    print(f"\nDaily returns:\n{df_ret}")


# ===========================================================================
# Mode 2: Standalone DBDataLoader
# ===========================================================================
def demo_standalone():
    """Use DBDataLoader directly without the qlib provider stack."""
    from qlib.contrib.data.db_provider import DBDataLoader

    loader = DBDataLoader()

    # --- Trading calendar ---
    cal = loader.get_trading_calendar(start_date="2024-01-01", end_date="2024-03-31")
    print(f"Trading days in 2024 Q1: {len(cal)}")

    # --- Stock list ---
    stocks = loader.get_stock_list()
    print(f"\nTotal stocks: {len(stocks)}")
    print(stocks[["ts_code", "name", "industry"]].head())

    # --- Daily prices ---
    prices = loader.get_daily_prices(
        instruments=["600000.SH", "000001.SZ"],
        start_date="2024-01-02",
        end_date="2024-01-15",
    )
    print(f"\nDaily prices shape: {prices.shape}")
    print(prices)

    # --- Factor values ---
    factors = loader.get_factor_values(
        instruments=["600000.SH"],
        start_date="2024-01-02",
        end_date="2024-01-10",
        factors=["roc_5d", "rsi_14", "macd_dif", "ret_1d"],
    )
    print(f"\nFactor values:\n{factors}")

    # --- Bulk load for ML pipeline ---
    df = loader.load_features(
        instruments=["SH600000", "SZ000001"],
        fields=["open", "close", "high", "low", "volume"],
        start_time="2024-01-02",
        end_time="2024-01-15",
    )
    print(f"\nBulk features shape: {df.shape}")
    print(df)


if __name__ == "__main__":
    print("=" * 60)
    print("Mode 2: Standalone DBDataLoader")
    print("=" * 60)
    demo_standalone()

    print("\n" + "=" * 60)
    print("Mode 1: Full qlib integration (uncomment to run)")
    print("=" * 60)
    # demo_full_qlib()
