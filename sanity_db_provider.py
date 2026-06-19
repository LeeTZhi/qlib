"""
Sanity check for DB Provider fixes:
  (1) backward-adjusted prices: $close * adj_factor when QLIB_DB_ADJUST_PRICES=1
  (2) PIT index membership: csi300 universe shrinks to ~300 names per snapshot date
"""
import os
import sys

sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from qlib.contrib.data.db_provider import (
    DBInstrumentProvider,
    DBDataLoader,
    qlib_to_ts_code,
)


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ------------------------------------------------------------------
# (1) PIT membership — csi300
# ------------------------------------------------------------------
banner("(1) PIT csi300 membership: stocks active at 2024-01-01 vs 2018-01-01")

prov = DBInstrumentProvider()
inst_dict = prov._load_instruments("csi300")
print(f"total stocks ever in csi300 (with PIT intervals): {len(inst_dict)}")

# Check membership at two snapshot dates
snap_dates = [pd.Timestamp("2018-01-15"), pd.Timestamp("2024-01-15")]
for d in snap_dates:
    cnt = sum(1 for spans in inst_dict.values() if any(s <= d <= e for s, e in spans))
    print(f"  active members at {d.date()}: {cnt}")

# Show first few intervals for SH600519 (Maotai, in csi300 long-time) and SH600999 (招商证券)
for code in ("SH600519", "SH600999", "SH601728"):
    spans = inst_dict.get(code)
    if spans:
        head = ", ".join(f"({s.date()}, {e.date()})" for s, e in spans[:3])
        print(f"  {code}: {len(spans)} interval(s), e.g. {head}{'...' if len(spans) > 3 else ''}")
    else:
        print(f"  {code}: not found in PIT membership")


# ------------------------------------------------------------------
# (2) Backward-adjusted price
# ------------------------------------------------------------------
banner("(2) Adjusted vs raw close — SH600519 (贵州茅台, has dividends)")

loader = DBDataLoader()

# Toggle ON
os.environ["QLIB_DB_ADJUST_PRICES"] = "1"
df_adj = loader.load_features(
    instruments=["SH600519"],
    fields=["open", "close", "volume"],
    start_time="2024-01-02",
    end_time="2024-01-15",
)
print(f"\n[adjusted] head:\n{df_adj.head()}")

# Toggle OFF
os.environ["QLIB_DB_ADJUST_PRICES"] = "0"
df_raw = loader.load_features(
    instruments=["SH600519"],
    fields=["open", "close", "volume"],
    start_time="2024-01-02",
    end_time="2024-01-15",
)
print(f"\n[raw]      head:\n{df_raw.head()}")

if not df_adj.empty and not df_raw.empty:
    common_idx = df_adj.index.intersection(df_raw.index)
    if len(common_idx) > 0:
        first_idx = common_idx[0]
        v_adj = df_adj.loc[first_idx, "close"]
        v_raw = df_raw.loc[first_idx, "close"]
        # In case of any residual duplicate rows, take the first scalar.
        sample_adj = float(v_adj.iloc[0]) if hasattr(v_adj, "iloc") else float(v_adj)
        sample_raw = float(v_raw.iloc[0]) if hasattr(v_raw, "iloc") else float(v_raw)
        ratio = sample_adj / sample_raw if sample_raw else float("nan")
        print(
            f"\n  first row ratio (adj_close / raw_close) = {sample_adj:.4f} / {sample_raw:.4f} = {ratio:.4f}"
        )
        if ratio > 1.0001:
            print("  ✓ adjusted price > raw price — adj_factor properly applied")
        elif abs(ratio - 1.0) < 1e-6:
            print(
                "  ⚠ ratio == 1.0 — adj_factor must be 1 for this date (no past dividends), "
                "or the JOIN missed."
            )

# Reset to default
os.environ["QLIB_DB_ADJUST_PRICES"] = "1"

print("\nSanity check complete.")
