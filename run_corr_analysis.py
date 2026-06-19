#!/usr/bin/env python
"""
A 实验：finance 因子 vs v1 (估值+流动性+资金流) 因子的横截面相关性矩阵

每个交易日内对所有 csi300 成分股计算两两 Pearson + Spearman 相关，再对全部
交易日做平均。同方向高相关（>0.5）说明因子冗余 / 重复编码同一信号。

输出：
  alpha158plus_v3_corr_full.csv  - 完整 22x22 相关矩阵
  alpha158plus_v3_corr_focus.csv - finance(7) x v1(15) 跨族矩阵
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha158_plus import Alpha158Plus, ALL_GROUPS, get_extra_features

warnings.filterwarnings("ignore")

MARKET = "csi300"
TRAIN_START = "2020-02-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-03-20"


def _load_db_env():
    env_path = "/data/StockData/code/stock_strategy_platform/.env"
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
    os.environ.setdefault("PGHOST", "localhost")
    os.environ.setdefault("CHHOST", "localhost")


def _xs_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """横截面相关：对每个 datetime 内的多只股票做相关，再对所有日期取均值."""
    daily = df.groupby(level="datetime").corr(method=method)
    # daily 是 (datetime, factor) x factor 多重索引，对 datetime 取均值
    return daily.groupby(level=1).mean()


def main() -> int:
    _load_db_env()
    from qlib.contrib.data.db_provider import init_qlib_with_db
    init_qlib_with_db(provider_uri="", region="cn")
    from qlib.data.dataset import DatasetH

    print("=" * 80)
    print("A 实验：finance vs v1 因子横截面相关")
    print("=" * 80)

    handler = Alpha158Plus(
        instruments=MARKET,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        factor_groups=ALL_GROUPS,
        # 用一个不重要的 label，反正这里不训模型
        label=(["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL"]),
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START, TEST_END),
        },
    )

    print("  loading test data...")
    df = dataset.prepare("test", col_set="feature")
    print(f"  shape: {df.shape}")

    # 我们只关心 22 个新因子（不需要 Alpha158 base 的 158 个）
    extra_fields, extra_names = get_extra_features(ALL_GROUPS)
    available = [n for n in extra_names if n in df.columns]
    missing = set(extra_names) - set(available)
    if missing:
        print(f"  ⚠ missing columns in dataset: {missing}")
    sub = df[available].dropna(how="all")
    print(f"  computing correlations on {len(available)} factors x {len(sub)} rows...")

    pearson = _xs_corr(sub, method="pearson")
    spearman = _xs_corr(sub, method="spearman")

    # 完整 22x22
    pearson.to_csv("alpha158plus_v3_corr_full.csv", encoding="utf-8-sig")
    print(f"\n[Pearson 横截面相关 — 完整 {len(available)}x{len(available)}]")
    print("(精确到 2 位)")
    print(pearson.round(2).to_string(float_format=lambda x: f"{x:+.2f}"))

    # 跨族 focus：finance(7) x v1(15)
    finance_names = [n for n in available if n in {"ROE", "NPM", "DEBT_TO_ASSETS",
                                                    "ASSETS_TURN", "LOG_TOTAL_ASSETS",
                                                    "ROE_DELTA_1Y", "NPM_DELTA_1Y"}]
    v1_names = [n for n in available if n not in finance_names]
    if finance_names and v1_names:
        focus_p = pearson.loc[finance_names, v1_names]
        focus_s = spearman.loc[finance_names, v1_names]
        focus_p.to_csv("alpha158plus_v3_corr_focus.csv", encoding="utf-8-sig")
        print(f"\n[Pearson 跨族相关 — finance ({len(finance_names)}) x v1 ({len(v1_names)})]")
        print(focus_p.round(2).to_string(float_format=lambda x: f"{x:+.2f}"))
        print(f"\n[Spearman 跨族相关 — finance x v1]")
        print(focus_s.round(2).to_string(float_format=lambda x: f"{x:+.2f}"))

        # 找 |corr| > 0.4 的 pair
        high = []
        for f in finance_names:
            for v in v1_names:
                p = focus_p.loc[f, v]
                s = focus_s.loc[f, v]
                if abs(p) > 0.4 or abs(s) > 0.4:
                    high.append((f, v, p, s))
        if high:
            print("\n[|corr| > 0.4 的因子对 — 强冗余候选]")
            for f, v, p, s in sorted(high, key=lambda x: -abs(x[2])):
                print(f"  {f:<18} ↔ {v:<18}  Pearson={p:+.2f}  Spearman={s:+.2f}")
        else:
            print("\n  (no |corr| > 0.4 — finance 与 v1 在横截面上基本独立)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
