#!/usr/bin/env python
"""
Tushare 互补因子 PoC：消融实验

跑 5 个配置，对比每个因子族的边际 IC 贡献：
  • baseline   — Alpha158 only
  • +valuation — Alpha158 + 估值因子族 (7)
  • +liquidity — Alpha158 + 流动性因子族 (4)
  • +moneyflow — Alpha158 + 资金流因子族 (4)
  • +all       — Alpha158 + 三族全开 (15)

口径与 analyze_daily.py 完全一致：
  train 2020-02-01 ~ 2022-12-31, valid 2023, test 2024-01-01 ~ 2026-03-20
  csi300 + LightGBM(num_boost_round=1000, early_stopping_rounds=50)

数据后端：DB Provider (PG + ClickHouse)，必要环境变量从 stock_strategy_platform/.env
读入。

输出：alpha158plus_poc_results.csv + stdout 表格
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Allow `import alpha158_plus` from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha158_plus import Alpha158Plus, ALL_GROUPS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MARKET = "csi300"
TRAIN_START = "2020-02-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-03-20"

# 训练 LightGBM
LGB_KWARGS = {
    "loss": "mse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 128,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

# 5 个消融配置
ABLATIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ("baseline",   ()),
    ("+valuation", ("valuation",)),
    ("+liquidity", ("liquidity",)),
    ("+moneyflow", ("moneyflow",)),
    ("+all",       ALL_GROUPS),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_db_env():
    """Load DB credentials from .env (best-effort)."""
    env_path = "/data/StockData/code/stock_strategy_platform/.env"
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
    # Provider expects host on localhost when running outside docker
    os.environ.setdefault("PGHOST", "localhost")
    os.environ.setdefault("CHHOST", "localhost")


def _init_qlib():
    from qlib.contrib.data.db_provider import init_qlib_with_db
    init_qlib_with_db(provider_uri="", region="cn")


def _build_handler(groups: Iterable[str]):
    return Alpha158Plus(
        instruments=MARKET,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        factor_groups=groups,
    )


def _train_and_score(name: str, groups: Iterable[str]) -> dict:
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH

    print(f"\n>>> {name}  (groups={tuple(groups) or 'none'})")
    handler = _build_handler(groups)
    n_features = len(handler.get_feature_config()[0])
    print(f"    feature count: {n_features}")

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START, TEST_END),
        },
    )

    model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": dict(LGB_KWARGS),
    })
    print("    [training...]")
    model.fit(dataset)
    best_iter = getattr(model.model, "best_iteration", None)
    print(f"    best_iter={best_iter}")

    pred = model.predict(dataset)
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    if merged.empty:
        return dict(name=name, n_feat=n_features, n_obs=0, ic=np.nan, ic_ir=np.nan,
                    rank_ic=np.nan, rank_ic_ir=np.nan, best_iter=best_iter)

    ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"]))
    rank_ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    out = dict(
        name=name,
        n_feat=n_features,
        n_obs=int(len(merged)),
        ic=float(ic.mean()),
        ic_ir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
        rank_ic=float(rank_ic.mean()),
        rank_ic_ir=float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() else np.nan,
        best_iter=best_iter,
    )
    print(
        f"    IC={out['ic']:+.4f}  IC_IR={out['ic_ir']:+.4f}  "
        f"RankIC={out['rank_ic']:+.4f}  RankICIR={out['rank_ic_ir']:+.4f}"
    )
    return out


def main() -> int:
    _load_db_env()
    _init_qlib()
    print("=" * 80)
    print("Alpha158Plus PoC — Tushare factor ablation")
    print("=" * 80)
    print(f"  train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  valid: {VALID_START} ~ {VALID_END}")
    print(f"  test : {TEST_START} ~ {TEST_END}")
    print(f"  market: {MARKET}")
    print("=" * 80)

    rows = []
    for name, groups in ABLATIONS:
        try:
            rows.append(_train_and_score(name, groups))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append(dict(name=name, error=str(exc)[:200]))

    df = pd.DataFrame(rows)
    df.to_csv("alpha158plus_poc_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Ablation summary  (vs baseline IC)")
    print("=" * 80)
    cols = [c for c in ["name", "n_feat", "best_iter", "n_obs",
                         "ic", "ic_ir", "rank_ic", "rank_ic_ir"] if c in df.columns]
    fmt = lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)
    print(df[cols].to_string(index=False, float_format=lambda x: fmt(x) if pd.notna(x) else "NaN"))

    if "ic" in df.columns and len(df) > 1 and pd.notna(df["ic"].iloc[0]):
        baseline_ic = df["ic"].iloc[0]
        baseline_ric = df["rank_ic"].iloc[0] if "rank_ic" in df.columns else None
        print("\nDelta vs baseline:")
        for _, r in df.iloc[1:].iterrows():
            d_ic = r["ic"] - baseline_ic if pd.notna(r.get("ic")) else float("nan")
            d_ric = r["rank_ic"] - baseline_ric if pd.notna(r.get("rank_ic")) else float("nan")
            print(f"  {r['name']:<12} ΔIC = {d_ic:+.4f}   ΔRankIC = {d_ric:+.4f}")

    print(f"\n结果已保存: alpha158plus_poc_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
