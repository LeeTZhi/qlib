#!/usr/bin/env python
"""
最终 backtest：5d label + Alpha158Plus(+all) + 现实交易成本

跑 4 个组合做对比：
  • csi300, baseline (Alpha158, 1d label)   -- 旧基准
  • csi300, full (Alpha158Plus +all, 5d label)
  • csi500, baseline (Alpha158, 1d label)
  • csi500, full (Alpha158Plus +all, 5d label)

每个组合都跑：训练 + 预测 + 含成本 backtest，输出 IC / 年化 / Sharpe / IR / MDD.

数据后端：DB Provider (PG + ClickHouse) -- PIT 成分股 + 后复权.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha158_plus import Alpha158Plus, ALL_GROUPS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TRAIN_START = "2020-02-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-03-20"

START_MONEY = 100_000

# 现实成本（与 run_daily_backtest_2024 一致）
EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0015,
    "close_cost": 0.0025,
    "min_cost": 5,
    "impact_cost": 0.1,
}

LGB_KWARGS = {
    "loss": "mse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 128,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

# 4 组实验：(experiment_name, market, benchmark, horizon, factor_groups, topk, n_drop)
# 注意 n_drop 与 horizon 对齐：1d 信号每天都更新，n_drop 大；5d 信号变化慢，n_drop 小避免无效换仓.
EXPERIMENTS = [
    ("csi300_baseline_1d_d3", "csi300", "SH000300", "1d", (),         30, 3),
    ("csi300_full_5d_d1",     "csi300", "SH000300", "5d", ALL_GROUPS, 30, 1),
    ("csi500_baseline_1d_d5", "csi500", "SH000905", "1d", (),         50, 5),
    ("csi500_full_5d_d1",     "csi500", "SH000905", "5d", ALL_GROUPS, 50, 1),
]

LABEL_EXPRS = {
    "1d":  ("Ref($close, -2)/Ref($close, -1) - 1",  "LABEL_1D"),
    "5d":  ("Ref($close, -6)/Ref($close, -1) - 1",  "LABEL_5D"),
    "10d": ("Ref($close, -11)/Ref($close, -1) - 1", "LABEL_10D"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _init_qlib():
    from qlib.contrib.data.db_provider import init_qlib_with_db
    init_qlib_with_db(provider_uri="", region="cn")


def _xs_ic(merged: pd.DataFrame, score_col="score", label_col="label", method="pearson"):
    daily = merged.groupby(level="datetime").apply(
        lambda x: x[score_col].corr(x[label_col], method=method)
    ).dropna()
    if daily.empty:
        return float("nan"), float("nan")
    mu = float(daily.mean())
    sd = float(daily.std())
    return mu, mu / sd if sd else float("nan")


def run_one(name: str, market: str, benchmark: str, horizon: str,
            factor_groups: Iterable[str], topk: int, n_drop: int) -> dict:
    """跑一组实验，返回指标 dict."""
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH
    from qlib.backtest import backtest as qlib_backtest

    label_expr, label_name = LABEL_EXPRS[horizon]
    print("\n" + "=" * 80)
    print(f"  {name}")
    print("=" * 80)
    print(f"  market   : {market}")
    print(f"  benchmark: {benchmark}")
    print(f"  horizon  : {horizon}  (label_name={label_name})")
    print(f"  groups   : {tuple(factor_groups) or 'baseline (Alpha158 only)'}")
    print(f"  TopK={topk}  n_drop={n_drop}")

    # ---- handler ----
    handler = Alpha158Plus(
        instruments=market,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        factor_groups=factor_groups,
        label=([label_expr], [label_name]),
    )
    n_features = len(handler.get_feature_config()[0])
    print(f"  features : {n_features}")

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START, TEST_END),
        },
    )

    # ---- model ----
    print("  [training...]")
    model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": dict(LGB_KWARGS),
    })
    model.fit(dataset)
    best_iter = getattr(model.model, "best_iteration", None)
    print(f"    best_iter = {best_iter}")

    # ---- signal IC ----
    pred = model.predict(dataset)
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    ic, ic_ir = _xs_ic(merged, "score", "label", "pearson")
    rank_ic, rank_ic_ir = _xs_ic(merged, "score", "label", "spearman")
    n_obs = int(len(merged))
    print(f"    IC = {ic:+.4f}   ICIR = {ic_ir:+.4f}")
    print(f"    Rank IC = {rank_ic:+.4f}   Rank ICIR = {rank_ic_ir:+.4f}")
    print(f"    n_obs = {n_obs}")

    # ---- backtest ----
    print("  [backtest...]")
    strategy_cfg = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": topk, "n_drop": n_drop},
    }
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    # Restrict the Exchange's universe to the actual market — without this, qlib
    # falls back to ``D.instruments()`` (= all 5000+ listed stocks) and triggers
    # thousands of per-instrument ClickHouse queries.
    exchange_kwargs = dict(EXCHANGE_KWARGS, codes=market)
    portfolio_result, _ = qlib_backtest(
        start_time=TEST_START,
        end_time=TEST_END,
        strategy=strategy_cfg,
        executor=executor_cfg,
        account=START_MONEY,
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )

    df = None
    if isinstance(portfolio_result, dict):
        for key, val in portfolio_result.items():
            cand = val[0] if isinstance(val, tuple) else val
            if isinstance(cand, pd.DataFrame) and "return" in cand.columns:
                df = cand
                break

    if df is None or df.empty:
        print("    !!! no portfolio dataframe")
        return {
            "name": name, "market": market, "horizon": horizon,
            "n_features": n_features, "best_iter": best_iter,
            "ic": ic, "ic_ir": ic_ir, "rank_ic": rank_ic, "rank_ic_ir": rank_ic_ir,
            "n_obs": n_obs,
            "annual_return": np.nan, "max_drawdown": np.nan, "sharpe": np.nan,
            "excess_annual": np.nan, "excess_ir": np.nan, "excess_mdd": np.nan,
            "total_return": np.nan, "n_days": 0, "win_rate": np.nan,
        }

    returns = df["return"].astype(float)
    bench   = df.get("bench", pd.Series(0.0, index=returns.index)).astype(float)
    excess  = returns - bench
    n = len(returns)

    total_return = float((1 + returns).prod() - 1)
    annual = float((1 + total_return) ** (252 / n) - 1) if n else np.nan
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    mdd = float(drawdown.min())
    vol = float(returns.std() * np.sqrt(252))
    sharpe = annual / vol if vol else np.nan
    win_rate = float((returns > 0).mean())

    bench_total = float((1 + bench).prod() - 1)
    bench_annual = float((1 + bench_total) ** (252 / n) - 1) if n else np.nan

    excess_total  = float((1 + excess).prod() - 1)
    excess_annual = float((1 + excess_total) ** (252 / n) - 1) if n else np.nan
    excess_vol    = float(excess.std() * np.sqrt(252))
    excess_ir     = excess_annual / excess_vol if excess_vol else np.nan
    excess_cum    = (1 + excess).cumprod()
    excess_dd     = (excess_cum - excess_cum.cummax()) / excess_cum.cummax()
    excess_mdd    = float(excess_dd.min())

    print(f"  >> 策略  : ann={annual:+.2%}  Sharpe={sharpe:+.2f}  MDD={mdd:+.2%}")
    print(f"  >> 基准  : ann={bench_annual:+.2%}  ({benchmark})")
    print(f"  >> 超额  : ann={excess_annual:+.2%}  IR={excess_ir:+.2f}  MDD={excess_mdd:+.2%}")
    print(f"  >> 期初/末: {START_MONEY:,.0f} → {START_MONEY*(1+total_return):,.0f}  ({total_return:+.2%})")

    return {
        "name": name, "market": market, "horizon": horizon,
        "n_features": n_features, "best_iter": best_iter,
        "n_obs": n_obs, "n_days": n,
        "ic": ic, "ic_ir": ic_ir, "rank_ic": rank_ic, "rank_ic_ir": rank_ic_ir,
        "total_return": total_return, "annual_return": annual,
        "max_drawdown": mdd, "sharpe": sharpe, "win_rate": win_rate,
        "bench_annual": bench_annual,
        "excess_total": excess_total, "excess_annual": excess_annual,
        "excess_ir": excess_ir, "excess_mdd": excess_mdd,
    }


def main() -> int:
    _load_db_env()
    _init_qlib()
    print("=" * 80)
    print("最终 backtest：5d label + Alpha158Plus(+all) vs baseline，csi300 & csi500")
    print("=" * 80)
    print(f"  train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  valid: {VALID_START} ~ {VALID_END}")
    print(f"  test : {TEST_START} ~ {TEST_END}")
    print(f"  costs: open={EXCHANGE_KWARGS['open_cost']:.2%}  close={EXCHANGE_KWARGS['close_cost']:.2%}  impact={EXCHANGE_KWARGS['impact_cost']}")
    print("=" * 80)

    rows = []
    for cfg in EXPERIMENTS:
        try:
            rows.append(run_one(*cfg))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append({"name": cfg[0], "error": str(exc)[:200]})

    df = pd.DataFrame(rows)
    df.to_csv("alpha158plus_final_backtest.csv", index=False, encoding="utf-8-sig")

    print("\n\n" + "=" * 80)
    print("汇总表")
    print("=" * 80)
    show_cols = [
        "name", "market", "horizon", "n_features", "best_iter",
        "ic", "rank_ic", "annual_return", "sharpe", "max_drawdown",
        "excess_annual", "excess_ir", "excess_mdd",
    ]
    have = [c for c in show_cols if c in df.columns]
    fmt = lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)
    print(df[have].to_string(index=False, float_format=lambda x: fmt(x) if pd.notna(x) else "NaN"))
    print("\n结果文件: alpha158plus_final_backtest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
