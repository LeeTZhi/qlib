#!/usr/bin/env python
"""
V3-lite 消融：去除冗余的 finance 因子（LOG_TOTAL_ASSETS / DEBT_TO_ASSETS）后重测

对比 4 组配置：
  • baseline           — Alpha158 (158 features)
  • +v1                — Alpha158 + valuation + liquidity + moneyflow (173 features)
  • +finance_lite      — Alpha158 + finance_lite (163 features, 5 个独立财务因子)
  • +v2_lite           — Alpha158 + v1 + finance_lite (178 features)

V3 (full finance) 的对照已经存于 alpha158plus_v3_results.csv：
  baseline   IC=+0.0145  excess=-7.15%
  +v1        IC=+0.0220  excess=+6.88%
  +finance   IC=+0.0237  excess=-5.54%
  +v2        IC=+0.0225  excess=+1.81%

如果"redundancy 假设"成立，预期 v3_lite 跑出来：
  +finance_lite  IC ≥ V3 +finance（去冗余不丢信号）
  +v2_lite       excess > V3 +v2 (+1.81%)，**理想情况 ≥ +v1 (+6.88%)**
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha158_plus import Alpha158Plus

warnings.filterwarnings("ignore")

# ---- 配置（与 v3 完全一致，仅 ABLATIONS 不同） ----
MARKET = "csi300"
BENCHMARK = "SH000300"
TRAIN_START = "2020-02-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-03-20"
START_MONEY = 100_000
TOPK, N_DROP = 30, 1

LABEL_EXPR = "Ref($close, -6)/Ref($close, -1) - 1"
LABEL_NAME = "LABEL_5D"

LGB_KWARGS = {
    "loss": "mse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 128,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0015,
    "close_cost": 0.0025,
    "min_cost": 5,
    "impact_cost": 0.1,
}

ABLATIONS = [
    ("baseline",      ()),
    ("+v1",           ("valuation", "liquidity", "moneyflow")),
    ("+finance_lite", ("finance_lite",)),
    ("+v2_lite",      ("valuation", "liquidity", "moneyflow", "finance_lite")),
]


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


def _xs_ic(merged: pd.DataFrame, score_col="score", label_col="label", method="pearson"):
    daily = merged.groupby(level="datetime").apply(
        lambda x: x[score_col].corr(x[label_col], method=method)
    ).dropna()
    if daily.empty:
        return np.nan, np.nan
    mu = float(daily.mean())
    sd = float(daily.std())
    return mu, mu / sd if sd else np.nan


def run_one(name: str, groups) -> dict:
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH
    from qlib.backtest import backtest as qlib_backtest

    print("\n" + "=" * 80)
    print(f"  {name}  groups={tuple(groups) or 'baseline'}")
    print("=" * 80)

    handler = Alpha158Plus(
        instruments=MARKET,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        factor_groups=groups,
        label=([LABEL_EXPR], [LABEL_NAME]),
    )
    n_features = len(handler.get_feature_config()[0])
    print(f"  features: {n_features}")

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START, TEST_END),
        },
    )

    print("  [training...]")
    model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": dict(LGB_KWARGS),
    })
    model.fit(dataset)
    best_iter = getattr(model.model, "best_iteration", None)
    print(f"    best_iter = {best_iter}")

    pred = model.predict(dataset)
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    ic, ic_ir = _xs_ic(merged, "score", "label", "pearson")
    rank_ic, rank_ic_ir = _xs_ic(merged, "score", "label", "spearman")
    print(f"    IC={ic:+.4f} IC_IR={ic_ir:+.4f}")
    print(f"    Rank IC={rank_ic:+.4f} Rank ICIR={rank_ic_ir:+.4f}")

    print("  [backtest...]")
    strategy_cfg = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": TOPK, "n_drop": N_DROP},
    }
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    portfolio_result, _ = qlib_backtest(
        start_time=TEST_START,
        end_time=TEST_END,
        strategy=strategy_cfg,
        executor=executor_cfg,
        account=START_MONEY,
        benchmark=BENCHMARK,
        exchange_kwargs=dict(EXCHANGE_KWARGS, codes=MARKET),
    )

    df = None
    if isinstance(portfolio_result, dict):
        for v in portfolio_result.values():
            cand = v[0] if isinstance(v, tuple) else v
            if isinstance(cand, pd.DataFrame) and "return" in cand.columns:
                df = cand
                break
    if df is None or df.empty:
        return {"name": name, "n_features": n_features, "best_iter": best_iter,
                "ic": ic, "rank_ic": rank_ic,
                "annual_return": np.nan, "sharpe": np.nan,
                "excess_annual": np.nan, "excess_ir": np.nan}

    returns = df["return"].astype(float)
    bench = df.get("bench", pd.Series(0.0, index=returns.index)).astype(float)
    excess = returns - bench
    n = len(returns)

    total = float((1 + returns).prod() - 1)
    annual = float((1 + total) ** (252 / n) - 1) if n else np.nan
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    mdd = float(drawdown.min())
    vol = float(returns.std() * np.sqrt(252))
    sharpe = annual / vol if vol else np.nan

    bench_total = float((1 + bench).prod() - 1)
    bench_annual = float((1 + bench_total) ** (252 / n) - 1) if n else np.nan
    excess_total = float((1 + excess).prod() - 1)
    excess_annual = float((1 + excess_total) ** (252 / n) - 1) if n else np.nan
    excess_vol = float(excess.std() * np.sqrt(252))
    excess_ir = excess_annual / excess_vol if excess_vol else np.nan

    print(f"  >> 策略 ann={annual:+.2%} Sharpe={sharpe:+.2f} MDD={mdd:+.2%}")
    print(f"  >> 基准 ann={bench_annual:+.2%}")
    print(f"  >> 超额 ann={excess_annual:+.2%} IR={excess_ir:+.2f}")

    return {
        "name": name, "n_features": n_features, "best_iter": best_iter,
        "ic": ic, "ic_ir": ic_ir, "rank_ic": rank_ic, "rank_ic_ir": rank_ic_ir,
        "annual_return": annual, "sharpe": sharpe, "max_drawdown": mdd,
        "bench_annual": bench_annual, "excess_annual": excess_annual,
        "excess_ir": excess_ir, "n_days": n,
    }


def main() -> int:
    _load_db_env()
    from qlib.contrib.data.db_provider import init_qlib_with_db
    init_qlib_with_db(provider_uri="", region="cn")
    print("=" * 80)
    print("V3-lite ablation: drop redundant finance factors")
    print("=" * 80)
    print(f"  market: {MARKET}, label: 5d, n_drop: {N_DROP}")
    print(f"  finance_lite drops: LOG_TOTAL_ASSETS, DEBT_TO_ASSETS  (corr w/ v1 > 0.5)")
    print("=" * 80)

    rows = []
    for name, groups in ABLATIONS:
        try:
            rows.append(run_one(name, groups))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append({"name": name, "error": str(exc)[:200]})

    df = pd.DataFrame(rows)
    df.to_csv("alpha158plus_v3_lite_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("V3-lite 汇总")
    print("=" * 80)
    show = ["name", "n_features", "best_iter", "ic", "rank_ic",
            "annual_return", "sharpe", "max_drawdown",
            "excess_annual", "excess_ir"]
    have = [c for c in show if c in df.columns]
    fmt = lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)
    print(df[have].to_string(index=False, float_format=lambda x: fmt(x) if pd.notna(x) else "NaN"))

    if "ic" in df.columns and len(df) >= 1:
        baseline_ic = df.loc[df["name"] == "baseline", "ic"].iloc[0]
        baseline_ric = df.loc[df["name"] == "baseline", "rank_ic"].iloc[0]
        baseline_excess = df.loc[df["name"] == "baseline", "excess_annual"].iloc[0]
        v1_excess = df.loc[df["name"] == "+v1", "excess_annual"].iloc[0] if "+v1" in df["name"].values else np.nan
        print("\n[Δ vs baseline]")
        for _, r in df.iterrows():
            if r["name"] == "baseline":
                continue
            d_ic = r["ic"] - baseline_ic
            d_ric = r["rank_ic"] - baseline_ric
            d_ex = r.get("excess_annual", np.nan) - baseline_excess if pd.notna(r.get("excess_annual")) else np.nan
            print(f"  {r['name']:<14} ΔIC={d_ic:+.4f}  ΔRankIC={d_ric:+.4f}  Δexcess={d_ex:+.2%}")
        if pd.notna(v1_excess):
            print("\n[Δ vs +v1（看 finance 是否在 v1 之上加值）]")
            for _, r in df.iterrows():
                if r["name"] in ("baseline", "+v1"):
                    continue
                d_ex = r.get("excess_annual", np.nan) - v1_excess if pd.notna(r.get("excess_annual")) else np.nan
                print(f"  {r['name']:<14} Δexcess vs +v1 = {d_ex:+.2%}")

    print("\n结果已保存: alpha158plus_v3_lite_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
