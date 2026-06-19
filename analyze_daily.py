#!/usr/bin/env python
"""
对 run_daily_backtest_2024.py 同一套配置做指标分析：
  - 重训模型（与 daily 脚本完全一致），算 IC / ICIR / Rank IC / Rank ICIR
  - 跑回测，计算总收益、年化、最大回撤、夏普比率、胜率
  - 顺手给出 best iteration（看 early stopping 触发情况）

用途：弥补 run_daily_backtest_2024.py 详细分析块里 strategy 变量未定义的 bug。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import qlib
from qlib.utils import init_instance_by_config

warnings.filterwarnings("ignore")

DATA_PATH = "/data/StockData/data/QuantExperiment/data/qlib_data_latest"
MARKET = "csi300"
BENCHMARK = "SH000300"
START_MONEY = 100_000

TRAIN_START = "2020-02-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-03-20"

EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0015,
    "close_cost": 0.0025,
    "min_cost": 5,
    "impact_cost": 0.1,
}


def main():
    qlib.init(provider_uri=DATA_PATH, region="cn")

    handler_kwargs = {
        "start_time": TRAIN_START,
        "end_time": TEST_END,
        "fit_start_time": TRAIN_START,
        "fit_end_time": TRAIN_END,
        "instruments": MARKET,
    }
    dataset_cfg = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_kwargs,
            },
            "segments": {
                "train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END),
                "test":  (TEST_START, TEST_END),
            },
        },
    }
    model_cfg = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 128,
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "verbose": -1,
        },
    }

    print("="*80)
    print("Re-running with the same config as run_daily_backtest_2024.py")
    print(f"  train  : {TRAIN_START} ~ {TRAIN_END}")
    print(f"  valid  : {VALID_START} ~ {VALID_END}  (used for early stopping)")
    print(f"  test   : {TEST_START} ~ {TEST_END}")
    print("="*80)

    dataset = init_instance_by_config(dataset_cfg)
    model = init_instance_by_config(model_cfg)
    print("\n[1/3] Training...")
    model.fit(dataset)
    best_iter = getattr(model.model, "best_iteration", None)
    print(f"      best iteration: {best_iter}")

    print("\n[2/3] Computing signal metrics...")
    pred = model.predict(dataset)
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    print(f"      n_obs (instrument-day pairs) = {len(merged)}")
    ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"]))
    rank_ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    print(f"      IC          mean={ic.mean():+.4f}  std={ic.std():+.4f}  ICIR={ic.mean()/ic.std():+.4f}")
    print(f"      Rank IC     mean={rank_ic.mean():+.4f}  std={rank_ic.std():+.4f}  RankICIR={rank_ic.mean()/rank_ic.std():+.4f}")

    print("\n[3/3] Running backtest...")
    from qlib.backtest import backtest

    strategy_cfg = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": 30, "n_drop": 3},
    }
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }

    portfolio_result, _ = backtest(
        start_time=TEST_START,
        end_time=TEST_END,
        strategy=strategy_cfg,
        executor=executor_cfg,
        account=START_MONEY,
        benchmark=BENCHMARK,
        exchange_kwargs=EXCHANGE_KWARGS,
    )

    # qlib backtest returns dict { '1day': (DataFrame, ...) }
    df = None
    for key, val in portfolio_result.items():
        candidate = val[0] if isinstance(val, tuple) else val
        if isinstance(candidate, pd.DataFrame) and "return" in candidate.columns:
            df = candidate
            break
    if df is None:
        print("!!! could not locate portfolio dataframe in backtest output")
        return

    returns = df["return"].astype(float)
    bench   = df.get("bench", pd.Series(0, index=returns.index)).astype(float)
    excess  = returns - bench
    n = len(returns)

    total_return = float((1 + returns).prod() - 1)
    annual = float((1 + total_return) ** (252 / n) - 1) if n else float("nan")
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    mdd = float(drawdown.min())
    vol = float(returns.std() * np.sqrt(252))
    sharpe = annual / vol if vol else float("nan")
    win_rate = float((returns > 0).mean())

    bench_total = float((1 + bench).prod() - 1)
    bench_annual = float((1 + bench_total) ** (252 / n) - 1) if n else float("nan")

    excess_total  = float((1 + excess).prod() - 1)
    excess_annual = float((1 + excess_total) ** (252 / n) - 1) if n else float("nan")
    excess_vol    = float(excess.std() * np.sqrt(252))
    excess_ir     = excess_annual / excess_vol if excess_vol else float("nan")
    excess_cum    = (1 + excess).cumprod()
    excess_drawdown = (excess_cum - excess_cum.cummax()) / excess_cum.cummax()
    excess_mdd    = float(excess_drawdown.min())

    print()
    print("="*80)
    print("回测结果摘要")
    print("="*80)
    print(f"  起步资金       : {START_MONEY:>12,.0f}")
    print(f"  最终资金       : {START_MONEY*(1+total_return):>12,.0f}")
    print(f"  交易天数       : {n}")
    print()
    print("  策略 (绝对收益)")
    print(f"    总收益率     : {total_return:>+8.2%}")
    print(f"    年化收益率   : {annual:>+8.2%}")
    print(f"    最大回撤     : {mdd:>+8.2%}")
    print(f"    年化波动率   : {vol:>+8.2%}")
    print(f"    夏普比率     : {sharpe:>+8.4f}")
    print(f"    胜率         : {win_rate:>+8.2%}")
    print()
    print("  基准 CSI300")
    print(f"    总收益率     : {bench_total:>+8.2%}")
    print(f"    年化收益率   : {bench_annual:>+8.2%}")
    print()
    print("  超额收益 (策略 - 基准)")
    print(f"    总超额收益   : {excess_total:>+8.2%}")
    print(f"    年化超额     : {excess_annual:>+8.2%}")
    print(f"    超额最大回撤 : {excess_mdd:>+8.2%}")
    print(f"    年化波动率   : {excess_vol:>+8.2%}")
    print(f"    信息比率(IR) : {excess_ir:>+8.4f}")
    print()
    print("  信号质量")
    print(f"    IC           : {float(ic.mean()):>+8.4f}")
    print(f"    ICIR         : {float(ic.mean()/ic.std()):>+8.4f}")
    print(f"    Rank IC      : {float(rank_ic.mean()):>+8.4f}")
    print(f"    Rank ICIR    : {float(rank_ic.mean()/rank_ic.std()):>+8.4f}")
    print(f"    Best iter    : {best_iter}")

    # save daily series
    out = pd.DataFrame({
        "account": START_MONEY * cum,
        "return": returns,
        "bench": bench,
        "excess": excess,
        "drawdown": drawdown,
    })
    out.to_csv("daily_metrics.csv", encoding="utf-8-sig")
    print(f"\n每日序列已保存：daily_metrics.csv")


if __name__ == "__main__":
    main()
