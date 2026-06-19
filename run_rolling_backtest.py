#!/usr/bin/env python
"""
滚动多窗口回测：用同一套 Alpha158 + LightGBM 配置在多个 train/valid/test 窗口上跑，
观察 IC / 年化收益 / 回撤的窗口间稳定性。

每个窗口的划分（默认 3+1+1 年）：
    test_end_date = 每个目标"测试结束日"
        train: [test_end - 5y, test_end - 2y - 1d]    (3 年)
        valid: [test_end - 2y,  test_end - 1y - 1d]    (1 年, 用于 early stopping)
        test:  [test_end - 1y,  test_end]              (1 年)

输出 rolling_results.csv，每行一个窗口的指标。

用法:
    python run_rolling_backtest.py                           # 默认 6 个滚动窗口
    python run_rolling_backtest.py 2024 2025                 # 仅在 2024、2025 年末跑测试
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import qlib
from qlib.utils import init_instance_by_config
from qlib.contrib.evaluate import risk_analysis

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DATA_PATH = "/data/StockData/data/QuantExperiment/data/qlib_data_latest"
MARKET = "csi300"
BENCHMARK = "SH000300"
START_MONEY = 100_000

# 默认窗口锚点（每个值代表一个窗口的"test 年度"，test 跑全年）
DEFAULT_TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# 窗口长度（年）
TRAIN_YEARS = 3
VALID_YEARS = 1
TEST_YEARS = 1


# ---------------------------------------------------------------------------
# 数据 / 模型 / 策略 配置工厂
# ---------------------------------------------------------------------------

@dataclass
class Window:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

    @property
    def label(self) -> str:
        return f"{self.test_start[:7]}_to_{self.test_end[:7]}"


def make_window(test_year: int) -> Window:
    """根据 test_year (e.g. 2024)生成默认 3+1+1 年窗口。"""
    test_start = pd.Timestamp(f"{test_year - TEST_YEARS + 1}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    valid_end = test_start - pd.Timedelta(days=1)
    valid_start = valid_end - pd.DateOffset(years=VALID_YEARS) + pd.Timedelta(days=1)
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS) + pd.Timedelta(days=1)
    return Window(
        train_start=train_start.strftime("%Y-%m-%d"),
        train_end=train_end.strftime("%Y-%m-%d"),
        valid_start=valid_start.strftime("%Y-%m-%d"),
        valid_end=valid_end.strftime("%Y-%m-%d"),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=test_end.strftime("%Y-%m-%d"),
    )


def build_dataset_config(win: Window) -> dict:
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": win.train_start,
                    "end_time": win.test_end,
                    "fit_start_time": win.train_start,
                    "fit_end_time": win.train_end,
                    "instruments": MARKET,
                },
            },
            "segments": {
                "train": (win.train_start, win.train_end),
                "valid": (win.valid_start, win.valid_end),
                "test":  (win.test_start, win.test_end),
            },
        },
    }


def build_model_config() -> dict:
    return {
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


def build_strategy_config(pred: pd.Series) -> dict:
    return {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": 30, "n_drop": 3},
    }


EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0015,
    "close_cost": 0.0025,
    "min_cost": 5,
    "impact_cost": 0.1,
}


# ---------------------------------------------------------------------------
# 单窗口跑通 + 指标提取
# ---------------------------------------------------------------------------

def evaluate_pred(pred: pd.Series, dataset) -> dict:
    """计算 IC / Rank IC 等横截面指标。"""
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    if merged.empty:
        return {"ic": np.nan, "ic_ir": np.nan, "rank_ic": np.nan, "rank_ic_ir": np.nan, "n_obs": 0}
    ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"]))
    rank_ic = merged.groupby(level="datetime").apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    return {
        "ic": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() else np.nan,
        "rank_ic": float(rank_ic.mean()),
        "rank_ic_ir": float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() else np.nan,
        "n_obs": int(len(merged)),
    }


def run_backtest_metrics(pred: pd.Series, win: Window) -> dict:
    """跑 backtest 拿到组合层级指标。"""
    from qlib.backtest import backtest as qlib_backtest

    strategy_cfg = build_strategy_config(pred)
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    try:
        portfolio_result, _ = qlib_backtest(
            start_time=win.test_start,
            end_time=win.test_end,
            strategy=strategy_cfg,
            executor=executor_cfg,
            account=START_MONEY,
            benchmark=BENCHMARK,
            exchange_kwargs=EXCHANGE_KWARGS,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "total_return": np.nan,
            "n_days": 0,
            "backtest_error": str(exc)[:200],
        }

    # qlib backtest 返回一个 dict-like，结构可能为 {'1day': (DataFrame, ...)}; 取 daily 收益序列
    returns = None
    if isinstance(portfolio_result, dict):
        for key, val in portfolio_result.items():
            df = val[0] if isinstance(val, tuple) else val
            if isinstance(df, pd.DataFrame) and "return" in df.columns:
                returns = df["return"].astype(float)
                break
    elif isinstance(portfolio_result, pd.DataFrame) and "return" in portfolio_result.columns:
        returns = portfolio_result["return"].astype(float)

    if returns is None or returns.empty:
        return {
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "total_return": np.nan,
            "n_days": 0,
        }

    total_return = float((1 + returns).prod() - 1)
    n = len(returns)
    annual_return = float((1 + total_return) ** (252 / n) - 1) if n else np.nan
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    max_drawdown = float(drawdown.min())
    vol = float(returns.std() * np.sqrt(252))
    sharpe = annual_return / vol if vol else np.nan

    return {
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "total_return": total_return,
        "n_days": n,
    }


def run_window(win: Window) -> dict:
    print(f"\n>>> Window {win.label}")
    print(f"    train: {win.train_start} ~ {win.train_end}")
    print(f"    valid: {win.valid_start} ~ {win.valid_end}")
    print(f"    test : {win.test_start} ~ {win.test_end}")

    dataset = init_instance_by_config(build_dataset_config(win))
    model = init_instance_by_config(build_model_config())

    print("    [training...]")
    model.fit(dataset)

    pred = model.predict(dataset)
    sig_metrics = evaluate_pred(pred, dataset)
    print(
        f"    IC={sig_metrics['ic']:+.4f} "
        f"IC_IR={sig_metrics['ic_ir']:+.4f} "
        f"RankIC={sig_metrics['rank_ic']:+.4f} "
        f"n_obs={sig_metrics['n_obs']}"
    )

    bt_metrics = run_backtest_metrics(pred, win)
    if not np.isnan(bt_metrics.get("annual_return", np.nan)):
        print(
            f"    AnnRet={bt_metrics['annual_return']:+.2%} "
            f"MDD={bt_metrics['max_drawdown']:+.2%} "
            f"Sharpe={bt_metrics['sharpe']:+.2f}"
        )

    row = {
        "window": win.label,
        "train_start": win.train_start,
        "train_end": win.train_end,
        "valid_start": win.valid_start,
        "valid_end": win.valid_end,
        "test_start": win.test_start,
        "test_end": win.test_end,
    }
    row.update(sig_metrics)
    row.update(bt_metrics)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) > 1:
        years = [int(x) for x in sys.argv[1:]]
    else:
        years = DEFAULT_TEST_YEARS

    qlib.init(provider_uri=DATA_PATH, region="cn")

    rows = []
    for y in years:
        win = make_window(y)
        try:
            rows.append(run_window(win))
        except Exception as exc:  # noqa: BLE001
            print(f"!!! window {win.label} failed: {exc}")
            rows.append({"window": win.label, "error": str(exc)[:200]})

    df = pd.DataFrame(rows)
    out_path = "rolling_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("滚动回测稳定性汇总")
    print("=" * 80)
    show_cols = [
        "window", "ic", "ic_ir", "rank_ic", "rank_ic_ir",
        "annual_return", "max_drawdown", "sharpe", "n_obs", "n_days",
    ]
    cols = [c for c in show_cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))
    if {"ic", "annual_return"}.issubset(df.columns):
        print(
            f"\nIC          mean={df['ic'].mean():+.4f} std={df['ic'].std():+.4f}"
        )
        print(
            f"AnnReturn   mean={df['annual_return'].mean():+.4f} std={df['annual_return'].std():+.4f}"
        )
    print(f"\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
