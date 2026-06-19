#!/usr/bin/env python
"""
Tushare 互补因子 PoC v2

(A) 多 horizon 消融对比：
    针对 1 / 5 / 10 日前向收益三个 label，跑 baseline / +valuation / +liquidity /
    +moneyflow / +all 共 5 个配置，对比每个因子族在不同尺度上的边际 IC.

(B) Univariate IC：
    每个新因子（共 15 个）单独 vs 1d/5d/10d label 的横截面 IC + Rank IC.
    用于剥离 Alpha158 干扰，验证因子本身有没有 alpha 信号.

输出:
    alpha158plus_poc_v2_ablation.csv   - 模型层级 IC（A 部分）
    alpha158plus_poc_v2_univariate.csv - 单因子 IC（B 部分）
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha158_plus import Alpha158Plus, ALL_GROUPS, get_extra_features

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

# 三个 label horizon
# 注意：Ref($close, -1) 是 T+1 的 close（执行价），分子是 T+1+H 的 close
HORIZONS = {
    "1d":  ("Ref($close, -2)/Ref($close, -1) - 1",  "LABEL_1D"),
    "5d":  ("Ref($close, -6)/Ref($close, -1) - 1",  "LABEL_5D"),
    "10d": ("Ref($close, -11)/Ref($close, -1) - 1", "LABEL_10D"),
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


def _build_handler(groups: Iterable[str], label_expr: str, label_name: str):
    return Alpha158Plus(
        instruments=MARKET,
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        factor_groups=groups,
        label=([label_expr], [label_name]),
    )


def _build_dataset(handler):
    from qlib.data.dataset import DatasetH
    return DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test":  (TEST_START, TEST_END),
        },
    )


def _xs_ic(df: pd.DataFrame, score_col: str, label_col: str, method: str = "pearson") -> Tuple[float, float, int]:
    """Cross-sectional IC, ICIR, n_days."""
    sub = df[[score_col, label_col]].dropna()
    if sub.empty:
        return float("nan"), float("nan"), 0
    daily = sub.groupby(level="datetime").apply(
        lambda x: x[score_col].corr(x[label_col], method=method)
    )
    daily = daily.dropna()
    if daily.empty:
        return float("nan"), float("nan"), 0
    mu = float(daily.mean())
    sd = float(daily.std())
    return mu, mu / sd if sd else float("nan"), int(len(daily))


# ---------------------------------------------------------------------------
# Part A: ablation grid
# ---------------------------------------------------------------------------

def run_ablation(name: str, groups: Iterable[str], horizon: str) -> dict:
    from qlib.utils import init_instance_by_config

    label_expr, label_name = HORIZONS[horizon]
    handler = _build_handler(groups, label_expr, label_name)
    dataset = _build_dataset(handler)

    n_features = len(handler.get_feature_config()[0])
    print(f"  >>> {name:<11} horizon={horizon:<3} n_feat={n_features}", end=" ", flush=True)

    model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": dict(LGB_KWARGS),
    })
    model.fit(dataset)
    best_iter = getattr(model.model, "best_iteration", None)

    pred = model.predict(dataset)
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    merged = pred.to_frame("score").join(label).dropna()
    if merged.empty:
        out = dict(name=name, horizon=horizon, n_feat=n_features, best_iter=best_iter,
                   ic=np.nan, ic_ir=np.nan, rank_ic=np.nan, rank_ic_ir=np.nan, n_days=0)
        print("(empty)")
        return out

    ic_mean, ic_ir, n_days = _xs_ic(merged, "score", "label", "pearson")
    ric_mean, ric_ir, _    = _xs_ic(merged, "score", "label", "spearman")
    out = dict(
        name=name, horizon=horizon, n_feat=n_features, best_iter=best_iter,
        ic=ic_mean, ic_ir=ic_ir, rank_ic=ric_mean, rank_ic_ir=ric_ir, n_days=n_days,
    )
    print(f"best_iter={best_iter}  IC={ic_mean:+.4f} RankIC={ric_mean:+.4f}")
    return out


# ---------------------------------------------------------------------------
# Part B: univariate IC of each new factor
# ---------------------------------------------------------------------------

def run_univariate_ic(horizon: str) -> List[dict]:
    """对每个新因子单独算 IC vs label_horizon."""
    label_expr, label_name = HORIZONS[horizon]
    # 用 +all 配置把所有 15 个新因子都拉进来
    handler = _build_handler(ALL_GROUPS, label_expr, label_name)
    dataset = _build_dataset(handler)
    extra_fields, extra_names = get_extra_features(ALL_GROUPS)

    # 直接拿 test 段的 raw 特征 + label
    df = dataset.prepare("test", col_set=["feature", "label"])
    if df.empty:
        return []

    # df.columns 是 MultiIndex: ('feature', name) / ('label', name)
    feat = df["feature"]
    lab  = df["label"][label_name]

    rows = []
    for fname in extra_names:
        if fname not in feat.columns:
            continue
        merged = pd.concat([feat[fname].rename("score"), lab.rename("label")], axis=1).dropna()
        ic_mean, ic_ir, n_days = _xs_ic(merged, "score", "label", "pearson")
        ric_mean, ric_ir, _ = _xs_ic(merged, "score", "label", "spearman")
        rows.append({
            "horizon": horizon,
            "factor": fname,
            "n_days": n_days,
            "ic": ic_mean, "ic_ir": ic_ir,
            "rank_ic": ric_mean, "rank_ic_ir": ric_ir,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _load_db_env()
    _init_qlib()
    print("=" * 80)
    print("Alpha158Plus PoC v2 — multi-horizon ablation + univariate IC")
    print("=" * 80)
    print(f"  train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  valid: {VALID_START} ~ {VALID_END}")
    print(f"  test : {TEST_START} ~ {TEST_END}")
    print(f"  market: {MARKET}")
    print(f"  horizons: {list(HORIZONS.keys())}")
    print("=" * 80)

    # --- Part B: univariate IC (do first, only need to load data once per horizon) ---
    print("\n[B] Univariate IC of each new factor")
    print("-" * 80)
    uni_rows: List[dict] = []
    for h in HORIZONS:
        rows = run_univariate_ic(h)
        uni_rows.extend(rows)

    uni_df = pd.DataFrame(uni_rows)
    if not uni_df.empty:
        uni_df.to_csv("alpha158plus_poc_v2_univariate.csv", index=False, encoding="utf-8-sig")
        # 透视成 horizon × factor 表
        for metric in ("ic", "rank_ic"):
            piv = uni_df.pivot(index="factor", columns="horizon", values=metric).reindex(
                index=[r["factor"] for r in uni_rows if r["horizon"] == "1d"]
            )
            print(f"\n[univariate {metric}]")
            print(piv.to_string(float_format=lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN"))

    # --- Part A: ablation grid ---
    print("\n\n[A] Ablation grid (model-level IC)")
    print("-" * 80)
    abl_rows: List[dict] = []
    for h in HORIZONS:
        for name, groups in ABLATIONS:
            try:
                abl_rows.append(run_ablation(name, groups, h))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                abl_rows.append(dict(name=name, horizon=h, error=str(exc)[:200]))

    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv("alpha158plus_poc_v2_ablation.csv", index=False, encoding="utf-8-sig")

    # 展示成 horizon × ablation 表
    print("\n" + "=" * 80)
    print("Model-level IC (horizon × ablation)")
    print("=" * 80)
    for metric in ("ic", "rank_ic"):
        if metric not in abl_df.columns:
            continue
        piv = abl_df.pivot(index="name", columns="horizon", values=metric)
        # 维持 ABLATIONS 顺序
        piv = piv.reindex(index=[a[0] for a in ABLATIONS])
        # 维持 horizon 顺序
        piv = piv[list(HORIZONS.keys())]
        print(f"\n[ablation {metric}]")
        print(piv.to_string(float_format=lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN"))

    # Delta 列
    if "ic" in abl_df.columns:
        print("\n[Δ IC vs baseline (per horizon)]")
        for h in HORIZONS:
            sub = abl_df[abl_df["horizon"] == h]
            base_ic = sub.loc[sub["name"] == "baseline", "ic"].iloc[0]
            for _, r in sub.iterrows():
                if r["name"] == "baseline":
                    continue
                d = r["ic"] - base_ic
                print(f"  horizon={h}  {r['name']:<11} ΔIC={d:+.4f}")

    print(f"\n结果文件:")
    print(f"  • alpha158plus_poc_v2_univariate.csv")
    print(f"  • alpha158plus_poc_v2_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
