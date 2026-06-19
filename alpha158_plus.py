"""
Alpha158Plus: Alpha158 + Tushare 互补因子.

新增的 3 类因子族（默认全部启用，可通过 ``factor_groups`` 做消融）：
  • valuation  (7 个) - 估值水平 + 估值动态
  • liquidity  (4 个) - 换手率 / 量比衍生
  • moneyflow  (4 个) - 主力资金 5 日净流入比

所有新因子都是表达式构造，由 qlib 的 Expression engine 解析。
要求 Provider 能解析 ``$pe_ttm``, ``$pb``, ``$ps_ttm``, ``$dv_ttm``, ``$circ_mv``,
``$turnover_rate_f``, ``$volume_ratio``, ``$amount``，以及 stock_moneyflow 表中的
``$net_mf_amount``, ``$buy_lg_amount``, ``$sell_lg_amount``, ``$buy_elg_amount``,
``$sell_elg_amount``, ``$buy_sm_amount``, ``$sell_sm_amount``。

最简单的用法：搭配 ``init_qlib_with_db`` 使用 PG + ClickHouse 后端。

示例：

    from alpha158_plus import Alpha158Plus
    handler = Alpha158Plus(
        instruments="csi300",
        start_time="2020-02-01",
        end_time="2026-03-20",
        fit_start_time="2020-02-01",
        fit_end_time="2022-12-31",
        factor_groups=["valuation", "liquidity", "moneyflow"],   # 默认全开
    )
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from qlib.contrib.data.handler import Alpha158


# ---------------------------------------------------------------------------
# Factor groups
# ---------------------------------------------------------------------------

ALL_GROUPS = ("valuation", "liquidity", "moneyflow", "finance")

def _valuation_factors() -> Tuple[List[str], List[str]]:
    """估值因子族：5 个静态 + 2 个动态 = 7 个."""
    fields = [
        # ---- 估值水平 ----
        "1.0 / $pe_ttm",                                        # 盈利收益率（低 PE 高分）
        "1.0 / $pb",                                            # 账面价值收益率
        "1.0 / $ps_ttm",                                        # 销售收益率
        "$dv_ttm",                                              # 股息率
        "Log($circ_mv)",                                        # 流通市值（取对数；小盘负相关）
        # ---- 估值动态 ----
        "1.0 / $pe_ttm - Ref(1.0 / $pe_ttm, 60)",               # EP 60 日变化（估值改善）
        "1.0 / $pb - Ref(1.0 / $pb, 60)",                       # BP 60 日变化
    ]
    names = [
        "EP_TTM",
        "BP",
        "SP_TTM",
        "DP",
        "LOG_CIRC_MV",
        "EP_DELTA_60",
        "BP_DELTA_60",
    ]
    return fields, names


def _liquidity_factors() -> Tuple[List[str], List[str]]:
    """流动性 / 关注度因子族：4 个."""
    fields = [
        "Mean($turnover_rate_f, 5)",                            # 5 日平均换手率
        "Mean($turnover_rate_f, 5) / (Mean($turnover_rate_f, 60) + 1e-6)",  # 流动性突变
        "Std($turnover_rate_f, 20)",                            # 20 日换手率标准差
        "Mean($volume_ratio, 5)",                               # 5 日平均量比
    ]
    names = [
        "TURN_5",
        "TURN_5_60_RATIO",
        "TURN_STD_20",
        "VR_MEAN_5",
    ]
    return fields, names


def _moneyflow_factors() -> Tuple[List[str], List[str]]:
    """资金流因子族：4 个 5 日比率因子.

    分母 ``Sum($amount, 5)`` 来自 stock_daily_prices（单位千元），分子来自
    stock_moneyflow（单位万元），存在 10x 量纲偏差，但是横截面比较时被
    cross-section z-score normalize 抹掉，不影响 IC.
    """
    fields = [
        "Sum($net_mf_amount, 5) / (Sum($amount, 5) + 1)",                          # 总主力净流入占比
        "Sum($buy_lg_amount - $sell_lg_amount, 5) / (Sum($amount, 5) + 1)",        # 大单净流入占比
        "Sum($buy_elg_amount - $sell_elg_amount, 5) / (Sum($amount, 5) + 1)",      # 超大单净流入（聪明钱）
        "(Sum($sell_sm_amount - $buy_sm_amount, 5)) / (Sum($amount, 5) + 1)",      # 散户反向（小单净卖出，正分代表散户在出货）
    ]
    names = [
        "MF_NET_RATIO_5",
        "MF_LG_NET_5",
        "MF_ELG_NET_5",
        "MF_SM_REV_5",
    ]
    return fields, names


def _finance_factors() -> Tuple[List[str], List[str]]:
    """财务因子族：5 个 level + 2 个 1 年同比变化 = 7 个.

    底层字段由 DBFeatureProvider 通过 ``stock_income`` + ``stock_balancesheet``
    联表 + 季度年化 + ann_date 公告日 PIT 滚动 提供（参见 db_provider.py
    的 ``FINA_DERIVED_FIELDS`` 和 ``_load_fina_per_instrument``）.

    动态因子用 ~250 个交易日的 Ref 近似一年同比，避免依赖更复杂的财报历史索引.
    """
    fields = [
        # ---- 财务水平 ----
        "$roe",                                  # 净资产收益率（年化）
        "$npm",                                  # 净利率（cumulative income / cumulative revenue）
        "$debt_to_assets",                       # 资产负债率
        "$assets_turn",                          # 总资产周转率（年化）
        "$log_total_assets",                     # log(总资产) - 规模因子
        # ---- 财务动态 ----
        "$roe - Ref($roe, 250)",                 # ROE 1 年同比变化（约 250 个交易日）
        "$npm - Ref($npm, 250)",                 # 净利率 1 年同比变化
    ]
    names = [
        "ROE",
        "NPM",
        "DEBT_TO_ASSETS",
        "ASSETS_TURN",
        "LOG_TOTAL_ASSETS",
        "ROE_DELTA_1Y",
        "NPM_DELTA_1Y",
    ]
    return fields, names


def _finance_lite_factors() -> Tuple[List[str], List[str]]:
    """精简财务因子族（5 个）：去掉与 v1 高度共线的 LOG_TOTAL_ASSETS 和 DEBT_TO_ASSETS.

    在 csi300 横截面上，``LOG_TOTAL_ASSETS`` 与 BP/EP_TTM/LOG_CIRC_MV 的相关
    都 > 0.5（运行 run_corr_analysis.py 可复现），``DEBT_TO_ASSETS`` 与 BP 也
    >0.5；本族保留的 5 个因子与 v1 的所有跨族 |corr| 都 < 0.4，是真正独立的
    财务信号.
    """
    fields = [
        "$roe",
        "$npm",
        "$assets_turn",
        "$roe - Ref($roe, 250)",
        "$npm - Ref($npm, 250)",
    ]
    names = [
        "ROE",
        "NPM",
        "ASSETS_TURN",
        "ROE_DELTA_1Y",
        "NPM_DELTA_1Y",
    ]
    return fields, names


_GROUP_BUILDERS = {
    "valuation": _valuation_factors,
    "liquidity": _liquidity_factors,
    "moneyflow": _moneyflow_factors,
    "finance":   _finance_factors,
    "finance_lite": _finance_lite_factors,
}


def get_extra_features(groups: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return (expressions, names) for the requested factor groups."""
    groups = tuple(groups)
    fields: List[str] = []
    names: List[str] = []
    for g in groups:
        if g not in _GROUP_BUILDERS:
            raise ValueError(f"unknown factor group: {g}")
        f, n = _GROUP_BUILDERS[g]()
        fields.extend(f)
        names.extend(n)
    return fields, names


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class Alpha158Plus(Alpha158):
    """Alpha158 + 估值 / 流动性 / 资金流 因子."""

    def __init__(
        self,
        *args,
        factor_groups: Iterable[str] | None = None,
        **kwargs,
    ):
        # Stash on instance so get_feature_config (called inside super().__init__)
        # can pick it up.  None == use all groups.
        if factor_groups is None:
            factor_groups = ALL_GROUPS
        self._extra_groups: Tuple[str, ...] = tuple(factor_groups)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------

    def get_feature_config(self):
        base_fields, base_names = super().get_feature_config()
        if not self._extra_groups:
            return base_fields, base_names
        extra_fields, extra_names = get_extra_features(self._extra_groups)
        return base_fields + extra_fields, base_names + extra_names


__all__ = [
    "Alpha158Plus",
    "ALL_GROUPS",
    "get_extra_features",
]
