#!/usr/bin/env python
"""
日频策略训练和回测脚本
- 训练期：~2023-12-31
- 回测期：2024-01-01 ~ 2026-03-20
- 起步资金：10万人民币
"""
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import init_instance_by_config
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_PATH = '/data/StockData/data/QuantExperiment/data/qlib_data_latest'
REGION = 'cn'  # 中国市场
MARKET = 'csi300'  # 沪深300
BENCHMARK = 'SH000300'  # 沪深300指数
START_MONEY = 100000  # 10万人民币

# 时间配置
# 训练 / 验证分离：valid 用作 LightGBM early stopping 的样本外观察集
# 起点 2020-02-01：因为 stock_index_weight 表最早数据为 2020-01-23，DB 后端
# 之前没有 PIT csi300 成分股，会返回空 universe。
TRAIN_START = '2020-02-01'  # 训练开始（PIT csi300 成员可用之后）
TRAIN_END   = '2022-12-31'  # 训练结束
VALID_START = '2023-01-01'  # 验证开始（独立一年，触发 early stopping）
VALID_END   = '2023-12-31'  # 验证结束
TEST_START  = '2024-01-01'  # 回测开始
TEST_END    = '2026-03-20'  # 回测结束

print("="*80)
print("日频策略训练与回测")
print("="*80)
print(f"数据路径: {DATA_PATH}")
print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
print(f"验证期: {VALID_START} ~ {VALID_END}  (独立样本外，用于 early stopping)")
print(f"回测期: {TEST_START} ~ {TEST_END}")
print(f"起步资金: {START_MONEY:,.0f} 人民币")
print(f"市场: {MARKET}, 基准: {BENCHMARK}")
print("="*80)

# 初始化 QLib
print("\n[1/6] 初始化 QLib...")
try:
    qlib.init(provider_uri=DATA_PATH, region=REGION)
    print("✓ QLib 初始化成功")
except Exception as e:
    print(f"✗ QLib 初始化失败: {e}")
    exit(1)

# 检查可用股票
print("\n[2/6] 检查可用股票...")
try:
    instruments = D.instruments(market=MARKET)
    print(f"✓ {MARKET} 共有 {len(instruments)} 只股票")
except Exception as e:
    print(f"✗ 获取股票列表失败: {e}")
    # 使用所有股票
    instruments = D.list_instruments(D.instruments())
    print(f"✓ 使用所有可用股票: {len(instruments)} 只")

# 配置数据处理器
print("\n[3/6] 配置数据处理器...")
data_handler_config = {
    'start_time': TRAIN_START,
    'end_time': TEST_END,
    'fit_start_time': TRAIN_START,
    'fit_end_time': TRAIN_END,
    'instruments': MARKET,
}

# 配置数据集
dataset_config = {
    'class': 'DatasetH',
    'module_path': 'qlib.data.dataset',
    'kwargs': {
        'handler': {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': data_handler_config,
        },
        'segments': {
            'train': (TRAIN_START, TRAIN_END),
            'valid': (VALID_START, VALID_END),  # 独立样本外，用于 early stopping
            'test':  (TEST_START, TEST_END),
        }
    }
}

try:
    dataset = init_instance_by_config(dataset_config)
    print("✓ 数据集创建成功")
except Exception as e:
    print(f"✗ 数据集创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 配置模型
print("\n[4/6] 训练模型...")
# 注意：qlib 的 LGBModel 用原生 LightGBM API，使用 num_boost_round 而非 sklearn 的 n_estimators
# 在 valid 与 train 真正不同的前提下，early_stopping_rounds 才会触发样本外早停
model_config = {
    'class': 'LGBModel',
    'module_path': 'qlib.contrib.model.gbdt',
    'kwargs': {
        'loss': 'mse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 128,
        'num_boost_round': 1000,        # 训练上限，由 early stopping 实际控制
        'early_stopping_rounds': 50,    # 50 轮验证集 l2 不下降则停止
        'verbose': -1,
    }
}

try:
    model = init_instance_by_config(model_config)
    print("开始训练模型...")
    model.fit(dataset)
    print("✓ 模型训练完成")
except Exception as e:
    print(f"✗ 模型训练失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 生成预测
print("\n[5/6] 生成预测...")
try:
    pred = model.predict(dataset)
    print(f"✓ 预测生成成功，形状: {pred.shape}")
    print(f"  预测范围: {pred.index.get_level_values('datetime').min()} ~ {pred.index.get_level_values('datetime').max()}")
except Exception as e:
    print(f"✗ 预测生成失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 执行回测
print("\n[6/6] 执行回测...")
print(f"回测期间: {TEST_START} ~ {TEST_END}")
print(f"起步资金: {START_MONEY:,.0f} 人民币")

# 配置策略
strategy_config = {
    'class': 'TopkDropoutStrategy',
    'module_path': 'qlib.contrib.strategy',
    'kwargs': {
        'signal': pred,
        'topk': 30,      # 持有30只股票
        'n_drop': 3,     # 每次调仓保留3只
    }
}

# 配置回测
backtest_config = {
    'start_time': TEST_START,
    'end_time': TEST_END,
    'account': START_MONEY,
    'benchmark': BENCHMARK,
    'exchange_kwargs': {
        'limit_threshold': 0.095,     # 涨跌停限制
        'deal_price': 'close',        # 使用收盘价成交（注意：实际成交价不可能正好等于 close）
        # A 股现实成本拆解（2023-08 印花税下调后）：
        #   买入 = 佣金 ~0.025% + 过户费 0.001% + 半个买卖价差 ~0.05% ≈ 0.15%
        #   卖出 = 佣金 ~0.025% + 印花税 0.05% + 过户费 0.001% + 半个价差 ~0.05% ≈ 0.25%
        # impact_cost: 市场冲击 = impact_cost * (trade_val / total_volume)^2，qlib 建议 0.1
        'open_cost': 0.0015,
        'close_cost': 0.0025,
        'min_cost': 5,
        'impact_cost': 0.1,
    }
}

try:
    # 配置executor
    executor_config = {
        'class': 'SimulatorExecutor',
        'module_path': 'qlib.backtest.executor',
        'kwargs': {
            'time_per_step': 'day',
            'generate_portfolio_metrics': True,
        }
    }

    # 执行回测
    print("开始回测...")
    portfolio_result, indicator_result = backtest(
        start_time=TEST_START,
        end_time=TEST_END,
        strategy=strategy_config,
        executor=executor_config,
        account=START_MONEY,
        benchmark=BENCHMARK,
        exchange_kwargs={
            'limit_threshold': 0.095,
            'deal_price': 'close',
            'open_cost': 0.0015,
            'close_cost': 0.0025,
            'min_cost': 5,
            'impact_cost': 0.1,
        }
    )

    print("\n" + "="*80)
    print("回测结果")
    print("="*80)

    # 解析结果
    if portfolio_result is not None:
        # 获取最终资产
        if hasattr(portfolio_result, 'get_assets'):
            final_assets = portfolio_result.get_assets()
            print(f"\n最终资产: {final_assets:,.2f} 人民币")
            print(f"总收益率: {(final_assets/START_MONEY - 1)*100:.2f}%")
        else:
            print("\n资产详情:")
            print(portfolio_result)

    # 显示风险指标
    if indicator_result is not None:
        print("\n风险指标:")
        for key, value in indicator_result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # 尝试获取更详细的收益分析
    try:
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.backtest import collect_data

        # 收集回测数据
        report_dict = collect_data(
            strategy=strategy,
            **backtest_config
        )

        if report_dict and 'portfolio' in report_dict:
            portfolio = report_dict['portfolio']
            returns = portfolio['return']

            # 计算详细指标
            total_return = (1 + returns).prod() - 1
            n_periods = len(returns)

            # 年化收益（日频，252个交易日）
            annual_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 夏普比率
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0

            # 胜率
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

            # 最终资产
            final_money = START_MONEY * (1 + total_return)

            print("\n" + "="*80)
            print("详细回测分析")
            print("="*80)
            print(f"\n资金变化:")
            print(f"  起步资金: {START_MONEY:,.2f} 人民币")
            print(f"  最终资金: {final_money:,.2f} 人民币")
            print(f"  盈利金额: {final_money - START_MONEY:,.2f} 人民币")
            print(f"  总收益率: {total_return*100:.2f}%")
            print(f"  年化收益率: {annual_return*100:.2f}%")

            print(f"\n风险指标:")
            print(f"  最大回撤: {max_drawdown*100:.2f}%")
            print(f"  年化波动率: {volatility*100:.2f}%")
            print(f"  夏普比率: {sharpe_ratio:.4f}")
            print(f"  胜率: {win_rate*100:.2f}%")
            print(f"  交易天数: {n_periods} 天")

            # 保存结果
            result_df = pd.DataFrame({
                '起步资金': [START_MONEY],
                '最终资金': [final_money],
                '盈利金额': [final_money - START_MONEY],
                '总收益率': [total_return],
                '年化收益率': [annual_return],
                '最大回撤': [max_drawdown],
                '夏普比率': [sharpe_ratio],
                '胜率': [win_rate],
                '交易天数': [n_periods]
            })

            result_file = '/data/openclaw_workspace/qlib/backtest_results_2024_2026.csv'
            result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 结果已保存到: {result_file}")

            # 保存每日收益
            returns_df = returns.to_frame('daily_return')
            returns_df['cumulative_return'] = (1 + returns_df['daily_return']).cumprod() - 1
            returns_df['asset_value'] = START_MONEY * (1 + returns_df['cumulative_return'])
            returns_file = '/data/openclaw_workspace/qlib/daily_returns_2024_2026.csv'
            returns_df.to_csv(returns_file, encoding='utf-8-sig')
            print(f"✓ 每日收益已保存到: {returns_file}")

    except Exception as e:
        print(f"\n详细分析出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("回测完成！")
    print("="*80)

except Exception as e:
    print(f"✗ 回测执行失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
