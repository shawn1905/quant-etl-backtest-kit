# Quant ETL & Backtest Kit

国内商品/股指期货与期权 **日盘后数据采集（ETL）+ 完整性校验 + 无前视同合约链/防过拟合回测框架**（脱敏可复用版本）。

## 目录结构

```text
quant-etl-backtest-kit/
├── etl/                        # 数据采集与质量门禁（支持 OpenCTP 主源 + TqSdk 自动兜底）
│   ├── run_eod_core.py         # 核心日线 ETL 串联入口（contracts -> 1d_fut -> 1d_opt -> basic -> integrity）
│   ├── update_contracts_openctp.py # 全市场期货存续合约日线（含 TqSdk 兜底与 OHL/OI 自检）
│   ├── derive_main_1d.py       # 无未来函数主连推导（D-1 最大 OI + 只向远月滚动，保留 physical contract）
│   ├── update_1dopt_openctp.py # 全市场期权日线（OpenCTP 主源 + TqSdk 兜底，含已到期期权分母隔离修复）
│   ├── update_basic_openctp.py # 合约基础信息（乘数/行权价/到期日/保证金率）
│   ├── collect_seat.py         # 交易所前 20 席位持仓排名增量采集（支持断点续传）
│   └── check_integrity.py      # 多层数据完整性门禁（覆盖率、时间轴、跨层一致性、合理性检查）
├── src/quant_backtest/         # 离线回测与统计学防过拟合底座（核心纯 Python 标准库）
│   ├── contracts.py            # 合约代码解析与按日主力合约判定
│   ├── fees.py                 # 品种乘数、最小变动价位与手续费加载器
│   ├── overfit_stats.py        # 试验登记、BH-FDR 多重校正、月块 Bootstrap、符号置换检验、二维参数高原
│   ├── opt_plateau_clean.py    # 同合约无跳空特征生成与严格成交量校验期权买方回测
│   ├── opt_credit_spread.py    # 封顶垂直期权信用价差回测（多档滑点/手续费与 6 门禁汇总）
│   └── family_pipeline.py      # 5 大策略族准入漏斗（GateSpec）+ 3 类撮合器（bar_order/ladder_grid/option_leg）+ 跨族组合评估
├── catalog/
│   └── fees_by_product.csv     # 公开交易所合约乘数/最小跳/参考费率表
├── examples/
│   └── demo_backtest.py        # 合成数据端到端回测示例
└── tests/                      # 全离线合成数据单测
```

## 快速开始

### 1. 运行离线单元测试与合成示例（无需额外数据源）

```bash
PYTHONPATH=src python3 -B -W error -m unittest discover -s tests -v
PYTHONPATH=src python3 -B examples/demo_backtest.py
```

### 2. 运行数据采集 ETL（可选）

ETL 依赖 `pandas`、`pyarrow`、`tqsdk`（仅兜底与席位采集使用）：

```bash
export QUANT_DATA_ROOT=/path/to/data_root
export TQ_USERNAME=your_tq_user
export TQ_PASSWORD=your_tq_password

python3 etl/run_eod_core.py --date 20261117
```

## 核心设计纪律

1. **禁止跨换月拼接收益**：所有日线收益与波动率均从 `1d_contracts` 同物理合约序列计算，避免未复权主连 `1d_fut` 换月跳空污染突破/波动率指标。
2. **严格成交可行性**：期权入/出场均校验 `vol > 0` 与到期前安全截断；限价阶梯挂单 (`ladder_grid`) 按同 bar 保守先跌后涨路径撮合，不做虚构成交。
3. **防过拟合六门禁**：所有回测结果均强制登记试验总数，统一通过样本数、1x/2x 成本净期望、月块 Bootstrap 95% 下限、`BH-FDR` 多重检验校正及去最好月稳健性检查。
