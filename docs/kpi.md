# 最小 KPI（用于“超越 RD-Agent”的证据化验收）

> 适用范围：`trading-system-v3`（IQFMP：`src/iqfmp` + 深改 Qlib：`vendor/qlib`）  
> 目标：把“全面超越 rd-agent”从口号变成可跑命令 + 可对比数字。

## 0. 基线与对标（必须先满足）

**KPI-0 Baseline Pinning（对标可复现）**

- **定义**：Qlib 官方、RD-Agent 官方、本仓库三份基线（Qlib 纯 fork / RD-Agent 纯 fork / Qlib 深改 fork）都要有可复现的版本标识（commit 或 tag）。
- **验收标准**：
  - `vendor/qlib`：记录其来源版本（例如“基于 Qlib 0.9.6”）+ 关联官方 commit/tag（若有网）。
  - `fork-project/qlib-main` 与 `fork-project/RD-Agent-main`：必须落为 **git 子模块**或 **vendor + VERSION 文件**（包含 upstream commit/tag）。
- **最小验证**：
  - 本地：`git status --porcelain=v1 -b`
  - 联网后（需要网络权限）：`git ls-remote --tags https://github.com/microsoft/qlib.git | head`、`git ls-remote --tags https://github.com/microsoft/RD-Agent.git | head`

## 1. 能力覆盖（Capability Matrix 量化）

**KPI-1 Coverage Score（能力覆盖度）**

- **定义**：把 Capability Matrix 的每个一级维度（数据/因子/agent/回测/评估/实验工程/线上执行&风控/安全）拆成可检查项，按状态计分：
  - 已具备 = 1
  - 部分具备 = 0.5
  - 缺失 = 0
  - 无法判断 = 不计入分母（但必须列出“消除不确定性”的最小步骤）
- **验收标准**（最小版本）：
  - IQFMP 覆盖度 ≥ RD-Agent 覆盖度（以相同清单评分）。
  - “Crypto 合约回测真实性”子项（见 KPI-4）单独要求 ≥ 80%。

## 2. 可复现性（端到端）

**KPI-2 Reproducibility**

- **定义**：一条命令可在干净环境（无网络也可）跑通核心链路的最小 demo。
- **验收标准（本仓库最小）**：
  - `pytest -q` 通过，且覆盖率阈值满足（由 `pyproject.toml` 约束）。
- **复现命令**：
  - `pytest -q`
- **增强版（需要服务）**：
  - 启动 API/DB 后跑 `scripts/full_pipeline_test.py`（依赖 `localhost:8000`）。

## 3. 研究效率（可测）

**KPI-3 Research Throughput**

- **定义**：从“数据 → 因子 → 评估 → 组合/策略 → 回测 → 报告”的自动化程度与耗时。
- **验收标准（最小）**：
  - 提供脚本/命令可在固定 seed 下跑完整闭环，并输出：
    - 生成因子数量
    - 通过质量门的因子数量
    - 运行耗时（wall clock）
    - 关键指标（IC/IR/Sharpe/MaxDD）
- **最小实现建议**：
  - 以 `tests/integration/*` 的固定数据/固定 seed 为基准，新增一个 `scripts/bench/research_throughput.py` 输出上述指标（后续迭代实现）。

## 4. Crypto 合约回测真实性（专项）

**KPI-4 Perp Backtest Realism（合约真实性）**

把“真实性”拆成 6 个可打勾子项（每项 0/0.5/1）：

1) 价格体系：trade/mark/index 的定义与使用；强平/结算触发基准明确  
2) 资金费率：结算频率、计费基准、对资金曲线影响（含测试）  
3) 保证金与爆仓：逐仓/全仓、维持保证金、强平逻辑、穿仓处理（含测试）  
4) 费用与滑点：maker/taker、冲击成本、延迟、部分成交（含测试）  
5) 订单与撮合：市价/限价/止损止盈、撮合规则、精度/最小下单量（含测试）  
6) 风控：最大杠杆/仓位、回撤熔断、黑天鹅保护（可选，含测试）

**验收标准（最小）**：
- 6 项合计得分 / 6 ≥ 0.8，并且每项都有：
  - 配置项/代码位置
  - 至少 1 个单元测试或一致性测试（synthetic data 可）

## 5. 稳定性（工程成熟度）

**KPI-5 Stability**

- **定义**：测试、覆盖率、关键路径错误可追溯性。
- **验收标准（本仓库最小）**：
  - `pytest -q` 全绿
  - 覆盖率阈值满足（当前由 `pyproject.toml` 控制）

