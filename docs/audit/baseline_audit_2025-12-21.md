# Baseline Audit（本地三份基线 + IQFMP）- 2025-12-21

> 本报告只引用可复现证据：本仓库文件/命令输出。  
> 生成方式：建议运行 `scripts/audit/run_local_audit.sh`（可选 `--with-tests`）。

## 1) 歧义消除声明（本项目/三份代码/对标基准如何定义）

- **本项目定义（按仓库现状）**：IQFMP 主体在 `src/iqfmp`，Qlib 深改 fork 在 `vendor/qlib`（被打包与测试引用）。  
  - 证据：`pyproject.toml:98`（wheel packages 包含 `vendor/qlib/qlib`）、`pyproject.toml:111`（pytest pythonpath 包含 `vendor/qlib`）
- **三份基线（本地）**：
  - B（Qlib 纯 fork）=`fork-project/qlib-main`（当前 **未被 git 跟踪**，仅作为本地对照快照）
  - C（RD-Agent 纯 fork）=`fork-project/RD-Agent-main`（当前 **未被 git 跟踪**，仅作为本地对照快照）
  - A（Qlib 深改 fork）=`vendor/qlib`（被 IQFMP 直接引用/打包）
  - 证据：`git status --porcelain=v1 -b` 显示 `?? fork-project/`；`PYTHONPATH=src:vendor/qlib python -c ...` 显示 qlib 路径为 `vendor/qlib/qlib/__init__.py`

## 2) 我在质疑你什么（关键假设逐条挑战）

1. **“全面超越 RD-Agent”如何验收？**  
   - 现状：`plan.md` 的 KPI 章节为空（`plan.md:781`）。  
   - 处置：已新增最小 KPI 定义：`docs/kpi.md`（对标必须先做 Baseline Pinning）。
2. **“建立在 Qlib 上”是否真实？**  
   - 现状：IQFMP 代码直接 `import qlib` 并使用 `qlib.backtest`/`qlib.data`；且打包包含 `vendor/qlib/qlib`。  
   - 风险：`src/iqfmp/core/qlib_crypto.py` 内存在 “pandas 实现 Qlib ops” 的 fallback（可能导致与 Qlib 原生实现偏差）。
3. **“合约深度优化”是否落地？**  
   - 现状：数据层已落地 funding/mark/index/liquidation 等采集与字段；回测侧仅覆盖 funding/fees/slippage 的“成本修正”，缺少保证金/强平/撮合等核心真实性机制（详见第 5 节与第 7 节缺口）。

## 3) Evidence Index（文件+命令+输出摘要，能复现）

**关键命令（本轮已跑通）**

```bash
# 1) 仓库信息
git rev-parse HEAD
git status --porcelain=v1 -b

# 2) Qlib deep fork 取证
PYTHONPATH=src:vendor/qlib python -c "import qlib; print(qlib.__file__); print(qlib.__version__)"
PYTHONPATH=src:vendor/qlib python -c "from qlib.constant import REG_CRYPTO; print(REG_CRYPTO)"

# 3) 测试与覆盖率
pytest -q

# 4) 深改 diff（排除 build artifacts）
diff -qr --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' --exclude='*.cpp' \
  fork-project/qlib-main/qlib vendor/qlib/qlib
```

**关键输出摘要（本轮实测）**

- 仓库 HEAD：`26ed4c61d3632c9773fc180e76eaa391c5c777c8`（`main`）
- Qlib deep fork 生效：`qlib.__file__=/Users/rocky243/trading-system-v3/vendor/qlib/qlib/__init__.py`，`qlib.__version__=0.9.6.99`
- 测试：`1191 passed`，覆盖率 `80.05%`（满足阈值）

**关键代码证据（路径+行号）**

- Qlib 深改版本与说明：`vendor/qlib/qlib/__init__.py:5`
- Crypto region：`vendor/qlib/qlib/constant.py:13`，`vendor/qlib/qlib/config.py:311`
- 24/7 交易日历：`vendor/qlib/qlib/utils/time.py:29`、`vendor/qlib/qlib/utils/time.py:69`
- IQFMP 使用 Qlib backtest：`src/iqfmp/agents/backtest_agent.py:55`
- IQFMP 初始化 Qlib（region 可设置为 crypto）：`src/iqfmp/core/qlib_init.py:29`、`src/iqfmp/core/qlib_init.py:132`
- RD-Agent 依赖 Qlib（场景模板）：`fork-project/RD-Agent-main/rdagent/scenarios/qlib/experiment/factor_data_template/generate.py:1`

## 4) Official Baseline Map（可用则给；不可用则说明限制+核验清单）

**限制**：当前环境 `network_access=restricted`，本轮未拉取官方仓库/文档，因此：

- 官方 Qlib/RD-Agent 的 commit/tag 未被记录
- 对标暂时锚定在本地 `fork-project/*` 快照

**联网后必须补做（最小核验清单）**

```bash
git ls-remote --tags https://github.com/microsoft/qlib.git | head -n 50
git ls-remote --tags https://github.com/microsoft/RD-Agent.git | head -n 50
# 记录：官方 tag/commit → 写入 docs/audit/official_baseline_map.md
```

## 5) Capability Matrix（表格）：Qlib纯fork vs rd-agent纯fork vs 我们

> 说明：以下均基于“本地代码存在 + 关键入口/调用证据”。无法从本地证据确认的条目标为“无法判断”并给最小验证步骤。

| 维度 | Qlib 纯 fork（B） | RD-Agent 纯 fork（C） | 我们（IQFMP + 深改 Qlib）（A） |
|---|---|---|---|
| 数据 | 已具备：`fork-project/qlib-main/qlib/data` | 部分具备：场景依赖 `pyqlib` 下载数据（`fork-project/RD-Agent-main/test/utils/test_env.py:24`） | 已具备：OHLCV + 衍生品采集（`src/iqfmp/data/derivatives.py:1`） + Qlib 数据抽象（`src/iqfmp/core/factor_engine.py` 需进一步抽取证据） |
| 因子 | 已具备：`qlib/contrib` 因子库 | 已具备：factor coder/runner（`fork-project/RD-Agent-main/rdagent/scenarios/qlib/developer/factor_runner.py:35`） | 已具备：LLM 因子生成与约束（`src/iqfmp/agents/factor_generation.py`）+ Qlib 表达式库（`src/iqfmp/evaluation/qlib_factor_library.py:151`） |
| Agent | 缺失（Qlib 本体非 agent 框架） | 已具备：`rdagent` 多场景 agent | 已具备：LangGraph 风格 orchestrator（`src/iqfmp/agents/orchestrator.py:130`） |
| 回测 | 已具备：`qlib/backtest`（但偏股票/现货语义） | 部分具备：通过 Docker + Qlib `qrun` 执行实验（见 `factor_runner.py`） | 部分具备：Qlib backtest + 成本修正（`src/iqfmp/evaluation/quality_gate.py:298`）；**合约保证金/强平/撮合缺失**（见第 7 节） |
| 评估 | 已具备：`qlib/contrib/eva` | 已具备：实验结果读取/比较（需进一步抽取） | 已具备：IC/稳定性/多检验校正（`src/iqfmp/evaluation/*`） |
| 实验工程 | 已具备：workflow/recorder（`qlib/workflow`） | 已具备：workspace + cache + docker env（`factor_runner.py`） | 已具备：pipeline + ledger + integration tests（`tests/integration/*`） |
| 线上执行/风控 | Qlib 有 online 子模块（`qlib/contrib/online`） | 已具备：env/health check 等（README/CLI） | 已具备：exchange adapter + risk controller（`src/iqfmp/exchange/risk.py:573`） |
| 安全 | 无法判断（需审计 Qlib sandbox/安全边界） | 已具备：Docker 隔离为主（需进一步抽取） | 已具备：RestrictedPython sandbox（`tests/unit/core/test_sandbox.py` 通过） |

## 6) Diff 审计摘要：A↔B、C↔B、A↔C（列出关键差异点）

### A（vendor/qlib）↔ B（fork-project/qlib-main）

排除 build artifacts 后，仅 7 处差异（可复现：`diff -qr ...`）：

- 版本策略：上游 `setuptools_scm` → 我们硬编码版本（`vendor/qlib/qlib/__init__.py:5`）
- 新增 crypto region：`REG_CRYPTO` + region config（`vendor/qlib/qlib/constant.py:13`、`vendor/qlib/qlib/config.py:311`）
- 24/7 日历：`CRYPTO_TIME`（`vendor/qlib/qlib/utils/time.py:29`）
- 客户端序列化：json → pickle（`vendor/qlib/qlib/data/client.py:97`）
- 新增 crypto data handler：`vendor/qlib/qlib/contrib/crypto/*`

### C（RD-Agent）↔ B（Qlib）

- RD-Agent 通过场景模板/runner 运行 Qlib（`fork-project/RD-Agent-main/rdagent/scenarios/qlib/experiment/*`），并依赖 `pyqlib` 数据下载（`fork-project/RD-Agent-main/test/utils/test_env.py:24`）。
- **对标点**：RD-Agent 的“因子/模型循环 + Docker 实验工程” vs IQFMP 的“Qlib deep fork + crypto 数据/评估/风控”。

### A（我们）↔ C（RD-Agent）

- 我们：深改 Qlib 增加 crypto region 与 24/7 日历，并新增 crypto fields（`vendor/qlib/...`）。  
- RD-Agent：agent 工程体系成熟、Docker/Workspace 体系完善；但其 Qlib 场景默认 cn_data（`generate.py:3`），对 crypto/perp 真实性未见本地证据。

## 6.5) Crypto 合约专项（必须覆盖：逐条取证）

> 状态口径：已具备 / 部分具备 / 缺失。每条给出代码位置与测试/验证方式。

### 价格体系：trade/mark/index；强平/结算触发基准

- **状态：部分具备**
- **证据（字段/采集）**：
  - funding_rates 表含 mark/index：`src/iqfmp/db/models.py:665`  
  - mark_prices 表含 mark/index/last/basis：`src/iqfmp/db/models.py:760`  
  - Qlib crypto handler 暴露 MARK_PRICE/INDEX_PRICE：`vendor/qlib/qlib/contrib/crypto/data/handler.py:58`
- **证据（回测使用）**：
  - 现有“成本回测”默认使用 `close`：`src/iqfmp/evaluation/quality_gate.py:402`  
  - 深改 Qlib region 默认 `deal_price="close"`：`vendor/qlib/qlib/config.py:312`  
  - 目前未见“强平/结算触发基准”在回测侧落地（仅数据与提示词层存在 `mark_price` 字段）
- **最小验证步骤**：
  - `rg -n "mark_price|index_price" src/iqfmp/evaluation src/iqfmp/agents src/iqfmp/strategy`（确认是否有回测级使用）
- **最小实现/测试建议**：
  - 实现 `PerpBacktestEngine` 支持 `price_basis = {trade, mark}`（交易用 trade/close，强平用 mark），并新增 synthetic 测试：当 `mark_price` 触发强平时资金曲线与仓位归零一致。

### 资金费率：结算频率、计费基准、对资金曲线影响

- **状态：已具备（成本层面）**
- **证据（对齐与结算）**：
  - funding 对齐与“8 小时结算”说明：`vendor/qlib/qlib/contrib/crypto/data/handler.py:421`  
  - 成本回测按 `[0,8,16]` 小时结算：`src/iqfmp/evaluation/quality_gate.py:283`、`src/iqfmp/evaluation/quality_gate.py:342`
  - 独立 backtest engine 也实现 funding 结算：`src/iqfmp/strategy/backtest.py:288`
- **测试证据**：
  - funding 对齐测试：`tests/unit/qlib_crypto/test_handler_complete.py:28`
- **缺口**：
  - funding 目前按 `-positions * funding_rate` 处理（`quality_gate.py:337`），未与杠杆/名义价值/保证金模型联动（合约真实性仍不足）。

### 保证金与爆仓：逐仓/全仓、维持保证金、强平逻辑假设、穿仓处理

- **状态：部分具备（监控/风控具备，回测缺失）**
- **证据（保证金与强平价格公式）**：
  - `PositionData.margin_used` / `liquidation_price`：`src/iqfmp/exchange/monitoring.py:77`、`src/iqfmp/exchange/monitoring.py:97`
- **测试证据**：
  - 强平价与保证金用量测试：`tests/unit/exchange/test_position_monitoring.py:81`
- **缺口（回测侧）**：
  - 现有回测未使用 `PositionData` 的强平逻辑；未实现逐仓/全仓切换；未定义穿仓处理（保险基金/自动减仓/归零等）。
- **最小实现/测试建议**：
  - 抽出“保证金账户模型”（isolated/cross）与“强平引擎”，复用 `PositionData` 公式；新增 synthetic 测试：不同杠杆与 mmr 下的强平价、穿仓处理路径。

### 费用与滑点：maker/taker、冲击成本模型、延迟、部分成交

- **状态：部分具备**
- **证据（费用/滑点）**：
  - Qlib exchange 支持 impact_cost/slippage 与费率：`vendor/qlib/qlib/backtest/exchange.py:107`
  - IQFMP 成本回测含 maker/taker fee + 动态滑点：`src/iqfmp/evaluation/quality_gate.py:279`、`src/iqfmp/evaluation/quality_gate.py:371`
- **证据（部分成交/订单执行-线上）**：
  - 线上执行含 partial fill handler、limit/stop：`src/iqfmp/exchange/execution.py:1`
- **缺口（回测侧）**：
  - 延迟与撮合层面未落地；部分成交仅存在于 Qlib exchange 的成交量约束语义（更接近股票/成交量限制），缺少基于 orderbook 的真实部分成交。
- **最小实现/测试建议**：
  - 在回测引擎引入 latency（bar 内延迟/滑点模型）与 partial fill（基于 depth/成交量）；新增一致性测试：同一信号在不同滑点/延迟参数下的单调性（成本更高 → 净收益不更高）。

### 订单与撮合：市价/限价/止损止盈、撮合规则、精度/最小下单量

- **状态：部分具备（线上具备，回测缺失）**
- **证据（线上订单类型）**：
  - order types：`src/iqfmp/exchange/adapter.py:79`、`src/iqfmp/exchange/execution.py:168`
- **证据（Qlib backtest 订单）**：
  - Qlib Order 无 price 字段（更像“时间区间市价成交”）：`vendor/qlib/qlib/backtest/decision.py:37`
- **缺口（回测侧）**：
  - 缺少 limit/stop 的回测撮合与精度/最小下单量规则。
- **最小实现/测试建议**：
  - 在 PerpBacktestEngine 中实现订单簿近似撮合（或 OHLCV 近似规则），并增加精度/最小下单量校验；新增测试：下单量小于最小单位应被拒单/归零处理一致。

### 风控：最大杠杆/仓位、回撤熔断、黑天鹅保护（可选）

- **状态：部分具备**
- **证据（硬阈值风控）**：
  - MAX_DRAWDOWN/MAX_LEVERAGE 等：`src/iqfmp/exchange/risk.py:583`
- **缺口（回测侧）**：
  - 回测引擎尚未把上述风控规则作为“交易前约束/熔断机制”接入（目前更多用于线上）。
- **最小实现/测试建议**：
  - 回测执行时接入 RiskController（或其等价纯函数版），新增测试：超过阈值后触发 “reduce_position/emergency_close_all” 的确定性行为。

## 7) Top 10 缺口（按“超越 rd-agent”的影响排序）

1. **合约保证金/强平模型缺失**：回测侧未实现维持保证金、强平触发、穿仓等（仅有成本修正 `quality_gate.py:298`）。  
2. **撮合/订单类型不足**：Qlib Order 不含限价/止损等（`vendor/qlib/qlib/backtest/decision.py:37`），缺少微观撮合规则。  
3. **mark/index price 未进入回测核心**：数据层有 mark/index（`models.py:760`），但回测仍用 close（`quality_gate.py:402`）。  
4. **orderbook 历史数据链路未落地**：能力声明存在，但 DB 检测与历史存储为 TODO（`field_capability.py:878`）。  
5. **baseline 未纳入版本控制**：`fork-project/*` 未被 git 跟踪（`git status`），导致对标不可复现。  
6. **vendor/qlib 含平台相关 artifacts**：`vendor/qlib/qlib/data/_libs/*.so` 等会破坏跨平台可复现（需治理策略）。  
7. **Qlib 运算一致性风险**：`qlib_crypto.py` 提供 pandas ops fallback，可能与 Qlib 原生实现偏差。  
8. **端到端 demo 依赖服务**：`scripts/full_pipeline_test.py` 需要运行 API/DB（需提供一键脚本与固定 seed）。  
9. **官方基线映射未完成**：网络受限导致官方 tag/commit 未 pin（第 4 节）。  
10. **RD-Agent 对比实验未自动化**：缺少“一键跑 RD-Agent Qlib 场景并输出同口径 KPI”脚本。

## 8) 路线图：0-2 周 / 2-6 周 / 6-12 周（每项含验收/回滚/风险）

### 0-2 周（消除不确定性 + 打通对标）

- 把 `fork-project/qlib-main`、`fork-project/RD-Agent-main` 变成可复现基线（子模块或 vendor + VERSION 文件）。  
  - 验收：仓库无 `?? fork-project/`；baseline 有 upstream commit/tag。  
  - 回滚：不改主逻辑，只改依赖组织。  
  - 风险：仓库体积/子模块管理复杂度上升。
- 建立 Perp 回测真实性测试（synthetic）：资金费率结算 + 强平触发 + 穿仓处理的最小一致性测试。  
  - 验收：新增 `tests/unit/perp_backtest/*`（或等价）并通过。  
  - 回滚：测试仅新增不影响生产。

### 2-6 周（核心真实性落地）

- 在 Qlib backtest 内新增 `PerpAccount/PerpExchange`（或等价抽象）：  
  - 支持保证金/维持保证金/强平/资金费率/手续费/滑点/精度。  
  - 验收：KPI-4 ≥ 0.8；示例策略的资金曲线可解释且可复现。  
  - 回滚：通过配置开关回退到现有 backtest。

### 6-12 周（全面超越 RD-Agent）

- 建立“同口径对比跑分”：IQFMP vs RD-Agent（同数据、同种子、同指标）自动产出报告。  
  - 验收：`scripts/bench/compare_rdagent.py`（或等价）输出 KPI 汇总 + diff。  
  - 回滚：仅为评测脚本，不影响主链路。

## 9) 本轮最小可行下一步（1-3步）：执行后显著降低不确定性

1. 运行本地审计脚本：`bash scripts/audit/run_local_audit.sh --with-tests`  
2. 若允许联网：补齐 Official Baseline Map（第 4 节命令）并写入 `docs/audit/official_baseline_map.md`  
3. 新增 1 个“合约强平”synthetic 测试用例（只依赖 pandas/numpy），把 KPI-4 的第 3 项从 0 提到 0.5+

## 10) 置信度声明：事实/推断/猜测分开写

- **事实（可直接复现验证）**：
  - IQFMP 使用 `vendor/qlib` 深改 Qlib（见第 3 节命令与路径）。  
  - `pytest -q` 通过，覆盖率满足阈值（见第 3 节）。  
  - 深改 Qlib 的代码差异点（crypto region + 24/7 日历 + crypto handler + pickle client）（见第 6 节）。  
- **推断（基于本地证据但仍需更细验证）**：
  - 当前回测的“合约真实性”主要停留在成本修正层（funding/fees/slippage），未覆盖保证金/强平/撮合；需要通过新增测试/实现验证提升到 100%。  
- **猜测（本轮不作为结论）**：
  - RD-Agent 官方版本/功能细节（需联网拉取官方证据后再下结论）。
