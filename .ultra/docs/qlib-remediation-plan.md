# IQFMP Qlib 架构整改计划

> **目标**: 将系统从 45/100 提升到 95/100 的 Qlib 合规性
> **原则**: Qlib 是唯一计算核心，其他都是增强插件
> **创建日期**: 2025-12-20

---

## 一、整改范围总览

### 1.1 需要修改的文件清单

| 优先级 | 文件 | 问题 | 预计工作量 |
|--------|------|------|-----------|
| **P0** | `qlib_stats.py` | 需要扩展统计函数 | 2小时 |
| **P1** | `alpha101.py` | scipy.stats.rankdata | 3小时 |
| **P1** | `alpha158.py` | scipy.stats.percentileofscore | 3小时 |
| **P1** | `alpha360.py` | scipy.stats.percentileofscore | 3小时 |
| **P1** | `sandbox.py` | 允许scipy/numpy/pandas | 1小时 |
| **P1** | `api/factors/service.py` | 直接使用scipy | 2小时 |
| **P2** | `risk_agent.py` | RiskCalculator独立类 | 2小时 |
| **P2** | `backtest_agent.py` | 手动指标计算 | 2小时 |
| **P2** | `redundancy_detector.py` | scipy聚类 | 3小时 |
| **P2** | `ic_decomposition.py` | scipy统计 | 2小时 |
| **P2** | `stability_analyzer.py` | scipy.linregress | 1小时 |
| **P2** | `alpha_benchmark.py` | scipy.spearmanr | 2小时 |
| **P3** | `factor_engine.py` | 本地操作符实现 | 4小时 |
| **P3** | `qlib_crypto.py` | from_exchange直接CCXT | 1小时 |
| **P3** | `factor_combiner.py` | scipy.stats | 1小时 |
| **P3** | `strategy/generator.py` | scipy.optimize | 1小时 |

**总计**: ~33小时工作量

---

## 二、Phase 0: 扩展 qlib_stats.py (基础设施)

### 2.1 需要添加的函数

```python
# 当前已有:
- normal_ppf(p) -> float
- normal_cdf(x) -> float
- QlibStatisticalEngine.calculate_mean/std/correlation/sharpe_ratio/max_drawdown

# 需要添加:
- spearman_correlation(x, y) -> float  # 替换 scipy.stats.spearmanr
- rank_percentile(series, window) -> Series  # 替换 scipy.stats.percentileofscore
- ts_rank(series, window) -> Series  # 替换 scipy.stats.rankdata
- linear_regression(x, y) -> tuple  # 替换 scipy.stats.linregress
- t_test_ind(a, b) -> tuple  # 替换 scipy.stats.ttest_ind
- hierarchical_cluster(distance_matrix, method) -> array  # 替换 scipy.cluster
- var_percentile(returns, confidence) -> float  # VaR计算
- expected_shortfall(returns, confidence) -> float  # ES计算
```

### 2.2 设计原则

1. **接口与Qlib一致**: 参数名和返回值与Qlib表达式兼容
2. **无scipy依赖**: 使用纯numpy/pandas或数学公式实现
3. **向Qlib迁移路径**: 当Qlib提供相应功能时可直接替换

---

## 三、Phase 1: CRITICAL 修复 (因子库 + 沙箱)

### 3.1 alpha101/158/360.py 重构策略

**当前实现**:
```python
from scipy.stats import rankdata

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_pct(x):
        ranked = rankdata(x, nan_policy='omit')
        return ranked[-1] / len(x[~np.isnan(x)])
    return series.rolling(window).apply(rank_pct, raw=True)
```

**目标实现**:
```python
from iqfmp.evaluation.qlib_stats import ts_rank, rank_percentile

# ts_rank 和 rank_percentile 由 qlib_stats.py 统一提供
# 内部使用 pandas rank() 方法，与 Qlib 的 Rank 操作符行为一致
```

### 3.2 sandbox.py 修复

**当前问题**:
```python
allowed_modules = ["pandas", "numpy", "scipy", ...]  # 允许绕过Qlib
```

**目标**:
```python
allowed_modules = [
    "qlib",  # 只允许Qlib
    "iqfmp.evaluation.qlib_stats",  # 统一统计接口
    "iqfmp.evaluation.qlib_factor_library",  # Qlib因子库
]

# 为 pandas/numpy 提供受限访问
restricted_globals = {
    "pd": PandasRestricted(),  # 只读操作
    "np": NumpyRestricted(),   # 只允许数组操作，禁止统计
}
```

### 3.3 api/factors/service.py 修复

替换所有 `from scipy import stats` 为 `from iqfmp.evaluation.qlib_stats import ...`

---

## 四、Phase 2: HIGH 修复 (Agent + 评估模块)

### 4.1 risk_agent.py 重构

**删除**:
```python
class RiskCalculator:  # 删除整个类 (~120行)
    def calculate_var(...)
    def calculate_expected_shortfall(...)
    def calculate_drawdown(...)
```

**替换为**:
```python
from qlib.contrib.evaluate import risk_analysis
from iqfmp.evaluation.qlib_stats import QlibRiskAnalyzer

class RiskCheckAgent:
    def __init__(self):
        self.risk_analyzer = QlibRiskAnalyzer()

    def _calculate_metrics(self, returns: pd.Series) -> RiskMetrics:
        metrics = self.risk_analyzer.analyze(returns)
        return RiskMetrics(
            var_95=metrics.var_95,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
        )
```

### 4.2 backtest_agent.py 重构

**删除**:
```python
def _calculate_metrics_manually(self, ...):  # 删除 (~60行)
    sharpe = mean_return / std_return * np.sqrt(...)
    ...
```

**替换为**:
```python
from iqfmp.evaluation.qlib_stats import QlibRiskAnalyzer

def _calculate_metrics(self, returns: pd.Series) -> BacktestMetrics:
    analyzer = QlibRiskAnalyzer()
    metrics = analyzer.analyze(returns)
    return BacktestMetrics.from_qlib_metrics(metrics)
```

### 4.3 redundancy_detector.py 重构

**删除**:
```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
```

**替换为**:
```python
from iqfmp.evaluation.qlib_stats import hierarchical_cluster, spearman_correlation

# 使用 qlib_stats 提供的聚类实现
def _cluster_factors(self, correlation_matrix):
    return hierarchical_cluster(
        1 - correlation_matrix,  # 距离矩阵
        method=self.config.linkage_method,
        threshold=self.config.correlation_threshold,
    )
```

### 4.4 ic_decomposition.py / stability_analyzer.py 重构

**删除**:
```python
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
```

**替换为**:
```python
from iqfmp.evaluation.qlib_stats import linear_regression

slope, intercept, r_value, p_value, std_err = linear_regression(x, y)
```

---

## 五、Phase 3: MEDIUM 修复 (引擎层)

### 5.1 factor_engine.py 重构

**删除本地操作符** (611-726行):
```python
def _op_mean(self, df, inner, window): ...  # 删除
def _op_std(self, df, inner, window): ...   # 删除
def _op_rsi(self, df, inner, window): ...   # 删除
def _op_macd(self, ...): ...                # 删除
```

**强制使用Qlib**:
```python
def compute_factor(self, expression: str) -> pd.Series:
    if not self._qlib_initialized:
        raise RuntimeError("Qlib必须初始化才能计算因子")

    # 使用 Qlib 表达式引擎
    from qlib.data import D
    return D.features(expressions=[expression])[expression]
```

### 5.2 qlib_crypto.py 修复

**删除**:
```python
@staticmethod
def from_exchange(exchange, symbol, ...):  # 删除整个方法
    import ccxt
    ...
```

**保留**: 只通过 `CryptoDataHandler` 加载已下载的数据

---

## 六、验收标准

### 6.1 代码检查

```bash
# 整改后应该返回 0 结果
grep -r "from scipy" src/iqfmp/ --include="*.py" | wc -l  # 应为 0
grep -r "import scipy" src/iqfmp/ --include="*.py" | wc -l  # 应为 0
```

### 6.2 测试覆盖

```bash
# 所有测试应该通过
pytest tests/ -v

# 覆盖率应该 >= 80%
pytest tests/ --cov=src/iqfmp --cov-report=term-missing
```

### 6.3 Qlib 集成度检查

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| scipy 导入数 | 0 | grep 检查 |
| 本地统计函数 | 0 | AST 分析 |
| Qlib 调用率 | 100% | 代码审查 |
| 测试通过率 | 100% | pytest |

---

## 七、风险评估

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 数值精度差异 | 中 | 低 | 添加精度测试对比 |
| 性能下降 | 低 | 中 | 性能基准测试 |
| Qlib API 变更 | 低 | 高 | 版本锁定 |

### 7.2 回滚计划

每个 Phase 完成后提交 git，如有问题可回滚:
```bash
git revert HEAD  # 回滚最后一次提交
```

---

## 八、执行计划

### 8.1 Phase 0 (今天)
- [ ] 扩展 qlib_stats.py
- [ ] 添加单元测试

### 8.2 Phase 1 (今天-明天)
- [ ] 重构 alpha101.py
- [ ] 重构 alpha158.py
- [ ] 重构 alpha360.py
- [ ] 修复 sandbox.py
- [ ] 修复 api/factors/service.py

### 8.3 Phase 2 (明天-后天)
- [ ] 重构 risk_agent.py
- [ ] 重构 backtest_agent.py
- [ ] 重构 redundancy_detector.py
- [ ] 重构 ic_decomposition.py
- [ ] 重构 stability_analyzer.py
- [ ] 重构 alpha_benchmark.py

### 8.4 Phase 3 (后续)
- [ ] 重构 factor_engine.py
- [ ] 修复 qlib_crypto.py
- [ ] 重构 factor_combiner.py
- [ ] 重构 strategy/generator.py

---

**状态**: 待执行
**负责人**: Claude Code
**开始时间**: 2025-12-20
