# Factor Mining 优化方案 v2.0

> **创建日期**: 2024-12-11
> **状态**: 实施中
> **优先级**: P0-P2

---

## 1. 问题分析

### 1.1 当前系统缺陷

| 问题 | 影响 | 严重程度 |
|------|------|---------|
| 无数据集选择 | 无法控制验证范围 | 高 |
| 无Walk-Forward验证 | 过拟合风险高 | **严重** |
| 基准因子库不完整 | 无法对比行业标准 | 高 |
| 模型选择单一 | 限制优化能力 | 中 |
| 无优化算法 | 无法自动调参 | 中 |
| 无IC分解诊断 | 无法定位失效原因 | 中 |

### 1.2 当前架构

```
用户请求 → MiningTaskCreateRequest → FactorService → FactorEngine → 评估
              ↓
         5个字段:
         - name
         - description
         - factor_families
         - target_count
         - auto_evaluate
```

**缺失的配置维度**:
1. 数据配置（时间范围、交易对、周期）
2. 基准配置（Alpha101/158/360、相关性阈值）
3. 模型配置（模型类型、优化算法）
4. 防过拟合配置（WF验证、动态阈值、IC衰减）

---

## 2. 优化方案

### 2.1 扩展后端 Schema

**文件**: `src/iqfmp/api/factors/schemas.py`

```python
# === 新增配置类 ===

class DataConfig(BaseModel):
    """数据集配置"""
    start_date: str = Field(default="2022-01-01", description="起始日期")
    end_date: str = Field(default="2024-12-01", description="结束日期")
    symbols: list[str] = Field(default=["BTC", "ETH"], description="交易对列表")
    timeframes: list[str] = Field(default=["4h", "1d"], description="时间周期")
    train_ratio: float = Field(default=0.6, ge=0.1, le=0.9, description="训练集比例")
    valid_ratio: float = Field(default=0.2, ge=0.0, le=0.4, description="验证集比例")
    test_ratio: float = Field(default=0.2, ge=0.1, le=0.4, description="测试集比例")


class BenchmarkConfig(BaseModel):
    """基准因子配置"""
    benchmark_set: str = Field(
        default="alpha158",
        description="基准因子库: alpha158, alpha101, alpha360, custom"
    )
    correlation_threshold: float = Field(
        default=0.70, ge=0.3, le=0.95,
        description="冗余判定阈值"
    )
    custom_factors: list[str] = Field(
        default_factory=list,
        description="自定义因子ID列表"
    )


class ModelConfig(BaseModel):
    """模型配置"""
    models: list[str] = Field(
        default=["lightgbm"],
        description="预测模型: lightgbm, xgboost, linear, catboost, ensemble"
    )
    optimization_method: str = Field(
        default="bayesian",
        description="优化算法: bayesian, genetic, grid, random, none"
    )
    max_trials: int = Field(default=100, ge=10, le=1000, description="优化轮数")
    early_stopping_rounds: int = Field(default=20, description="早停轮数")


class RobustnessConfig(BaseModel):
    """防过拟合配置"""
    # Walk-Forward 验证
    use_walk_forward: bool = Field(default=True, description="启用WF验证")
    wf_window_size: int = Field(default=252, description="训练窗口(天)")
    wf_step_size: int = Field(default=63, description="滚动步长(天)")
    wf_min_train_samples: int = Field(default=126, description="最小训练样本")

    # 动态阈值
    use_dynamic_threshold: bool = Field(default=True, description="启用动态阈值")
    min_sharpe: float = Field(default=1.5, description="最低Sharpe阈值")
    confidence_level: float = Field(default=0.95, description="置信水平")

    # IC衰减检测
    use_ic_decay_detection: bool = Field(default=True, description="启用IC衰减检测")
    max_half_life: int = Field(default=60, description="最大半衰期(天)")

    # 冗余因子过滤
    use_redundancy_filter: bool = Field(default=True, description="启用冗余过滤")
    cluster_threshold: float = Field(default=0.85, description="聚类阈值")


# === 扩展的请求类 ===

class MiningTaskCreateRequestV2(BaseModel):
    """扩展的因子挖掘任务请求"""
    # 基础配置（保持向后兼容）
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    factor_families: list[str] = Field(default_factory=list)
    target_count: int = Field(default=10, ge=1, le=100)
    auto_evaluate: bool = Field(default=True)

    # 新增配置
    data_config: DataConfig = Field(default_factory=DataConfig)
    benchmark_config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    model_config: ModelConfig = Field(default_factory=ModelConfig)
    robustness_config: RobustnessConfig = Field(default_factory=RobustnessConfig)
```

### 2.2 Walk-Forward 验证框架

**新文件**: `src/iqfmp/evaluation/walk_forward_validator.py`

```python
"""Walk-Forward 验证框架

防止过拟合的核心机制，模拟实际交易环境。
"""

from dataclasses import dataclass
from typing import Iterator, Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置"""
    initial_train_size: int = 252 * 2  # 2年初始训练
    test_size: int = 63  # ~3个月测试
    step_size: int = 21  # 每月滚动
    min_train_samples: int = 252  # 最小训练样本

    # 嵌套交叉验证
    inner_cv_folds: int = 5  # 超参优化用

    # 评估阈值
    min_oos_ic: float = 0.02  # 最小OOS IC
    max_ic_degradation: float = 0.5  # 最大IC降幅 (50%)


@dataclass
class WalkForwardSplit:
    """单个WF分割"""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    window_idx: int


@dataclass
class WalkForwardResult:
    """WF验证结果"""
    # 核心指标
    avg_train_ic: float
    avg_oos_ic: float  # 关键：样本外IC
    ic_degradation: float  # OOS降幅
    oos_ir: float  # 样本外IR

    # 分布统计
    min_oos_ic: float
    max_oos_ic: float
    oos_ic_std: float

    # 稳健性判定
    ic_consistency: float  # IC稳定性得分 (0-1)
    passes_robustness: bool  # 是否通过稳健性测试

    # 详细结果
    window_results: list[dict]


class WalkForwardValidator:
    """Walk-Forward 验证器"""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(self, n_samples: int) -> Iterator[WalkForwardSplit]:
        """生成WF分割"""
        train_end = self.config.initial_train_size
        window_idx = 0

        while train_end + self.config.test_size <= n_samples:
            test_end = train_end + self.config.test_size

            yield WalkForwardSplit(
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                window_idx=window_idx,
            )

            train_end += self.config.step_size
            window_idx += 1

    def validate(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> WalkForwardResult:
        """执行WF验证"""
        n_samples = len(factor_values)
        window_results = []

        for split in self.generate_splits(n_samples):
            # 训练集IC
            train_factor = factor_values.iloc[split.train_start:split.train_end]
            train_returns = forward_returns.iloc[split.train_start:split.train_end]
            train_ic = self._calculate_ic(train_factor, train_returns)

            # 测试集IC (OOS)
            test_factor = factor_values.iloc[split.test_start:split.test_end]
            test_returns = forward_returns.iloc[split.test_start:split.test_end]
            oos_ic = self._calculate_ic(test_factor, test_returns)

            window_results.append({
                'window': split.window_idx,
                'train_ic': train_ic,
                'oos_ic': oos_ic,
                'degradation': (train_ic - oos_ic) / abs(train_ic) if train_ic != 0 else 0,
            })

        return self._build_result(window_results)

    def _calculate_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算IC (Spearman相关系数)"""
        valid_mask = ~(factor.isna() | returns.isna())
        if valid_mask.sum() < 10:
            return 0.0
        return stats.spearmanr(factor[valid_mask], returns[valid_mask])[0]

    def _build_result(self, window_results: list[dict]) -> WalkForwardResult:
        """构建结果"""
        if not window_results:
            return WalkForwardResult(
                avg_train_ic=0, avg_oos_ic=0, ic_degradation=0,
                oos_ir=0, min_oos_ic=0, max_oos_ic=0, oos_ic_std=0,
                ic_consistency=0, passes_robustness=False, window_results=[]
            )

        train_ics = [r['train_ic'] for r in window_results]
        oos_ics = [r['oos_ic'] for r in window_results]

        avg_train_ic = np.mean(train_ics)
        avg_oos_ic = np.mean(oos_ics)
        oos_ic_std = np.std(oos_ics)

        # IC降幅
        ic_degradation = (avg_train_ic - avg_oos_ic) / abs(avg_train_ic) if avg_train_ic != 0 else 0

        # IR (IC均值/IC标准差)
        oos_ir = avg_oos_ic / oos_ic_std if oos_ic_std > 0 else 0

        # IC一致性 (正IC窗口比例)
        positive_ratio = sum(1 for ic in oos_ics if ic > 0) / len(oos_ics)
        ic_consistency = positive_ratio

        # 稳健性判定
        passes_robustness = (
            avg_oos_ic >= self.config.min_oos_ic and
            ic_degradation <= self.config.max_ic_degradation and
            ic_consistency >= 0.6  # 至少60%窗口IC为正
        )

        return WalkForwardResult(
            avg_train_ic=avg_train_ic,
            avg_oos_ic=avg_oos_ic,
            ic_degradation=ic_degradation,
            oos_ir=oos_ir,
            min_oos_ic=min(oos_ics),
            max_oos_ic=max(oos_ics),
            oos_ic_std=oos_ic_std,
            ic_consistency=ic_consistency,
            passes_robustness=passes_robustness,
            window_results=window_results,
        )
```

### 2.3 多算法优化框架

**新文件**: `src/iqfmp/models/optimizer.py`

```python
"""多算法优化框架

支持贝叶斯优化、遗传算法等自动超参调优。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler, RandomSampler


class OptimizationMethod(str, Enum):
    NONE = "none"
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class OptimizationConfig:
    method: OptimizationMethod = OptimizationMethod.BAYESIAN
    objective: str = "sharpe"  # sharpe, ic, ir, calmar
    max_trials: int = 100
    timeout: int = 3600  # 秒
    n_startup_trials: int = 20
    early_stopping_rounds: int = 20


@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    optimization_history: list[dict]


class FactorOptimizer:
    """多算法因子优化器"""

    # 默认参数空间
    DEFAULT_PARAM_SPACE = {
        'lightgbm': {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 500),
            'min_child_samples': (5, 50),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
        },
        'xgboost': {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 500),
            'min_child_weight': (1, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
        },
    }

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study: Optional[optuna.Study] = None

    def optimize(
        self,
        objective_func: Callable[[Dict], float],
        param_space: Optional[Dict] = None,
        model_type: str = "lightgbm",
    ) -> OptimizationResult:
        """执行优化"""
        if param_space is None:
            param_space = self.DEFAULT_PARAM_SPACE.get(model_type, {})

        # 选择采样器
        if self.config.method == OptimizationMethod.BAYESIAN:
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                multivariate=True,
            )
        elif self.config.method == OptimizationMethod.RANDOM:
            sampler = RandomSampler()
        else:
            sampler = TPESampler()  # 默认

        # 创建study
        self.study = optuna.create_study(
            sampler=sampler,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
            ),
        )

        # 包装目标函数
        def wrapped_objective(trial: optuna.Trial) -> float:
            params = {}
            for name, (low, high) in param_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))

            return objective_func(params)

        # 执行优化
        self.study.optimize(
            wrapped_objective,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout,
            show_progress_bar=False,
        )

        return self._build_result()

    def _build_result(self) -> OptimizationResult:
        """构建结果"""
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                })

        return OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            optimization_history=history,
        )
```

### 2.4 IC分解诊断

**新文件**: `src/iqfmp/evaluation/ic_decomposition.py`

```python
"""IC分解诊断

理解因子失效的根本原因。
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class ICDecomposition:
    """IC分解结果"""
    total_ic: float

    # 按时间分解
    ic_by_month: Dict[str, float]
    ic_by_quarter: Dict[str, float]

    # 按市值分解
    large_cap_ic: float
    mid_cap_ic: float
    small_cap_ic: float

    # 按波动率制度分解
    high_vol_ic: float
    low_vol_ic: float

    # 诊断指标
    regime_shift_detected: bool
    ic_decay_rate: float  # 每月衰减率
    predicted_half_life: int  # 预测半衰期(天)

    # 建议
    diagnosis: str
    recommendations: list[str]


class ICDecomposer:
    """IC分解器"""

    def __init__(self, market_cap_column: str = "market_cap"):
        self.market_cap_column = market_cap_column

    def decompose(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> ICDecomposition:
        """分解IC"""
        # 总IC
        total_ic = self._calc_ic(factor_values, forward_returns)

        # 按时间分解
        ic_by_month = self._decompose_by_time(factor_values, forward_returns, 'M')
        ic_by_quarter = self._decompose_by_time(factor_values, forward_returns, 'Q')

        # 按市值分解（如果有特征）
        large_cap_ic, mid_cap_ic, small_cap_ic = 0.0, 0.0, 0.0
        if features is not None and self.market_cap_column in features.columns:
            large_cap_ic, mid_cap_ic, small_cap_ic = self._decompose_by_market_cap(
                factor_values, forward_returns, features
            )

        # 按波动率制度分解
        high_vol_ic, low_vol_ic = self._decompose_by_volatility(
            factor_values, forward_returns
        )

        # 诊断
        regime_shift = self._detect_regime_shift(ic_by_month)
        decay_rate = self._calculate_decay_rate(ic_by_month)
        half_life = int(-np.log(0.5) / decay_rate) if decay_rate > 0 else 999

        # 生成诊断和建议
        diagnosis, recommendations = self._generate_diagnosis(
            total_ic, ic_by_month, regime_shift, decay_rate
        )

        return ICDecomposition(
            total_ic=total_ic,
            ic_by_month=ic_by_month,
            ic_by_quarter=ic_by_quarter,
            large_cap_ic=large_cap_ic,
            mid_cap_ic=mid_cap_ic,
            small_cap_ic=small_cap_ic,
            high_vol_ic=high_vol_ic,
            low_vol_ic=low_vol_ic,
            regime_shift_detected=regime_shift,
            ic_decay_rate=decay_rate,
            predicted_half_life=half_life,
            diagnosis=diagnosis,
            recommendations=recommendations,
        )

    def _calc_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算IC"""
        valid = ~(factor.isna() | returns.isna())
        if valid.sum() < 10:
            return 0.0
        return stats.spearmanr(factor[valid], returns[valid])[0]

    def _decompose_by_time(
        self,
        factor: pd.Series,
        returns: pd.Series,
        freq: str,
    ) -> Dict[str, float]:
        """按时间分解"""
        result = {}
        for period, group in returns.groupby(returns.index.to_period(freq)):
            ic = self._calc_ic(factor.loc[group.index], group)
            result[str(period)] = ic
        return result

    def _decompose_by_market_cap(
        self,
        factor: pd.Series,
        returns: pd.Series,
        features: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """按市值分解"""
        cap = features[self.market_cap_column]
        q70 = cap.quantile(0.7)
        q30 = cap.quantile(0.3)

        large_mask = cap >= q70
        mid_mask = (cap >= q30) & (cap < q70)
        small_mask = cap < q30

        return (
            self._calc_ic(factor[large_mask], returns[large_mask]),
            self._calc_ic(factor[mid_mask], returns[mid_mask]),
            self._calc_ic(factor[small_mask], returns[small_mask]),
        )

    def _decompose_by_volatility(
        self,
        factor: pd.Series,
        returns: pd.Series,
    ) -> tuple[float, float]:
        """按波动率分解"""
        # 使用历史波动率
        rolling_vol = returns.rolling(20).std()
        median_vol = rolling_vol.median()

        high_vol_mask = rolling_vol >= median_vol
        low_vol_mask = rolling_vol < median_vol

        return (
            self._calc_ic(factor[high_vol_mask], returns[high_vol_mask]),
            self._calc_ic(factor[low_vol_mask], returns[low_vol_mask]),
        )

    def _detect_regime_shift(self, ic_by_month: Dict[str, float]) -> bool:
        """检测制度转换（Chow检验简化版）"""
        if len(ic_by_month) < 6:
            return False

        values = list(ic_by_month.values())
        n = len(values)
        first_half = np.mean(values[:n//2])
        second_half = np.mean(values[n//2:])

        # 简单判断：两半的IC均值差异显著
        return abs(first_half - second_half) > 0.03

    def _calculate_decay_rate(self, ic_by_month: Dict[str, float]) -> float:
        """计算IC衰减率"""
        if len(ic_by_month) < 3:
            return 0.0

        values = list(ic_by_month.values())
        # 简单线性回归斜率
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        return max(0, -slope)  # 返回正的衰减率

    def _generate_diagnosis(
        self,
        total_ic: float,
        ic_by_month: Dict[str, float],
        regime_shift: bool,
        decay_rate: float,
    ) -> tuple[str, list[str]]:
        """生成诊断和建议"""
        recommendations = []

        if total_ic < 0.02:
            diagnosis = "因子预测能力弱"
            recommendations.append("考虑改进因子构造逻辑或选择其他因子家族")
        elif regime_shift:
            diagnosis = "检测到制度转换"
            recommendations.append("因子在不同市场制度下表现不一致")
            recommendations.append("考虑添加制度过滤器或动态调整因子权重")
        elif decay_rate > 0.01:
            diagnosis = "因子IC存在显著衰减"
            recommendations.append(f"预计半衰期: {int(-np.log(0.5)/decay_rate)}天")
            recommendations.append("考虑定期重新训练或添加自适应机制")
        else:
            diagnosis = "因子表现稳健"
            recommendations.append("可以进入生产环境测试")

        return diagnosis, recommendations
```

### 2.5 冗余因子检测

**新文件**: `src/iqfmp/evaluation/redundancy_detector.py`

```python
"""冗余因子检测

识别并去除高相关因子。
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class RedundancyReport:
    """冗余检测报告"""
    total_factors: int
    retained_factors: List[str]
    removed_factors: List[str]
    redundant_groups: List[Dict]
    factor_reduction_ratio: float
    correlation_matrix: pd.DataFrame


class RedundancyDetector:
    """冗余因子检测器"""

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        ic_weight: float = 0.7,
        stability_weight: float = 0.3,
    ):
        self.corr_threshold = correlation_threshold
        self.ic_weight = ic_weight
        self.stability_weight = stability_weight

    def detect(
        self,
        factors_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> RedundancyReport:
        """检测冗余因子

        Args:
            factors_df: 因子值矩阵 (index=日期, columns=因子名)
            metrics_df: 因子指标 (index=因子名, columns=[ic, ir, stability_score])

        Returns:
            冗余检测报告
        """
        # 计算相关性矩阵
        corr_matrix = factors_df.corr().abs()

        # 层次聚类
        clusters = self._cluster_factors(corr_matrix)

        # 每个簇内选择最优因子
        retained = []
        removed = []
        redundant_groups = []

        for cluster_id in np.unique(clusters):
            cluster_factors = factors_df.columns[clusters == cluster_id].tolist()

            if len(cluster_factors) == 1:
                retained.append(cluster_factors[0])
                continue

            # 按综合得分选择最优
            best_factor = self._select_best_in_cluster(cluster_factors, metrics_df)
            retained.append(best_factor)
            removed.extend([f for f in cluster_factors if f != best_factor])

            redundant_groups.append({
                'cluster_factors': cluster_factors,
                'best_factor': best_factor,
                'avg_correlation': corr_matrix.loc[
                    cluster_factors, cluster_factors
                ].values[np.triu_indices(len(cluster_factors), k=1)].mean(),
            })

        return RedundancyReport(
            total_factors=len(factors_df.columns),
            retained_factors=retained,
            removed_factors=removed,
            redundant_groups=redundant_groups,
            factor_reduction_ratio=len(removed) / len(factors_df.columns),
            correlation_matrix=corr_matrix,
        )

    def _cluster_factors(self, corr_matrix: pd.DataFrame) -> np.ndarray:
        """层次聚类"""
        # 将相关性转换为距离
        distance_matrix = 1 - corr_matrix.values
        np.fill_diagonal(distance_matrix, 0)

        # 转换为压缩形式
        condensed = squareform(distance_matrix)

        # 层次聚类
        Z = linkage(condensed, method='ward')

        # 切割树以获得聚类
        clusters = fcluster(Z, t=1 - self.corr_threshold, criterion='distance')

        return clusters

    def _select_best_in_cluster(
        self,
        factors: List[str],
        metrics_df: pd.DataFrame,
    ) -> str:
        """选择簇内最优因子"""
        scores = {}

        for factor in factors:
            if factor in metrics_df.index:
                ic = abs(metrics_df.loc[factor, 'ic']) if 'ic' in metrics_df.columns else 0
                stability = metrics_df.loc[factor, 'stability_score'] if 'stability_score' in metrics_df.columns else 0
                scores[factor] = self.ic_weight * ic + self.stability_weight * stability
            else:
                scores[factor] = 0

        return max(scores, key=scores.get)
```

---

## 3. 前端UI改造

### 3.1 FactorMiningPage 新增配置区域

**文件**: `dashboard/src/pages/FactorMiningPage.tsx`

新增以下配置 section:

```tsx
// === 数据配置 ===
<div className="space-y-4">
  <h3 className="font-semibold text-gray-900 dark:text-gray-100">数据配置</h3>

  {/* 时间范围 */}
  <div className="grid grid-cols-2 gap-4">
    <div>
      <Label>开始日期</Label>
      <Input type="date" value={startDate} onChange={...} />
    </div>
    <div>
      <Label>结束日期</Label>
      <Input type="date" value={endDate} onChange={...} />
    </div>
  </div>

  {/* 交易对选择 */}
  <div>
    <Label>交易对</Label>
    <div className="flex flex-wrap gap-2">
      {SYMBOLS.map(symbol => (
        <Button
          key={symbol}
          variant={selectedSymbols.includes(symbol) ? 'default' : 'outline'}
          onClick={() => toggleSymbol(symbol)}
        >
          {symbol}
        </Button>
      ))}
    </div>
  </div>

  {/* 时间周期 */}
  <div>
    <Label>时间周期</Label>
    <div className="flex gap-2">
      {TIMEFRAMES.map(tf => (
        <Button
          key={tf}
          variant={selectedTimeframes.includes(tf) ? 'default' : 'outline'}
          onClick={() => toggleTimeframe(tf)}
        >
          {tf}
        </Button>
      ))}
    </div>
  </div>
</div>

// === 基准配置 ===
<div className="space-y-4">
  <h3 className="font-semibold">基准因子库</h3>

  {/* 基准选择 */}
  <div className="flex gap-2">
    {['alpha158', 'alpha101', 'alpha360'].map(bench => (
      <Button
        key={bench}
        variant={benchmarkSet === bench ? 'default' : 'outline'}
        onClick={() => setBenchmarkSet(bench)}
      >
        {bench.toUpperCase()}
      </Button>
    ))}
  </div>

  {/* 相关性阈值 */}
  <div>
    <Label>相关性阈值</Label>
    <Slider
      value={correlationThreshold}
      onValueChange={setCorrelationThreshold}
      min={0.3}
      max={0.95}
      step={0.05}
    />
    <span className="text-sm text-gray-500">{correlationThreshold.toFixed(2)}</span>
  </div>
</div>

// === 模型配置 ===
<div className="space-y-4">
  <h3 className="font-semibold">模型配置</h3>

  {/* 优化算法 */}
  <div>
    <Label>优化算法</Label>
    <div className="flex gap-2">
      {['bayesian', 'genetic', 'grid', 'none'].map(method => (
        <Button
          key={method}
          variant={optimizationMethod === method ? 'default' : 'outline'}
          onClick={() => setOptimizationMethod(method)}
        >
          {method === 'bayesian' ? '贝叶斯' :
           method === 'genetic' ? '遗传算法' :
           method === 'grid' ? '网格搜索' : '不优化'}
        </Button>
      ))}
    </div>
  </div>

  {/* 优化轮数 */}
  {optimizationMethod !== 'none' && (
    <div>
      <Label>优化轮数</Label>
      <Slider
        value={maxTrials}
        onValueChange={setMaxTrials}
        min={10}
        max={500}
        step={10}
      />
      <span>{maxTrials}</span>
    </div>
  )}
</div>

// === 防过拟合配置 ===
<div className="space-y-4">
  <h3 className="font-semibold">防过拟合</h3>

  {/* Walk-Forward */}
  <div className="flex items-center justify-between">
    <div>
      <Label>Walk-Forward 验证</Label>
      <p className="text-sm text-gray-500">模拟实际交易环境，防止过拟合</p>
    </div>
    <Switch checked={useWalkForward} onCheckedChange={setUseWalkForward} />
  </div>

  {useWalkForward && (
    <div className="grid grid-cols-2 gap-4 pl-4">
      <div>
        <Label>训练窗口</Label>
        <Input type="number" value={wfWindowSize} onChange={...} />
        <span className="text-xs text-gray-500">天</span>
      </div>
      <div>
        <Label>滚动步长</Label>
        <Input type="number" value={wfStepSize} onChange={...} />
        <span className="text-xs text-gray-500">天</span>
      </div>
    </div>
  )}

  {/* 动态阈值 */}
  <div className="flex items-center justify-between">
    <div>
      <Label>动态阈值调整</Label>
      <p className="text-sm text-gray-500">根据试验次数调整Sharpe阈值</p>
    </div>
    <Switch checked={useDynamicThreshold} onCheckedChange={setUseDynamicThreshold} />
  </div>

  {/* IC衰减检测 */}
  <div className="flex items-center justify-between">
    <div>
      <Label>IC衰减检测</Label>
      <p className="text-sm text-gray-500">检测因子预测能力的时间衰减</p>
    </div>
    <Switch checked={useICDecay} onCheckedChange={setUseICDecay} />
  </div>

  {/* 冗余过滤 */}
  <div className="flex items-center justify-between">
    <div>
      <Label>冗余因子过滤</Label>
      <p className="text-sm text-gray-500">自动去除高相关冗余因子</p>
    </div>
    <Switch checked={useRedundancyFilter} onCheckedChange={setUseRedundancyFilter} />
  </div>
</div>
```

---

## 4. 实施计划

### Phase 1: 后端Schema扩展 (Day 1)

- [ ] 创建 `DataConfig`, `BenchmarkConfig`, `ModelConfig`, `RobustnessConfig`
- [ ] 扩展 `MiningTaskCreateRequest` 为 `MiningTaskCreateRequestV2`
- [ ] 更新 router 支持新参数
- [ ] 向后兼容旧API

### Phase 2: Walk-Forward验证 (Day 2-3)

- [ ] 实现 `WalkForwardValidator`
- [ ] 集成到 `FactorEvaluator`
- [ ] 添加OOS指标到评估结果
- [ ] 单元测试

### Phase 3: 多算法优化 (Day 4-5)

- [ ] 实现 `FactorOptimizer`
- [ ] 集成Optuna
- [ ] 添加遗传算法支持
- [ ] 参数空间配置

### Phase 4: IC分解与冗余检测 (Day 6)

- [ ] 实现 `ICDecomposer`
- [ ] 实现 `RedundancyDetector`
- [ ] 集成到评估流程

### Phase 5: 前端UI (Day 7-8)

- [ ] 数据配置UI
- [ ] 基准配置UI
- [ ] 模型配置UI
- [ ] 防过拟合配置UI
- [ ] 状态管理

### Phase 6: 测试与优化 (Day 9-10)

- [ ] E2E测试
- [ ] 性能优化
- [ ] 文档更新

---

## 5. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Optuna依赖冲突 | 中 | 使用可选依赖 |
| WF验证计算量大 | 高 | 添加缓存、并行化 |
| 前端状态复杂 | 中 | 使用React Hook Form |
| API不兼容 | 高 | 保持v1 API，新增v2 |

---

## 6. 成功指标

- [ ] OOS IC >= 0.02
- [ ] IC降幅 <= 50%
- [ ] 因子冗余率 <= 30%
- [ ] 优化收敛时间 < 30分钟
- [ ] 前端配置项完整度 100%
