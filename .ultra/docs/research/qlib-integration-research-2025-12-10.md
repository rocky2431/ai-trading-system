# Qlib Integration Research Report

**Date**: 2025-12-10
**Mode**: Focused Technology Research (Mode 2)
**Topic**: Microsoft Qlib quantitative investment platform integration patterns

---

## Executive Summary

Microsoft Qlib is a comprehensive AI-oriented quantitative investment platform providing: (1) Alpha101/158/360 factor libraries with 158-360 pre-built technical indicators, (2) Expression engine for programmatic factor calculation, (3) Model zoo with LightGBM/XGBoost/LSTM, and (4) Complete backtest framework. For FastAPI integration, Qlib's modular design allows direct API-level usage of factor computation (`D.features()`), model training (`model.fit()`), and backtesting (`backtest_daily()`). RD-Agent uses Qlib primarily for factor evaluation and backtesting validation.

---

## 1. Qlib Components Overview

### 1.1 Architecture Diagram

```
+------------------------------------------------------------------+
|                        Qlib Platform                              |
+------------------------------------------------------------------+
|  Expression Engine  |  Data Layer  |  Model Layer  |  Backtest   |
|  - Operators        |  - D.features|  - LGBModel   |  - Strategy |
|  - Custom factors   |  - Handler   |  - XGBModel   |  - Executor |
|  - Alpha101/158/360 |  - Loader    |  - LSTM       |  - Analysis |
+------------------------------------------------------------------+
|                      qlib.init() Provider                         |
+------------------------------------------------------------------+
```

### 1.2 Factor Libraries Comparison

| Library | Features | Characteristics | Use Case |
|---------|----------|-----------------|----------|
| **Alpha360** | 360 | Raw 60-day price/volume time series | Deep learning (LSTM, Transformer) |
| **Alpha158** | 158 | Engineered technical indicators | GBDT, tabular models |
| **Alpha101** | 101 | WorldQuant formula-based factors | Research, custom strategies |

---

## 2. What Can Be Directly Used

### 2.1 Expression Engine (Factor Calculation)

**Capability**: Programmatically execute factor expressions without complex code.

```python
import qlib
from qlib.data import D

# Initialize Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# Execute factor expressions
factors = D.features(
    instruments=['SH600519', 'SZ000001'],
    fields=[
        '$close',                                    # Raw field
        'Ref($close, 5)',                           # 5-day lag
        'Mean($close, 20)',                         # 20-day moving average
        'Std($close, 20)/$close',                   # Normalized volatility
        '(Mean($close,20)+2*Std($close,20)-$close)/Mean($close,20)',  # Bollinger Band
        'Corr($close, Log($volume+1), 20)',         # Price-volume correlation
    ],
    start_time='2023-01-01',
    end_time='2024-01-01'
)
```

**Available Operators**:
- **Element**: `Abs`, `Sign`, `Log`, `Mask`, `Not`
- **Pair**: `Add`, `Sub`, `Mul`, `Div`, `Power`, `Greater`, `Less`, `And`, `Or`, `If`
- **Time-series**: `Ref`, `Mean`, `Sum`, `Std`, `Var`, `Skew`, `Kurt`, `Max`, `Min`, `Rank`, `Delta`, `Slope`, `EMA`, `WMA`
- **Cross-section**: `Corr`, `Cov`, `TResample`

### 2.2 Alpha158 Data Handler

**Complete Factor Set** (158 factors organized by category):

```python
from qlib.contrib.data.handler import Alpha158

handler_config = {
    "start_time": "2020-01-01",
    "end_time": "2024-01-01",
    "fit_start_time": "2020-01-01",
    "fit_end_time": "2022-12-31",
    "instruments": "csi300",
}

handler = Alpha158(**handler_config)

# Get all features
features = handler.fetch(col_set="feature")
labels = handler.fetch(col_set="label")
```

**Alpha158 Factor Categories**:

| Category | Factors | Description |
|----------|---------|-------------|
| **KBAR** | 9 | OHLC price ratios: `($close-$open)/$open`, `($high-$low)/$open` |
| **Price** | 25 | Multi-window price lags: `OPEN0-59`, `HIGH0-59`, `LOW0-59`, `CLOSE0-59` |
| **Rolling** | 124 | Technical indicators: ROC, MA, STD, BETA, RSQR, RESI, CORR, CORD, RSV, etc. |

**Key Rolling Factors**:
```python
# Factor expression examples from Alpha158
FACTORS = {
    'ROC5':   'Ref($close, 5)/$close - 1',           # 5-day momentum
    'MA5':    'Mean($close, 5)/$close',              # 5-day MA normalized
    'STD5':   'Std($close, 5)/$close',               # 5-day volatility
    'BETA5':  'Slope($close, 5)/$close',             # 5-day beta
    'RSQR5':  'Rsquare($close, 5)',                  # 5-day R-squared
    'RESI5':  'Resi($close, 5)/$close',              # 5-day residual
    'CORR5':  'Corr($close, Log($volume+1), 5)',     # Price-volume correlation
    'WVMA5':  'Std(Abs($close/Ref($close,1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close,1)-1)*$volume, 5)+1e-12)',
}
```

### 2.3 Alpha360 Data Handler

**Raw Time Series Features** (360 features for deep learning):

```python
from qlib.contrib.data.handler import Alpha360

handler = Alpha360(
    instruments="csi300",
    start_time="2020-01-01",
    end_time="2024-01-01",
)

# 360 features: 6 fields x 60 days
# CLOSE0-CLOSE59, OPEN0-OPEN59, HIGH0-HIGH59, LOW0-LOW59, VWAP0-VWAP59, VOLUME0-VOLUME59
```

**Feature Generation Pattern**:
```python
# Each feature is normalized by current close
fields = [f"Ref(${field}, {d})/$close" for field in ['close','open','high','low','vwap']
          for d in range(60)]
# Volume normalized separately
volume_fields = [f"Ref($volume, {d})/($volume+1e-12)" for d in range(60)]
```

### 2.4 Model Zoo

**Available Models**:

```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.model.pytorch_lstm import LSTM

# LightGBM Example
model = LGBModel(
    loss="mse",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    early_stopping_rounds=50,
)

# Train
model.fit(dataset)

# Predict
predictions = model.predict(dataset, segment="test")
```

**Model Interface**:

```python
class Model:
    def fit(self, dataset: DatasetH, **kwargs):
        """Train model on dataset"""
        pass

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """Generate predictions"""
        pass

    def finetune(self, dataset: DatasetH, num_boost_round: int = 10):
        """Continue training with new data (for LGBModel)"""
        pass
```

### 2.5 Backtest Framework

**Complete Backtest Pipeline**:

```python
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis

# Strategy configuration
strategy = TopkDropoutStrategy(
    topk=50,           # Hold top 50 stocks
    n_drop=5,          # Drop 5 worst performers daily
    signal=pred_score, # pd.Series with prediction scores
)

# Executor configuration
exec_config = {
    "time_per_step": "day",
    "generate_portfolio_metrics": True,
}

# Backtest configuration
backtest_config = {
    "start_time": "2023-01-01",
    "end_time": "2024-01-01",
    "account": 100000000,
    "benchmark": "SH000300",  # CSI300
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,  # 9.5% price limit
        "deal_price": "close",
        "open_cost": 0.0005,       # 0.05% commission
        "close_cost": 0.0015,      # 0.15% stamp duty
        "min_cost": 5,
    },
}

# Run backtest
portfolio_metrics, indicators = backtest(
    executor=executor.SimulatorExecutor(**exec_config),
    strategy=strategy,
    **backtest_config
)

# Analyze results
report, positions = portfolio_metrics.get("day")
analysis = risk_analysis(report)
```

**Simplified Backtest**:

```python
from qlib.contrib.evaluate import backtest_daily

report, positions = backtest_daily(
    start_time="2023-01-01",
    end_time="2024-01-01",
    strategy=strategy,
)
```

---

## 3. RD-Agent Integration with Qlib

### 3.1 What RD-Agent Uses

| Component | Usage | Purpose |
|-----------|-------|---------|
| **Alpha158DL** | Feature loader | Engineered features (RESI5, WVMA5, RSQR5, KLEN) |
| **StaticDataLoader** | Load precomputed factors | `combined_factors_df.parquet` |
| **Backtest Pipeline** | Validate factors/models | TopkDropoutStrategy with 50 stocks |
| **Performance Metrics** | IC, ICIR, returns | Factor quality evaluation |

### 3.2 RD-Agent Configuration Pattern

```yaml
# RD-Agent Qlib configuration
qlib_init:
  provider_uri: ~/.qlib/qlib_data/cn_data
  region: cn

dataset:
  class: DatasetH
  module_path: qlib.data.dataset
  kwargs:
    handler:
      class: DataHandlerLP
      module_path: qlib.data.dataset.handler
      kwargs:
        data_loader:
          class: NestedDataLoader
          kwargs:
            - class: Alpha158DL  # Engineered features
            - class: StaticDataLoader  # Custom factors
              kwargs:
                path: combined_factors_df.parquet
        infer_processors:
          - class: RobustZScoreNorm
            kwargs: {clip_outlier: true}
          - class: Fillna
        learn_processors:
          - class: DropnaLabel
          - class: CSZScoreNorm
```

### 3.3 Factor Evaluation Flow

```python
# RD-Agent factor evaluation pattern
def evaluate_factor(factor_expression: str, dataset: DatasetH) -> dict:
    """Evaluate a single factor using Qlib backtest"""

    # 1. Compute factor values
    factor_values = D.features(
        instruments=dataset.instruments,
        fields=[factor_expression],
        start_time=dataset.start_time,
        end_time=dataset.end_time,
    )

    # 2. Calculate IC (Information Coefficient)
    returns = D.features(
        instruments=dataset.instruments,
        fields=['Ref($close, -1)/$close - 1'],  # Next-day return
        start_time=dataset.start_time,
        end_time=dataset.end_time,
    )

    ic = factor_values.corrwith(returns, axis=0).mean()

    # 3. Run backtest
    report, _ = backtest_daily(
        start_time=dataset.test_start,
        end_time=dataset.test_end,
        strategy=TopkDropoutStrategy(topk=50, n_drop=5, signal=factor_values),
    )

    return {
        'IC': ic,
        'ICIR': ic / factor_values.corrwith(returns, axis=0).std(),
        'annual_return': report['return'].mean() * 252,
        'sharpe': report['return'].mean() / report['return'].std() * np.sqrt(252),
    }
```

---

## 4. FastAPI Integration Patterns

### 4.1 Architecture Design

```
+------------------------------------------------------------------+
|                      FastAPI Application                          |
+------------------------------------------------------------------+
|  /factors/calculate  |  /models/train  |  /backtest/run          |
+------------------------------------------------------------------+
|                       Qlib Service Layer                          |
|  - QlibFactorService  - QlibModelService  - QlibBacktestService  |
+------------------------------------------------------------------+
|                         Qlib Core                                 |
|  - D.features()       - Model.fit()       - backtest()           |
+------------------------------------------------------------------+
```

### 4.2 Factor Calculation API

```python
# services/qlib_factor_service.py
import qlib
from qlib.data import D
from typing import List, Dict, Any
import pandas as pd

class QlibFactorService:
    _initialized = False

    @classmethod
    def init(cls, provider_uri: str = '~/.qlib/qlib_data/cn_data', region: str = 'cn'):
        if not cls._initialized:
            qlib.init(provider_uri=provider_uri, region=region)
            cls._initialized = True

    @classmethod
    def calculate_factors(
        cls,
        instruments: List[str],
        expressions: List[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Calculate factor values using Qlib expression engine"""
        cls.init()
        return D.features(
            instruments=instruments,
            fields=expressions,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def get_alpha158_features(
        cls,
        instruments: str,  # e.g., 'csi300'
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Get all Alpha158 features"""
        cls.init()
        from qlib.contrib.data.handler import Alpha158

        handler = Alpha158(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )
        return handler.fetch(col_set="feature")


# api/factors.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

router = APIRouter(prefix="/factors", tags=["factors"])

class FactorRequest(BaseModel):
    instruments: List[str]
    expressions: List[str]
    start_time: str
    end_time: str

class FactorResponse(BaseModel):
    data: Dict[str, Any]
    columns: List[str]
    shape: List[int]

@router.post("/calculate", response_model=FactorResponse)
async def calculate_factors(request: FactorRequest):
    """Calculate factor values from expressions"""
    try:
        df = QlibFactorService.calculate_factors(
            instruments=request.instruments,
            expressions=request.expressions,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        return FactorResponse(
            data=df.to_dict(orient='split'),
            columns=list(df.columns),
            shape=list(df.shape),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alpha158")
async def get_alpha158(
    instruments: str = "csi300",
    start_time: str = "2023-01-01",
    end_time: str = "2024-01-01",
):
    """Get Alpha158 pre-built features"""
    df = QlibFactorService.get_alpha158_features(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )
    return {
        "data": df.head(100).to_dict(orient='records'),
        "total_rows": len(df),
        "columns": list(df.columns),
    }
```

### 4.3 Model Training API

```python
# services/qlib_model_service.py
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
import pickle
from pathlib import Path

class QlibModelService:

    @staticmethod
    def train_lightgbm(
        dataset_config: dict,
        model_config: dict,
        save_path: str,
    ) -> dict:
        """Train LightGBM model and save"""

        # Create dataset
        dataset = init_instance_by_config(dataset_config)

        # Create model
        model = LGBModel(**model_config)

        # Train
        model.fit(dataset)

        # Save model
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        # Generate test predictions
        predictions = model.predict(dataset, segment="test")

        return {
            "model_path": save_path,
            "test_samples": len(predictions),
            "prediction_range": [float(predictions.min()), float(predictions.max())],
        }

    @staticmethod
    def predict(model_path: str, dataset_config: dict) -> pd.Series:
        """Load model and generate predictions"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        dataset = init_instance_by_config(dataset_config)
        return model.predict(dataset, segment="test")


# api/models.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])

class TrainRequest(BaseModel):
    dataset_config: dict
    model_config: dict = {
        "loss": "mse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 1000,
    }

@router.post("/train/lightgbm")
async def train_lightgbm(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train LightGBM model (async)"""
    model_id = f"lgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = f"models/{model_id}.pkl"

    background_tasks.add_task(
        QlibModelService.train_lightgbm,
        dataset_config=request.dataset_config,
        model_config=request.model_config,
        save_path=save_path,
    )

    return {"model_id": model_id, "status": "training"}
```

### 4.4 Backtest API

```python
# services/qlib_backtest_service.py
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis
import pandas as pd

class QlibBacktestService:

    @staticmethod
    def run_backtest(
        predictions: pd.Series,
        start_time: str,
        end_time: str,
        topk: int = 50,
        n_drop: int = 5,
        account: float = 100000000,
        benchmark: str = "SH000300",
    ) -> dict:
        """Run backtest with TopkDropoutStrategy"""

        strategy = TopkDropoutStrategy(
            topk=topk,
            n_drop=n_drop,
            signal=predictions,
        )

        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }

        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": account,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

        portfolio_metrics, indicators = backtest(
            executor=executor.SimulatorExecutor(**executor_config),
            strategy=strategy,
            **backtest_config,
        )

        report, positions = portfolio_metrics.get("day")
        analysis = risk_analysis(report)

        return {
            "annual_return": float(analysis.get("annualized_return", {}).get("mean", 0)),
            "sharpe_ratio": float(analysis.get("information_ratio", {}).get("mean", 0)),
            "max_drawdown": float(analysis.get("max_drawdown", {}).get("mean", 0)),
            "total_trades": len(positions),
            "report": report.to_dict(orient='records')[-30:],  # Last 30 days
        }


# api/backtest.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    model_id: str
    start_time: str
    end_time: str
    topk: int = 50
    n_drop: int = 5

@router.post("/run")
async def run_backtest(request: BacktestRequest):
    """Run backtest with trained model predictions"""
    # Load predictions
    predictions = QlibModelService.predict(
        model_path=f"models/{request.model_id}.pkl",
        dataset_config={...},
    )

    result = QlibBacktestService.run_backtest(
        predictions=predictions,
        start_time=request.start_time,
        end_time=request.end_time,
        topk=request.topk,
        n_drop=request.n_drop,
    )

    return result
```

### 4.5 Complete FastAPI Application

```python
# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import qlib

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Qlib on startup
    qlib.init(
        provider_uri='~/.qlib/qlib_data/cn_data',
        region='cn',
    )
    yield
    # Cleanup on shutdown

app = FastAPI(
    title="Qlib Trading API",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
from api import factors, models, backtest
app.include_router(factors.router)
app.include_router(models.router)
app.include_router(backtest.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "qlib_initialized": True}
```

---

## 5. Key API Reference

### 5.1 Data API

| API | Purpose | Example |
|-----|---------|---------|
| `qlib.init()` | Initialize provider | `qlib.init(provider_uri='~/.qlib', region='cn')` |
| `D.features()` | Calculate features | `D.features(instruments, fields, start, end)` |
| `D.instruments()` | Get instrument list | `D.instruments('csi300')` |
| `D.calendar()` | Get trading calendar | `D.calendar(start, end, freq='day')` |

### 5.2 Handler API

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Alpha158` | 158 engineered features | `fetch(col_set='feature')` |
| `Alpha360` | 360 raw time series | `fetch(col_set='feature')` |
| `DataHandlerLP` | Custom handlers | `fetch()`, `get_cols()` |

### 5.3 Model API

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `LGBModel` | LightGBM wrapper | `fit()`, `predict()`, `finetune()` |
| `XGBModel` | XGBoost wrapper | `fit()`, `predict()` |
| `LSTM` | PyTorch LSTM | `fit()`, `predict()` |

### 5.4 Backtest API

| Function/Class | Purpose | Key Parameters |
|----------------|---------|----------------|
| `backtest()` | Full backtest | `executor`, `strategy`, `start_time`, `end_time` |
| `backtest_daily()` | Simple backtest | `strategy`, `start_time`, `end_time` |
| `TopkDropoutStrategy` | Stock selection | `topk`, `n_drop`, `signal` |
| `SimulatorExecutor` | Order execution | `time_per_step`, `generate_portfolio_metrics` |

---

## 6. Risk Assessment

### 6.1 Critical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data Dependency** | Qlib requires specific data format | Pre-process data to Qlib format or use StaticDataLoader |
| **Memory Usage** | Large datasets consume memory | Use streaming or batch processing |
| **Thread Safety** | `qlib.init()` is not thread-safe | Initialize once at startup |

### 6.2 Integration Challenges

| Challenge | Solution |
|-----------|----------|
| Custom data sources | Use `StaticDataLoader` with parquet files |
| Real-time factor calculation | Cache expressions, use incremental updates |
| Async compatibility | Run Qlib in ThreadPoolExecutor |

---

## 7. Recommendation

### Primary: Direct Qlib Integration

**Confidence**: High

**Rationale**:
1. Expression engine provides flexible factor calculation without complex code
2. Alpha158/360 handlers offer pre-built, battle-tested factors
3. Model zoo and backtest framework enable end-to-end workflow

### Implementation Steps

1. **Install Qlib**: `pip install pyqlib`
2. **Download data**: `python -m qlib.run.get_data qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data`
3. **Create service layer**: Wrap Qlib APIs in service classes
4. **Build FastAPI endpoints**: Expose factor/model/backtest APIs
5. **Add caching**: Cache frequent factor calculations

### Expected Outcomes

- **Factor calculation**: 158-360 features in <1s per batch
- **Model training**: LightGBM training <5 min for 10 years data
- **Backtest**: Daily strategy evaluation <30s
- **Development time**: 2-3 days for basic integration

---

## Sources

- [Microsoft Qlib GitHub](https://github.com/microsoft/qlib)
- [Qlib Documentation - Data Layer](https://qlib.readthedocs.io/en/latest/component/data.html)
- [Qlib Documentation - Model](https://qlib.readthedocs.io/en/latest/component/model.html)
- [Qlib Documentation - Strategy](https://qlib.readthedocs.io/en/latest/component/strategy.html)
- [Qlib API Reference](https://qlib.readthedocs.io/en/latest/reference/api.html)
- [RD-Agent Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)
- [Qlib Alpha158 Handler Source](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py)
- [Qlib Data Loader Source](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
- [Qlib LGBModel Source](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/gbdt.py)
