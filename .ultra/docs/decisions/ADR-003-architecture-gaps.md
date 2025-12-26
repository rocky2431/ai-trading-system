# ADR-003: Architecture Gap Analysis and Migration Roadmap

## Status
**PROPOSED** - Comprehensive architecture audit findings

## Date
2025-12-26

## Context

External team audit identified several architecture deviations from original spec. This ADR documents gaps that require significant refactoring and proposes a migration roadmap.

---

## 1. Backtest Engine Gap

### Current State
- `core/backtest_engine.py`: Pandas-based backtester (700+ lines)
- `core/crypto_backtest.py`: Pandas-based crypto backtester
- Custom portfolio management, signal handling, fee calculation

### Spec Requirement
- Use Qlib's `backtest` module for backtesting
- Leverage Qlib's `Executor` and `Strategy` classes
- Integration with Qlib's risk control framework

### Gap Analysis
| Component | Current | Required |
|-----------|---------|----------|
| Backtest Engine | pandas vectorized | Qlib `backtest` |
| Order Execution | Custom logic | Qlib `Executor` |
| Risk Control | Custom `RiskManager` | Qlib `RiskModel` |
| Strategy | Custom signals | Qlib `Strategy` |

### Migration Effort: HIGH
- 2-3 weeks for full Qlib backtest integration
- Requires preserving backward compatibility
- Risk: Different backtest results during transition

---

## 2. RL Module Gap

### Current State
- `rl/ppo.py`: Custom PPO implementation
- `rl/a2c.py`: Custom A2C implementation
- `rl/sac.py`: Custom SAC implementation
- Standalone implementations not integrated with Qlib RL

### Spec Requirement
- Use Qlib's RL order execution framework
- Leverage `qlib.contrib.rl` (PPO, OPDS, TWAP examples)
- Integration with Qlib's order execution environment

### Gap Analysis
| Component | Current | Required |
|-----------|---------|----------|
| RL Algorithms | Custom standalone | Qlib `contrib.rl` |
| Environment | Not integrated | Qlib `OrderExecutionEnv` |
| Training | Custom loop | Qlib RL workflow |
| Order Execution | Not connected | Qlib `Executor` |

### Migration Effort: MEDIUM-HIGH
- 1-2 weeks to adapt to Qlib RL framework
- Custom implementations may be kept for experimentation

---

## 3. Deployment Infrastructure Gap

### Current State
- Database models defined in `db/models.py`
- TimescaleDB hypertable SQL documented but not automated
- No deployment scripts for production setup

### Spec Requirement
- Automated TimescaleDB hypertable creation
- Data ingestion pipelines for OHLCV data
- Redis/Qdrant enforcement in production

### Missing Components
```
scripts/
├── init_timescale.py     # Create hypertables
├── ingest_ohlcv.py       # Data ingestion
├── setup_qdrant.py       # Vector DB setup
└── migrate_db.py         # Database migrations
```

### Migration Effort: LOW
- 1 week for deployment scripts
- Can be done incrementally

---

## 4. Qlib Alpha DataHandler Gap

### Current State
- Custom factor expressions in `qlib_factor_library.py`
- Manual Alpha158/Alpha360 expression definitions
- `use_d_features=True` now enabled (fixed)

### Spec Requirement
- Use Qlib's native Alpha158 DataHandler
- Use Qlib's Alpha360 DataHandler
- Leverage `qrun` workflow for model training

### Gap Analysis
| Component | Current | Required |
|-----------|---------|----------|
| Alpha158 | Manual expressions | Qlib `Alpha158` DataHandler |
| Alpha360 | Partial expressions | Qlib `Alpha360` DataHandler |
| Model Training | Custom pipeline | Qlib `qrun` workflow |
| Hyperparameter | Custom Optuna | Qlib hyperparameter search |

### Migration Effort: MEDIUM
- 1-2 weeks for full DataHandler integration
- Expression compatibility already verified

---

## 5. Summary of Fixes Already Applied

| Issue | Status | Fix Applied |
|-------|--------|-------------|
| LangGraph orchestration | ADR-002 | Migration plan documented |
| Security chain | ✅ FIXED | ASTSecurityChecker on all eval/exec |
| API JWT auth | ✅ FIXED | Dependencies on all protected routes |
| Rate limiting | ✅ FIXED | RateLimitMiddleware added |
| Vector DB required | ✅ FIXED | `require_vector_db=True` |
| PostgreSQL ledger | ✅ FIXED | PostgresStorage as default |
| D.features() | ✅ FIXED | `use_d_features=True` |
| Dynamic thresholds API | ✅ EXISTS | `/api/v1/metrics/thresholds` |

---

## 6. Recommended Migration Order

### Phase 1: Immediate (Week 1)
1. ✅ Security chain enforcement (DONE)
2. ✅ API security (JWT + Rate limiting) (DONE)
3. ✅ Default configuration fixes (DONE)

### Phase 2: Short-term (Weeks 2-3)
4. Deployment scripts (TimescaleDB, Qdrant, Redis)
5. Data ingestion automation
6. LangGraph migration (ADR-002)

### Phase 3: Medium-term (Weeks 4-6)
7. Qlib Alpha DataHandler integration
8. Qlib backtest module migration
9. RL module Qlib integration

### Phase 4: Long-term (Weeks 7+)
10. Full Qlib workflow (`qrun`) integration
11. Model registry and versioning
12. Production monitoring dashboard

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backtest result changes | HIGH | HIGH | Run parallel comparison tests |
| RL training differences | MEDIUM | MEDIUM | Keep custom implementations as fallback |
| Data migration issues | LOW | HIGH | Maintain CSV fallback |
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after |

---

## 8. Files Requiring Major Refactoring

### Backtest Migration
- `core/backtest_engine.py` → Qlib `backtest`
- `core/crypto_backtest.py` → Qlib crypto adapter

### RL Migration
- `rl/ppo.py` → Qlib `contrib.rl`
- `rl/a2c.py` → Qlib `contrib.rl`
- `rl/sac.py` → Qlib `contrib.rl`

### DataHandler Migration
- `evaluation/qlib_factor_library.py` → Qlib DataHandler
- `core/factor_engine.py` → Simplified with DataHandler

---

## Decision

Proceed with phased migration approach:
1. **Phase 1**: ✅ Completed in this commit
2. **Phase 2**: Priority for next sprint
3. **Phase 3-4**: Plan after Phase 2 validation

## References
- [ADR-002: LangGraph Migration](./ADR-002-langgraph-migration.md)
- [Qlib Backtest Module](https://qlib.readthedocs.io/en/latest/component/backtest.html)
- [Qlib RL Order Execution](https://qlib.readthedocs.io/en/latest/component/rl.html)
