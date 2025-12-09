# Project Constitution - IQFMP 智能量化因子挖掘平台

This document defines the core principles and standards for this project. It serves as the foundation for all development decisions.

> **Configuration Values**: All numeric thresholds and limits are defined in `.ultra/config.json`. This document provides the **qualitative principles** only.

## Project Vision

构建一个端到端的自动化量化研究平台，从因子生成到策略部署全流程自动化，超越RD-Agent的专业级加密货币量化研究系统。

## Development Principles

### 1. Specification-Driven
- Specifications are the source of truth
- Code derives from specs, not vice versa
- Changes require spec updates first

### 2. Test-First Development
- Write tests before implementation
- Coverage targets defined in config.json (overall ≥85%, critical paths 100%)
- Integration tests with real services (except external APIs)

### 3. Anti-Overfitting First
- 所有因子/策略必须通过多市场、多时间段验证
- 研究账本 (Research Ledger) 记录所有试验
- 试验次数越多，统计阈值越严格 (Deflated Sharpe Ratio)
- 稳定性分析优先于绝对收益

### 4. Minimal Abstraction
- Use frameworks directly, avoid unnecessary wrappers
- Abstraction only when pattern repeats ≥3 times
- Favor composition over inheritance

### 5. Anti-Future-Proofing
- Build only for current requirements
- No speculative features
- Refactor when new requirements emerge

### 6. Library-First
- Extract reusable logic to libraries
- Core components: Factor Engine, Strategy Engine, Execution Engine
- Internal packages for shared code

### 7. Single Source of Truth
- One canonical representation per concept
- Specs in `.ultra/specs/`
- Factor definitions in structured format
- Decisions in ADRs

### 8. Explicit Decisions
- All architecture decisions documented
- Factor selection criteria explicit and auditable
- Trade-offs acknowledged

### 9. Living Documentation
- Documentation evolves with code
- Factor library continuously updated
- Specs updated with every feature

## Quality Standards

### Code Quality
- SOLID principles enforced (see config.json for thresholds)
- DRY: No duplication (max 3 duplicate lines)
- KISS: Low complexity (max cyclomatic complexity 10)
- YAGNI: Only current requirements
- Function size: max 50 lines, nesting: max 3 levels

### Testing
- Test coverage targets: overall ≥85%, critical paths 100%
- Six-dimensional coverage required:
  1. Functional (core logic)
  2. Boundary (edge cases)
  3. Exception (error handling)
  4. Performance (load tests)
  5. Security (injection prevention)
  6. Compatibility (cross-platform)

### Quantitative Quality
- Factor IC (Information Coefficient) ≥ 0.03
- Sharpe Ratio ≥ 1.0 (after costs)
- Maximum Drawdown ≤ 20%
- Stability Score ≥ 0.6 (cross-market consistency)
- Turnover within reasonable bounds

### Frontend (Dashboard)
- Avoid default fonts (Inter/Roboto/Open Sans)
- Use design tokens/CSS variables
- Prefer established UI libraries (Ant Design recommended)
- Core Web Vitals: LCP<2.5s, INP<200ms, CLS<0.1

## Technology Constraints

### Must Use
- Python 3.12+ for all backend/research code
- Qlib as research kernel (non-Docker, direct integration)
- LangGraph + LangChain for Agent orchestration
- ccxt for exchange connectivity
- TimescaleDB for time-series data
- Redis for caching and real-time data
- Qdrant for factor vector similarity
- React + TypeScript for dashboard

### Must NOT
- No Docker isolation for Qlib (direct execution for speed)
- No single-market-only factor validation
- No untracked factor experiments

## Anti-Overfitting Controls

### 1. Multi-Dimensional Validation
- Time: Train (T0-T1) → Valid (T1-T2) → Test (T2-T3)
- Market: BTC/ETH + Altcoins + Small caps
- Frequency: 1m / 5m / 1h / 1d
- Regime: Bull/Bear × High/Low volatility

### 2. Research Ledger
- Every factor experiment logged
- Global experiment count (N) tracked
- Statistical thresholds tighten with N

### 3. Stability Requirements
- Time stability: Positive IC ratio across periods
- Market stability: Consistent across asset groups
- Environment stability: Works in different regimes

### 4. Implementability Checks
- Turnover limits enforced
- Capacity estimation required
- Transaction cost modeling mandatory

## Git Workflow

- Branch naming: `feat/task-{id}-{description}`, `fix/bug-{id}-{description}`
- Commit format: Conventional Commits
- Independent branches: Each task gets its own branch
- Immediate merge: Merge to main after task completion

## Architecture Decision Process

All significant technical decisions must:
1. Be documented as ADRs in `.ultra/docs/decisions/`
2. Include context, decision, rationale, consequences
3. Be reviewed and approved
4. Link back to requirements in specs

## Factor Family Definitions

Factors must belong to one or more families:
1. **Trend/Momentum**: MA, MACD, multi-timeframe momentum
2. **Volatility/Risk**: Realized vol, range, GARCH proxies
3. **Liquidity**: Spread, depth, volume/market cap
4. **Derivatives Structure**: Basis, funding rate, OI dynamics
5. **Cross-Asset Breadth**: Beta-adjusted returns, rotation
6. **Sentiment**: On-chain flows, social metrics (future)

---

Last Updated: 2025-12-09
