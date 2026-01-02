# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Start infrastructure (TimescaleDB, Redis, Qdrant)
docker compose -f docker-compose.dev.yml up -d

# Start backend (port 8000)
uvicorn iqfmp.api.main:app --reload --port 8000

# Start frontend (port 5173)
cd dashboard && npm run dev

# Or use the convenience script
./scripts/start_dev.sh
```

### Testing
```bash
# Run all tests (requires 80% coverage)
pytest

# Run specific test file
pytest tests/unit/exchange/test_order_execution.py -v

# Run tests by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only

# Skip coverage requirement for quick iteration
pytest --no-cov tests/unit/core/test_sandbox.py
```

### Code Quality
```bash
ruff check src tests    # Lint
ruff format src tests   # Format
mypy src                # Type check
```

### Database
```bash
alembic upgrade head     # Run migrations
alembic revision -m "description"  # Create migration
```

## Architecture

### Backend (src/iqfmp/)

**Multi-Agent System** (`agents/`): LangGraph-based orchestrator with specialized agents (Hypothesis, FactorGeneration, Evaluation, Risk, Strategy, Backtest). The `orchestrator.py` implements a StateGraph pattern with checkpoint persistence to PostgreSQL.

**Factor Evaluation Pipeline** (`evaluation/`): Research trials flow through `research_ledger.py` for persistence, `factor_evaluator.py` for IC/IR metrics, `purged_cv.py` for anti-overfitting CV, and `walk_forward_validator.py` for out-of-sample testing.

**Exchange Integration** (`exchange/`): CCXT-based adapters in `adapter.py`, order execution with idempotency in `execution.py`, margin calculations in `margin.py`, and risk controls in `risk.py`. Critical state (idempotency cache, partial fills) uses Redis for persistence.

**Code Sandbox** (`core/sandbox.py`): RestrictedPython-based execution of LLM-generated factor code with AST analysis for security scanning.

**Three-Layer Security** (`core/`):
1. `security.py` - AST-based static analysis (pre-execution)
2. `sandbox.py` - RestrictedPython runtime isolation
3. `review.py` - Human review gate for production deployment

### Frontend (dashboard/)

React 19 + TypeScript + Vite. State management via Zustand stores (`store/`). API client in `api/client.ts` with Bearer token auth and automatic 401 token refresh.

### Key Patterns

1. **Decimal for Financial Data**: All price/quantity/amount fields use `Decimal`, not `float`. API schemas use `DecimalStr` type with proper JSON serialization.

2. **Async Database Sessions**: Use `get_async_session()` for FastAPI routes, `sync_session()` context manager for Celery tasks.

3. **Critical State Persistence**: Per architecture rules, critical state (idempotency, checkpoints, research trials, partial fills) must be persisted to PostgreSQL/Redis, never in-memory only.

4. **Pydantic Schemas**: All API models in `*/schemas.py` files. SQLAlchemy ORM models separate in `db/models.py`.

5. **LLM Integration** (`llm/`):
   - `provider.py` - OpenRouter API with model registry, fallback chains, and auto-continue
   - `cache.py` - Two-tier caching (Redis + PostgreSQL) for prompt deduplication
   - `retry.py` - Exponential backoff with error classification
   - `validation/` - JSON schema validation for structured outputs

6. **Vector Store** (`vector/`): Qdrant-based similarity search for factor deduplication. `SimilaritySearcher` prevents regenerating similar factors.

7. **WebSocket Updates** (`api/system/websocket.py`): Real-time broadcasts for agent updates, task progress, factor creation, and evaluation completion.

## Environment

Required `.env` variables:
- `OPENROUTER_API_KEY` - LLM provider
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` - Database (default port 5433)
- `REDIS_HOST`, `REDIS_PORT` - Cache
- `QDRANT_URL` - Vector database (optional, defaults to localhost:6333)

Optional:
- `RESEARCH_LEDGER_STRICT=true` - Prevent MemoryStorage fallback in production
- `BINANCE_API_KEY`, `BINANCE_SECRET_KEY` - Exchange credentials

## Conventions

- Python 3.12+ with strict mypy
- Conventional commits (`feat:`, `fix:`, `refactor:`)
- Tests required for new functionality
- No TODO/FIXME in production code
- Use `logger` instead of `print()` for all logging
- Validate at construction time (dataclass `__post_init__`, Pydantic `model_validator`)
