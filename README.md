# AI Trading System

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AI-powered quantitative trading platform with autonomous factor discovery, multi-agent orchestration, and production-grade persistence.**

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

AI Trading System (IQFMP - Intelligent Quantitative Factor Mining Platform) is a comprehensive platform for automated quantitative trading research and execution. It leverages Large Language Models (LLMs) and multi-agent systems to discover, validate, and deploy trading factors with minimal human intervention.

### Key Highlights

- **Autonomous Factor Discovery**: LLM-powered hypothesis generation and factor coding
- **Multi-Agent Orchestration**: LangGraph-based collaboration between specialized agents
- **Production-Grade Persistence**: PostgreSQL/TimescaleDB with strict mode enforcement
- **Rigorous Evaluation**: Purged K-Fold CV, Walk-Forward validation, IC/IR analysis
- **Real-time Dashboard**: React 19 + TypeScript frontend with live monitoring
- **Exchange Integration**: CCXT-based cryptocurrency trading (Binance, OKX, etc.)

---

## Features

### 🤖 Multi-Agent System

| Agent | Role |
|-------|------|
| **Hypothesis Agent** | Generates trading hypotheses based on market insights |
| **Factor Generation Agent** | Translates hypotheses into executable factor code |
| **Evaluation Agent** | Validates factors with multi-dimensional metrics |
| **Risk Agent** | Assesses portfolio risk and position sizing |
| **Strategy Agent** | Combines factors into tradable strategies |
| **Backtest Agent** | Simulates historical performance |

### 📊 Factor Evaluation Pipeline

- **IC/Rank IC Analysis**: Information coefficient calculation
- **IR (Information Ratio)**: Risk-adjusted return metrics
- **Sharpe Ratio**: Portfolio performance measurement
- **Maximum Drawdown**: Risk assessment
- **Purged K-Fold CV**: Anti-overfitting cross-validation
- **Walk-Forward Validation**: Out-of-sample testing
- **Alpha158/Alpha360 Benchmark**: Qlib factor library comparison

### 🔒 Security & Sandboxing

- **RestrictedPython Sandbox**: Safe execution of LLM-generated code
- **Human Review Gate**: Manual approval for production deployment
- **AST-based Code Analysis**: Pre-execution security scanning
- **Rate Limiting**: API protection against abuse

### 💾 Data Infrastructure

- **TimescaleDB**: Time-series optimized PostgreSQL
- **Redis**: Caching and real-time data streaming
- **Qdrant**: Vector database for factor similarity search
- **Research Ledger**: Experiment tracking with dynamic thresholds

### 📈 Trading Capabilities

- **CCXT Integration**: 100+ cryptocurrency exchanges
- **Order Management**: Limit, market, and advanced order types
- **Position Management**: Real-time portfolio tracking
- **Risk Controls**: Stop-loss, take-profit, position limits
- **Emergency System**: Circuit breakers and kill switches

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI Trading System                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Dashboard  │  │   FastAPI   │  │   Celery    │  │  WebSocket  │    │
│  │  (React 19) │◄─┤   Backend   │◄─┤   Workers   │◄─┤   Server    │    │
│  └─────────────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘    │
│                          │                 │                             │
│  ┌───────────────────────┴─────────────────┴───────────────────────┐    │
│  │                    Multi-Agent Orchestrator                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │    │
│  │  │Hypothesis│ │ Factor   │ │Evaluation│ │   Risk   │ │Strategy│ │    │
│  │  │  Agent   │ │Generator │ │  Agent   │ │  Agent   │ │ Agent  │ │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │    │
│  └─────────────────────────────┬────────────────────────────────────┘    │
│                                │                                         │
│  ┌─────────────────────────────┴────────────────────────────────────┐   │
│  │                        Core Services                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │   │
│  │  │ Research │ │  Factor  │ │ Sandbox  │ │ Backtest │ │ Signal │  │   │
│  │  │  Ledger  │ │Evaluator │ │ Executor │ │  Engine  │ │Converter│  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                │                                         │
│  ┌─────────────────────────────┴────────────────────────────────────┐   │
│  │                      Data Infrastructure                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │TimescaleDB│ │  Redis   │ │  Qdrant  │ │   CCXT   │             │   │
│  │  │(PostgreSQL)│ │ (Cache)  │ │ (Vector) │ │(Exchange)│             │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
- **Python 3.12+** - Core language
- **FastAPI** - High-performance async API framework
- **LangGraph** - Multi-agent orchestration
- **LangChain** - LLM integration framework
- **Celery** - Distributed task queue

### Frontend
- **React 19** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool
- **TailwindCSS** - Utility-first CSS
- **Zustand** - State management
- **Radix UI** - Accessible components

### Database & Storage
- **PostgreSQL/TimescaleDB** - Primary database with time-series optimization
- **Redis** - Caching, sessions, and Pub/Sub
- **Qdrant** - Vector similarity search

### Quantitative
- **Qlib** - Microsoft's quantitative research platform (deep fork)
- **Pandas/NumPy** - Data manipulation
- **LightGBM** - ML-based signal generation
- **CCXT** - Cryptocurrency exchange library

### DevOps
- **Docker & Docker Compose** - Containerization
- **GitHub Actions** - CI/CD
- **Prometheus & Grafana** - Monitoring

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system

# Start infrastructure services
docker compose up -d

# Install Python dependencies
pip install -e ".[dev]"

# Install frontend dependencies
cd dashboard && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OpenRouter, Exchange credentials, etc.)

# Run database migrations
alembic upgrade head

# Start the backend
uvicorn iqfmp.api.main:app --reload --port 8000

# In another terminal, start the frontend
cd dashboard && npm run dev
```

### Environment Variables

```bash
# LLM Configuration
OPENROUTER_API_KEY=your_openrouter_key
LLM_MODEL=deepseek/deepseek-chat

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/iqfmp
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# Exchange (Optional)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret

# Security
JWT_SECRET_KEY=your_jwt_secret
RESEARCH_LEDGER_STRICT=true
```

---

## Project Structure

```
ai-trading-system/
├── src/iqfmp/                 # Main Python package
│   ├── agents/                # LangGraph agents
│   │   ├── hypothesis_agent.py
│   │   ├── factor_generation.py
│   │   ├── evaluation_agent.py
│   │   ├── risk_agent.py
│   │   ├── strategy_agent.py
│   │   └── orchestrator.py
│   ├── api/                   # FastAPI routes
│   │   ├── auth/              # Authentication
│   │   ├── factors/           # Factor management
│   │   ├── research/          # Research ledger API
│   │   ├── backtest/          # Backtesting API
│   │   └── main.py            # App entry point
│   ├── core/                  # Core business logic
│   │   ├── rd_loop.py         # Research-Development loop
│   │   ├── sandbox.py         # Code execution sandbox
│   │   ├── backtest_engine.py # Backtesting engine
│   │   └── signal_converter.py
│   ├── evaluation/            # Factor evaluation
│   │   ├── factor_evaluator.py
│   │   ├── research_ledger.py
│   │   ├── alpha_benchmark.py # Alpha158/360 benchmarks
│   │   ├── purged_cv.py       # Purged K-Fold CV
│   │   └── walk_forward_validator.py
│   ├── exchange/              # Trading integration
│   │   ├── adapter.py         # CCXT adapter
│   │   ├── execution.py       # Order execution
│   │   └── risk.py            # Risk management
│   ├── llm/                   # LLM integration
│   │   ├── provider.py        # Multi-provider support
│   │   └── prompts/           # Prompt templates
│   ├── db/                    # Database models
│   └── vector/                # Vector store
├── dashboard/                 # React frontend
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── pages/             # Route pages
│   │   ├── api/               # API clients
│   │   ├── hooks/             # Custom hooks
│   │   └── store/             # Zustand stores
│   └── package.json
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── vendor/                    # Vendored dependencies (Qlib fork)
├── docker-compose.yml         # Development services
├── pyproject.toml             # Python project config
└── README.md
```

---

## API Reference

### Authentication

```bash
# Register
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "securepassword"
}

# Login
POST /api/v1/auth/login
# Returns JWT token
```

### Factor Management

```bash
# List factors
GET /api/v1/factors

# Generate new factor
POST /api/v1/factors/generate
{
  "hypothesis": "Momentum effect in crypto markets"
}

# Evaluate factor
POST /api/v1/factors/{factor_id}/evaluate
```

### Research Ledger

```bash
# List trials
GET /api/v1/research/trials

# Get statistics
GET /api/v1/research/stats

# Get dynamic threshold
GET /api/v1/research/threshold
```

### Backtesting

```bash
# Run backtest
POST /api/v1/backtest/run
{
  "strategy_id": "uuid",
  "start_date": "2024-01-01",
  "end_date": "2024-12-01",
  "initial_capital": 100000
}
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/iqfmp --cov-report=html

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/e2e/               # End-to-end tests

# Run specific test file
pytest tests/unit/core/test_sandbox.py -v
```

### Test Coverage Requirements

- Overall coverage: ≥80%
- Critical paths: 100% (authentication, trading, risk)
- Branch coverage: ≥75%

---

## Development

### Code Quality

```bash
# Linting
ruff check src tests

# Formatting
ruff format src tests

# Type checking
mypy src

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Architecture Principles

1. **Production-Grade Persistence**: All environments use PostgreSQL/TimescaleDB
2. **No Silent Fallbacks**: Strict mode prevents MemoryStorage fallback
3. **Dependency Injection**: Components accept injected dependencies for testing
4. **SOLID Principles**: Single responsibility, dependency inversion

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit with conventional commits (`feat:`, `fix:`, `docs:`)
6. Push and create a Pull Request

---

## Roadmap

- [x] Multi-agent factor discovery
- [x] Research ledger with PostgreSQL persistence
- [x] Sandbox code execution
- [x] Alpha158/360 benchmarking
- [x] Purged K-Fold cross-validation
- [x] React dashboard
- [ ] Live trading integration
- [ ] Reinforcement learning agents
- [ ] Multi-exchange arbitrage
- [ ] Mobile application

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Qlib](https://github.com/microsoft/qlib) - Microsoft's quantitative research platform
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange library

---

<div align="center">

**Built with ❤️ for quantitative traders**

[Report Bug](https://github.com/yourusername/ai-trading-system/issues) • [Request Feature](https://github.com/yourusername/ai-trading-system/issues)

</div>
