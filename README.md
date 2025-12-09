# IQFMP - Intelligent Quantitative Factor Mining Platform

A platform for AI-driven quantitative factor discovery and trading.

## Features

- **Multi-Agent Orchestration**: LangGraph-based agent collaboration for factor discovery
- **Factor Generation**: LLM-powered factor code generation with safety checks
- **Factor Evaluation**: Multi-dimensional validation with anti-overfitting mechanisms
- **Research Ledger**: Experiment tracking with dynamic thresholds
- **Live Trading**: ccxt-based exchange integration (Phase 3+)

## Tech Stack

- **Backend**: Python 3.12, FastAPI, LangGraph
- **Database**: TimescaleDB, Redis, Qdrant
- **LLM**: OpenRouter (DeepSeek/Claude/GPT)
- **Quant**: Qlib integration

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/iqfmp/trading-system-v3.git
cd trading-system-v3

# Start development services
docker compose up -d

# Install Python dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
uvicorn iqfmp.api.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/iqfmp --cov-report=html

# Run specific test file
pytest tests/unit/test_project_structure.py -v
```

## Project Structure

```
trading-system-v3/
├── src/iqfmp/           # Main package
│   ├── agents/          # Agent implementations
│   ├── core/            # Core engines
│   ├── data/            # Data layer
│   ├── llm/             # LLM integration
│   ├── api/             # FastAPI routes
│   ├── models/          # Data models
│   └── utils/           # Utilities
├── tests/               # Test suite
├── dashboard/           # React dashboard (TBD)
├── docker-compose.yml   # Development services
└── pyproject.toml       # Project configuration
```

## Development

### Code Quality

```bash
# Run linter
ruff check src tests

# Run formatter
ruff format src tests

# Run type checker
mypy src

# Install pre-commit hooks
pre-commit install
```

## License

MIT License
