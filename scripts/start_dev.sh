#!/bin/bash
# IQFMP Development Environment Startup Script
# Usage: ./scripts/start_dev.sh [--infra-only|--backend-only|--all]
#
# Options:
#   --infra-only    Start only infrastructure (DB, Redis, Qdrant)
#   --backend-only  Start only backend (assumes infra is running)
#   --all           Start everything including frontend (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default mode
MODE="${1:---all}"

cd "$PROJECT_ROOT"

# ==================== Phase 1: Infrastructure ====================
start_infrastructure() {
    log_info "Starting infrastructure services (TimescaleDB + Redis + Qdrant)..."

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Start containers
    docker compose -f docker-compose.dev.yml up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."

    # Wait for TimescaleDB
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker exec iqfmp-timescaledb-dev pg_isready -U iqfmp > /dev/null 2>&1; then
            log_success "TimescaleDB is ready"
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_error "TimescaleDB failed to start within ${max_attempts}s"
        exit 1
    fi

    # Wait for Redis
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker exec iqfmp-redis-dev redis-cli ping > /dev/null 2>&1; then
            log_success "Redis is ready"
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done

    # Wait for Qdrant
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            log_success "Qdrant is ready"
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Qdrant health check timed out, but continuing..."
    fi

    log_success "All infrastructure services are running"
}

# ==================== Phase 2: Database Init ====================
init_database() {
    log_info "Initializing database..."

    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        log_error "Virtual environment not found. Run: python3 -m venv .venv && pip install -e ."
        exit 1
    fi

    # Set environment variables
    export DATABASE_URL="postgresql://iqfmp:iqfmp@localhost:5433/iqfmp"
    export REDIS_URL="redis://localhost:6379/0"
    export QDRANT_HOST="localhost"
    export QDRANT_PORT="6333"
    export PYTHONPATH="src:vendor/qlib"

    # Run database initialization
    if [ -f "scripts/init_db.py" ]; then
        python scripts/init_db.py
        log_success "Database initialized"
    else
        log_warn "init_db.py not found, skipping database initialization"
    fi
}

# ==================== Phase 2.5: Qlib Verification ====================
verify_qlib() {
    log_info "Verifying Qlib backtest engine availability..."

    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    # Set PYTHONPATH
    export PYTHONPATH="src:vendor/qlib:$PYTHONPATH"

    # Check Qlib availability
    QLIB_CHECK=$(python3 -c "
try:
    from iqfmp.agents.backtest_agent import QLIB_AVAILABLE
    print('AVAILABLE' if QLIB_AVAILABLE else 'UNAVAILABLE')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

    if [ "$QLIB_CHECK" = "AVAILABLE" ]; then
        log_success "Qlib backtest engine: READY"
    elif [[ "$QLIB_CHECK" == ERROR:* ]]; then
        log_error "Qlib check failed: ${QLIB_CHECK#ERROR:}"
        log_info "Attempting to install missing Qlib dependencies..."
        pip install gym cvxpy
        # Retry check
        QLIB_RETRY=$(python3 -c "
try:
    from iqfmp.agents.backtest_agent import QLIB_AVAILABLE
    print('AVAILABLE' if QLIB_AVAILABLE else 'UNAVAILABLE')
except:
    print('UNAVAILABLE')
" 2>&1)
        if [ "$QLIB_RETRY" = "AVAILABLE" ]; then
            log_success "Qlib backtest engine: READY (after installing dependencies)"
        else
            log_error "CRITICAL: Qlib is REQUIRED for backtesting but unavailable!"
            log_error "Please run: pip install gym cvxpy"
            exit 1
        fi
    else
        log_error "CRITICAL: Qlib is REQUIRED for backtesting but unavailable!"
        log_error "Please ensure vendor/qlib is in PYTHONPATH and dependencies are installed"
        log_error "Run: pip install gym cvxpy"
        exit 1
    fi
}

# ==================== Phase 3: Backend ====================
start_backend() {
    log_info "Starting backend API server..."

    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        log_error "Virtual environment not found. Run: python3 -m venv .venv && pip install -e ."
        exit 1
    fi

    # Set environment variables
    export DATABASE_URL="postgresql://iqfmp:iqfmp@localhost:5433/iqfmp"
    # Also set individual PG* vars for psycopg2 compatibility
    export PGHOST="localhost"
    export PGPORT="5433"
    export PGUSER="iqfmp"
    export PGPASSWORD="iqfmp"
    export PGDATABASE="iqfmp"
    export REDIS_URL="redis://localhost:6379/0"
    export QDRANT_HOST="localhost"
    export QDRANT_PORT="6333"
    export PYTHONPATH="src:vendor/qlib:$PYTHONPATH"

    # M2 FIX: Qlib initialization environment
    export QLIB_AUTO_INIT="true"
    export QLIB_DATA_DIR="$HOME/.qlib/qlib_data"

    # Start uvicorn in background
    log_info "Backend will start at http://localhost:8000"
    log_info "API docs available at http://localhost:8000/docs"

    python -m uvicorn iqfmp.api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid

    # Wait for backend to be ready
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend is ready"
            return
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    log_warn "Backend health check timed out, check logs for errors"
}

# ==================== Phase 3.5: Celery Worker (C2 Fix) ====================
start_celery_worker() {
    log_info "Starting Celery worker for task queue..."

    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        log_error "Virtual environment not found. Run: python3 -m venv .venv && pip install -e ."
        exit 1
    fi

    # Set environment variables (same as backend)
    export DATABASE_URL="postgresql://iqfmp:iqfmp@localhost:5433/iqfmp"
    export REDIS_URL="redis://localhost:6379/0"
    export QDRANT_HOST="localhost"
    export QDRANT_PORT="6333"
    export PYTHONPATH="src:vendor/qlib:$PYTHONPATH"
    export QLIB_AUTO_INIT="true"
    export QLIB_DATA_DIR="$HOME/.qlib/qlib_data"

    # Start Celery worker in background with all queues
    log_info "Celery worker processing queues: high, default, low"
    celery -A iqfmp.celery_app.app worker -l info -Q high,default,low &
    CELERY_PID=$!
    echo $CELERY_PID > .celery.pid

    # Wait for Celery to be ready
    sleep 3
    if kill -0 "$CELERY_PID" 2>/dev/null; then
        log_success "Celery worker started (PID: $CELERY_PID)"
    else
        log_warn "Celery worker may have failed to start, check logs"
    fi
}

# ==================== Phase 4: Frontend ====================
start_frontend() {
    log_info "Starting frontend development server..."

    if [ ! -d "dashboard" ]; then
        log_warn "Frontend directory 'dashboard' not found, skipping"
        return
    fi

    cd dashboard

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi

    log_info "Frontend will start at http://localhost:5173"
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend.pid

    cd ..
    log_success "Frontend started"
}

# ==================== Main ====================
case "$MODE" in
    --infra-only)
        log_info "Starting infrastructure only..."
        start_infrastructure
        ;;
    --backend-only)
        log_info "Starting backend only (assumes infrastructure is running)..."
        init_database
        start_backend
        ;;
    --all|*)
        log_info "Starting complete IQFMP development environment..."
        echo ""
        echo "┌────────────────────────────────────────────────────────────┐"
        echo "│  IQFMP - Intelligent Quantitative Factor Mining Platform  │"
        echo "│  Development Environment                                   │"
        echo "└────────────────────────────────────────────────────────────┘"
        echo ""

        start_infrastructure
        echo ""
        init_database
        echo ""
        verify_qlib
        echo ""
        start_backend
        echo ""
        start_celery_worker  # C2 FIX: Start Celery worker for persistent task queue
        echo ""
        start_frontend

        echo ""
        echo "┌────────────────────────────────────────────────────────────┐"
        echo "│  All services started!                                     │"
        echo "│                                                            │"
        echo "│  Backend API:  http://localhost:8000                       │"
        echo "│  API Docs:     http://localhost:8000/docs                  │"
        echo "│  Frontend:     http://localhost:5173                       │"
        echo "│  Celery:       Worker processing tasks                     │"
        echo "│  TimescaleDB:  localhost:5433                              │"
        echo "│  Redis:        localhost:6379                              │"
        echo "│  Qdrant:       localhost:6333                              │"
        echo "│                                                            │"
        echo "│  To stop: ./scripts/stop_dev.sh                            │"
        echo "└────────────────────────────────────────────────────────────┘"
        ;;
esac
