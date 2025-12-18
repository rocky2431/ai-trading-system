#!/bin/bash
# IQFMP Development Environment Stop Script
# Usage: ./scripts/stop_dev.sh [--infra|--all]

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

MODE="${1:---all}"

cd "$PROJECT_ROOT"

# Stop frontend
stop_frontend() {
    if [ -f ".frontend.pid" ]; then
        PID=$(cat .frontend.pid)
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log_success "Frontend stopped (PID: $PID)"
        fi
        rm -f .frontend.pid
    else
        # Try to find and kill npm dev process
        pkill -f "vite.*dashboard" 2>/dev/null || true
        log_info "Frontend process killed (if running)"
    fi
}

# Stop backend
stop_backend() {
    if [ -f ".backend.pid" ]; then
        PID=$(cat .backend.pid)
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log_success "Backend stopped (PID: $PID)"
        fi
        rm -f .backend.pid
    else
        # Try to find and kill uvicorn process
        pkill -f "uvicorn.*iqfmp" 2>/dev/null || true
        log_info "Backend process killed (if running)"
    fi
}

# Stop Celery worker (C2 FIX)
stop_celery() {
    if [ -f ".celery.pid" ]; then
        PID=$(cat .celery.pid)
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log_success "Celery worker stopped (PID: $PID)"
        fi
        rm -f .celery.pid
    else
        # Try to find and kill celery process
        pkill -f "celery.*iqfmp" 2>/dev/null || true
        log_info "Celery process killed (if running)"
    fi
}

# Stop infrastructure
stop_infrastructure() {
    log_info "Stopping infrastructure services..."
    docker compose -f docker-compose.dev.yml down
    log_success "Infrastructure stopped"
}

case "$MODE" in
    --infra)
        stop_infrastructure
        ;;
    --all|*)
        log_info "Stopping all IQFMP services..."
        stop_frontend
        stop_backend
        stop_infrastructure
        echo ""
        log_success "All services stopped"
        ;;
esac
