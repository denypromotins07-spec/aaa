# =============================================================================
# NEXUS-OMEGA MASTER MAKEFILE
# Single-command orchestration for multi-billion-dollar trading infrastructure
# =============================================================================
# 
# PORT ALLOCATION (NO COLLISIONS):
# ---------------------------------
# - Rust HFT Kernel / Telemetry WebSocket: localhost:8080
# - Python Ray Cluster Head:              localhost:6379 (Redis), 8265 (Dashboard)
# - Next.js Frontend:                     localhost:3000
# - Rust FFI Python Binding:              Internal (no port)
# =============================================================================

.PHONY: all install build boot-singularity dev clean check help

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SHELL := /bin/bash
PYTHON_VENV := .venv
RUST_TARGET := x86_64-unknown-linux-gnu
FRONTEND_DIR := frontend
BACKEND_DIR := backend
CRATES_DIR := crates

# Ports (verify no collisions)
RUST_TELEMETRY_PORT := 8080
RAY_REDIS_PORT := 6379
RAY_DASHBOARD_PORT := 8265
NEXTJS_PORT := 3000

# Colors for output
COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_RED := \033[31m

# -----------------------------------------------------------------------------
# DEFAULT TARGET
# -----------------------------------------------------------------------------
all: help

help:
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)NEXUS-OMEGA Build System$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)Available targets:$(COLOR_RESET)"
	@echo "  make install           - Set up Python venv, Rust toolchain, npm deps"
	@echo "  make build             - Compile Rust workspace + Next.js frontend"
	@echo "  make boot-singularity  - Start ALL services (tmux orchestration)"
	@echo "  make dev               - Start development servers (hot reload)"
	@echo "  make check             - Run clippy + typecheck without building"
	@echo "  make clean             - Remove all build artifacts"
	@echo ""
	@echo "$(COLOR_YELLOW)Port Allocation:$(COLOR_RESET)"
	@echo "  Rust Telemetry WS:  localhost:$(RUST_TELEMETRY_PORT)"
	@echo "  Ray Redis:          localhost:$(RAY_REDIS_PORT)"
	@echo "  Ray Dashboard:      localhost:$(RAY_DASHBOARD_PORT)"
	@echo "  Next.js Frontend:   localhost:$(NEXTJS_PORT)"
	@echo ""

# -----------------------------------------------------------------------------
# INSTALL - Set up complete development environment
# -----------------------------------------------------------------------------
install: install-python install-rust install-frontend
	@echo "$(COLOR_GREEN)✓ Installation complete$(COLOR_RESET)"

install-python:
	@echo "$(COLOR_BLUE)[Python] Setting up virtual environment...$(COLOR_RESET)"
	@if [ ! -d "$(PYTHON_VENV)" ]; then \
		python3 -m venv $(PYTHON_VENV); \
	fi
	@source $(PYTHON_VENV)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r $(BACKEND_DIR)/requirements.txt && \
		echo "$(COLOR_GREEN)✓ Python dependencies installed$(COLOR_RESET)"

install-rust:
	@echo "$(COLOR_BLUE)[Rust] Checking toolchain...$(COLOR_RESET)"
	@rustup default stable 2>/dev/null || true
	@rustup component add rustfmt clippy 2>/dev/null || true
	@rustup target add $(RUST_TARGET) 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Rust toolchain ready$(COLOR_RESET)"

install-frontend:
	@echo "$(COLOR_BLUE)[Frontend] Installing npm dependencies...$(COLOR_RESET)"
	@cd $(FRONTEND_DIR) && npm install
	@echo "$(COLOR_GREEN)✓ Frontend dependencies installed$(COLOR_RESET)"

# -----------------------------------------------------------------------------
# BUILD - Compile everything in release mode
# -----------------------------------------------------------------------------
build: build-rust build-frontend
	@echo "$(COLOR_GREEN)✓ Build complete$(COLOR_RESET)"

build-rust:
	@echo "$(COLOR_BLUE)[Rust] Compiling workspace in release mode...$(COLOR_RESET)"
	@cargo build --release --workspace
	@echo "$(COLOR_GREEN)✓ Rust workspace compiled$(COLOR_RESET)"

build-frontend:
	@echo "$(COLOR_BLUE)[Frontend] Building Next.js application...$(COLOR_RESET)"
	@cd $(FRONTEND_DIR) && npm run build
	@echo "$(COLOR_GREEN)✓ Frontend built$(COLOR_RESET)"

# -----------------------------------------------------------------------------
# CHECK - Fast validation without full build
# -----------------------------------------------------------------------------
check: check-rust check-frontend

check-rust:
	@echo "$(COLOR_BLUE)[Rust] Running clippy and fmt check...$(COLOR_RESET)"
	@cargo clippy --workspace -- -D warnings
	@cargo fmt --check
	@echo "$(COLOR_GREEN)✓ Rust checks passed$(COLOR_RESET)"

check-frontend:
	@echo "$(COLOR_BLUE)[Frontend] Running ESLint and typecheck...$(COLOR_RESET)"
	@cd $(FRONTEND_DIR) && npm run lint
	@echo "$(COLOR_GREEN)✓ Frontend checks passed$(COLOR_RESET)"

# -----------------------------------------------------------------------------
# BOOT-SINGULARITY - Start all services via tmux
# -----------------------------------------------------------------------------
boot-singularity: kill-existing
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)AWAKENING NEXUS-OMEGA$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)Starting services on isolated ports...$(COLOR_RESET)"
	@echo "  - Rust HFT Kernel:    localhost:$(RUST_TELEMETRY_PORT)"
	@echo "  - Ray Cluster:        localhost:$(RAY_REDIS_PORT) / $(RAY_DASHBOARD_PORT)"
	@echo "  - Next.js Frontend:   localhost:$(NEXTJS_PORT)"
	@echo ""
	
	# Create tmux session
	@tmux new-session -d -s nexus-omega
	
	# Pane 0: Rust HFT Kernel & Telemetry WebSocket Server
	@tmux send-keys -t nexus-omega:0.0 \
		"source $(PYTHON_VENV)/bin/activate && cargo run --release -p nexus_telemetry" C-m
	@echo "$(COLOR_GREEN)✓ Rust Telemetry Server starting on :$(RUST_TELEMETRY_PORT)$(COLOR_RESET)"
	
	# Split window for Ray Cluster
	@tmux split-window -t nexus-omega:0 -h
	@tmux send-keys -t nexus-omega:0.1 \
		"source $(PYTHON_VENV)/bin/activate && ray start --head --port=$(RAY_REDIS_PORT) --dashboard-port=$(RAY_DASHBOARD_PORT) --include-dashboard=true" C-m
	@echo "$(COLOR_GREEN)✓ Ray Cluster Head starting on :$(RAY_REDIS_PORT)$(COLOR_RESET)"
	
	# Split window for Next.js Frontend
	@tmux split-window -t nexus-omega:0 -v
	@tmux send-keys -t nexus-omega:0.2 \
		"cd $(FRONTEND_DIR) && npm run dev -- -p $(NEXTJS_PORT)" C-m
	@echo "$(COLOR_GREEN)✓ Next.js Frontend starting on :$(NEXTJS_PORT)$(COLOR_RESET)"
	
	# Attach to session
	@echo ""
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)All services initializing...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Attach with: tmux attach -t nexus-omega$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)============================================$(COLOR_RESET)"
	@tmux attach -t nexus-omega

# -----------------------------------------------------------------------------
# DEV - Development mode with hot reload (alternative to boot-singularity)
# -----------------------------------------------------------------------------
dev: kill-existing
	@echo "$(COLOR_BLUE)[Dev Mode] Starting with hot reload...$(COLOR_RESET)"
	@tmux new-session -d -s nexus-dev
	@tmux send-keys -t nexus-dev:0.0 "cargo watch -x 'run -p nexus_telemetry'" C-m
	@tmux split-window -t nexus-dev:0 -h
	@tmux send-keys -t nexus-dev:0.1 "cd $(FRONTEND_DIR) && npm run dev" C-m
	@tmux attach -t nexus-dev

# -----------------------------------------------------------------------------
# CLEAN - Remove all build artifacts
# -----------------------------------------------------------------------------
clean:
	@echo "$(COLOR_BLUE)[Clean] Removing build artifacts...$(COLOR_RESET)"
	@cargo clean
	@rm -rf $(FRONTEND_DIR)/.next
	@rm -rf $(FRONTEND_DIR)/node_modules
	@rm -rf $(PYTHON_VENV)
	@rm -rf target
	@echo "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)"

# -----------------------------------------------------------------------------
# KILL-EXISTING - Terminate any running instances
# -----------------------------------------------------------------------------
kill-existing:
	@echo "$(COLOR_YELLOW)[Kill] Terminating existing processes...$(COLOR_RESET)"
	@pkill -f "nexus_telemetry" 2>/dev/null || true
	@pkill -f "ray.*start" 2>/dev/null || true
	@pkill -f "next dev" 2>/dev/null || true
	@tmux kill-session -t nexus-omega 2>/dev/null || true
	@tmux kill-session -t nexus-dev 2>/dev/null || true
	@sleep 1

# -----------------------------------------------------------------------------
# TEST - Run all tests
# -----------------------------------------------------------------------------
test: test-rust test-python

test-rust:
	@echo "$(COLOR_BLUE)[Test] Running Rust tests...$(COLOR_RESET)"
	@cargo test --workspace

test-python:
	@echo "$(COLOR_BLUE)[Test] Running Python tests...$(COLOR_RESET)"
	@source $(PYTHON_VENV)/bin/activate && pytest core_engine/

# -----------------------------------------------------------------------------
# LINT - Run all linters
# -----------------------------------------------------------------------------
lint: lint-rust lint-frontend

lint-rust:
	@cargo clippy --workspace -- -D warnings
	@cargo fmt --check

lint-frontend:
	@cd $(FRONTEND_DIR) && npm run lint

# -----------------------------------------------------------------------------
# FORMAT - Format all code
# -----------------------------------------------------------------------------
format:
	@cargo fmt --all
	@cd $(FRONTEND_DIR) && npx prettier --write "src/**/*.{ts,tsx}"
