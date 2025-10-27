# Hermes Backend Makefile

# Use 'venv' as the virtual environment directory name for consistency.
VENV_DIR=venv

.PHONY: help install dev dev-ngrok test lint clean

help: ## Show this help message
	@echo "Hermes Backend - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed. Please install it from https://astral.sh/uv/install.sh"; exit 1; }
	uv venv $(VENV_DIR)
	uv sync

dev: ## Start development server (local only)
	uv run python run.py

dev-ngrok: ## Start development server with ngrok tunnel (for Prism testing)
	./scripts/development/dev_with_ngrok.sh

ngrok: ## Start only ngrok tunnel (server must be running separately)
	./scripts/development/start_ngrok.sh

ngrok-status: ## Show ngrok tunnel status
	@uv run python scripts/development/get_ngrok_url.py || echo "ngrok not running"

ngrok-url: ## Get ngrok URL and print export statements
	@uv run python scripts/development/get_ngrok_url.py --env

ngrok-update: ## Update .env.local with current ngrok URL
	@uv run python scripts/development/get_ngrok_url.py --update

test: ## Run tests
	uv run pytest

lint: ## Run linters
	uv run flake8 app/

clean: ## Clean up temporary files and the virtual environment
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f .env.local
	rm -f /tmp/ngrok.log
	rm -rf $(VENV_DIR)

install-ngrok: ## Install ngrok (macOS only)
	@command -v brew >/dev/null 2>&1 || { echo "Error: Homebrew not installed. Install from https://brew.sh"; exit 1; }
	brew install ngrok/ngrok/ngrok
	@echo ""
	@echo "✅ ngrok installed!"
	@echo "Next step: Authenticate ngrok"
	@echo "  1. Sign up at https://dashboard.ngrok.com/signup"
	@echo "  2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken"
	@echo "  3. Run: ngrok config add-authtoken <your-token>"

setup-ngrok: ## Setup ngrok authentication (interactive)
	@echo "Setting up ngrok authentication..."
	@echo ""
	@echo "Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
	@echo ""
	@read -p "Enter your ngrok authtoken: " token && ngrok config add-authtoken $$token
	@echo ""
	@echo "✅ ngrok authenticated!"
	@echo "Run 'make dev-ngrok' to start development with ngrok"

prism-test: ## Quick test Prism WebSocket locally
	@echo "Testing Prism WebSocket..."
	@command -v wscat >/dev/null 2>&1 || { echo "Installing wscat..."; npm install -g wscat; }
	@echo ""
	@echo "Connect to: ws://localhost:8080/api/v1/prism/start-session"
	@echo "Send: {\"action\": \"start\", \"meeting_url\": \"https://meet.google.com/abc-defg-hij\"}"
	@echo ""
	wscat -c ws://localhost:8080/api/v1/prism/start-session

.DEFAULT_GOAL := help
