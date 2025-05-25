.PHONY: dev dev-detach prod prod-detach stop clean help

help: ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

dev: ## Start the development server with hot reloading
	docker compose -f docker-compose.dev.yml up --build

dev-detach: ## Start the development server in detached mode
	docker compose -f docker-compose.dev.yml up -d --build

prod: ## Start the production server
	docker compose up --build

prod-detach: ## Start the production server in detached mode
	docker compose up -d --build

stop: ## Stop all running containers
	docker compose down
	docker compose -f docker-compose.dev.yml down

clean: ## Stop containers and remove images
	docker compose down --rmi all
	docker compose -f docker-compose.dev.yml down --rmi all

logs: ## View logs of running containers
	docker compose -f docker-compose.dev.yml logs -f 