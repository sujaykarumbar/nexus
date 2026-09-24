.PHONY: help install test lint run-api run-web docker-up docker-down clean

help:
	@echo "NEXUS Platform - Developer Commands"
	@echo "======================================"
	@echo "make install      Install backend and frontend dependencies"
	@echo "make test         Run backend tests"
	@echo "make run-api      Run FastAPI backend server locally"
	@echo "make run-web      Run Vite frontend server locally"
	@echo "make docker-up    Start full stack with Docker Compose"
	@echo "make docker-down  Stop all Docker containers"
	@echo "make clean        Remove cache and build artifacts"

install:
	pip install -r requirements.txt
	cd apps/web && npm install

test:
	pytest tests/backend -v

run-api:
	uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000

run-web:
	cd apps/web && npm run dev

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf apps/web/dist apps/web/node_modules/.vite
