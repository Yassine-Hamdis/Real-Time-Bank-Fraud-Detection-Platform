# Makefile

.PHONY: up down restart logs health clean train \
        psql kafka-topics lint validate \
        logs-spark logs-producer logs-kafka

# ── COLORS ────────────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# ═══════════════════════════════════════════════════════════════
# CORE COMMANDS
# ═══════════════════════════════════════════════════════════════

## Start the full platform
up:
	@echo "$(GREEN)🚀 Starting Fraud Detection Platform...$(NC)"
	@docker-compose up --build -d
	@echo "$(GREEN)⏳ Waiting for services to initialize...$(NC)"
	@sleep 30
	@make health

## Stop everything
down:
	@echo "$(YELLOW)⛔ Stopping platform...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Platform stopped$(NC)"

## Stop and remove all data volumes
clean:
	@echo "$(RED)🗑️  Removing all data (volumes included)...$(NC)"
	@docker-compose down -v
	@echo "$(GREEN)✅ Clean complete$(NC)"

## Rebuild and restart everything
restart:
	@make down
	@make up

## Restart specific service
restart-spark:
	@docker-compose restart spark

restart-producer:
	@docker-compose restart producer


# ═══════════════════════════════════════════════════════════════
# LOGS
# ═══════════════════════════════════════════════════════════════

## Tail all logs
logs:
	@docker-compose logs -f

## Tail spark logs
logs-spark:
	@docker-compose logs -f spark

## Tail producer logs
logs-producer:
	@docker-compose logs -f producer

## Tail kafka logs
logs-kafka:
	@docker-compose logs -f kafka


# ═══════════════════════════════════════════════════════════════
# HEALTH & MONITORING
# ═══════════════════════════════════════════════════════════════

## Check health of all services
health:
	@echo "\n$(GREEN)═══ FRAUD DETECTION PLATFORM HEALTH ═══$(NC)\n"
	@echo "$(GREEN)Containers:$(NC)"
	@docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
	@echo "\n$(GREEN)Kafka Topics:$(NC)"
	@docker exec kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--list 2>/dev/null || echo "  Kafka not ready yet"
	@echo "\n$(GREEN)Prometheus Targets:$(NC)"
	@curl -s http://localhost:9090/api/v1/targets 2>/dev/null | \
		python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data['data']['activeTargets']:
        status = '✅' if t['health'] == 'up' else '❌'
        print(f'  {status} {t[\"labels\"][\"job\"]} -> {t[\"health\"]}')
except:
    print('  Prometheus not ready yet')
"
	@echo "\n$(GREEN)Transaction Count:$(NC)"
	@docker exec postgres psql -U fraud_user -d fraud_detection \
		-c "SELECT COUNT(*) as total, \
		        SUM(is_fraud_predicted::int) as fraud \
		    FROM transactions;" 2>/dev/null || \
		echo "  PostgreSQL not ready yet"
	@echo "\n$(GREEN)Metrics Endpoints:$(NC)"
	@curl -s http://localhost:8000/metrics | \
		grep producer_transactions_sent_total | \
		head -1 | \
		awk '{print "  Producer txns sent:", $$2}' || \
		echo "  Producer metrics not ready"
	@curl -s http://localhost:8001/metrics | \
		grep spark_transactions_processed_total | \
		head -1 | \
		awk '{print "  Spark txns processed:", $$2}' || \
		echo "  Spark metrics not ready"
	@echo "\n$(GREEN)Service URLs:$(NC)"
	@echo "  📊 Grafana:    http://localhost:3000"
	@echo "  ⚡ Kafka UI:   http://localhost:8080"
	@echo "  🔥 Spark UI:   http://localhost:4040"
	@echo "  📈 Prometheus: http://localhost:9090"


# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════

## Open PostgreSQL shell
psql:
	@docker exec -it postgres \
		psql -U fraud_user -d fraud_detection

## Show recent transactions
show-transactions:
	@docker exec postgres psql -U fraud_user -d fraud_detection -c "\
		SELECT \
			transaction_id, \
			ROUND(amount::numeric, 2) AS amount, \
			ROUND(fraud_probability::numeric, 4) AS score, \
			is_fraud_predicted AS fraud, \
			country, \
			processed_at \
		FROM transactions \
		ORDER BY processed_at DESC \
		LIMIT 10;"

## Show fraud summary
show-fraud:
	@docker exec postgres psql -U fraud_user -d fraud_detection -c "\
		SELECT \
			COUNT(*)                              AS total, \
			SUM(is_fraud_predicted::int)          AS detected, \
			SUM(is_fraud_ground_truth)            AS actual, \
			ROUND(AVG(fraud_probability)::numeric, 4) AS avg_score, \
			ROUND(AVG(processing_time_ms)::numeric, 2) AS avg_ms \
		FROM transactions;"

## Show batch metrics
show-batches:
	@docker exec postgres psql -U fraud_user -d fraud_detection -c "\
		SELECT \
			batch_id, \
			batch_size, \
			fraud_count, \
			ROUND((fraud_rate * 100)::numeric, 3) AS fraud_pct, \
			ROUND(processing_ms::numeric, 0) AS ms, \
			processed_at \
		FROM batch_metrics \
		ORDER BY batch_id DESC \
		LIMIT 10;"


# ═══════════════════════════════════════════════════════════════
# KAFKA
# ═══════════════════════════════════════════════════════════════

## List Kafka topics
kafka-topics:
	@docker exec kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--list

## Describe Kafka topics
kafka-describe:
	@docker exec kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--describe \
		--topic raw_transactions
	@docker exec kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--describe \
		--topic fraud_alerts

## Show message count per partition
kafka-offsets:
	@echo "raw_transactions offsets:"
	@docker exec kafka kafka-run-class \
		kafka.tools.GetOffsetShell \
		--broker-list localhost:9092 \
		--topic raw_transactions
	@echo "\nfraud_alerts offsets:"
	@docker exec kafka kafka-run-class \
		kafka.tools.GetOffsetShell \
		--broker-list localhost:9092 \
		--topic fraud_alerts


# ═══════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════

## Train the ML model
train:
	@echo "$(GREEN)🤖 Training XGBoost model...$(NC)"
	@python model/train.py
	@echo "$(GREEN)✅ Model training complete$(NC)"

## Check Python syntax
lint:
	@echo "$(GREEN)🔍 Checking Python syntax...$(NC)"
	@python -m py_compile model/train.py && \
		echo "  ✅ model/train.py"
	@python -m py_compile producer/producer.py && \
		echo "  ✅ producer/producer.py"
	@python -m py_compile spark/streaming_job.py && \
		echo "  ✅ spark/streaming_job.py"
	@echo "$(GREEN)✅ All files pass syntax check$(NC)"

## Validate pipeline end-to-end
validate:
	@echo "$(GREEN)🔬 Validating pipeline...$(NC)"
	@echo "\n1. Checking model files..."
	@test -f model/fraud_model.pkl && \
		echo "  ✅ fraud_model.pkl exists" || \
		echo "  ❌ fraud_model.pkl MISSING — run: make train"
	@test -f model/scaler.pkl && \
		echo "  ✅ scaler.pkl exists" || \
		echo "  ❌ scaler.pkl MISSING — run: make train"
	@echo "\n2. Checking dataset..."
	@test -f data/creditcard.csv && \
		echo "  ✅ creditcard.csv exists" || \
		echo "  ❌ creditcard.csv MISSING"
	@echo "\n3. Checking Docker..."
	@docker info > /dev/null 2>&1 && \
		echo "  ✅ Docker running" || \
		echo "  ❌ Docker not running"
	@echo "\n4. Checking containers..."
	@make health
	@echo "\n$(GREEN)✅ Validation complete$(NC)"

## Show help
help:
	@echo "$(GREEN)Fraud Detection Platform — Available Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Core:$(NC)"
	@echo "  make up              Start full platform"
	@echo "  make down            Stop platform"
	@echo "  make clean           Stop + remove all data"
	@echo "  make restart         Rebuild and restart"
	@echo ""
	@echo "$(YELLOW)Logs:$(NC)"
	@echo "  make logs            All service logs"
	@echo "  make logs-spark      Spark logs only"
	@echo "  make logs-producer   Producer logs only"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make health          Full health check"
	@echo "  make show-fraud      Fraud summary from DB"
	@echo "  make show-batches    Batch metrics from DB"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make train           Train ML model"
	@echo "  make lint            Check Python syntax"
	@echo "  make validate        Validate full pipeline"
	@echo "  make psql            Open PostgreSQL shell"
	@echo ""
	@echo "$(YELLOW)URLs:$(NC)"
	@echo "  Grafana    http://localhost:3000"
	@echo "  Kafka UI   http://localhost:8080"
	@echo "  Spark UI   http://localhost:4040"
	@echo "  Prometheus http://localhost:9090"