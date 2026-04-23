# Run this in your terminal


# Create all directories
mkdir -p data notebooks model producer spark \
         monitoring/grafana/dashboards k8s/kafka \
         k8s/spark k8s/producer k8s/monitoring

# Create empty files
touch docker-compose.yml .env .gitignore README.md
touch model/train.py
touch producer/Dockerfile producer/requirements.txt producer/producer.py
touch spark/Dockerfile spark/requirements.txt spark/streaming_job.py
touch monitoring/prometheus.yml
touch monitoring/grafana/dashboards/fraud_dashboard.json

echo "✅ Project structure created"