FROM python:3.10-slim

# Metadata
LABEL maintainer="fraud-detection"
LABEL description="Kafka transaction producer"

# Set working directory
WORKDIR /app

# Install dependencies first (Docker cache optimization)
# If requirements.txt doesn't change → this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY producer.py .

# Data and model will be mounted as volumes
# not copied into the image

# Expose Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; \
                 urllib.request.urlopen('http://localhost:8000')" \
  || exit 1

# Run the producer
CMD ["python", "producer.py"]