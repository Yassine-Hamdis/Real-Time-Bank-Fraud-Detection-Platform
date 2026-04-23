# spark/Dockerfile

FROM python:3.10-slim-bookworm

LABEL description="Spark Structured Streaming fraud scorer"

# ── SYSTEM DEPENDENCIES ───────────────────────────────────────
# Spark needs Java
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    wget curl procps ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ── JAVA ENV ──────────────────────────────────────────────────
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── PYTHON DEPENDENCIES ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── SPARK KAFKA CONNECTOR ─────────────────────────────────────
# Download Spark-Kafka connector JAR
# Must match your Spark + Kafka + Scala versions
ENV SPARK_VERSION=3.4.1
ENV SCALA_VERSION=2.12
ENV KAFKA_CONNECTOR_VERSION=3.4.1

RUN wget -q \
    https://repo1.maven.org/maven2/org/apache/spark/\
spark-sql-kafka-0-10_${SCALA_VERSION}/${SPARK_VERSION}/\
spark-sql-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar \
    -O /app/spark-sql-kafka.jar

RUN wget -q \
    https://repo1.maven.org/maven2/org/apache/kafka/\
kafka-clients/3.4.0/kafka-clients-3.4.0.jar \
    -O /app/kafka-clients.jar

RUN wget -q \
    https://repo1.maven.org/maven2/org/apache/spark/\
spark-token-provider-kafka-0-10_${SCALA_VERSION}/${SPARK_VERSION}/\
spark-token-provider-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar \
    -O /app/spark-token-provider-kafka.jar

RUN wget -q \
    https://repo1.maven.org/maven2/org/apache/commons/\
commons-pool2/2.11.1/commons-pool2-2.11.1.jar \
    -O /app/commons-pool2.jar

# ── APP CODE ──────────────────────────────────────────────────
COPY streaming_job.py .

# ── PORTS ─────────────────────────────────────────────────────
# 4040 → Spark UI
# 8001 → Prometheus metrics
EXPOSE 4040 8001

# ── HEALTHCHECK ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
    CMD curl -f http://localhost:4040 || exit 1

CMD ["python", "streaming_job.py"]