import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
    IntegerType, LongType,
    TimestampType
)

from kafka import KafkaProducer
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge
)
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
)
KAFKA_TOPIC_INPUT  = os.getenv(
    'KAFKA_TOPIC_TRANSACTIONS', 'raw_transactions'
)
KAFKA_TOPIC_ALERTS = os.getenv(
    'KAFKA_TOPIC_ALERTS', 'fraud_alerts'
)
MODEL_PATH         = os.getenv(
    'MODEL_PATH', '/model/fraud_model.pkl'
)
SCALER_PATH        = os.getenv(
    'SCALER_PATH', '/model/scaler.pkl'
)
FRAUD_THRESHOLD    = float(os.getenv(
    'FRAUD_THRESHOLD', '0.5'
))
BATCH_INTERVAL     = int(os.getenv(
    'SPARK_BATCH_INTERVAL', '5'
))
PROMETHEUS_PORT    = int(os.getenv(
    'PROMETHEUS_PORT', '8001'
))
POSTGRES_HOST      = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT      = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB        = os.getenv('POSTGRES_DB', 'fraud_detection')
POSTGRES_USER      = os.getenv('POSTGRES_USER', 'fraud_user')
POSTGRES_PASSWORD  = os.getenv('POSTGRES_PASSWORD', 'fraud_pass')

# ── FEATURE COLUMNS ───────────────────────────────────────────
V_FEATURES    = [f'V{i}' for i in range(1, 29)]
ALL_FEATURES  = ['Time', 'Amount'] + V_FEATURES  # 30 features

# ── PROMETHEUS METRICS ────────────────────────────────────────
transactions_processed = Counter(
    'spark_transactions_processed_total',
    'Total transactions processed by Spark'
)
fraud_detected = Counter(
    'spark_fraud_detected_total',
    'Total fraud transactions detected'
)
inference_duration = Histogram(
    'spark_inference_duration_seconds',
    'Time to run ML inference on a batch',
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0]
)
batch_size_metric = Histogram(
    'spark_batch_size_transactions',
    'Number of transactions per micro-batch',
    buckets=[1, 5, 10, 50, 100, 500, 1000, 5000]
)
batch_processing_time = Histogram(
    'spark_batch_processing_time_seconds',
    'Total time to process one micro-batch',
    buckets=[.1, .5, 1.0, 2.5, 5.0, 10.0, 30.0]
)
fraud_score_distribution = Histogram(
    'spark_fraud_score_distribution',
    'Distribution of fraud probability scores',
    buckets=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
)
current_fraud_rate = Gauge(
    'spark_current_fraud_rate',
    'Fraud rate in the last batch (0.0 to 1.0)'
)
db_write_errors = Counter(
    'spark_db_write_errors_total',
    'Total database write errors'
)
kafka_alert_errors = Counter(
    'spark_kafka_alert_errors_total',
    'Total Kafka alert send errors'
)


# ════════════════════════════════════════════════════════════════
# MODEL MANAGER
# ════════════════════════════════════════════════════════════════
class ModelManager:
    """
    Loads and manages the ML model
    Designed to be serializable for Spark broadcast
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self._model      = None
        self._scaler     = None

    def load(self):
        """Load model and scaler from disk"""
        log.info(f"🤖 Loading model from {self.model_path}")
        self._model  = joblib.load(self.model_path)
        self._scaler = joblib.load(self.scaler_path)
        log.info("✅ Model loaded successfully")
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Run inference on a batch of transactions
        Returns array of fraud probabilities
        """
        # Scale Amount and Time
        features_df = features_df.copy()
        features_df[['Amount', 'Time']] = self._scaler.transform(
            features_df[['Amount', 'Time']]
        )

        # Predict probabilities
        probabilities = self._model.predict_proba(
            features_df[ALL_FEATURES]
        )[:, 1]  # index 1 = fraud class

        return probabilities


# ════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Handles PostgreSQL connections and writes"""

    def __init__(self):
        self.conn   = None
        self.cursor = None

    def connect(self, retries: int = 10, wait: int = 5):
        """Connect to PostgreSQL with retry logic"""
        for attempt in range(1, retries + 1):
            try:
                log.info(
                    f"🗄️  Connecting to PostgreSQL "
                    f"(attempt {attempt}/{retries})"
                )
                self.conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    dbname=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    connect_timeout=10
                )
                self.cursor = self.conn.cursor()
                log.info("✅ Connected to PostgreSQL")
                return self
            except Exception as e:
                log.warning(
                    f"   DB not ready: {e}. "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError("Could not connect to PostgreSQL")

    def create_tables(self):
        """Create tables if they don't exist"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id                   SERIAL PRIMARY KEY,
                transaction_id       VARCHAR(20) UNIQUE,
                processed_at         TIMESTAMP DEFAULT NOW(),
                amount               DOUBLE PRECISION,
                fraud_probability    DOUBLE PRECISION,
                is_fraud_predicted   BOOLEAN,
                is_fraud_ground_truth INTEGER,
                merchant_id          VARCHAR(20),
                card_last_four       VARCHAR(4),
                country              VARCHAR(2),
                processing_time_ms   DOUBLE PRECISION
            );

            CREATE TABLE IF NOT EXISTS fraud_alerts (
                id                SERIAL PRIMARY KEY,
                transaction_id    VARCHAR(20),
                alerted_at        TIMESTAMP DEFAULT NOW(),
                fraud_probability DOUBLE PRECISION,
                amount            DOUBLE PRECISION,
                merchant_id       VARCHAR(20),
                country           VARCHAR(2)
            );

            CREATE TABLE IF NOT EXISTS batch_metrics (
                id              SERIAL PRIMARY KEY,
                batch_id        BIGINT,
                processed_at    TIMESTAMP DEFAULT NOW(),
                batch_size      INTEGER,
                fraud_count     INTEGER,
                fraud_rate      DOUBLE PRECISION,
                processing_ms   DOUBLE PRECISION
            );
        """)
        self.conn.commit()
        log.info("✅ Database tables ready")

    def insert_transactions(self, records: list):
        """Batch insert transactions"""
        if not records:
            return
        try:
            execute_batch(
                self.cursor,
                """
                INSERT INTO transactions (
                    transaction_id,
                    amount,
                    fraud_probability,
                    is_fraud_predicted,
                    is_fraud_ground_truth,
                    merchant_id,
                    card_last_four,
                    country,
                    processing_time_ms
                ) VALUES (
                    %(transaction_id)s,
                    %(amount)s,
                    %(fraud_probability)s,
                    %(is_fraud_predicted)s,
                    %(is_fraud_ground_truth)s,
                    %(merchant_id)s,
                    %(card_last_four)s,
                    %(country)s,
                    %(processing_time_ms)s
                )
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                records,
                page_size=500
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            db_write_errors.inc()
            log.error(f"❌ DB insert error: {e}")

    def insert_batch_metric(self, batch_id: int,
                            batch_size: int,
                            fraud_count: int,
                            processing_ms: float):
        """Record batch-level metrics"""
        try:
            fraud_rate = (
                fraud_count / batch_size
                if batch_size > 0 else 0
            )
            self.cursor.execute(
                """
                INSERT INTO batch_metrics
                (batch_id, batch_size, fraud_count,
                 fraud_rate, processing_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (batch_id, batch_size, fraud_count,
                 fraud_rate, processing_ms)
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            log.error(f"❌ Batch metric insert error: {e}")


# ════════════════════════════════════════════════════════════════
# ALERT PRODUCER
# ════════════════════════════════════════════════════════════════
class AlertProducer:
    """Sends fraud alerts back to Kafka"""

    def __init__(self):
        self.producer = None

    def connect(self, retries: int = 10, wait: int = 5):
        for attempt in range(1, retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: (
                        json.dumps(v).encode('utf-8')
                    )
                )
                log.info("✅ Alert producer connected to Kafka")
                return self
            except Exception as e:
                log.warning(
                    f"Alert producer not ready: {e}. "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            "Could not connect alert producer to Kafka"
        )

    def send_alert(self, transaction: dict):
        """Send fraud alert to Kafka topic"""
        try:
            alert = {
                'alert_id'        : f"ALERT-{time.time_ns()}",
                'transaction_id'  : transaction['transaction_id'],
                'timestamp'       : time.time(),
                'fraud_probability': transaction['fraud_probability'],
                'amount'          : transaction['amount'],
                'merchant_id'     : transaction['merchant_id'],
                'country'         : transaction['country'],
                'card_last_four'  : transaction['card_last_four'],
                'severity'        : self._get_severity(
                    transaction['fraud_probability']
                )
            }
            self.producer.send(KAFKA_TOPIC_ALERTS, value=alert)
        except Exception as e:
            kafka_alert_errors.inc()
            log.error(f"❌ Alert send error: {e}")

    @staticmethod
    def _get_severity(probability: float) -> str:
        if probability >= 0.9:
            return 'CRITICAL'
        elif probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'


# ════════════════════════════════════════════════════════════════
# KAFKA MESSAGE SCHEMA
# ════════════════════════════════════════════════════════════════
def get_transaction_schema() -> StructType:
    """
    Define the schema for messages coming from Kafka
    Spark uses this to parse JSON efficiently
    """
    v_fields = [
        StructField(f'V{i}', DoubleType(), True)
        for i in range(1, 29)
    ]

    return StructType([
        StructField('transaction_id',
                    StringType(), False),
        StructField('timestamp',
                    DoubleType(), True),
        StructField('timestamp_iso',
                    StringType(), True),
        StructField('Time',
                    DoubleType(), True),
        StructField('Amount',
                    DoubleType(), True),
        *v_fields,
        StructField('is_fraud_ground_truth',
                    IntegerType(), True),
        StructField('merchant_id',
                    StringType(), True),
        StructField('card_last_four',
                    StringType(), True),
        StructField('country',
                    StringType(), True),
    ])


# ════════════════════════════════════════════════════════════════
# SPARK SESSION
# ════════════════════════════════════════════════════════════════
def create_spark_session() -> SparkSession:
    """Create and configure Spark session"""

    # JAR files for Kafka connector
    jars = ','.join([
        '/app/spark-sql-kafka.jar',
        '/app/kafka-clients.jar',
        '/app/spark-token-provider-kafka.jar',
        '/app/commons-pool2.jar',
    ])

    spark = (
        SparkSession.builder
        .appName('FraudDetectionStreaming')
        .config('spark.jars', jars)

        # Streaming configs
        .config('spark.streaming.stopGracefullyOnShutdown', 'true')
        .config('spark.sql.streaming.checkpointLocation',
                '/tmp/spark-checkpoints')

        # Performance configs
        .config('spark.sql.shuffle.partitions', '4')
        .config('spark.default.parallelism', '4')

        # Memory configs
        .config('spark.driver.memory', '2g')
        .config('spark.executor.memory', '2g')

        # Log level
        .config('spark.ui.enabled', 'true')
        .config('spark.ui.port', '4040')

        .getOrCreate()
    )

    # Reduce Spark's verbose logging
    spark.sparkContext.setLogLevel('WARN')

    log.info("✅ Spark session created")
    log.info(f"   Spark version : {spark.version}")
    log.info(f"   Spark UI      : http://localhost:4040")

    return spark


# ════════════════════════════════════════════════════════════════
# BATCH PROCESSOR
# ════════════════════════════════════════════════════════════════
def process_batch(batch_df, batch_id: int,
                  model_manager: ModelManager,
                  db_manager: DatabaseManager,
                  alert_producer: AlertProducer):
    """
    Called by Spark for each micro-batch
    This is where the ML scoring happens

    batch_df  → Spark DataFrame with transactions from Kafka
    batch_id  → incrementing batch number
    """
    batch_start = time.time()

    # Skip empty batches
    if batch_df.isEmpty():
        log.debug(f"Batch {batch_id}: empty, skipping")
        return

    # Convert Spark DataFrame → Pandas
    # (needed for XGBoost inference)
    pdf = batch_df.toPandas()
    batch_size = len(pdf)

    log.info(
        f"\n{'='*50}\n"
        f"📦 Batch {batch_id} | Size: {batch_size} transactions"
    )

    # ── ML INFERENCE ──────────────────────────────────────────
    inference_start = time.time()

    try:
        fraud_probabilities = model_manager.predict(
            pdf[ALL_FEATURES]
        )
    except Exception as e:
        log.error(f"❌ Inference error in batch {batch_id}: {e}")
        return

    inference_elapsed = time.time() - inference_start
    inference_duration.observe(inference_elapsed)

    # ── ADD PREDICTIONS TO DATAFRAME ──────────────────────────
    pdf['fraud_probability']  = fraud_probabilities
    pdf['is_fraud_predicted'] = (
        fraud_probabilities >= FRAUD_THRESHOLD
    ).astype(bool)
    pdf['processing_time_ms'] = inference_elapsed * 1000

    # ── CALCULATE BATCH STATS ─────────────────────────────────
    fraud_count = int(pdf['is_fraud_predicted'].sum())
    fraud_rate  = fraud_count / batch_size if batch_size > 0 else 0

    log.info(f"   Inference time : {inference_elapsed*1000:.1f}ms")
    log.info(f"   Fraud detected : {fraud_count}/{batch_size} "
             f"({fraud_rate*100:.2f}%)")

    # ── UPDATE PROMETHEUS ──────────────────────────────────────
    transactions_processed.inc(batch_size)
    fraud_detected.inc(fraud_count)
    batch_size_metric.observe(batch_size)
    current_fraud_rate.set(fraud_rate)

    for score in fraud_probabilities:
        fraud_score_distribution.observe(float(score))

    # ── SEND FRAUD ALERTS TO KAFKA ────────────────────────────
    fraud_rows = pdf[pdf['is_fraud_predicted'] == True]
    for _, row in fraud_rows.iterrows():
        alert_producer.send_alert({
            'transaction_id'  : row['transaction_id'],
            'fraud_probability': float(row['fraud_probability']),
            'amount'          : float(row['Amount']),
            'merchant_id'     : row['merchant_id'],
            'country'         : row['country'],
            'card_last_four'  : row['card_last_four'],
        })

    if fraud_count > 0:
        log.info(f"   🚨 Sent {fraud_count} alerts to Kafka")

    # ── LOG HIGH CONFIDENCE FRAUD ─────────────────────────────
    high_confidence = pdf[pdf['fraud_probability'] >= 0.9]
    if len(high_confidence) > 0:
        log.info(f"   ⚠️  HIGH CONFIDENCE fraud:")
        for _, row in high_confidence.iterrows():
            log.info(
                f"      {row['transaction_id']} | "
                f"Amount: ${row['Amount']:.2f} | "
                f"Score: {row['fraud_probability']:.4f} | "
                f"Country: {row['country']}"
            )

    # ── WRITE TO DATABASE ─────────────────────────────────────
    records = []
    for _, row in pdf.iterrows():
        records.append({
            'transaction_id'      : row['transaction_id'],
            'amount'              : float(row['Amount']),
            'fraud_probability'   : float(row['fraud_probability']),
            'is_fraud_predicted'  : bool(row['is_fraud_predicted']),
            'is_fraud_ground_truth': int(row['is_fraud_ground_truth']),
            'merchant_id'         : row['merchant_id'],
            'card_last_four'      : row['card_last_four'],
            'country'             : row['country'],
            'processing_time_ms'  : float(row['processing_time_ms']),
        })

    db_manager.insert_transactions(records)

    # ── RECORD BATCH METRICS ──────────────────────────────────
    batch_elapsed    = time.time() - batch_start
    batch_elapsed_ms = batch_elapsed * 1000

    batch_processing_time.observe(batch_elapsed)
    db_manager.insert_batch_metric(
        batch_id, batch_size, fraud_count, batch_elapsed_ms
    )

    log.info(
        f"   Total batch time: {batch_elapsed_ms:.1f}ms\n"
        f"{'='*50}"
    )


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("FRAUD DETECTION STREAMING JOB STARTING")
    log.info("=" * 60)

    # ── START PROMETHEUS ──────────────────────────────────────
    start_http_server(PROMETHEUS_PORT)
    log.info(f"📊 Prometheus metrics on port {PROMETHEUS_PORT}")

    # ── LOAD MODEL ────────────────────────────────────────────
    model_manager = ModelManager(MODEL_PATH, SCALER_PATH).load()

    # ── CONNECT TO POSTGRES ───────────────────────────────────
    db_manager = DatabaseManager().connect()
    db_manager.create_tables()

    # ── CONNECT ALERT PRODUCER ────────────────────────────────
    alert_producer = AlertProducer().connect()

    # ── CREATE SPARK SESSION ──────────────────────────────────
    spark = create_spark_session()

    # ── DEFINE SCHEMA ─────────────────────────────────────────
    schema = get_transaction_schema()

    # ── READ STREAM FROM KAFKA ────────────────────────────────
    log.info(f"📡 Connecting to Kafka topic: {KAFKA_TOPIC_INPUT}")

    raw_stream = (
        spark.readStream
        .format('kafka')
        .option(
            'kafka.bootstrap.servers',
            KAFKA_BOOTSTRAP_SERVERS
        )
        .option('subscribe', KAFKA_TOPIC_INPUT)
        .option('startingOffsets', 'latest')
        .option('maxOffsetsPerTrigger', 10000)
        .option('failOnDataLoss', 'false')
        .load()
    )

    # ── PARSE JSON MESSAGES ───────────────────────────────────
    # Kafka value is binary → convert to string → parse JSON
    parsed_stream = (
        raw_stream
        .select(
            F.from_json(
                F.col('value').cast('string'),
                schema
            ).alias('data'),
            F.col('timestamp').alias('kafka_timestamp'),
            F.col('partition'),
            F.col('offset')
        )
        .select(
            'data.*',
            'kafka_timestamp',
            'partition',
            'offset'
        )
        # Drop rows with null transaction_id
        .filter(F.col('transaction_id').isNotNull())
    )

    log.info("✅ Stream schema defined")
    log.info(f"   Batch interval: {BATCH_INTERVAL} seconds")
    log.info(f"   Fraud threshold: {FRAUD_THRESHOLD}")

    # ── START STREAMING QUERY ─────────────────────────────────
    query = (
        parsed_stream.writeStream
        .foreachBatch(
            lambda df, batch_id: process_batch(
                df, batch_id,
                model_manager,
                db_manager,
                alert_producer
            )
        )
        .trigger(
            processingTime=f'{BATCH_INTERVAL} seconds'
        )
        .option(
            'checkpointLocation',
            '/tmp/spark-checkpoints/fraud-streaming'
        )
        .start()
    )

    log.info("🚀 Streaming query started")
    log.info(f"   Query ID: {query.id}")
    log.info("   Waiting for data from Kafka...\n")

    # ── WAIT FOR TERMINATION ──────────────────────────────────
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        log.info("\n⛔ Stopping streaming job...")
        query.stop()
        spark.stop()
        log.info("✅ Streaming job stopped cleanly")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    main()