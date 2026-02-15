"""
Message Sink - Kafka to Elasticsearch Consumer with LQR Control

This is the main consumer component that:
1. Collects system state metrics (CPU, memory, I/O, queue length)
2. Uses the controller to compute optimal batch_size and poll_interval
3. Consumes messages from Kafka with controlled parameters
4. Writes data to Elasticsearch in batches
5. Logs all metrics for analysis
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from elasticsearch import Elasticsearch, helpers

from metrics_collector import MetricsCollector, SystemState
from lqr_controller import create_controller, BaseController, ControlOutput, StaticController

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('message-sink')


@dataclass
class SinkMetrics:
    """Metrics logged each cycle for analysis."""
    timestamp: str
    # State variables
    queue_length: int
    cpu_util: float
    mem_util: float
    io_ops: float
    # Control variables
    batch_size: int
    poll_interval_ms: int
    control_mode: str
    # Performance metrics
    messages_consumed: int
    messages_indexed: int
    cycle_duration_ms: float
    indexing_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_line(self) -> str:
        values = [str(v) for v in asdict(self).values()]
        return ','.join(values)

    @staticmethod
    def csv_header() -> str:
        return ','.join([
            'timestamp', 'queue_length', 'cpu_util', 'mem_util', 'io_ops',
            'batch_size', 'poll_interval_ms', 'control_mode',
            'messages_consumed', 'messages_indexed', 'cycle_duration_ms', 'indexing_duration_ms'
        ])


class MessageSink:
    """
    Kafka to Elasticsearch consumer with adaptive control.
    """

    def __init__(self):
        # Kafka configuration
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'cdc.postgres.changes')
        self.kafka_group_id = os.getenv('KAFKA_GROUP_ID', 'message-sink-group')

        # Elasticsearch configuration
        self.es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        self.es_port = int(os.getenv('ELASTICSEARCH_PORT', 9200))
        self.es_index = os.getenv('ELASTICSEARCH_INDEX', 'cqrs_read')

        # Control configuration
        self.control_mode = os.getenv('CONTROL_MODE', 'static')
        self.metrics_log_interval = int(os.getenv('METRICS_LOG_INTERVAL_SEC', 5))

        # Initialize components
        self.consumer: Optional[KafkaConsumer] = None
        self.es_client: Optional[Elasticsearch] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.controller: Optional[BaseController] = None

        # Runtime state
        self.running = False
        self.total_messages_consumed = 0
        self.total_messages_indexed = 0
        self.last_metrics_log_time = 0

        # Metrics log file
        self.metrics_log_path = os.getenv('METRICS_LOG_PATH', '/app/logs/sink_metrics.csv')

        # External control override file (used by sysid experiment)
        self.control_override_path = os.getenv('CONTROL_OVERRIDE_PATH', '/app/logs/control_override.json')

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def connect_kafka(self) -> bool:
        """Connect to Kafka."""
        try:
            self.consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_servers.split(','),
                group_id=self.kafka_group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                # These will be overridden by controller
                max_poll_records=100,
            )
            logger.info(f"Connected to Kafka at {self.kafka_servers}, topic: {self.kafka_topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    def connect_elasticsearch(self) -> bool:
        """Connect to Elasticsearch and create index if needed."""
        try:
            self.es_client = Elasticsearch(
                [{'host': self.es_host, 'port': self.es_port, 'scheme': 'http'}],
                request_timeout=30,
            )

            # Check connection
            if not self.es_client.ping():
                raise Exception("Elasticsearch ping failed")

            # Create index if it doesn't exist
            if not self.es_client.indices.exists(index=self.es_index):
                # Load index mapping from file if exists
                mapping_file = '/app/init/init-elasticsearch.json'
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        mapping = json.load(f)
                    self.es_client.indices.create(index=self.es_index, body=mapping)
                else:
                    self.es_client.indices.create(index=self.es_index)
                logger.info(f"Created Elasticsearch index: {self.es_index}")
            else:
                logger.info(f"Elasticsearch index already exists: {self.es_index}")

            logger.info(f"Connected to Elasticsearch at {self.es_host}:{self.es_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

    def initialize_components(self) -> bool:
        """Initialize all components."""
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            kafka_servers=self.kafka_servers,
            kafka_topic=self.kafka_topic,
            kafka_group_id=self.kafka_group_id,
        )

        # Initialize controller
        self.controller = create_controller(self.control_mode)
        logger.info(f"Initialized controller in '{self.control_mode}' mode")

        # Initialize metrics log
        self._init_metrics_log()

        return True

    def _init_metrics_log(self):
        """Initialize the metrics log file."""
        try:
            log_dir = os.path.dirname(self.metrics_log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Write header if file doesn't exist
            if not os.path.exists(self.metrics_log_path):
                with open(self.metrics_log_path, 'w') as f:
                    f.write(SinkMetrics.csv_header() + '\n')
                logger.info(f"Created metrics log: {self.metrics_log_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize metrics log: {e}")

    def _check_control_override(self) -> Optional[ControlOutput]:
        """
        Check for external control override file.
        Used during system identification experiments to inject known control values.
        File format: {"batch_size": 100, "poll_interval": 1000}
        """
        try:
            if os.path.exists(self.control_override_path):
                with open(self.control_override_path, 'r') as f:
                    override = json.load(f)
                return ControlOutput(
                    batch_size=int(override['batch_size']),
                    poll_interval_ms=int(override['poll_interval']),
                    mode='sysid'
                )
        except Exception:
            pass
        return None

    def _log_metrics(self, metrics: SinkMetrics):
        """Log metrics to file and console."""
        try:
            # Log to file
            with open(self.metrics_log_path, 'a') as f:
                f.write(metrics.to_csv_line() + '\n')

            # Log to console periodically
            current_time = time.time()
            if current_time - self.last_metrics_log_time >= self.metrics_log_interval:
                logger.info(
                    f"State: queue={metrics.queue_length}, cpu={metrics.cpu_util:.1f}%, "
                    f"mem={metrics.mem_util:.1f}%, io={metrics.io_ops:.0f} B/s | "
                    f"Control: batch={metrics.batch_size}, poll={metrics.poll_interval_ms}ms | "
                    f"Perf: consumed={metrics.messages_consumed}, indexed={metrics.messages_indexed}, "
                    f"cycle={metrics.cycle_duration_ms:.0f}ms"
                )
                self.last_metrics_log_time = current_time

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def process_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Process messages from Kafka and prepare for Elasticsearch indexing.
        """
        actions = []
        for msg in messages:
            try:
                # Extract data from CDC message
                data = msg.get('data', {})
                operation = msg.get('operation', 'INSERT')

                # Prepare Elasticsearch document
                doc = {
                    '_index': self.es_index,
                    '_id': data.get('id', msg.get('lsn', str(time.time()))),
                    '_source': {
                        **data,
                        '_operation': operation,
                        '_source_lsn': msg.get('lsn'),
                        'synced_at': datetime.utcnow().isoformat(),
                    }
                }

                # Handle DELETE operations
                if operation == 'DELETE':
                    doc['_op_type'] = 'delete'
                else:
                    doc['_op_type'] = 'index'

                actions.append(doc)

            except Exception as e:
                logger.warning(f"Failed to process message: {e}")

        return actions

    def bulk_index(self, actions: List[Dict]) -> int:
        """
        Bulk index documents to Elasticsearch.
        Returns number of successfully indexed documents.
        """
        if not actions:
            return 0

        try:
            success, errors = helpers.bulk(
                self.es_client,
                actions,
                raise_on_error=False,
                raise_on_exception=False,
            )

            if errors:
                logger.warning(f"Bulk indexing had {len(errors)} errors")
                for error in errors[:5]:  # Log first 5 errors
                    logger.warning(f"  Error: {error}")

            return success

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return 0

    def run(self):
        """Main consumer loop."""
        logger.info("Starting Message Sink...")

        # Connect to services
        while not self.connect_kafka():
            logger.warning("Retrying Kafka connection in 5 seconds...")
            time.sleep(5)

        while not self.connect_elasticsearch():
            logger.warning("Retrying Elasticsearch connection in 5 seconds...")
            time.sleep(5)

        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return

        logger.info("Message Sink running. Press Ctrl+C to stop.")

        self.running = True
        while self.running:
            cycle_start = time.time()

            try:
                # 1. Collect current system state
                state = self.metrics_collector.collect_state()

                # 2. Compute optimal control (or use external override for sysid)
                override = self._check_control_override()
                control = override if override else self.controller.compute_control(state.to_dict())

                # 3. Poll messages with controlled parameters
                # Note: KafkaConsumer doesn't support changing max_poll_records after creation
                # We use timeout to approximate poll_interval behavior
                poll_start = time.time()
                message_batch = self.consumer.poll(
                    timeout_ms=control.poll_interval_ms,
                    max_records=control.batch_size,
                )
                poll_duration = (time.time() - poll_start) * 1000

                # 4. Process and index messages
                messages = []
                for tp, records in message_batch.items():
                    for record in records:
                        messages.append(record.value)

                messages_consumed = len(messages)
                self.total_messages_consumed += messages_consumed

                # 5. Bulk index to Elasticsearch
                index_start = time.time()
                if messages:
                    actions = self.process_messages(messages)
                    messages_indexed = self.bulk_index(actions)
                    self.total_messages_indexed += messages_indexed
                else:
                    messages_indexed = 0
                index_duration = (time.time() - index_start) * 1000

                cycle_duration = (time.time() - cycle_start) * 1000

                # 6. Log metrics
                metrics = SinkMetrics(
                    timestamp=datetime.utcnow().isoformat(),
                    queue_length=state.queue_length,
                    cpu_util=state.cpu_util,
                    mem_util=state.mem_util,
                    io_ops=state.io_ops,
                    batch_size=control.batch_size,
                    poll_interval_ms=control.poll_interval_ms,
                    control_mode=control.mode,
                    messages_consumed=messages_consumed,
                    messages_indexed=messages_indexed,
                    cycle_duration_ms=cycle_duration,
                    indexing_duration_ms=index_duration,
                )
                self._log_metrics(metrics)

            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")

        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")

        if self.es_client:
            self.es_client.close()
            logger.info("Elasticsearch client closed")

        if self.metrics_collector:
            self.metrics_collector.close()

        logger.info(f"Total messages consumed: {self.total_messages_consumed}")
        logger.info(f"Total messages indexed: {self.total_messages_indexed}")


if __name__ == '__main__':
    sink = MessageSink()
    sink.run()
