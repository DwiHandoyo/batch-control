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
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from elasticsearch import Elasticsearch, helpers

from metrics_collector import MetricsCollector, SystemState
from controllers import create_controller, BaseController, ControlOutput, StaticController

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
    indexing_time_rate: float
    io_write_ops: float
    os_cpu_percent: int
    os_mem_used_percent: int
    gc_time_rate: float
    write_queue_size: int
    # Control variables
    batch_size: int
    poll_interval_ms: int
    control_mode: str
    # Performance metrics
    messages_consumed: int
    messages_indexed: int
    cycle_duration_ms: float
    indexing_duration_ms: float
    avg_latency_ms: float  # mean(synced_at - updated_at) per batch

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_line(self) -> str:
        values = [str(v) for v in asdict(self).values()]
        return ','.join(values)

    @staticmethod
    def csv_header() -> str:
        return ','.join([
            'timestamp', 'queue_length', 'cpu_util', 'mem_util',
            'indexing_time_rate', 'io_write_ops',
            'os_cpu_percent', 'os_mem_used_percent',
            'gc_time_rate', 'write_queue_size',
            'batch_size', 'poll_interval_ms', 'control_mode',
            'messages_consumed', 'messages_indexed', 'cycle_duration_ms', 'indexing_duration_ms',
            'avg_latency_ms'
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

        # Async indexing
        self.index_queue = queue.Queue()
        self.index_worker = None
        self.last_index_count = 0
        self.last_index_duration = 0.0
        self.last_avg_latency_ms = 0.0

        # Metrics log file
        self.metrics_log_path = os.getenv('METRICS_LOG_PATH', '/app/logs/sink_metrics.csv')

        # External control override file (used by sysid experiment)
        self.control_override_path = os.getenv('CONTROL_OVERRIDE_PATH', '/app/logs/control_override.json')

        # Model file paths (for runtime controller switching)
        self.sysid_json_path = os.getenv('SYSID_JSON', None)
        self.ann_model_json_path = os.getenv('ANN_MODEL_JSON', None)

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
        """Initialize the metrics log file with header."""
        try:
            log_dir = os.path.dirname(self.metrics_log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Always write fresh file with header on startup
            with open(self.metrics_log_path, 'w') as f:
                f.write(SinkMetrics.csv_header() + '\n')
            logger.info(f"Initialized metrics log: {self.metrics_log_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize metrics log: {e}")

    def _switch_controller(self, mode: str):
        """Switch the internal controller to a new mode at runtime.

        Supports Q-variant modes like 'lqr_q1', 'ann_q2', etc.
        The Q-variant parsing is handled by create_controller().
        """
        if mode == self.control_mode:
            return
        kwargs = {}

        # Determine base mode for sysid/ann path logic
        base_mode = mode
        q_key = None
        if '_q' in mode:
            parts = mode.rsplit('_', 1)
            base_mode = parts[0]
            q_key = parts[1].upper()  # e.g. "Q1"

        if base_mode in ('lqr', 'rule_based') and self.sysid_json_path:
            kwargs['sysid_json'] = self.sysid_json_path
        if base_mode == 'ann':
            # Use per-Q env var if available, else fallback to default
            if q_key:
                ann_path = os.getenv(f'ANN_MODEL_{q_key}', self.ann_model_json_path)
            else:
                ann_path = self.ann_model_json_path
            if ann_path:
                kwargs['ann_model_json'] = ann_path

        self.controller = create_controller(mode, **kwargs)
        self.control_mode = mode
        logger.info(f"Controller switched to '{mode}'")

    def _check_control_override(self):
        """
        Check for external control override file.

        Supports three override modes:
        1. {"pause": true} / {"pause": true, "seek_to_end": true} — pause/drain
        2. {"controller_mode": "pid"} — switch internal controller, then run autonomously
        3. {"batch_size": 100, "poll_interval": 1000} — direct control injection (sysid)

        Returns 'pause', 'seek_to_end', ControlOutput, or None.
        """
        try:
            if os.path.exists(self.control_override_path):
                with open(self.control_override_path, 'r') as f:
                    override = json.load(f)
                if override.get('pause', False):
                    if override.get('seek_to_end', False):
                        return 'seek_to_end'
                    return 'pause'
                # Controller mode switch: switch internal controller and remove override
                if 'controller_mode' in override:
                    self._switch_controller(override['controller_mode'])
                    try:
                        os.remove(self.control_override_path)
                    except OSError:
                        pass
                    return None  # let the newly-switched controller run
                return ControlOutput(
                    batch_size=int(override['batch_size']),
                    poll_interval_ms=int(override['poll_interval']),
                    mode='sysid'
                )
        except Exception:
            pass
        return None

    def _index_worker_loop(self):
        """Worker thread: takes messages from queue and bulk indexes to ES."""
        while self.running or not self.index_queue.empty():
            try:
                messages = self.index_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                start = time.time()
                actions, avg_latency = self.process_messages(messages)
                indexed = self.bulk_index(actions)
                duration = (time.time() - start) * 1000
                self.total_messages_indexed += indexed
                self.last_index_count = indexed
                self.last_index_duration = duration
                self.last_avg_latency_ms = avg_latency
            except Exception as e:
                logger.error(f"Index worker error: {e}", exc_info=True)
            finally:
                self.index_queue.task_done()

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
                    f"mem={metrics.mem_util:.1f}%, idx_rate={metrics.indexing_time_rate:.1f}ms/s, io={metrics.io_write_ops:.1f}ops/s | "
                    f"OS: cpu={metrics.os_cpu_percent}%, mem={metrics.os_mem_used_percent}% | "
                    f"GC: {metrics.gc_time_rate:.1f}ms/s, wq={metrics.write_queue_size} | "
                    f"Control: batch={metrics.batch_size}, poll={metrics.poll_interval_ms}ms | "
                    f"Perf: consumed={metrics.messages_consumed}, indexed={metrics.messages_indexed}, "
                    f"cycle={metrics.cycle_duration_ms:.0f}ms"
                )
                self.last_metrics_log_time = current_time

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def process_messages(self, messages: List[Dict]) -> tuple:
        """
        Process messages from Kafka and prepare for Elasticsearch indexing.

        Returns:
            (actions, avg_latency_ms): ES bulk actions and mean end-to-end latency
        """
        actions = []
        latencies = []
        now = datetime.utcnow()
        for msg in messages:
            try:
                # Extract data from CDC message
                data = msg.get('data', {})
                operation = msg.get('operation', 'INSERT')

                # Compute end-to-end latency: now - updated_at (PostgreSQL timestamp)
                updated_at_str = data.get('updated_at')
                if updated_at_str:
                    try:
                        # Parse ISO format (with or without timezone)
                        ua = updated_at_str.replace('Z', '+00:00')
                        if '+' in ua or ua.endswith('Z'):
                            from datetime import timezone
                            updated_at = datetime.fromisoformat(ua).replace(tzinfo=None)
                        else:
                            updated_at = datetime.fromisoformat(ua)
                        latency_ms = (now - updated_at).total_seconds() * 1000
                        if latency_ms >= 0:
                            latencies.append(latency_ms)
                    except (ValueError, TypeError):
                        pass

                synced_at = now.isoformat()

                # Prepare Elasticsearch document
                doc = {
                    '_index': self.es_index,
                    '_id': data.get('id', msg.get('lsn', str(time.time()))),
                    '_source': {
                        **data,
                        '_operation': operation,
                        '_source_lsn': msg.get('lsn'),
                        'synced_at': synced_at,
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

        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        return actions, avg_latency_ms

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
                refresh=True,
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

        # Start async index worker
        self.running = True
        self.index_worker = threading.Thread(target=self._index_worker_loop, daemon=True)
        self.index_worker.start()
        logger.info("Index worker thread started")
        while self.running:
            cycle_start = time.time()

            try:
                # 1. Collect current system state
                state = self.metrics_collector.collect_state()

                # 2. Compute optimal control (or use external override for sysid)
                override = self._check_control_override()
                if override == 'seek_to_end':
                    self.consumer.seek_to_end()
                    # Must poll after seek to update internal offsets before commit
                    self.consumer.poll(timeout_ms=0)
                    self.consumer.commit()
                    logger.info("Seeked to end of topic (queue drained)")
                    time.sleep(0.5)
                    continue
                if override == 'pause':
                    time.sleep(0.5)
                    continue
                state_dict = state.to_dict()
                state_dict['avg_latency_ms'] = self.last_avg_latency_ms
                control = override if override else self.controller.compute_control(state_dict)

                # 3. Poll until batch_size fulfilled (with timeout + override check)
                messages = []
                poll_deadline = time.time() + (control.poll_interval_ms / 1000)
                while len(messages) < control.batch_size:
                    # Check for override (pause/seek) inside polling loop
                    inner_override = self._check_control_override()
                    if inner_override in ('pause', 'seek_to_end'):
                        break
                    if time.time() > poll_deadline:
                        break  # don't block forever
                    remaining = control.batch_size - len(messages)
                    batch = self.consumer.poll(
                        timeout_ms=500,
                        max_records=remaining,
                    )
                    for tp, records in batch.items():
                        for record in records:
                            messages.append(record.value)

                messages_consumed = len(messages)
                self.total_messages_consumed += messages_consumed

                # 5. Submit to async index worker
                if messages:
                    self.index_queue.put(messages)
                messages_indexed = self.last_index_count
                index_duration = self.last_index_duration

                cycle_duration = (time.time() - cycle_start) * 1000

                # 6. Log metrics
                metrics = SinkMetrics(
                    timestamp=datetime.utcnow().isoformat(),
                    queue_length=state.queue_length,
                    cpu_util=state.cpu_util,
                    mem_util=state.mem_util,
                    indexing_time_rate=state.indexing_time_rate,
                    io_write_ops=state.io_write_ops,
                    os_cpu_percent=state.os_cpu_percent,
                    os_mem_used_percent=state.os_mem_used_percent,
                    gc_time_rate=state.gc_time_rate,
                    write_queue_size=state.write_queue_size,
                    batch_size=control.batch_size,
                    poll_interval_ms=control.poll_interval_ms,
                    control_mode=control.mode,
                    messages_consumed=messages_consumed,
                    messages_indexed=messages_indexed,
                    cycle_duration_ms=cycle_duration,
                    indexing_duration_ms=index_duration,
                    avg_latency_ms=self.last_avg_latency_ms,
                )
                self._log_metrics(metrics)

                # Enforce cycle_time (poll_interval_ms)
                cycle_elapsed = (time.time() - cycle_start) * 1000
                remaining = control.poll_interval_ms - cycle_elapsed
                if remaining > 0:
                    time.sleep(remaining / 1000)

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

        # Wait for index worker to finish remaining items
        if self.index_worker and self.index_worker.is_alive():
            logger.info("Waiting for index worker to finish...")
            self.index_queue.join()
            self.index_worker.join(timeout=30)

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
