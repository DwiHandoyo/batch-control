"""
Metrics Collector - Collects system metrics from Elasticsearch and Kafka.
Provides state variables for the LQR controller:
- queue_length: Kafka consumer lag
- cpu_util: Elasticsearch process CPU utilization (%)
- mem_util: Elasticsearch JVM heap utilization (%)
- indexing_time_rate: ES indexing time rate (ms/s)
- io_write_ops: ES disk I/O write operations per second
- os_cpu_percent: OS-level CPU utilization (%) — for comparison
- os_mem_used_percent: OS-level RAM used (%) — for comparison
- gc_time_rate: GC time rate (ms/s) — overhead of garbage collection
- write_queue_size: Write thread pool queue depth — ES internal backpressure
"""

import os
import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from kafka import KafkaConsumer, TopicPartition

logger = logging.getLogger('metrics-collector')


@dataclass
class SystemState:
    """Represents the current state of the system for control purposes."""
    queue_length: int           # Kafka consumer lag (number of messages)
    cpu_util: float             # ES process CPU utilization (%)
    mem_util: float             # ES JVM heap used (%)
    indexing_time_rate: float   # ms of indexing per second
    io_write_ops: float         # ES disk I/O write operations per second
    os_cpu_percent: int         # OS-level CPU (%) — for comparison
    os_mem_used_percent: int    # OS-level RAM used (%) — for comparison
    gc_time_rate: float         # ms of GC per second
    write_queue_size: int       # write thread pool queue depth
    timestamp: datetime         # When this state was captured

    def to_dict(self) -> Dict[str, Any]:
        return {
            'queue_length': self.queue_length,
            'cpu_util': self.cpu_util,
            'mem_util': self.mem_util,
            'indexing_time_rate': self.indexing_time_rate,
            'io_write_ops': self.io_write_ops,
            'os_cpu_percent': self.os_cpu_percent,
            'os_mem_used_percent': self.os_mem_used_percent,
            'gc_time_rate': self.gc_time_rate,
            'write_queue_size': self.write_queue_size,
            'timestamp': self.timestamp.isoformat(),
        }

    def to_vector(self) -> list:
        """Convert state to vector format for LQR controller."""
        return [
            self.queue_length,
            self.cpu_util,
            self.mem_util,
            self.indexing_time_rate,
            self.io_write_ops,
            self.os_cpu_percent,
            self.os_mem_used_percent,
            self.gc_time_rate,
            self.write_queue_size,
        ]


class MetricsCollector:
    """
    Collects metrics from Elasticsearch _nodes/stats API and Kafka.
    """

    def __init__(
        self,
        kafka_servers: str = None,
        kafka_topic: str = None,
        kafka_group_id: str = None,
    ):
        # Kafka configuration
        self.kafka_servers = kafka_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = kafka_topic or os.getenv('KAFKA_TOPIC', 'cdc.postgres.changes')
        self.kafka_group_id = kafka_group_id or os.getenv('KAFKA_GROUP_ID', 'message-sink-group')

        # Kafka admin client for lag calculation
        self._kafka_consumer: Optional[KafkaConsumer] = None

        # Elasticsearch stats configuration
        self.es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')

        # Cache for indexing_time rate calculation
        self._last_indexing_time_ms: Optional[int] = None
        self._last_indexing_time_ts: Optional[float] = None

        # Cache for CPU rate calculation
        self._last_cpu_time_ms: Optional[int] = None
        self._last_cpu_time_ts: Optional[float] = None

        # Cache for disk I/O write ops rate calculation
        self._last_io_write_ops: Optional[int] = None
        self._last_io_write_ts: Optional[float] = None

        # Cache for GC time rate calculation
        self._last_gc_time_ms: Optional[int] = None
        self._last_gc_time_ts: Optional[float] = None

    def get_kafka_consumer_lag(self) -> int:
        """
        Get the consumer lag (number of messages behind) for the Kafka topic.
        """
        try:
            if self._kafka_consumer is None:
                self._kafka_consumer = KafkaConsumer(
                    bootstrap_servers=self.kafka_servers.split(','),
                    group_id=self.kafka_group_id,
                    enable_auto_commit=False,
                )

            partitions = self._kafka_consumer.partitions_for_topic(self.kafka_topic)
            if not partitions:
                return 0

            total_lag = 0
            for partition in partitions:
                tp = TopicPartition(self.kafka_topic, partition)

                end_offsets = self._kafka_consumer.end_offsets([tp])
                end_offset = end_offsets.get(tp, 0)

                committed = self._kafka_consumer.committed(tp)
                committed_offset = committed if committed else 0

                lag = end_offset - committed_offset
                total_lag += max(lag, 0)

            return total_lag

        except Exception as e:
            logger.warning(f"Error getting Kafka consumer lag: {e}")
            return 0

    def get_es_native_stats(self) -> tuple:
        """
        Fetch ES-native metrics from _nodes/stats API.
        Returns (cpu_util, mem_util, indexing_time_rate, io_write_ops, os_cpu_percent, os_mem_used_percent, gc_time_rate, write_queue_size).
        """
        try:
            url = f"{self.es_url}/_nodes/stats/process,jvm,indices,fs,os,thread_pool"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                logger.warning(f"ES _nodes/stats returned status {response.status_code}")
                return (0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)

            data = response.json()
            nodes = data.get('nodes', {})
            if not nodes:
                return (0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)

            # Single-node: get the first (and only) node
            node = next(iter(nodes.values()))

            # CPU utilization — rate from process.cpu.total_in_millis
            cpu_time_ms = node.get('process', {}).get('cpu', {}).get('total_in_millis', 0)
            now = time.time()

            cpu_util = 0.0
            if self._last_cpu_time_ms is not None and self._last_cpu_time_ts is not None:
                elapsed = now - self._last_cpu_time_ts
                if elapsed > 0:
                    delta_ms = cpu_time_ms - self._last_cpu_time_ms
                    cpu_util = max((delta_ms / (elapsed * 1000)) * 100, 0.0)

            self._last_cpu_time_ms = cpu_time_ms
            self._last_cpu_time_ts = now

            # Memory utilization (JVM heap used %)
            mem_util = node.get('jvm', {}).get('mem', {}).get('heap_used_percent', 0)

            # Indexing time rate (ms/sec) — requires rate calculation
            indexing_time_ms = node.get('indices', {}).get('indexing', {}).get('index_time_in_millis', 0)

            indexing_time_rate = 0.0
            if self._last_indexing_time_ms is not None and self._last_indexing_time_ts is not None:
                elapsed = now - self._last_indexing_time_ts
                if elapsed > 0:
                    delta_ms = indexing_time_ms - self._last_indexing_time_ms
                    indexing_time_rate = max(delta_ms / elapsed, 0.0)

            self._last_indexing_time_ms = indexing_time_ms
            self._last_indexing_time_ts = now

            # Disk I/O write ops rate (ops/sec) from fs.io_stats
            io_write_ops_total = node.get('fs', {}).get('io_stats', {}).get('total', {}).get('write_operations', 0)

            io_write_ops = 0.0
            if self._last_io_write_ops is not None and self._last_io_write_ts is not None:
                elapsed = now - self._last_io_write_ts
                if elapsed > 0:
                    delta = io_write_ops_total - self._last_io_write_ops
                    io_write_ops = max(delta / elapsed, 0.0)

            self._last_io_write_ops = io_write_ops_total
            self._last_io_write_ts = now

            # OS-level CPU and memory (snapshot, for comparison)
            os_cpu_percent = node.get('os', {}).get('cpu', {}).get('percent', 0)
            os_mem_used_percent = node.get('os', {}).get('mem', {}).get('used_percent', 0)

            # GC time rate (ms/sec) — rate from young gen GC time
            gc_time_ms = node.get('jvm', {}).get('gc', {}).get('collectors', {}).get('young', {}).get('collection_time_in_millis', 0)

            gc_time_rate = 0.0
            if self._last_gc_time_ms is not None and self._last_gc_time_ts is not None:
                elapsed = now - self._last_gc_time_ts
                if elapsed > 0:
                    delta_ms = gc_time_ms - self._last_gc_time_ms
                    gc_time_rate = max(delta_ms / elapsed, 0.0)

            self._last_gc_time_ms = gc_time_ms
            self._last_gc_time_ts = now

            # Write thread pool queue depth (snapshot)
            write_queue_size = node.get('thread_pool', {}).get('write', {}).get('queue', 0)

            return (cpu_util, mem_util, indexing_time_rate, io_write_ops, os_cpu_percent, os_mem_used_percent, gc_time_rate, write_queue_size)

        except Exception as e:
            logger.warning(f"Error fetching ES native stats: {e}")
            return (0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)

    def collect_state(self) -> SystemState:
        """
        Collect current system state from all sources.
        Returns a SystemState object with all metrics.
        """
        # Get Kafka consumer lag
        queue_length = self.get_kafka_consumer_lag()

        # Get ES-native metrics
        cpu_util, mem_util, indexing_time_rate, io_write_ops, os_cpu_percent, os_mem_used_percent, gc_time_rate, write_queue_size = self.get_es_native_stats()

        state = SystemState(
            queue_length=queue_length,
            cpu_util=cpu_util,
            mem_util=mem_util,
            indexing_time_rate=indexing_time_rate,
            io_write_ops=io_write_ops,
            os_cpu_percent=os_cpu_percent,
            os_mem_used_percent=os_mem_used_percent,
            gc_time_rate=gc_time_rate,
            write_queue_size=write_queue_size,
            timestamp=datetime.utcnow(),
        )

        logger.debug(f"Collected state: {state.to_dict()}")
        return state

    def close(self):
        """Clean up resources."""
        if self._kafka_consumer:
            self._kafka_consumer.close()
            logger.info("Kafka consumer closed")


# Standalone test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    collector = MetricsCollector()

    print("Collecting metrics every 2 seconds. Press Ctrl+C to stop.")
    try:
        while True:
            state = collector.collect_state()
            print(f"\nState: {state.to_dict()}")
            print(f"Vector: {state.to_vector()}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        collector.close()
