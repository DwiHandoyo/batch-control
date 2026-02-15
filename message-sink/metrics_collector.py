"""
Metrics Collector - Collects system metrics from cAdvisor and Kafka.
Provides state variables for the LQR controller:
- queue_length: Kafka consumer lag
- cpu_util: Elasticsearch container CPU utilization (%)
- mem_util: Elasticsearch container memory utilization (%)
- io_ops: Elasticsearch container I/O operations (bytes/sec)
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
    queue_length: int       # Kafka consumer lag (number of messages)
    cpu_util: float         # CPU utilization percentage (0-100)
    mem_util: float         # Memory utilization percentage (0-100)
    io_ops: float           # I/O operations (bytes per second)
    timestamp: datetime     # When this state was captured

    def to_dict(self) -> Dict[str, Any]:
        return {
            'queue_length': self.queue_length,
            'cpu_util': self.cpu_util,
            'mem_util': self.mem_util,
            'io_ops': self.io_ops,
            'timestamp': self.timestamp.isoformat(),
        }

    def to_vector(self) -> list:
        """Convert state to vector format for LQR controller."""
        return [
            self.queue_length,
            self.cpu_util,
            self.mem_util,
            self.io_ops,
        ]


class MetricsCollector:
    """
    Collects metrics from cAdvisor and Kafka for system state observation.
    """

    def __init__(
        self,
        cadvisor_url: str = None,
        elasticsearch_container: str = None,
        kafka_servers: str = None,
        kafka_topic: str = None,
        kafka_group_id: str = None,
    ):
        # cAdvisor configuration
        self.cadvisor_url = cadvisor_url or os.getenv('CADVISOR_URL', 'http://localhost:8080')
        self.es_container = elasticsearch_container or os.getenv(
            'ELASTICSEARCH_CONTAINER_NAME', 'elasticsearch-read'
        )

        # Kafka configuration
        self.kafka_servers = kafka_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = kafka_topic or os.getenv('KAFKA_TOPIC', 'cdc.postgres.changes')
        self.kafka_group_id = kafka_group_id or os.getenv('KAFKA_GROUP_ID', 'message-sink-group')

        # Cache for rate calculations
        self._last_cpu_stats: Optional[Dict] = None
        self._last_io_stats: Optional[Dict] = None
        self._last_collect_time: Optional[float] = None

        # Kafka admin client for lag calculation
        self._kafka_consumer: Optional[KafkaConsumer] = None

        # Cache for container ID (resolved from container name)
        self._container_id: Optional[str] = None

    def _resolve_container_id(self) -> Optional[str]:
        """
        Resolve container name to container ID.
        First tries Docker API, then falls back to cAdvisor scanning.
        """
        if self._container_id:
            return self._container_id

        # Method 1: Try Docker API via unix socket
        container_id = self._resolve_via_docker_api()
        if container_id:
            self._container_id = container_id
            return container_id

        # Method 2: If es_container looks like a container ID, use it directly
        if len(self.es_container) >= 12 and all(c in '0123456789abcdef' for c in self.es_container.lower()):
            self._container_id = self.es_container
            logger.info(f"Using provided container ID: {self.es_container[:12]}...")
            return self._container_id

        # Method 3: Search in cAdvisor subcontainers
        container_id = self._resolve_via_cadvisor()
        if container_id:
            self._container_id = container_id
            return container_id

        logger.warning(f"Could not find container '{self.es_container}'")
        return None

    def _resolve_via_docker_api(self) -> Optional[str]:
        """Try to resolve container ID using Docker API."""
        try:
            # Docker API endpoint - works with container name or ID
            docker_url = f"http://localhost:2375/containers/{self.es_container}/json"
            response = requests.get(docker_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                container_id = data.get('Id', '')
                if container_id:
                    logger.info(f"Resolved via Docker API: {container_id[:12]}...")
                    return container_id
        except:
            pass

        # Try unix socket via requests-unixsocket if available
        try:
            import requests_unixsocket
            session = requests_unixsocket.Session()
            url = f"http+unix://%2Fvar%2Frun%2Fdocker.sock/containers/{self.es_container}/json"
            response = session.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                container_id = data.get('Id', '')
                if container_id:
                    logger.info(f"Resolved via Docker socket: {container_id[:12]}...")
                    return container_id
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Docker socket lookup failed: {e}")

        return None

    def _resolve_via_cadvisor(self) -> Optional[str]:
        """Search for container in cAdvisor subcontainers."""
        try:
            url = f"{self.cadvisor_url}/api/v1.3/subcontainers/docker"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                return None

            containers = response.json()

            # Search for container by partial ID match
            logger.info(f"Searching for container '{self.es_container}' in cAdvisor...")
            for container in containers:
                container_path = container.get('name', '')
                if '/docker/' in container_path and container_path != '/docker' and '/buildx' not in container_path:
                    container_id = container_path.split('/docker/')[-1]
                    if container_id and '/' not in container_id:
                        if self.es_container.lower() in container_id.lower():
                            logger.info(f"Found container ID: {container_id[:12]}...")
                            return container_id

            return None

        except Exception as e:
            logger.debug(f"cAdvisor lookup failed: {e}")
            return None

    def _get_cadvisor_container_stats(self) -> Optional[Dict]:
        """Fetch container stats from cAdvisor API."""
        try:
            # First, resolve container name to ID
            container_id = self._resolve_container_id()

            if not container_id:
                # Fallback: try to get stats from all containers and find by name pattern
                return self._get_stats_from_subcontainers()

            # cAdvisor API endpoint for container stats using container ID
            url = f"{self.cadvisor_url}/api/v1.3/containers/docker/{container_id}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                # Response is a single container object with 'stats' array
                if data and 'stats' in data:
                    stats = data['stats']
                    if stats:
                        # Return the most recent stats
                        return stats[-1]
            else:
                logger.warning(f"cAdvisor returned status {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch cAdvisor stats: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing cAdvisor response: {e}")
            return None

    def _get_stats_from_subcontainers(self) -> Optional[Dict]:
        """
        Fallback method: Get stats from subcontainers endpoint.
        This is less efficient but works when we can't resolve container ID.
        """
        try:
            url = f"{self.cadvisor_url}/api/v1.3/subcontainers/docker"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                return None

            containers = response.json()

            # Find the container with most memory usage (likely Elasticsearch)
            # or match by partial name/id
            target_container = None
            for container in containers:
                container_path = container.get('name', '')
                if '/docker/' in container_path and container_path != '/docker':
                    # Skip buildx and other system containers
                    if 'buildx' in container_path:
                        continue

                    stats = container.get('stats', [])
                    if stats:
                        # Check if this might be our target by memory usage
                        # Elasticsearch typically uses more memory
                        mem_usage = stats[-1].get('memory', {}).get('usage', 0)
                        if mem_usage > 500_000_000:  # > 500MB
                            if target_container is None:
                                target_container = container
                            else:
                                # Compare memory usage
                                current_mem = target_container.get('stats', [{}])[-1].get('memory', {}).get('usage', 0)
                                if mem_usage > current_mem:
                                    target_container = container

            if target_container and target_container.get('stats'):
                container_id = target_container.get('name', '').split('/docker/')[-1]
                logger.info(f"Using container with highest memory: {container_id[:12]}...")
                self._container_id = container_id
                return target_container['stats'][-1]

            return None

        except Exception as e:
            logger.error(f"Error getting stats from subcontainers: {e}")
            return None

    def _get_cadvisor_machine_stats(self) -> Optional[Dict]:
        """Fetch machine info from cAdvisor for total resources."""
        try:
            url = f"{self.cadvisor_url}/api/v1.3/machine"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.warning(f"Failed to fetch machine stats: {e}")
            return None

    def get_cpu_utilization(self, stats: Dict) -> float:
        """
        Calculate CPU utilization percentage from cAdvisor stats.
        Uses the difference between current and previous readings.
        """
        try:
            current_time = time.time()
            current_usage = stats.get('cpu', {}).get('usage', {}).get('total', 0)

            if self._last_cpu_stats is None:
                self._last_cpu_stats = {
                    'usage': current_usage,
                    'time': current_time,
                }
                return 0.0

            # Calculate CPU usage over the interval
            usage_diff = current_usage - self._last_cpu_stats['usage']
            time_diff = current_time - self._last_cpu_stats['time']

            # Update cache
            self._last_cpu_stats = {
                'usage': current_usage,
                'time': current_time,
            }

            if time_diff <= 0:
                return 0.0

            # Convert nanoseconds to percentage
            # CPU usage is in nanoseconds, time is in seconds
            # For 1 CPU: 1 second = 1,000,000,000 nanoseconds = 100%
            cpu_percent = (usage_diff / (time_diff * 1e9)) * 100

            # Clamp to reasonable range
            return min(max(cpu_percent, 0.0), 100.0)

        except Exception as e:
            logger.warning(f"Error calculating CPU utilization: {e}")
            return 0.0

    def get_memory_utilization(self, stats: Dict) -> float:
        """
        Calculate memory utilization percentage from cAdvisor stats.
        """
        try:
            memory_stats = stats.get('memory', {})
            usage = memory_stats.get('usage', 0)
            limit = memory_stats.get('limit', 0)

            # If no limit set, try to get from machine info
            if limit == 0 or limit > 1e15:  # Very large number means no limit
                machine = self._get_cadvisor_machine_stats()
                if machine:
                    limit = machine.get('memory_capacity', 0)

            if limit <= 0:
                return 0.0

            mem_percent = (usage / limit) * 100
            return min(max(mem_percent, 0.0), 100.0)

        except Exception as e:
            logger.warning(f"Error calculating memory utilization: {e}")
            return 0.0

    def get_io_operations(self, stats: Dict) -> float:
        """
        Calculate I/O operations (bytes per second) from cAdvisor stats.
        """
        try:
            current_time = time.time()
            diskio = stats.get('diskio', {})

            # Sum all read and write bytes
            io_stats = diskio.get('io_service_bytes', [])
            current_bytes = 0
            for stat in io_stats:
                current_bytes += stat.get('stats', {}).get('Read', 0)
                current_bytes += stat.get('stats', {}).get('Write', 0)

            if self._last_io_stats is None:
                self._last_io_stats = {
                    'bytes': current_bytes,
                    'time': current_time,
                }
                return 0.0

            # Calculate bytes per second
            bytes_diff = current_bytes - self._last_io_stats['bytes']
            time_diff = current_time - self._last_io_stats['time']

            # Update cache
            self._last_io_stats = {
                'bytes': current_bytes,
                'time': current_time,
            }

            if time_diff <= 0:
                return 0.0

            return max(bytes_diff / time_diff, 0.0)

        except Exception as e:
            logger.warning(f"Error calculating I/O operations: {e}")
            return 0.0

    def get_kafka_consumer_lag(self) -> int:
        """
        Get the consumer lag (number of messages behind) for the Kafka topic.
        """
        try:
            # Create a temporary consumer to check offsets
            if self._kafka_consumer is None:
                self._kafka_consumer = KafkaConsumer(
                    bootstrap_servers=self.kafka_servers.split(','),
                    group_id=self.kafka_group_id,
                    enable_auto_commit=False,
                )

            # Get partitions for the topic
            partitions = self._kafka_consumer.partitions_for_topic(self.kafka_topic)
            if not partitions:
                return 0

            total_lag = 0
            for partition in partitions:
                tp = TopicPartition(self.kafka_topic, partition)

                # Get end offset (latest message)
                end_offsets = self._kafka_consumer.end_offsets([tp])
                end_offset = end_offsets.get(tp, 0)

                # Get committed offset (last processed)
                committed = self._kafka_consumer.committed(tp)
                committed_offset = committed if committed else 0

                # Calculate lag
                lag = end_offset - committed_offset
                total_lag += max(lag, 0)

            return total_lag

        except Exception as e:
            logger.warning(f"Error getting Kafka consumer lag: {e}")
            return 0

    def collect_state(self) -> SystemState:
        """
        Collect current system state from all sources.
        Returns a SystemState object with all metrics.
        """
        # Get cAdvisor stats
        stats = self._get_cadvisor_container_stats()

        if stats:
            cpu_util = self.get_cpu_utilization(stats)
            mem_util = self.get_memory_utilization(stats)
            io_ops = self.get_io_operations(stats)
        else:
            logger.warning("Could not get cAdvisor stats, using defaults")
            cpu_util = 0.0
            mem_util = 0.0
            io_ops = 0.0

        # Get Kafka consumer lag
        queue_length = self.get_kafka_consumer_lag()

        state = SystemState(
            queue_length=queue_length,
            cpu_util=cpu_util,
            mem_util=mem_util,
            io_ops=io_ops,
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
