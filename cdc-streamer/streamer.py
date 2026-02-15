"""
CDC Streamer - PostgreSQL to Kafka
Captures changes from PostgreSQL using logical replication and streams to Kafka.
This is a pure streaming component without any control logic.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import LogicalReplicationConnection
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('cdc-streamer')


class CDCStreamer:
    """
    Change Data Capture Streamer from PostgreSQL to Kafka.
    Uses PostgreSQL logical replication to capture changes.
    """

    def __init__(self):
        # PostgreSQL configuration
        self.pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.pg_port = int(os.getenv('POSTGRES_PORT', 5432))
        self.pg_user = os.getenv('POSTGRES_USER', 'postgres')
        self.pg_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        self.pg_database = os.getenv('POSTGRES_DB', 'cqrs_write')

        # Kafka configuration
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'cdc.postgres.changes')

        # Replication configuration
        self.slot_name = os.getenv('REPLICATION_SLOT_NAME', 'cdc_slot')
        self.poll_interval = float(os.getenv('POLL_INTERVAL_SEC', 1))

        # Connections
        self.pg_conn: Optional[Any] = None
        self.pg_cursor: Optional[Any] = None
        self.kafka_producer: Optional[KafkaProducer] = None

        # Stats
        self.messages_sent = 0
        self.last_lsn = None

    def connect_postgres(self) -> bool:
        """Establish connection to PostgreSQL."""
        try:
            # Regular connection for setup
            conn_params = {
                'host': self.pg_host,
                'port': self.pg_port,
                'user': self.pg_user,
                'password': self.pg_password,
                'database': self.pg_database,
            }

            self.pg_conn = psycopg2.connect(**conn_params)
            self.pg_conn.autocommit = True
            self.pg_cursor = self.pg_conn.cursor()

            logger.info(f"Connected to PostgreSQL at {self.pg_host}:{self.pg_port}/{self.pg_database}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    def setup_replication_slot(self) -> bool:
        """Create logical replication slot if it doesn't exist."""
        try:
            # Check if slot exists
            self.pg_cursor.execute(
                "SELECT slot_name FROM pg_replication_slots WHERE slot_name = %s",
                (self.slot_name,)
            )
            if self.pg_cursor.fetchone():
                logger.info(f"Replication slot '{self.slot_name}' already exists")
                return True

            # Create slot using test_decoding (built-in plugin)
            # Note: wal2json requires extension installation, using test_decoding for simplicity
            self.pg_cursor.execute(
                sql.SQL("SELECT pg_create_logical_replication_slot(%s, %s)"),
                (self.slot_name, 'test_decoding')
            )
            logger.info(f"Created replication slot '{self.slot_name}' with test_decoding plugin")
            return True

        except psycopg2.errors.DuplicateObject:
            logger.info(f"Replication slot '{self.slot_name}' already exists")
            return True
        except Exception as e:
            logger.error(f"Failed to create replication slot: {e}")
            return False

    def connect_kafka(self) -> bool:
        """Establish connection to Kafka."""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                retry_backoff_ms=500,
            )
            logger.info(f"Connected to Kafka at {self.kafka_servers}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    def parse_test_decoding(self, data: str) -> Optional[Dict[str, Any]]:
        """
        Parse test_decoding output format.
        Example formats:
        - table public.orders: INSERT: id[integer]:1 customer_name[character varying]:'Alice Johnson' quantity[integer]:2 unit_price[numeric]:150000.00 metadata[jsonb]:'{"color": "black"}'
        - table public.orders: UPDATE: id[integer]:1 status[character varying]:'shipped'
        - table public.orders: DELETE: id[integer]:1
        """
        try:
            if not data.startswith('table '):
                return None

            # Parse table name and operation
            parts = data.split(': ', 2)
            if len(parts) < 3:
                return None

            table_part = parts[0]  # "table public.orders"
            operation = parts[1]   # "INSERT", "UPDATE", "DELETE"
            columns_part = parts[2]  # "id[integer]:1 customer_name[character varying]:'Alice Johnson' ..."

            table_name = table_part.replace('table ', '')

            # Parse columns using a state-machine approach
            # Handles: name[type]:value or name[type]:'quoted value with spaces'
            columns = {}
            pos = 0
            text = columns_part

            while pos < len(text):
                # Skip whitespace
                while pos < len(text) and text[pos] == ' ':
                    pos += 1
                if pos >= len(text):
                    break

                # Find column name: everything before '['
                bracket_start = text.find('[', pos)
                if bracket_start == -1:
                    break
                col_name = text[pos:bracket_start]

                # Find type: everything between '[' and ']:'
                colon_after_bracket = text.find(']:', bracket_start)
                if colon_after_bracket == -1:
                    break
                col_type = text[bracket_start + 1:colon_after_bracket]

                # Value starts after ']:'
                val_start = colon_after_bracket + 2

                if val_start >= len(text):
                    columns[col_name] = None
                    break

                if text[val_start] == "'":
                    # Quoted value — find matching closing quote
                    # Closing quote is a ' followed by space+name[ or end-of-string
                    search_pos = val_start + 1
                    while search_pos < len(text):
                        quote_pos = text.find("'", search_pos)
                        if quote_pos == -1:
                            # No closing quote found, take rest of string
                            quote_pos = len(text)
                            break
                        # Check if this quote is the end: followed by ' '+col_def or end
                        after = quote_pos + 1
                        if after >= len(text):
                            break  # end of string
                        if text[after] == ' ':
                            # Check if next token looks like a column def (contains '[')
                            next_bracket = text.find('[', after)
                            next_space = text.find(' ', after + 1)
                            if next_bracket != -1 and (next_space == -1 or next_bracket < next_space):
                                break  # this quote is the real end
                            # Could be space inside value, keep going
                            search_pos = quote_pos + 1
                        else:
                            search_pos = quote_pos + 1

                    raw_value = text[val_start + 1:quote_pos]
                    pos = quote_pos + 1

                    # Convert quoted value
                    columns[col_name] = self._convert_value(raw_value, col_type)
                else:
                    # Unquoted value — ends at next space or end of string
                    space_pos = text.find(' ', val_start)
                    if space_pos == -1:
                        raw_value = text[val_start:]
                        pos = len(text)
                    else:
                        raw_value = text[val_start:space_pos]
                        pos = space_pos + 1

                    columns[col_name] = self._convert_value(raw_value, col_type)

            return {
                'table': table_name,
                'operation': operation,
                'data': columns,
                'timestamp': datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Failed to parse change data: {e}, raw: {data[:200]}")
            return None

    def _convert_value(self, value: str, col_type: str) -> Any:
        """Convert string value to appropriate Python type."""
        if value == 'null':
            return None

        try:
            if col_type in ('integer', 'bigint', 'smallint'):
                return int(value)
            elif col_type in ('real', 'double precision', 'numeric'):
                return float(value)
            elif col_type == 'boolean':
                return value.lower() == 'true'
            elif col_type == 'jsonb' or col_type == 'json':
                return json.loads(value)
            else:
                return value
        except:
            return value

    def poll_changes(self) -> int:
        """
        Poll for changes from PostgreSQL replication slot.
        Returns number of changes processed.
        """
        try:
            # Get changes from slot
            self.pg_cursor.execute(
                sql.SQL("SELECT lsn, xid, data FROM pg_logical_slot_get_changes(%s, NULL, NULL)"),
                (self.slot_name,)
            )
            changes = self.pg_cursor.fetchall()

            count = 0
            for lsn, xid, data in changes:
                # Parse the change
                parsed = self.parse_test_decoding(data)
                if parsed is None:
                    continue

                # Add metadata
                parsed['lsn'] = str(lsn)
                parsed['xid'] = xid

                # Send to Kafka
                key = f"{parsed['table']}:{parsed.get('data', {}).get('id', 'unknown')}"
                self.send_to_kafka(key, parsed)
                count += 1
                self.last_lsn = lsn

            if count > 0:
                logger.info(f"Processed {count} changes, last LSN: {self.last_lsn}")

            return count

        except Exception as e:
            logger.error(f"Error polling changes: {e}")
            return 0

    def send_to_kafka(self, key: str, message: Dict[str, Any]) -> bool:
        """Send message to Kafka topic."""
        try:
            future = self.kafka_producer.send(
                self.kafka_topic,
                key=key,
                value=message
            )
            # Wait for send to complete (with timeout)
            future.get(timeout=10)
            self.messages_sent += 1
            logger.debug(f"Sent message to Kafka: {key}")
            return True

        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False

    def run(self):
        """Main streaming loop."""
        logger.info("Starting CDC Streamer...")

        # Connect to PostgreSQL
        while not self.connect_postgres():
            logger.warning("Retrying PostgreSQL connection in 5 seconds...")
            time.sleep(5)

        # Setup replication slot
        while not self.setup_replication_slot():
            logger.warning("Retrying replication slot setup in 5 seconds...")
            time.sleep(5)

        # Connect to Kafka
        while not self.connect_kafka():
            logger.warning("Retrying Kafka connection in 5 seconds...")
            time.sleep(5)

        logger.info(f"CDC Streamer running. Polling every {self.poll_interval} seconds.")
        logger.info(f"Publishing to Kafka topic: {self.kafka_topic}")

        # Main loop
        try:
            while True:
                self.poll_changes()
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down CDC Streamer...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up connections."""
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
            logger.info("Kafka producer closed")

        if self.pg_cursor:
            self.pg_cursor.close()
        if self.pg_conn:
            self.pg_conn.close()
            logger.info("PostgreSQL connection closed")

        logger.info(f"Total messages sent: {self.messages_sent}")


if __name__ == '__main__':
    streamer = CDCStreamer()
    streamer.run()
