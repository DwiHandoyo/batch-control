"""
Closed-Loop Experiment — Test 11 controllers × 4 load patterns.

Tests each controller mode (static, rule_based, pid, lqr_q1-q4, ann_q1-q4)
against each load pattern (step, ramp, impulse, periodic_step) with continuous
load injection via PostgreSQL CDC → Kafka.

Load patterns (thesis chapter-3 §III.4.2):
  - step:           constant injection at base_rate from t=0
  - ramp:           linear increase from 0 to base_rate over duration
  - impulse:        burst at base_rate for first 15% of duration, then 0
  - periodic_step:  alternates base_rate and 0 every period_sec seconds

Usage:
    python closed_loop_experiment.py --sysid-json <path> --ann-model <path> [options]
    python closed_loop_experiment.py --modes static pid lqr --load-patterns step ramp
"""

import os
import sys
import csv
import time
import json
import signal
import random
import string
import logging
import argparse
import threading
import numpy as np
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'message-sink'))
from metrics_collector import MetricsCollector, SystemState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('closed-loop')

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')

STATE_VARS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops', 'indexing_time_rate']
CONTROL_VARS = ['batch_size', 'poll_interval']

STATUSES = ['pending', 'confirmed', 'processing', 'shipped', 'delivered']


# ─── Load Pattern Definitions ────────────────────────────────────────
# Each pattern returns the injection rate (msg/s) at time t.
# Patterns are parameterized via kwargs from CLI.

def _rate_step(t: float, **kwargs) -> int:
    """Constant 1000 msg/s for 30s. Total = 30,000."""
    rate = kwargs.get('step_rate', 1000)
    duration = kwargs.get('step_duration', 30)
    return rate if t < duration else 0


def _rate_ramp(t: float, **kwargs) -> int:
    """Linear 0→2000 msg/s over 30s. Total = 30,000."""
    peak_rate = kwargs.get('ramp_peak_rate', 2000)
    duration = kwargs.get('ramp_duration', 30)
    if t >= duration:
        return 0
    return max(1, int(peak_rate * t / duration))


def _rate_impulse(t: float, **kwargs) -> int:
    """Burst 10,000 msg/s for 1s. Total = 10,000."""
    rate = kwargs.get('impulse_rate', 10000)
    burst_sec = kwargs.get('impulse_burst_sec', 1)
    return rate if t < burst_sec else 0


def _rate_periodic_step(t: float, **kwargs) -> int:
    """Alternating 1500/500 msg/s, 3s each, for 30s. Total = 30,000."""
    high_rate = kwargs.get('periodic_high_rate', 1500)
    low_rate = kwargs.get('periodic_low_rate', 500)
    half_period = kwargs.get('periodic_half_period', 3)
    duration = kwargs.get('periodic_duration', 30)
    if t >= duration:
        return 0
    phase = int(t / half_period) % 2
    return high_rate if phase == 0 else low_rate


def _rate_step_low(t: float, **kwargs) -> int:
    """Low rate step: 100 msg/s for 60s. Total = 6,000."""
    rate = kwargs.get('step_low_rate', 100)
    duration = kwargs.get('step_low_duration', 60)
    return rate if t < duration else 0


LOAD_PATTERNS: Dict[str, Callable] = {
    'step': _rate_step,
    'ramp': _rate_ramp,
    'impulse': _rate_impulse,
    'periodic_step': _rate_periodic_step,
    'step_low': _rate_step_low,
}

# Duration of injection phase per pattern (for LoadInjector timeout)
PATTERN_DURATIONS: Dict[str, str] = {
    'step': 'step_duration',
    'ramp': 'ramp_duration',
    'impulse': 'impulse_burst_sec',
    'periodic_step': 'periodic_duration',
}
PATTERN_DURATION_DEFAULTS: Dict[str, float] = {
    'step': 30, 'ramp': 30, 'impulse': 1, 'periodic_step': 30,
}


# ─── Load Injector ────────────────────────────────────────────────────

class LoadInjector:
    """
    Injects load into PostgreSQL via UPDATE queries in a separate thread.
    UPDATEs trigger CDC → Kafka messages, creating load for the sink.
    Rate profile is determined by the pattern function (step, ramp, impulse, etc.)
    """

    def __init__(
        self,
        pg_config: Dict,
        pattern_name: str,
        pattern_kwargs: Optional[Dict] = None,
    ):
        self.pg_config = pg_config
        self.pattern_fn = LOAD_PATTERNS[pattern_name]
        self.pattern_name = pattern_name
        self.pattern_kwargs = pattern_kwargs or {}

        self.pg_conn = None
        self.thread = None
        self.running = False
        self.done = False  # True when injection phase complete
        self.total_injected = 0
        self.current_rate = 0
        self.injection_log: List[tuple] = []

    def start(self):
        """Start the load injection thread."""
        self.pg_conn = psycopg2.connect(**self.pg_config)
        cur = self.pg_conn.cursor()
        cur.execute("SELECT MIN(id), MAX(id) FROM orders")
        self._id_min, self._id_max = cur.fetchone()
        self._update_offset = self._id_min
        cur.close()
        self.running = True
        self.done = False
        self.total_injected = 0
        self.current_rate = 0
        self.injection_log = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"LoadInjector started: pattern={self.pattern_name}")

    def stop(self):
        """Stop the load injection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=15)
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
        logger.info(f"LoadInjector stopped: total_injected={self.total_injected}")

    TICK_INTERVAL = 0.1  # 100ms tick for smooth injection

    def _run(self):
        """Main injection loop — 100ms ticks for smooth message production."""
        cursor = self.pg_conn.cursor()
        start_time = time.time()

        while self.running:
            tick_start = time.time()
            t = tick_start - start_time

            target_rate = self.pattern_fn(t, **self.pattern_kwargs)
            self.current_rate = target_rate

            if target_rate <= 0:
                if self.total_injected > 0:
                    future_rate = self.pattern_fn(t + 1, **self.pattern_kwargs)
                    if future_rate <= 0:
                        break
            else:
                # Inject fraction per tick: rate * tick_interval
                chunk = max(1, int(target_rate * self.TICK_INTERVAL))
                actual = self._do_updates(cursor, chunk)
                self.total_injected += actual
                self.injection_log.append((round(t, 2), target_rate, actual))

            # Sleep remainder of tick
            elapsed = time.time() - tick_start
            sleep_time = max(0, self.TICK_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.done = True
        self.current_rate = 0
        cursor.close()

    def _do_updates(self, cursor, count: int) -> int:
        """Execute UPDATE queries using id range (fast, no random)."""
        total_updated = 0
        remaining = count

        while remaining > 0:
            chunk = min(remaining, 5000)
            id_start = self._update_offset
            id_end = id_start + chunk - 1
            if id_end > self._id_max:
                self._update_offset = self._id_min
                id_start = self._id_min
                id_end = id_start + chunk - 1
            try:
                cursor.execute("""
                    UPDATE orders
                    SET status = (ARRAY['pending','confirmed','processing','shipped','delivered'])
                                 [floor(random()*5+1)::int],
                        updated_at = NOW()
                    WHERE id BETWEEN %s AND %s
                """, (id_start, id_end))
                self.pg_conn.commit()
                total_updated += chunk
            except Exception as e:
                self.pg_conn.rollback()
                logger.warning(f"Injector update failed: {e}")
            self._update_offset = id_end + 1
            remaining -= chunk

        return total_updated


# ─── Experiment Runner ────────────────────────────────────────────────

class ClosedLoopExperiment:
    """
    Closed-loop experiment: tests all controller modes against all load patterns.
    """

    def __init__(
        self,
        kafka_servers: str = 'localhost:29092',
        kafka_topic: str = 'cdc.public.orders',
        kafka_group_id: str = 'message-sink-group',
        sample_interval: float = 1.0,
        control_override_path: str = None,
        pg_host: str = 'localhost',
        pg_port: int = 5433,
        pg_db: str = 'cqrs_write',
        pg_user: str = 'postgres',
        pg_password: str = 'postgres',
        controller_modes: List[str] = None,
        load_patterns: List[str] = None,
        pattern_kwargs: Dict = None,
        max_test_duration: int = 600,
        repeats: int = 1,
        randomize_order: bool = False,
        seed: int = 42,
    ):
        self.sample_interval = sample_interval
        self.running = True

        self.control_override_path = control_override_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'message-sink', 'logs', 'control_override.json'
        )

        # Sink metrics CSV path (written by sink each cycle)
        self.sink_metrics_path = os.path.join(
            os.path.dirname(self.control_override_path), 'sink_metrics.csv'
        )

        self.metrics = MetricsCollector(
            kafka_servers=kafka_servers,
            kafka_topic=kafka_topic,
            kafka_group_id=kafka_group_id,
        )

        # PostgreSQL config for main connection + injector threads
        self.pg_config = {
            'host': pg_host, 'port': pg_port, 'dbname': pg_db,
            'user': pg_user, 'password': pg_password,
        }
        self.pg_conn = psycopg2.connect(**self.pg_config)
        logger.info(f"Connected to PostgreSQL at {pg_host}:{pg_port}/{pg_db}")

        self.controller_modes = controller_modes or [
            'static', 'rule_based',
            'pid',       # MIMO 2x2 PID
            'ann_cw_q1', 'ann_cw_q2', 'ann_cw_q4',
            'lqr_q1', 'lqr_q2', 'lqr_q4',
            # 'passive',  # DISABLED — no consumption baseline (enable when needed)
        ]
        self.load_patterns = load_patterns or [
            'step', 'ramp', 'impulse', 'periodic_step', 'step_low'
        ]
        self.pattern_kwargs = pattern_kwargs or {}
        self.max_test_duration = max_test_duration
        # Fixed time window per pattern (for fair J comparison)
        # Cost window = pattern duration + 10s buffer for final drain observation.
        # Derived from pattern_kwargs so CLI args like --step-duration propagate here.
        self.pattern_test_duration = {
            'step':          pattern_kwargs.get('step_duration', 30) + 10,
            'ramp':          pattern_kwargs.get('ramp_duration', 30) + 10,
            'impulse':       pattern_kwargs.get('impulse_burst_sec', 1) + 19,
            'periodic_step': pattern_kwargs.get('periodic_duration', 30) + 10,
            'step_low':      pattern_kwargs.get('step_low_duration', 60) + 20,
        }
        self.repeats = repeats
        self.randomize_order = randomize_order
        self.seed = seed

        self.data: List[Dict] = []

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Interrupted, saving data...")
        self.running = False

    # ─── Control Override Helpers ──────────────────────────────

    def set_pause(self, pause: bool):
        """Pause/unpause the sink."""
        override = {'pause': pause, 'batch_size': 1, 'poll_interval': 1000}
        with open(self.control_override_path, 'w') as f:
            json.dump(override, f)

    def _get_wal_lag_bytes(self) -> int:
        """Get WAL replication lag in bytes (0 = CDC fully caught up)."""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT COALESCE(
                    pg_current_wal_lsn() - confirmed_flush_lsn, 0
                )::bigint
                FROM pg_replication_slots
                WHERE slot_name = 'debezium_slot'
            """)
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"Failed to get WAL lag: {e}")
            return -1

    def drain_queue(self, wal_timeout=300, wal_threshold=100000):
        """
        Wait for CDC to flush WAL, then seek to end in Kafka.

        1. Wait until WAL replication lag is stable and below threshold
        2. Seek to end in Kafka to skip all published messages
        3. Verify queue near 0
        """
        # Phase 1: Wait for WAL flush (lag < threshold and stable)
        start = time.time()
        prev_lag = None
        stable_count = 0
        while time.time() - start < wal_timeout:
            lag = self._get_wal_lag_bytes()
            if lag < wal_threshold:
                # Check stability (lag not growing)
                if prev_lag is not None and lag <= prev_lag:
                    stable_count += 1
                    if stable_count >= 2:
                        logger.info(f"  WAL flushed (lag={lag} bytes, stable)")
                        break
                else:
                    stable_count = 0
            else:
                stable_count = 0
            prev_lag = lag
            elapsed = time.time() - start
            if int(elapsed) % 15 == 0:  # log every 15s
                logger.info(f"  Waiting for WAL flush [{elapsed:.0f}s]: lag={lag} bytes")
            time.sleep(5)
        else:
            lag = self._get_wal_lag_bytes()
            logger.warning(f"  WAL flush timeout after {wal_timeout}s (lag={lag} bytes)")

        # Phase 2: Seek to end in Kafka
        override = {'pause': True, 'seek_to_end': True, 'batch_size': 1, 'poll_interval': 1000}
        with open(self.control_override_path, 'w') as f:
            json.dump(override, f)
        time.sleep(3)

        # Phase 3: Verify
        state = self.metrics.collect_state()
        logger.info(f"  Post-drain queue={state.queue_length}")
        self.set_pause(True)

    def switch_controller_mode(self, mode: str):
        """Signal sink to switch its internal controller to the given mode.
        For 'passive' mode: keep sink paused (no controller switch needed).
        """
        if mode == 'passive':
            # Sink stays paused — no controller switch.
            # set_pause(True) was already called before this.
            return
        override = {'controller_mode': mode}
        with open(self.control_override_path, 'w') as f:
            json.dump(override, f)
        # Wait for sink to read override and delete the file
        for _ in range(20):
            time.sleep(0.5)
            if not os.path.exists(self.control_override_path):
                break
        else:
            logger.warning(f"  Controller switch to {mode} may not have been processed")

    def unpause_sink(self):
        """Remove control override to let the sink's controller run freely."""
        try:
            os.remove(self.control_override_path)
        except FileNotFoundError:
            pass

    # ─── Sink Metrics Reading ─────────────────────────────────

    def _read_latest_sink_metrics(self) -> Dict[str, str]:
        """Read the last line of sink_metrics.csv for actual control actions."""
        try:
            if not os.path.exists(self.sink_metrics_path):
                return {}
            with open(self.sink_metrics_path, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                return {}
            header = lines[0].strip().split(',')
            last_line = lines[-1].strip().split(',')
            if len(header) == len(last_line):
                return dict(zip(header, last_line))
        except Exception:
            pass
        return {}

    # ─── Sample Collection ────────────────────────────────────

    def collect_sample(
        self,
        controller_mode: str,
        load_pattern: str,
        elapsed_sec: float,
        trial_num: int,
        injector: Optional[LoadInjector] = None,
    ) -> Dict:
        """Collect one sample: state + control actions + injection info."""
        state = self.metrics.collect_state()
        sink = self._read_latest_sink_metrics()

        sample = {
            'timestamp': datetime.utcnow().isoformat(),
            'controller_mode': controller_mode,
            'load_pattern': load_pattern,
            'trial_num': trial_num,
            'elapsed_sec': round(elapsed_sec, 1),
            # State variables
            'queue_length': state.queue_length,
            'cpu_util': state.cpu_util,
            'container_mem_pct': state.container_mem_pct,
            'io_write_ops': state.io_write_ops,
            'indexing_time_rate': state.indexing_time_rate,
            'os_cpu_percent': state.os_cpu_percent,
            'os_mem_used_percent': state.os_mem_used_percent,
            # Control actions (from sink's actual controller output)
            'batch_size': int(sink.get('batch_size', 0)),
            'poll_interval_ms': int(sink.get('poll_interval_ms', 0)),
            'inv_poll_interval': 1000.0 / max(int(sink.get('poll_interval_ms', 1000)), 1),
            # Throughput metrics
            'messages_consumed': int(sink.get('messages_consumed', 0)),
            'messages_indexed': int(sink.get('messages_indexed', 0)),
            'cycle_duration_ms': float(sink.get('cycle_duration_ms', 0)),
            'avg_latency_ms': float(sink.get('avg_latency_ms', 0)),
            # Load injection info
            'injection_rate': injector.current_rate if injector else 0,
            'total_injected': injector.total_injected if injector else 0,
        }
        self.data.append(sample)
        return sample

    # ─── Single Controller + Pattern Test ─────────────────────

    def run_controller_test(
        self,
        mode: str,
        pattern_name: str,
        trial_num: int = 1,
    ) -> List[Dict]:
        """Run a single closed-loop test.

        Test ends when injector is done AND queue drains to ~0,
        or max_test_duration is reached.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {mode} + {pattern_name} (trial {trial_num})")
        logger.info(f"{'=' * 60}")

        # 1. Pause sink (so it exits polling loop and can read override)
        logger.info("  Pausing sink...")
        self.set_pause(True)
        time.sleep(2)

        # 2. Verify clean state
        wal_lag = self._get_wal_lag_bytes()
        state = self.metrics.collect_state()
        logger.info(f"  Pre-test: WAL lag={wal_lag} bytes, queue={state.queue_length}")

        # 3. Switch controller mode (sink is paused, will read override)
        logger.info(f"  Switching to {mode} controller...")
        self.switch_controller_mode(mode)

        # 3. Create and start load injector
        injector = LoadInjector(
            pg_config=self.pg_config,
            pattern_name=pattern_name,
            pattern_kwargs=self.pattern_kwargs,
        )
        logger.info(f"  Starting {pattern_name} injection...")
        injector.start()

        # For 'passive' mode: keep sink paused — observe raw queue growth without consumption.
        # This shows the natural injection pattern as a baseline.
        if mode == 'passive':
            logger.info("  PASSIVE mode: sink stays paused (no consumption)")
        else:
            self.unpause_sink()

        # 4. Collect metrics for fixed time window (fair J comparison)
        test_duration = self.pattern_test_duration.get(pattern_name, 40)
        logger.info(f"  Collecting for {test_duration}s (fixed window)...")
        start_time = time.time()
        samples_before = len(self.data)
        sample_count = 0

        while self.running and (time.time() - start_time) < test_duration:
            elapsed = time.time() - start_time
            sample = self.collect_sample(
                mode, pattern_name, elapsed, trial_num, injector
            )
            sample_count += 1

            if sample_count % 15 == 0:
                logger.info(
                    f"  [{elapsed:.0f}s] queue={sample['queue_length']}, "
                    f"cpu={sample['cpu_util']:.1f}%, "
                    f"batch={sample['batch_size']}, poll={sample['poll_interval_ms']}ms, "
                    f"inj={injector.total_injected}, "
                    f"latency={sample['avg_latency_ms']:.0f}ms"
                )

            time.sleep(self.sample_interval)

        # 5. Stop injector
        injector.stop()

        samples = self.data[samples_before:]
        cost_time = time.time() - start_time
        logger.info(f"  Cost window: {len(samples)} samples in {cost_time:.0f}s")
        logger.info(f"  Injected {injector.total_injected}")

        # 6. Drain — NO TIMEOUT — must fully empty queue + WAL
        # For passive mode: seek to end (discard accumulated messages — no drain needed,
        # passive is only for observing production pattern, not consumption).
        if mode == 'passive':
            logger.info("  PASSIVE done: seeking to end (discarding accumulated queue)...")
            self.drain_queue()  # drain_queue uses seek_to_end internally
            drain_time = time.time() - (time.time() - 3)  # near-instant
            total_time = cost_time + 3
            logger.info(f"  Drain: ~3s (seek_to_end), Total completion: {total_time:.0f}s")
            return self.data[samples_before:]

        drain_start = time.time()
        stable_count = 0
        while self.running:
            state = self.metrics.collect_state()
            q = state.queue_length
            wal = self._get_wal_lag_bytes()
            if q < 50:
                # Queue empty — check stability over 5s
                time.sleep(5)
                q2 = self.metrics.collect_state().queue_length
                if q2 < 50:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0
            else:
                stable_count = 0
                if int(time.time() - drain_start) % 30 == 0:
                    logger.info(f"  Draining: queue={q}, elapsed={time.time()-drain_start:.0f}s")
            time.sleep(1)

        drain_time = time.time() - drain_start
        total_time = cost_time + drain_time
        logger.info(f"  Drain: {drain_time:.0f}s, Total completion: {total_time:.0f}s")

        return samples

    # ─── Full Experiment ──────────────────────────────────────

    def run(self, output_dir: str):
        """Run all controller × pattern tests and save results."""
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

        # Build test schedule: iterate patterns → controllers (rolling order).
        # Rolling: each pattern starts from a different controller (cyclic shift by
        # pattern index). This distributes temporal bias evenly — no single controller
        # is always tested first or last across all patterns.
        # Example (9 controllers, 5 patterns):
        #   Step:          static[0], rule[1], pid[2], ann1[3], ...
        #   Ramp:          rule[1],   pid[2],  ann1[3], ...
        #   Impulse:       pid[2],    ann1[3], ...
        schedule = []
        modes_base = list(self.controller_modes)
        n_modes = len(modes_base)
        for trial in range(1, self.repeats + 1):
            for pat_idx, pattern in enumerate(self.load_patterns):
                if self.randomize_order:
                    rng = np.random.RandomState(self.seed + trial + pat_idx)
                    modes = list(modes_base)
                    rng.shuffle(modes)
                else:
                    # Rolling: shift by pattern index
                    modes = [modes_base[(j + pat_idx) % n_modes] for j in range(n_modes)]
                for mode in modes:
                    schedule.append((mode, pattern, trial))

        total_tests = len(schedule)
        est_time = total_tests * (self.max_test_duration // 2 + 60)  # rough estimate
        logger.info(f"Experiment plan: {total_tests} tests "
                    f"({len(self.controller_modes)} controllers × "
                    f"{len(self.load_patterns)} patterns × "
                    f"{self.repeats} repeats)")
        logger.info(f"Estimated time: ~{est_time // 60} min")

        # Save metadata
        metadata = {
            'run_id': os.path.basename(output_dir),
            'experiment_type': 'closed_loop_fixed_count',
            'start_time': datetime.utcnow().isoformat(),
            'controller_modes': self.controller_modes,
            'load_patterns': self.load_patterns,
            'pattern_kwargs': self.pattern_kwargs,
            'max_test_duration_sec': self.max_test_duration,
            'sample_interval_sec': self.sample_interval,
            'repeats': self.repeats,
            'randomize_order': self.randomize_order,
            'seed': self.seed,
            'schedule': [
                {'mode': m, 'pattern': p, 'trial': t}
                for m, p, t in schedule
            ],
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Run tests
        for i, (mode, pattern, trial) in enumerate(schedule):
            if not self.running:
                break
            logger.info(f"\n[{i+1}/{total_tests}] mode={mode}, "
                        f"pattern={pattern}, trial={trial}")
            self.run_controller_test(mode, pattern, trial)

        # Save data
        self._save_data(output_dir)

        # Generate comparison
        self._generate_comparison(output_dir)

        # Cleanup
        self.set_pause(True)
        if os.path.exists(self.control_override_path):
            os.remove(self.control_override_path)
        self.pg_conn.close()

        metadata['end_time'] = datetime.utcnow().isoformat()
        metadata['total_samples'] = len(self.data)
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nExperiment complete! {len(self.data)} total samples")
        logger.info(f"Results: {output_dir}")

    def _save_data(self, output_dir: str):
        """Save all data to CSV."""
        if not self.data:
            return

        csv_path = os.path.join(output_dir, 'results', 'closed_loop_data.csv')
        fieldnames = list(self.data[0].keys())

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

        logger.info(f"Saved {len(self.data)} samples to {csv_path}")

    def _generate_comparison(self, output_dir: str):
        """Generate a comparison summary grouped by load pattern."""
        if not self.data:
            return

        import pandas as pd
        df = pd.DataFrame(self.data)

        report_path = os.path.join(output_dir, 'results', 'comparison_summary.txt')
        lines = []
        lines.append("Closed-Loop Experiment Comparison Summary")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append(f"Pattern params: {self.pattern_kwargs}")
        lines.append("=" * 100)

        patterns = df['load_pattern'].unique()
        modes = df['controller_mode'].unique()

        for pattern in patterns:
            lines.append(f"\n{'=' * 100}")
            lines.append(f"  LOAD PATTERN: {pattern.upper()}")
            lines.append(f"{'=' * 100}")

            # Header
            header = f"{'Metric':<30s}"
            for mode in modes:
                header += f" {mode:>14s}"
            lines.append(header)
            lines.append("-" * (30 + 15 * len(modes)))

            def add_metric(name, values):
                row = f"{name:<30s}"
                for v in values:
                    if v is None:
                        row += f" {'N/A':>14s}"
                    elif isinstance(v, float):
                        row += f" {v:>14.2f}"
                    elif isinstance(v, int):
                        row += f" {v:>14d}"
                    else:
                        row += f" {str(v):>14s}"
                lines.append(row)

            # Compute metrics per mode
            backlog_means = []
            backlog_maxs = []
            throughputs = []
            mean_cpus = []
            mean_mems = []
            mean_ios = []
            batch_means = []
            batch_stds = []
            poll_means = []
            poll_stds = []

            for mode in modes:
                mdf = df[(df['controller_mode'] == mode) &
                         (df['load_pattern'] == pattern)].copy()
                mdf = mdf.sort_values('elapsed_sec')

                if len(mdf) == 0:
                    backlog_means.append(None)
                    backlog_maxs.append(None)
                    throughputs.append(None)
                    mean_cpus.append(None)
                    mean_mems.append(None)
                    mean_ios.append(None)
                    batch_means.append(None)
                    batch_stds.append(None)
                    poll_means.append(None)
                    poll_stds.append(None)
                    continue

                backlog_means.append(mdf['queue_length'].mean())
                backlog_maxs.append(int(mdf['queue_length'].max()))

                # Throughput from messages_indexed
                duration = mdf['elapsed_sec'].max() - mdf['elapsed_sec'].min()
                if duration > 0 and mdf['messages_indexed'].sum() > 0:
                    throughputs.append(mdf['messages_indexed'].sum() / duration)
                else:
                    throughputs.append(0.0)

                mean_cpus.append(mdf['cpu_util'].mean())
                mean_mems.append(mdf['container_mem_pct'].mean() if 'container_mem_pct' in mdf.columns else 0)
                mean_ios.append(mdf['io_write_ops'].mean())

                batch_means.append(mdf['batch_size'].mean())
                batch_stds.append(mdf['batch_size'].std() if len(mdf) > 1 else 0.0)
                poll_means.append(mdf['poll_interval_ms'].mean())
                poll_stds.append(mdf['poll_interval_ms'].std() if len(mdf) > 1 else 0.0)

            add_metric("Backlog mean", backlog_means)
            add_metric("Backlog max", backlog_maxs)
            add_metric("Throughput (msg/s)", throughputs)
            add_metric("CPU util mean (%)", mean_cpus)
            add_metric("Mem util mean (%)", mean_mems)
            add_metric("IO write ops mean", mean_ios)
            add_metric("Batch size mean", batch_means)
            add_metric("Batch size std", batch_stds)
            add_metric("Poll interval mean (ms)", poll_means)
            add_metric("Poll interval std", poll_stds)

        lines.append(f"\n{'=' * 100}")
        lines.append("NOTE: Run metrics_analysis.py on the output CSV for full thesis")
        lines.append("metrics (rise time, settling time, overshoot, cost J, regret).")
        lines.append(f"{'=' * 100}")

        report = '\n'.join(lines)
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n{report}")
        logger.info(f"Comparison saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Closed-loop experiment: 5 controllers × 4 load patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment (20 tests)
  python closed_loop_experiment.py --sysid-json <path> --ann-model <path>

  # Quick test: one controller, one pattern
  python closed_loop_experiment.py --modes static --load-patterns step --test-duration 60

  # Selected controllers and patterns
  python closed_loop_experiment.py --modes static pid lqr --load-patterns step ramp

  # With repeats and randomization
  python closed_loop_experiment.py --repeats 3 --randomize-order --base-rate 500
        """
    )
    parser.add_argument('--sysid-json', default=None,
                        help='Path to sysid_matrices_*.json (for LQR + rule_based)')
    parser.add_argument('--ann-model', default=None,
                        help='Path to ann_model_*.json (for ANN controller, legacy single mode)')
    parser.add_argument('--ann-model-q1', default=None,
                        help='Path to ANN model trained with Q1 oracle')
    parser.add_argument('--ann-model-q2', default=None,
                        help='Path to ANN model trained with Q2 oracle')
    parser.add_argument('--ann-model-q4', default=None,
                        help='Path to ANN model trained with Q4 oracle')
    parser.add_argument('--ann-universal', default=None,
                        help='Path to universal cost-weighted ANN model (for ann_cw_* modes)')
    parser.add_argument('--capacity-json', default=None,
                        help='Path to capacity_benchmark_*.json (for LQR B-matrix override from benchmark)')
    parser.add_argument('--modes', nargs='+',
                        default=[
                            'static', 'rule_based', 'pid',
                            'ann_cw_q1', 'ann_cw_q2', 'ann_cw_q4',
                            'lqr_q1', 'lqr_q2', 'lqr_q4',
                        ],
                        help='Controller modes to test (default: 9 modes)')
    parser.add_argument('--load-patterns', nargs='+',
                        default=['step', 'ramp', 'impulse', 'periodic_step', 'step_low'],
                        help='Load patterns to test (default: all 5)')
    # Pattern parameters
    parser.add_argument('--step-rate', type=int, default=1000,
                        help='Step pattern: constant rate msg/s (default: 1000)')
    parser.add_argument('--step-duration', type=int, default=30,
                        help='Step pattern: duration in seconds (default: 30)')
    parser.add_argument('--ramp-peak-rate', type=int, default=2000,
                        help='Ramp pattern: peak rate msg/s (default: 2000)')
    parser.add_argument('--ramp-duration', type=int, default=30,
                        help='Ramp pattern: duration in seconds (default: 30)')
    parser.add_argument('--impulse-rate', type=int, default=10000,
                        help='Impulse pattern: burst rate msg/s (default: 10000)')
    parser.add_argument('--impulse-burst-sec', type=int, default=1,
                        help='Impulse pattern: burst duration in seconds (default: 1)')
    parser.add_argument('--periodic-high-rate', type=int, default=1500,
                        help='Periodic pattern: high rate msg/s (default: 1500)')
    parser.add_argument('--periodic-low-rate', type=int, default=500,
                        help='Periodic pattern: low rate msg/s (default: 500)')
    parser.add_argument('--periodic-half-period', type=int, default=3,
                        help='Periodic pattern: half-period in seconds (default: 3)')
    parser.add_argument('--periodic-duration', type=int, default=30,
                        help='Periodic pattern: total duration in seconds (default: 30)')
    parser.add_argument('--step-low-rate', type=int, default=100,
                        help='Step low pattern: constant rate msg/s (default: 100)')
    parser.add_argument('--step-low-duration', type=int, default=60,
                        help='Step low pattern: duration in seconds (default: 60)')
    parser.add_argument('--max-test-duration', type=int, default=600,
                        help='Max test duration in seconds including drain (default: 600)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of repeats (default: 1)')
    parser.add_argument('--randomize-order', action='store_true',
                        help='Randomize controller test order within each pattern')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: auto-generated in runs/)')

    # Connection params
    parser.add_argument('--kafka-servers', default='localhost:29092')
    parser.add_argument('--kafka-topic', default='cdc.public.orders')
    parser.add_argument('--pg-host', default='localhost')
    parser.add_argument('--pg-port', type=int, default=5433)

    args = parser.parse_args()

    # Validate load patterns
    for p in args.load_patterns:
        if p not in LOAD_PATTERNS:
            parser.error(f"Unknown load pattern '{p}'. "
                         f"Choose from: {list(LOAD_PATTERNS.keys())}")

    # Validate model files for requested modes
    has_lqr = any(m.startswith('lqr') or m == 'rule_based' for m in args.modes)
    has_ann = any(m.startswith('ann') for m in args.modes)
    if has_lqr and not args.sysid_json:
        logger.warning("No --sysid-json provided; LQR/rule_based will use defaults")
    if has_ann:
        ann_q_models = {
            'Q1': args.ann_model_q1,
            'Q2': args.ann_model_q2,
            'Q4': args.ann_model_q4,
        }
        # Check that requested ann_qN (specialized) modes have model files
        for mode in args.modes:
            if mode.startswith('ann_q'):
                q_key = mode.split('_')[1].upper()  # "Q1"
                if not ann_q_models.get(q_key) and not args.ann_model:
                    parser.error(f"Mode '{mode}' requires --ann-model-{q_key.lower()} <path>")
        # Check ann_cw_* modes need a universal model
        has_ann_cw = any(m.startswith('ann_cw') for m in args.modes)
        if has_ann_cw and not args.ann_universal:
            parser.error("ann_cw_* modes require --ann-universal <path>")

    # Set env vars so sink can find model files when switching
    if args.sysid_json:
        os.environ['SYSID_JSON'] = args.sysid_json
    if args.ann_model:
        os.environ['ANN_MODEL_JSON'] = args.ann_model
    if args.ann_universal:
        os.environ['ANN_UNIVERSAL_JSON'] = args.ann_universal
    if args.capacity_json:
        os.environ['CAPACITY_JSON'] = args.capacity_json
    # Per-Q ANN model env vars
    for q_key, path in [('Q1', args.ann_model_q1), ('Q2', args.ann_model_q2),
                        ('Q4', args.ann_model_q4)]:
        if path:
            os.environ[f'ANN_MODEL_{q_key}'] = path

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or os.path.join(RUNS_DIR, f'closed_loop_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Run
    pattern_kwargs = {
        'step_rate': args.step_rate,
        'step_duration': args.step_duration,
        'ramp_peak_rate': args.ramp_peak_rate,
        'ramp_duration': args.ramp_duration,
        'impulse_rate': args.impulse_rate,
        'impulse_burst_sec': args.impulse_burst_sec,
        'periodic_high_rate': args.periodic_high_rate,
        'periodic_low_rate': args.periodic_low_rate,
        'periodic_half_period': args.periodic_half_period,
        'periodic_duration': args.periodic_duration,
        'step_low_rate': args.step_low_rate,
        'step_low_duration': args.step_low_duration,
    }

    experiment = ClosedLoopExperiment(
        kafka_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        sample_interval=args.sample_interval,
        controller_modes=args.modes,
        load_patterns=args.load_patterns,
        pattern_kwargs=pattern_kwargs,
        max_test_duration=args.max_test_duration,
        repeats=args.repeats,
        randomize_order=args.randomize_order,
        seed=args.seed,
    )

    experiment.run(output_dir)


if __name__ == '__main__':
    main()
