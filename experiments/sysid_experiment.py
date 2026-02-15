"""
System Identification Experiment Runner

Collects data for identifying the relationship between control variables and state variables
in the CQRS synchronization system.

Model: x[k+1] = A·x[k] + B·u[k]

State x = [queue_length, cpu_util, mem_util, io_ops]
Control u = [batch_size, poll_interval]

Methodology:
1. Apply known control inputs (u) in a step-wise pattern
2. Record state transitions (x[k] -> x[k+1]) at each timestep
3. Data will be used for least squares regression to identify A and B matrices

Experiment Design:
- Phase 1: Vary batch_size while keeping poll_interval constant
- Phase 2: Vary poll_interval while keeping batch_size constant
- Phase 3: Vary both simultaneously (validation)

Each phase has a settling period between steps to observe transient response.
"""

import os
import sys
import csv
import time
import json
import signal
import logging
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'message-sink'))
from metrics_collector import MetricsCollector, SystemState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sysid-experiment')


class SysIdExperiment:
    """
    System Identification Experiment Runner.

    Writes control overrides and records state transitions.
    The message-sink reads the override file and applies the control values.
    """

    def __init__(
        self,
        cadvisor_url: str = 'http://localhost:8080',
        kafka_servers: str = 'localhost:29092',
        kafka_topic: str = 'cdc.postgres.changes',
        kafka_group_id: str = 'sysid-observer',
        es_container_name: str = 'elasticsearch-read',
        sample_interval: float = 1.0,
        output_dir: str = './results',
        control_override_path: str = None,
    ):
        self.sample_interval = sample_interval
        self.output_dir = output_dir
        self.running = True

        # Path to write control overrides (shared volume with message-sink)
        self.control_override_path = control_override_path or os.path.join(
            os.path.dirname(__file__), '..', 'message-sink', 'logs', 'control_override.json'
        )

        # Initialize metrics collector for state observation
        os.environ['CADVISOR_URL'] = cadvisor_url
        os.environ['ELASTICSEARCH_CONTAINER_NAME'] = es_container_name
        self.metrics = MetricsCollector(
            kafka_servers=kafka_servers,
            kafka_topic=kafka_topic,
            kafka_group_id=kafka_group_id,
        )

        # Data storage
        self.data: List[Dict] = []

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Interrupted, saving data...")
        self.running = False

    def set_control(self, batch_size: int, poll_interval: int):
        """Write control override file for message-sink to read."""
        override = {'batch_size': batch_size, 'poll_interval': poll_interval}
        with open(self.control_override_path, 'w') as f:
            json.dump(override, f)

    def clear_control_override(self):
        """Remove control override file so message-sink uses its own controller."""
        if os.path.exists(self.control_override_path):
            os.remove(self.control_override_path)
            logger.info("Control override cleared")

    def collect_sample(self, batch_size: int, poll_interval: int, phase: str) -> Dict:
        """Collect one state sample with the current control settings."""
        state = self.metrics.collect_state()
        sample = {
            'timestamp': datetime.utcnow().isoformat(),
            'phase': phase,
            # State variables x[k]
            'queue_length': state.queue_length,
            'cpu_util': state.cpu_util,
            'mem_util': state.mem_util,
            'io_ops': state.io_ops,
            # Control variables u[k]
            'batch_size': batch_size,
            'poll_interval': poll_interval,
        }
        self.data.append(sample)
        return sample

    def run_step_experiment(
        self,
        control_steps: List[Tuple[int, int]],
        step_duration: int,
        settle_duration: int,
        phase_name: str,
    ):
        """
        Run a step experiment with given control values.

        Args:
            control_steps: List of (batch_size, poll_interval) to apply
            step_duration: How long to hold each step (seconds)
            settle_duration: Settling time between steps (seconds)
            phase_name: Name for this experiment phase
        """
        logger.info(f"=== Phase: {phase_name} ===")
        logger.info(f"Steps: {control_steps}")
        logger.info(f"Step duration: {step_duration}s, Settle: {settle_duration}s")

        for i, (batch_size, poll_interval) in enumerate(control_steps):
            if not self.running:
                break

            logger.info(f"Step {i+1}/{len(control_steps)}: batch_size={batch_size}, poll_interval={poll_interval}ms")

            # Set control override for message-sink
            self.set_control(batch_size, poll_interval)

            # Collect samples during this step
            step_start = time.time()
            sample_count = 0

            while self.running and (time.time() - step_start) < step_duration:
                sample = self.collect_sample(batch_size, poll_interval, phase_name)
                sample_count += 1

                if sample_count % 10 == 0:
                    logger.info(
                        f"  Sample {sample_count}: queue={sample['queue_length']}, "
                        f"cpu={sample['cpu_util']:.1f}%, mem={sample['mem_util']:.1f}%, "
                        f"io={sample['io_ops']:.0f}"
                    )

                time.sleep(self.sample_interval)

            logger.info(f"  Collected {sample_count} samples")

            # Settling time (still collect data, marked as 'settle')
            if settle_duration > 0 and i < len(control_steps) - 1:
                logger.info(f"  Settling for {settle_duration}s...")
                settle_start = time.time()
                while self.running and (time.time() - settle_start) < settle_duration:
                    self.collect_sample(batch_size, poll_interval, f"{phase_name}_settle")
                    time.sleep(self.sample_interval)

    def save_data(self, filename: str):
        """Save collected data to CSV."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        if not self.data:
            logger.warning("No data to save")
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

        logger.info(f"Saved {len(self.data)} samples to {filepath}")

    def cleanup(self):
        self.clear_control_override()
        self.metrics.close()


def build_phase_steps():
    """Build control steps for each experiment phase."""
    # Phase 1: Vary batch_size, keep poll_interval constant
    fixed_poll = 1000  # ms
    batch_steps = [
        (10, fixed_poll),
        (50, fixed_poll),
        (100, fixed_poll),
        (200, fixed_poll),
        (500, fixed_poll),
        (100, fixed_poll),  # return to baseline
    ]

    # Phase 2: Vary poll_interval, keep batch_size constant
    fixed_batch = 100
    poll_steps = [
        (fixed_batch, 200),
        (fixed_batch, 500),
        (fixed_batch, 1000),
        (fixed_batch, 2000),
        (fixed_batch, 5000),
        (fixed_batch, 1000),  # return to baseline
    ]

    # Phase 3: Cartesian product of all values (n x m)
    batch_values = [10, 50, 100, 200, 500]
    poll_values = [200, 500, 1000, 2000, 5000]
    combined_steps = [(b, p) for b in batch_values for p in poll_values]

    return {
        'vary_batch': batch_steps,
        'vary_poll': poll_steps,
        'vary_both': combined_steps,
    }


AVAILABLE_PHASES = ['vary_batch', 'vary_poll', 'vary_both']


def run_full_sysid(args):
    """Run the system identification experiment for selected phases."""
    exp = SysIdExperiment(
        cadvisor_url=args.cadvisor_url,
        kafka_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        es_container_name=args.es_container,
        sample_interval=args.sample_interval,
        output_dir=args.output_dir,
        control_override_path=args.control_override_path,
    )

    phases_to_run = args.phases
    all_steps = build_phase_steps()

    try:
        for phase_name in phases_to_run:
            steps = all_steps[phase_name]
            exp.run_step_experiment(
                control_steps=steps,
                step_duration=args.step_duration,
                settle_duration=args.settle_duration,
                phase_name=phase_name,
            )

        # Save all data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        phase_suffix = '_'.join(phases_to_run)
        exp.save_data(f'sysid_data_{phase_suffix}_{timestamp}.csv')

    finally:
        exp.cleanup()


def main():
    parser = argparse.ArgumentParser(description='System Identification Experiment')
    parser.add_argument('--cadvisor-url', default='http://localhost:8080')
    parser.add_argument('--kafka-servers', default='localhost:29092')
    parser.add_argument('--kafka-topic', default='cdc.postgres.changes')
    parser.add_argument('--es-container', default='elasticsearch-read')
    parser.add_argument('--sample-interval', type=float, default=1.0,
                        help='Time between state samples (seconds)')
    parser.add_argument('--step-duration', type=int, default=60,
                        help='Duration of each control step (seconds)')
    parser.add_argument('--settle-duration', type=int, default=10,
                        help='Settling time between steps (seconds)')
    parser.add_argument('--output-dir', default='./results',
                        help='Directory for output files')
    parser.add_argument('--control-override-path', default=None,
                        help='Path to control override file (shared with message-sink)')
    parser.add_argument('--phases', nargs='+', default=AVAILABLE_PHASES,
                        choices=AVAILABLE_PHASES,
                        help='Phases to run (default: all)')

    args = parser.parse_args()
    run_full_sysid(args)


if __name__ == '__main__':
    main()
