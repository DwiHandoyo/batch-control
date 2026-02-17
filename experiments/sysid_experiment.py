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
        kafka_group_id: str = 'message-sink-group',
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


def build_phase_steps(config=None):
    """
    Build control steps for each experiment phase.

    Args:
        config: GridConfig instance. If None, uses default 5×5 configuration
                for backward compatibility.

    Returns:
        dict with keys 'vary_batch', 'vary_poll', 'vary_both'
    """
    # Import GridConfig here to avoid circular imports
    from grid_config import GridConfig

    if config is None:
        config = GridConfig()  # Default 5×5 for backward compatibility

    # Generate grid values
    batch_values = config.generate_batch_values()
    poll_values = config.generate_poll_values()

    # Compute baseline (middle value)
    baseline_batch = batch_values[len(batch_values) // 2]
    baseline_poll = poll_values[len(poll_values) // 2]

    # Phase 1: Vary batch_size, keep poll_interval constant
    batch_steps = [(b, baseline_poll) for b in batch_values]
    batch_steps.append((baseline_batch, baseline_poll))  # return to baseline

    # Phase 2: Vary poll_interval, keep batch_size constant
    poll_steps = [(baseline_batch, p) for p in poll_values]
    poll_steps.append((baseline_batch, baseline_poll))  # return to baseline

    # Phase 3: Cartesian product of all values (full grid)
    combined_steps = [(b, p) for b in batch_values for p in poll_values]

    return {
        'vary_batch': batch_steps,
        'vary_poll': poll_steps,
        'vary_both': combined_steps,
    }


AVAILABLE_PHASES = ['vary_batch', 'vary_poll', 'vary_both']


def run_full_sysid(args, config=None):
    """
    Run the system identification experiment for selected phases.

    Args:
        args: Parsed command-line arguments
        config: GridConfig instance (optional, defaults to 5×5)
    """
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
    all_steps = build_phase_steps(config)

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


def build_grid_config_from_args(args):
    """
    Build GridConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        GridConfig instance
    """
    from grid_config import GridConfig, PRESETS

    # Priority: preset > individual args
    if hasattr(args, 'preset') and args.preset:
        return PRESETS[args.preset]

    return GridConfig(
        batch_min=args.batch_min,
        batch_max=args.batch_max,
        batch_points=args.batch_points,
        poll_min=args.poll_min,
        poll_max=args.poll_max,
        poll_points=args.poll_points,
        spacing=args.spacing,
    )


def print_grid_info(config, step_duration, settle_duration):
    """
    Print formatted grid information.

    Args:
        config: GridConfig instance
        step_duration: Duration of each step (seconds)
        settle_duration: Settling time between steps (seconds)
    """
    batch_vals = config.generate_batch_values()
    poll_vals = config.generate_poll_values()

    print("=" * 70)
    print("GRID CONFIGURATION")
    print("=" * 70)
    print(f"\nBatch Size Values ({len(batch_vals)} points, {config.spacing} spacing):")
    print(f"  Range: {batch_vals[0]} - {batch_vals[-1]}")
    print(f"  Values: {batch_vals}")

    print(f"\nPoll Interval Values ({len(poll_vals)} points, {config.spacing} spacing):")
    print(f"  Range: {poll_vals[0]} - {poll_vals[-1]} ms")
    print(f"  Values: {poll_vals}")

    print(f"\nGrid Size: {len(batch_vals)} × {len(poll_vals)} = {len(batch_vals) * len(poll_vals)} combinations")

    durations = config.estimate_duration(step_duration, settle_duration)
    print(f"\nEstimated Duration (step={step_duration}s, settle={settle_duration}s):")
    print(f"  Phase 1 (vary_batch): {durations['vary_batch']//60} min")
    print(f"  Phase 2 (vary_poll): {durations['vary_poll']//60} min")
    print(f"  Phase 3 (vary_both): {durations['vary_both']//60} min ({durations['vary_both']//3600:.1f} hr)")
    print(f"  TOTAL: {durations['total']//60} min ({durations['total']//3600:.1f} hr)")
    print("=" * 70)


def print_experiment_plan(config, args):
    """
    Print experiment plan for dry-run.

    Args:
        config: GridConfig instance
        args: Parsed command-line arguments
    """
    print_grid_info(config, args.step_duration, args.settle_duration)

    print("\nPHASES TO RUN:")
    for phase in args.phases:
        print(f"  - {phase}")

    steps = build_phase_steps(config)
    print("\nSTEPS PREVIEW:")
    for phase in args.phases:
        phase_steps = steps[phase]
        print(f"\n{phase}: {len(phase_steps)} steps")
        if len(phase_steps) <= 10:
            for i, (b, p) in enumerate(phase_steps):
                print(f"  {i+1}. batch_size={b}, poll_interval={p}ms")
        else:
            # Show first 3 and last 3
            for i, (b, p) in enumerate(phase_steps[:3]):
                print(f"  {i+1}. batch_size={b}, poll_interval={p}ms")
            print(f"  ... ({len(phase_steps) - 6} more steps)")
            for i, (b, p) in enumerate(phase_steps[-3:], start=len(phase_steps)-2):
                print(f"  {i}. batch_size={b}, poll_interval={p}ms")


def main():
    parser = argparse.ArgumentParser(
        description='System Identification Experiment with Configurable Grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default 5×5 grid (backward compatible)
  python sysid_experiment.py

  # Use preset configurations
  python sysid_experiment.py --preset quick    # 3×3 grid, fast testing
  python sysid_experiment.py --preset 10x10    # 10×10 grid, ~2.4 hours
  python sysid_experiment.py --preset 20x20    # 20×20 grid, ~8.6 hours

  # Custom grid with CLI arguments
  python sysid_experiment.py --batch-points 15 --poll-points 15

  # Logarithmic spacing for wide ranges
  python sysid_experiment.py --batch-min 1 --batch-max 1000 \\
                             --batch-points 20 --spacing log

  # Show grid without running
  python sysid_experiment.py --preset 20x20 --show-grid

  # Dry run to see experiment plan
  python sysid_experiment.py --preset 20x20 --dry-run

  # Run only Phase 3 (grid validation)
  python sysid_experiment.py --preset 20x20 --phases vary_both
        """
    )

    # Existing arguments
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

    # NEW: Grid configuration arguments
    grid_group = parser.add_argument_group('Grid Configuration')

    grid_group.add_argument(
        '--preset',
        choices=['5x5', '10x10', '20x20', 'quick'],
        help='Use preset grid configuration (overrides other grid args)'
    )

    grid_group.add_argument('--batch-min', type=int, default=10,
                           help='Minimum batch_size value (default: 10)')
    grid_group.add_argument('--batch-max', type=int, default=500,
                           help='Maximum batch_size value (default: 500)')
    grid_group.add_argument('--batch-points', type=int, default=5,
                           help='Number of batch_size values (default: 5)')

    grid_group.add_argument('--poll-min', type=int, default=200,
                           help='Minimum poll_interval in ms (default: 200)')
    grid_group.add_argument('--poll-max', type=int, default=5000,
                           help='Maximum poll_interval in ms (default: 5000)')
    grid_group.add_argument('--poll-points', type=int, default=5,
                           help='Number of poll_interval values (default: 5)')

    grid_group.add_argument('--spacing', choices=['linear', 'log'], default='linear',
                           help='Grid spacing method (default: linear)')

    # Utility arguments
    grid_group.add_argument('--show-grid', action='store_true',
                           help='Print grid values and exit')
    grid_group.add_argument('--dry-run', action='store_true',
                           help='Show experiment plan without running')

    args = parser.parse_args()

    # Build GridConfig from arguments
    config = build_grid_config_from_args(args)

    # Validate configuration
    is_valid, error = config.validate()
    if not is_valid:
        logger.error(f"Invalid configuration: {error}")
        sys.exit(1)

    # Handle utility modes
    if args.show_grid:
        print_grid_info(config, args.step_duration, args.settle_duration)
        sys.exit(0)

    if args.dry_run:
        print_experiment_plan(config, args)
        sys.exit(0)

    # Run experiment
    run_full_sysid(args, config)


if __name__ == '__main__':
    main()
