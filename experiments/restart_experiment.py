"""
Restart Phase 3 Experiment (no data generator)
Stops existing processes and starts fresh in an organized run directory.
Queue is pre-filled per step by sysid_experiment.py itself.
"""
import subprocess
import sys
import time
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from grid_config import PRESETS

print("=" * 70)
print("RESTARTING 10x10 PHASE 3 EXPERIMENT")
print("=" * 70)

# 1. Stop existing processes
print("\n1. Stopping existing processes...")
try:
    subprocess.run(["pkill", "-f", "sysid_experiment.py"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "data_generator.py"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    print("   [OK] Processes stopped")
except:
    print("   [-] No processes to stop")

# 2. Create run directory
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = Path(f"./runs/run_{run_id}")
(run_dir / "logs").mkdir(parents=True, exist_ok=True)
(run_dir / "results").mkdir(parents=True, exist_ok=True)

print(f"\n2. Created run directory: {run_dir}")

# Save initial metadata with actual grid values
grid = PRESETS["10x10"]
batch_values = grid.generate_batch_values()
poll_values = grid.generate_poll_values()

metadata = {
    "run_id": run_id,
    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "grid_config": {
        "preset": "10x10",
        "batch_min": grid.batch_min, "batch_max": grid.batch_max, "batch_points": grid.batch_points,
        "poll_min": grid.poll_min, "poll_max": grid.poll_max, "poll_points": grid.poll_points,
        "spacing": grid.spacing,
        "batch_size_values": batch_values,
        "poll_interval_values": poll_values,
    },
    "experiment_params": {
        "phases": ["vary_both"],
        "step_duration": 60,
        "settle_duration": 10,
        "total_steps": len(batch_values) * len(poll_values),
    },
    "data_generator": "none (queue pre-filled per step by experiment runner)"
}
with open(run_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 3. Start experiment (no data generator — queue pre-filled per step)
output_dir = str(run_dir / "results")
exp_log_path = str(run_dir / "logs" / "experiment.log")
print(f"\n   Starting experiment (10x10)...")
exp_log = open(exp_log_path, "w")
exp_process = subprocess.Popen(
    [
        sys.executable, "sysid_experiment.py",
        "--preset", "10x10",
        "--phases", "vary_both",
        "--step-duration", "60",
        "--settle-duration", "10",
        "--output-dir", output_dir
    ],
    stdout=exp_log,
    stderr=subprocess.STDOUT,
)
print(f"   [OK] Experiment started (PID: {exp_process.pid})")
print("\n4. Waiting for initialization (20 seconds)...")
time.sleep(20)

# Check experiment started successfully
if exp_process.poll() is None:
    print("   [OK] Experiment running successfully!")
else:
    print("   [ERROR] Experiment failed to start")
    print(f"   Check {exp_log_path} for errors")
    sys.exit(1)

exp_log.close()

# 5. Create/update 'latest' symlink (or junction on Windows)
latest_link = Path("./runs/latest")
try:
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_dir() and not latest_link.is_symlink():
            os.rmdir(latest_link)
        else:
            latest_link.unlink()
    # On Windows, use junction instead of symlink (no admin required)
    if sys.platform == 'win32':
        subprocess.run(["cmd", "/c", "mklink", "/J", str(latest_link), str(run_dir)],
                       capture_output=True)
    else:
        latest_link.symlink_to(run_dir.name, target_is_directory=True)
    print(f"\n   [OK] Updated 'latest' link -> {run_dir.name}")
except Exception as e:
    print(f"\n   [-] Could not create 'latest' link: {e}")

# 6. Show status
total_steps = len(batch_values) * len(poll_values)
est_hours = total_steps * 70 / 3600
print("\n" + "=" * 70)
print("EXPERIMENT STARTED SUCCESSFULLY")
print("=" * 70)
print(f"\nRun directory: {run_dir}")
print(f"Experiment PID: {exp_process.pid}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Data generator: NONE (queue pre-filled per step)")
print(f"\nEstimated completion: ~{est_hours:.1f} hours ({total_steps} steps × ~70s/step)")
print(f"\nMonitor progress:")
print(f"  tail -f {exp_log_path}")
print(f"\nVisualize results after completion:")
print(f"  python sysid_visualize.py {run_dir}/results/*.csv")
print("\n" + "=" * 70)
