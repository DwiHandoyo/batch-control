"""Launch 20x20 experiment in background"""
import subprocess
import sys
import time
from datetime import datetime, timedelta

print("=" * 70)
print("LAUNCHING 20x20 PHASE 3 EXPERIMENT IN BACKGROUND")
print("=" * 70)

# Build command
cmd = [
    sys.executable,  # Use same python interpreter
    "sysid_experiment.py",
    "--preset", "20x20",
    "--phases", "vary_both",
    "--step-duration", "60",
    "--settle-duration", "10",
    "--output-dir", "./results/grid_20x20"
]

# Open log file
log_file = open("sysid_20x20_phase3.log", "w")

# Start process in background
print(f"\nCommand: {' '.join(cmd)}")
print(f"Log file: sysid_20x20_phase3.log")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

process = subprocess.Popen(
    cmd,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    cwd=".",
)

print(f"Process ID: {process.pid}")

# Estimate completion time
duration_hours = 7.8
completion_time = datetime.now() + timedelta(hours=duration_hours)
print(f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: ~{duration_hours} hours (400 steps × 70 seconds/step)")

# Wait a bit and check initial status
print("\nWaiting for initialization (20 seconds)...")
time.sleep(20)

# Check if process is still running
if process.poll() is None:
    print("✓ Process is running successfully!")
    print(f"\nTo monitor progress:")
    print(f"  tail -f sysid_20x20_phase3.log")
    print(f"\nTo check if running:")
    print(f"  ps aux | grep {process.pid}")
else:
    print(f"✗ Process terminated with code: {process.returncode}")
    print("\nCheck sysid_20x20_phase3.log for errors")

log_file.close()

print("\n" + "=" * 70)
print("Experiment launched. Returning control to terminal.")
print("=" * 70)
