"""
Restart 20x20 Phase 3 Experiment
Stops existing processes, clears logs, and restarts fresh
"""
import subprocess
import sys
import time
import os
from datetime import datetime

print("=" * 70)
print("RESTARTING 20x20 PHASE 3 EXPERIMENT")
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

# 2. Create new log files with timestamp
print("\n2. Creating new log files...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
data_gen_log_name = f"data_gen_20x20_{timestamp}.log"
sysid_log_name = f"sysid_20x20_phase3_{timestamp}.log"
print(f"   [OK] Data generator log: {data_gen_log_name}")
print(f"   [OK] Experiment log: {sysid_log_name}")

# 3. Start data generator (UPDATE-only mode)
print("\n3. Starting data generator (150 updates/sec for 8.3 hours)...")
data_gen_log = open(data_gen_log_name, "w")
data_gen_process = subprocess.Popen(
    [sys.executable, "data_generator.py", "mixed", "--rate", "150", "--duration", "29800", "--update-ratio", "1.0"],
    stdout=data_gen_log,
    stderr=subprocess.STDOUT,
)
print(f"   [OK] Data generator started (PID: {data_gen_process.pid})")
time.sleep(5)

# Check data generator started successfully
if data_gen_process.poll() is None:
    print("   [OK] Data generator running")
else:
    print("   [ERROR] Data generator failed to start")
    sys.exit(1)

# 4. Start experiment
print("\n4. Starting experiment (20x20 Phase 3)...")
exp_log = open(sysid_log_name, "w")
exp_process = subprocess.Popen(
    [
        sys.executable, "sysid_experiment.py",
        "--preset", "20x20",
        "--phases", "vary_both",
        "--step-duration", "60",
        "--settle-duration", "10",
        "--output-dir", "./results/grid_20x20"
    ],
    stdout=exp_log,
    stderr=subprocess.STDOUT,
)
print(f"   [OK] Experiment started (PID: {exp_process.pid})")
print("\n5. Waiting for initialization (20 seconds)...")
time.sleep(20)

# Check experiment started successfully
if exp_process.poll() is None:
    print("   [OK] Experiment running successfully!")
else:
    print("   [ERROR] Experiment failed to start")
    print(f"   Check {sysid_log_name} for errors")
    sys.exit(1)

data_gen_log.close()
exp_log.close()

# 6. Show status
print("\n" + "=" * 70)
print("EXPERIMENT RESTARTED SUCCESSFULLY")
print("=" * 70)
print(f"\nData Generator PID: {data_gen_process.pid}")
print(f"Experiment PID: {exp_process.pid}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nEstimated completion: ~7.8 hours from now")
print(f"\nMonitor progress:")
print(f"  tail -f {sysid_log_name}")
print(f"  tail -f {data_gen_log_name}")
print("\n" + "=" * 70)
