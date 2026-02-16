"""
Seed Database with Initial Data for UPDATE-Only Experiments
Pre-populates the database with rows that can be updated during experiments
"""
import subprocess
import sys
import time
from datetime import datetime

print("=" * 70)
print("SEEDING DATABASE WITH INITIAL DATA")
print("=" * 70)

# Configuration
NUM_ROWS = 10000
RATE = 1000  # rows/sec
DURATION = NUM_ROWS // RATE  # Calculate duration needed

print(f"\nTarget: {NUM_ROWS:,} rows")
print(f"Rate: {RATE} rows/sec")
print(f"Estimated time: {DURATION} seconds")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run data generator in constant mode
print("\nStarting data generator...")
log_file = open("seed_database.log", "w")

process = subprocess.Popen(
    [sys.executable, "data_generator.py", "constant", "--rate", str(RATE), "--duration", str(DURATION)],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

print(f"Process ID: {process.pid}")

# Wait for completion
print(f"\nSeeding in progress... (this will take ~{DURATION} seconds)")
process.wait()

log_file.close()

# Check result
if process.returncode == 0:
    print("\n" + "=" * 70)
    print("[OK] DATABASE SEEDED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nRows inserted: ~{NUM_ROWS:,}")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nYou can now run UPDATE-only experiments using:")
    print("  python restart_experiment.py")
    print("\n" + "=" * 70)
else:
    print("\n" + "=" * 70)
    print("[ERROR] SEEDING FAILED")
    print("=" * 70)
    print(f"Return code: {process.returncode}")
    print("Check seed_database.log for errors")
    sys.exit(1)
