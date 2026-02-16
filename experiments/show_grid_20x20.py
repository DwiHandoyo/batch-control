"""Quick script to show 20x20 Phase 3 experiment details"""
from grid_config import PRESETS

config = PRESETS['20x20']

print("=" * 70)
print("EXPERIMENT: 20x20 Grid - Phase 3 Only")
print("=" * 70)

batch_vals = config.generate_batch_values()
poll_vals = config.generate_poll_values()

print(f"\nBatch Size Values ({len(batch_vals)} points, {config.spacing} spacing):")
print(f"  Range: {batch_vals[0]} - {batch_vals[-1]}")
print(f"  First 5: {batch_vals[:5]}")
print(f"  Last 5: {batch_vals[-5:]}")

print(f"\nPoll Interval Values ({len(poll_vals)} points, {config.spacing} spacing):")
print(f"  Range: {poll_vals[0]} - {poll_vals[-1]} ms")
print(f"  First 5: {poll_vals[:5]}")
print(f"  Last 5: {poll_vals[-5:]}")

grid_size = len(batch_vals) * len(poll_vals)
print(f"\nGrid Size: {len(batch_vals)} × {len(poll_vals)} = {grid_size} combinations")

step_duration = 60  # seconds
settle_duration = 10  # seconds
durations = config.estimate_duration(step_duration, settle_duration)

phase3_duration_sec = durations['vary_both']
phase3_duration_min = phase3_duration_sec // 60
phase3_duration_hr = phase3_duration_sec / 3600

print(f"\nEstimated Duration (Phase 3 only):")
print(f"  Steps: {grid_size}")
print(f"  Time per step: {step_duration + settle_duration} seconds")
print(f"  Total: {phase3_duration_sec} seconds = {phase3_duration_min} minutes = {phase3_duration_hr:.1f} hours")

# Data generation requirements
data_rate = 150  # rows/sec
total_records = data_rate * phase3_duration_sec
data_duration_with_buffer = phase3_duration_sec + 1800  # Add 30 min buffer

print(f"\nData Generation Requirements:")
print(f"  Rate: {data_rate} rows/sec")
print(f"  Duration needed: {phase3_duration_sec} sec ({phase3_duration_hr:.1f} hr)")
print(f"  With buffer (+30 min): {data_duration_with_buffer} sec ({data_duration_with_buffer/3600:.1f} hr)")
print(f"  Total records: ~{total_records:,.0f}")

print("\n" + "=" * 70)
print("COMMANDS TO RUN:")
print("=" * 70)
print("\n# Terminal 1: Data Generator (background)")
print(f"cd experiments")
print(f"python data_generator.py constant --rate {data_rate} --duration {data_duration_with_buffer} &")
print(f"\n# Terminal 2: System Identification Experiment (background)")
print(f"python sysid_experiment.py --preset 20x20 --phases vary_both \\")
print(f"  --step-duration {step_duration} --settle-duration {settle_duration} &")
print("\n" + "=" * 70)
