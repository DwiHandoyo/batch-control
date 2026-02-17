"""Simple test script for grid configuration"""
from grid_config import GridConfig, PRESETS

print("=" * 70)
print("TEST 1: Backward Compatibility (Default 5x5)")
print("=" * 70)
config = GridConfig()
batch_vals = config.generate_batch_values()
poll_vals = config.generate_poll_values()
print(f"Batch values: {batch_vals}")
print(f"Poll values: {poll_vals}")
print(f"Grid size: {len(batch_vals)} × {len(poll_vals)} = {len(batch_vals) * len(poll_vals)}")

# Simulate build_phase_steps
baseline_batch = batch_vals[len(batch_vals) // 2]
baseline_poll = poll_vals[len(poll_vals) // 2]
phase1_steps = len(batch_vals) + 1  # + 1 for baseline return
phase2_steps = len(poll_vals) + 1
phase3_steps = len(batch_vals) * len(poll_vals)
print(f"\nExpected steps:")
print(f"  Phase 1: {phase1_steps} steps (original: 6)")
print(f"  Phase 2: {phase2_steps} steps (original: 6)")
print(f"  Phase 3: {phase3_steps} steps (original: 25)")

print("\n" + "=" * 70)
print("TEST 2: Preset 20x20")
print("=" * 70)
config20 = PRESETS['20x20']
batch_vals20 = config20.generate_batch_values()
poll_vals20 = config20.generate_poll_values()
print(f"Batch values: {batch_vals20}")
print(f"Poll values: {poll_vals20}")
print(f"Grid size: {len(batch_vals20)} × {len(poll_vals20)} = {len(batch_vals20) * len(poll_vals20)}")

durations = config20.estimate_duration(60, 10)
print(f"\nDuration estimates (60s step, 10s settle):")
print(f"  Phase 1: {durations['vary_batch']//60} min")
print(f"  Phase 2: {durations['vary_poll']//60} min")
print(f"  Phase 3: {durations['vary_both']//60} min ({durations['vary_both']//3600:.1f} hr)")
print(f"  TOTAL: {durations['total']//60} min ({durations['total']//3600:.1f} hr)")

print("\n" + "=" * 70)
print("TEST 3: Logarithmic Spacing")
print("=" * 70)
config_log = GridConfig(
    batch_min=10,
    batch_max=1000,
    batch_points=10,
    poll_min=100,
    poll_max=10000,
    poll_points=10,
    spacing='log'
)
batch_log = config_log.generate_batch_values()
poll_log = config_log.generate_poll_values()
print(f"Batch values (log): {batch_log}")
print(f"Poll values (log): {poll_log}")

print("\n" + "=" * 70)
print("TEST 4: Validation")
print("=" * 70)
# Test valid config
is_valid, error = config20.validate()
print(f"20x20 preset validation: {is_valid} (error: {error if error else 'none'})")

# Test invalid config
invalid_config = GridConfig(batch_min=-10, batch_max=500, batch_points=5)
is_valid, error = invalid_config.validate()
print(f"Invalid config (negative min): {is_valid} (error: {error})")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
