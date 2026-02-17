"""
Grid Configuration Module for System Identification Experiments

Provides flexible grid generation for control parameters (batch_size, poll_interval)
with support for linear and logarithmic spacing.
"""

import numpy as np
from typing import Tuple, List


def generate_linear_grid(min_val: int, max_val: int, n_points: int) -> List[int]:
    """
    Generate linearly spaced integer values.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        n_points: Number of points to generate

    Returns:
        List of n_points integer values, linearly spaced from min_val to max_val

    Example:
        >>> generate_linear_grid(10, 500, 5)
        [10, 132, 255, 377, 500]
    """
    if n_points == 1:
        return [(min_val + max_val) // 2]

    # Use numpy linspace for uniform spacing
    values = np.linspace(min_val, max_val, n_points)
    values = [int(round(v)) for v in values]

    # Ensure exact min and max
    values[0] = min_val
    values[-1] = max_val

    return values


def generate_log_grid(min_val: int, max_val: int, n_points: int) -> List[int]:
    """
    Generate logarithmically spaced integer values.

    Useful for parameters that span multiple orders of magnitude.
    Provides more resolution at lower values.

    Args:
        min_val: Minimum value (must be > 0)
        max_val: Maximum value
        n_points: Number of points to generate

    Returns:
        List of n_points integer values, logarithmically spaced

    Example:
        >>> generate_log_grid(10, 1000, 7)
        [10, 21, 46, 100, 215, 464, 1000]

    Raises:
        ValueError: If min_val <= 0 (log spacing requires positive values)
    """
    if n_points == 1:
        return [(min_val + max_val) // 2]

    if min_val <= 0:
        raise ValueError("Logarithmic spacing requires min_val > 0")

    # Use numpy logspace (base 10)
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_values = np.linspace(log_min, log_max, n_points)
    values = [int(round(10 ** lv)) for lv in log_values]

    # Ensure exact min and max
    values[0] = min_val
    values[-1] = max_val

    # Remove duplicates while maintaining order
    seen = set()
    unique_values = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)

    # If duplicates reduced count, add interpolated points to fill gaps
    while len(unique_values) < n_points:
        # Find largest gap and add midpoint
        gaps = []
        for i in range(len(unique_values) - 1):
            gap_size = unique_values[i + 1] - unique_values[i]
            gaps.append((gap_size, i))

        if not gaps:
            break

        max_gap_size, max_gap_idx = max(gaps)
        if max_gap_size <= 1:
            # No more room to add points
            break

        # Add midpoint
        new_val = (unique_values[max_gap_idx] + unique_values[max_gap_idx + 1]) // 2
        if new_val not in seen and new_val != unique_values[max_gap_idx] and new_val != unique_values[max_gap_idx + 1]:
            seen.add(new_val)
            unique_values.insert(max_gap_idx + 1, new_val)
        else:
            break

    return sorted(unique_values)[:n_points]


class GridConfig:
    """
    Configuration for system identification experiment grid.

    Defines the parameter ranges and grid resolution for control variables:
    - batch_size: Number of messages consumed per poll
    - poll_interval: Polling timeout in milliseconds

    Attributes:
        batch_min: Minimum batch_size value
        batch_max: Maximum batch_size value
        batch_points: Number of batch_size values in grid
        poll_min: Minimum poll_interval value (ms)
        poll_max: Maximum poll_interval value (ms)
        poll_points: Number of poll_interval values in grid
        spacing: Grid spacing method ('linear' or 'log')
    """

    def __init__(
        self,
        batch_min: int = 10,
        batch_max: int = 500,
        batch_points: int = 5,
        poll_min: int = 200,
        poll_max: int = 5000,
        poll_points: int = 5,
        spacing: str = 'linear',
    ):
        """
        Initialize grid configuration.

        Args:
            batch_min: Minimum batch_size (default: 10)
            batch_max: Maximum batch_size (default: 500)
            batch_points: Number of batch_size points (default: 5)
            poll_min: Minimum poll_interval in ms (default: 200)
            poll_max: Maximum poll_interval in ms (default: 5000)
            poll_points: Number of poll_interval points (default: 5)
            spacing: 'linear' or 'log' (default: 'linear')
        """
        self.batch_min = batch_min
        self.batch_max = batch_max
        self.batch_points = batch_points
        self.poll_min = poll_min
        self.poll_max = poll_max
        self.poll_points = poll_points
        self.spacing = spacing

    def generate_batch_values(self) -> List[int]:
        """Generate batch_size values based on configuration."""
        if self.spacing == 'linear':
            return generate_linear_grid(self.batch_min, self.batch_max, self.batch_points)
        elif self.spacing == 'log':
            return generate_log_grid(self.batch_min, self.batch_max, self.batch_points)
        else:
            raise ValueError(f"Invalid spacing method: {self.spacing}. Must be 'linear' or 'log'.")

    def generate_poll_values(self) -> List[int]:
        """Generate poll_interval values based on configuration."""
        if self.spacing == 'linear':
            return generate_linear_grid(self.poll_min, self.poll_max, self.poll_points)
        elif self.spacing == 'log':
            return generate_log_grid(self.poll_min, self.poll_max, self.poll_points)
        else:
            raise ValueError(f"Invalid spacing method: {self.spacing}. Must be 'linear' or 'log'.")

    def estimate_duration(self, step_duration: int, settle_duration: int) -> dict:
        """
        Estimate experiment duration for each phase.

        Args:
            step_duration: Duration of each control step in seconds
            settle_duration: Settling time between steps in seconds

        Returns:
            Dictionary with duration estimates (in seconds) for each phase
        """
        time_per_step = step_duration + settle_duration

        # Phase 1: n_batch points + 1 baseline return
        phase1_steps = self.batch_points + 1
        phase1_duration = phase1_steps * time_per_step

        # Phase 2: n_poll points + 1 baseline return
        phase2_steps = self.poll_points + 1
        phase2_duration = phase2_steps * time_per_step

        # Phase 3: n_batch × n_poll (Cartesian product)
        phase3_steps = self.batch_points * self.poll_points
        phase3_duration = phase3_steps * time_per_step

        total_duration = phase1_duration + phase2_duration + phase3_duration

        return {
            'vary_batch': phase1_duration,
            'vary_poll': phase2_duration,
            'vary_both': phase3_duration,
            'total': total_duration,
        }

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string
        """
        # Check batch_size parameters
        if self.batch_min <= 0:
            return False, "batch_min must be positive"
        if self.batch_max <= self.batch_min:
            return False, "batch_max must be greater than batch_min"
        if self.batch_points < 1:
            return False, "batch_points must be at least 1"

        # Check poll_interval parameters
        if self.poll_min <= 0:
            return False, "poll_min must be positive"
        if self.poll_max <= self.poll_min:
            return False, "poll_max must be greater than poll_min"
        if self.poll_points < 1:
            return False, "poll_points must be at least 1"

        # Check spacing
        if self.spacing not in ['linear', 'log']:
            return False, f"spacing must be 'linear' or 'log', got '{self.spacing}'"

        # Warning checks (not errors, but good to know)
        warnings = []

        if self.batch_max > 1000:
            warnings.append(f"batch_max={self.batch_max} is very large")

        if self.poll_min < 100:
            warnings.append(f"poll_min={self.poll_min}ms may be too fast for system stability")

        grid_size = self.batch_points * self.poll_points
        if grid_size > 500:
            warnings.append(f"grid size {grid_size} is very large, experiment may take very long")

        # All checks passed
        return True, ""

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"GridConfig(batch={self.batch_min}-{self.batch_max} ({self.batch_points} points), "
            f"poll={self.poll_min}-{self.poll_max}ms ({self.poll_points} points), "
            f"spacing={self.spacing})"
        )


# Preset configurations for common use cases
PRESETS = {
    '5x5': GridConfig(
        batch_min=10,
        batch_max=500,
        batch_points=5,
        poll_min=200,
        poll_max=5000,
        poll_points=5,
        spacing='linear',
    ),
    '10x10': GridConfig(
        batch_min=10,
        batch_max=500,
        batch_points=10,
        poll_min=200,
        poll_max=5000,
        poll_points=10,
        spacing='linear',
    ),
    '20x20': GridConfig(
        batch_min=10,
        batch_max=2000,
        batch_points=20,
        poll_min=50,
        poll_max=1000,
        poll_points=20,
        spacing='linear',
    ),
    'quick': GridConfig(
        batch_min=50,
        batch_max=200,
        batch_points=3,
        poll_min=500,
        poll_max=2000,
        poll_points=3,
        spacing='linear',
    ),
}
