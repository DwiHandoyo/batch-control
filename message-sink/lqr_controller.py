"""
LQR Controller - Linear Quadratic Regulator for synchronization parameter control.

This module provides the control logic for adaptively adjusting:
- batch_size: Number of messages to process per batch
- poll_interval: Time between polling Kafka (in milliseconds)

Based on the system state:
- queue_length: Kafka consumer lag
- cpu_util: Elasticsearch CPU utilization
- mem_util: Elasticsearch memory utilization
- io_ops: Elasticsearch I/O operations

The controller minimizes the quadratic cost function:
    J = sum(x'Qx + u'Ru)

Where:
    x = state deviation from target
    u = control input deviation from nominal
    Q = state cost matrix
    R = control cost matrix
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger('lqr-controller')


@dataclass
class ControlOutput:
    """Output from the controller."""
    batch_size: int
    poll_interval_ms: int
    mode: str  # 'static' or 'lqr'


class BaseController(ABC):
    """Abstract base class for controllers."""

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        min_poll_interval_ms: int = 100,
        max_poll_interval_ms: int = 10000,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_poll_interval_ms = min_poll_interval_ms
        self.max_poll_interval_ms = max_poll_interval_ms

    @abstractmethod
    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute control output given current state."""
        pass

    def _clamp_control(self, batch_size: float, poll_interval: float) -> Tuple[int, int]:
        """Clamp control values to valid ranges."""
        batch_size = int(np.clip(batch_size, self.min_batch_size, self.max_batch_size))
        poll_interval = int(np.clip(poll_interval, self.min_poll_interval_ms, self.max_poll_interval_ms))
        return batch_size, poll_interval


class StaticController(BaseController):
    """
    Static controller - uses fixed batch_size and poll_interval.
    Used as baseline for comparison with LQR controller.
    """

    def __init__(
        self,
        batch_size: int = 100,
        poll_interval_ms: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.poll_interval_ms = poll_interval_ms
        logger.info(f"StaticController initialized: batch_size={batch_size}, poll_interval_ms={poll_interval_ms}")

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Return fixed control values regardless of state."""
        return ControlOutput(
            batch_size=self.batch_size,
            poll_interval_ms=self.poll_interval_ms,
            mode='static'
        )


class LQRController(BaseController):
    """
    Linear Quadratic Regulator controller for optimal synchronization control.

    State vector x = [queue_length, cpu_util, mem_util, io_ops]
    Control vector u = [batch_size, poll_interval]

    The optimal control law is: u = -K @ (x - x_target)
    Where K is the optimal gain matrix computed from the Riccati equation.
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        u_nominal: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # System dimensions
        self.n_states = 4   # queue_length, cpu_util, mem_util, io_ops
        self.n_controls = 2  # batch_size, poll_interval

        # Target state (desired equilibrium)
        # Default: low queue, moderate CPU/mem usage, moderate I/O
        self.x_target = x_target if x_target is not None else np.array([
            0.0,    # queue_length: target 0 (no lag)
            50.0,   # cpu_util: target 50%
            50.0,   # mem_util: target 50%
            1e6,    # io_ops: target 1 MB/s
        ])

        # Nominal control (operating point)
        self.u_nominal = u_nominal if u_nominal is not None else np.array([
            100.0,   # batch_size
            1000.0,  # poll_interval_ms
        ])

        # System matrices (to be identified from experiments)
        # These are placeholders - actual values should come from system identification
        self.A = A if A is not None else np.eye(self.n_states)
        self.B = B if B is not None else np.zeros((self.n_states, self.n_controls))

        # Cost matrices
        # Q: penalize state deviations (higher = more aggressive tracking)
        self.Q = Q if Q is not None else np.diag([
            1.0,    # queue_length weight
            0.1,    # cpu_util weight
            0.1,    # mem_util weight
            0.01,   # io_ops weight (normalized)
        ])

        # R: penalize control effort (higher = smoother control)
        self.R = R if R is not None else np.diag([
            0.01,   # batch_size change weight
            0.001,  # poll_interval change weight
        ])

        # Compute optimal gain matrix
        self.K = self._solve_dare()

        logger.info("LQRController initialized")
        logger.info(f"Target state: {self.x_target}")
        logger.info(f"Nominal control: {self.u_nominal}")
        logger.info(f"Gain matrix K:\n{self.K}")

    def _solve_dare(self) -> np.ndarray:
        """
        Solve the Discrete Algebraic Riccati Equation (DARE) to get optimal gain K.

        The DARE is: A'PA - P - A'PB(R + B'PB)^(-1)B'PA + Q = 0

        Returns the optimal gain: K = (R + B'PB)^(-1)B'PA
        """
        try:
            from scipy import linalg

            # Solve DARE
            P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

            # Compute optimal gain
            K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

            return K

        except Exception as e:
            logger.warning(f"Failed to solve DARE: {e}. Using zero gain (static behavior).")
            return np.zeros((self.n_controls, self.n_states))

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """
        Compute optimal control given current state.

        Args:
            state: Dictionary with keys 'queue_length', 'cpu_util', 'mem_util', 'io_ops'

        Returns:
            ControlOutput with batch_size and poll_interval_ms
        """
        # Convert state dict to vector
        x = np.array([
            state.get('queue_length', 0),
            state.get('cpu_util', 0),
            state.get('mem_util', 0),
            state.get('io_ops', 0),
        ])

        # Compute state deviation from target
        x_error = x - self.x_target

        # Normalize io_ops for better scaling (divide by 1e6 to get MB/s scale)
        x_error[3] = x_error[3] / 1e6

        # Compute optimal control deviation: delta_u = -K @ x_error
        delta_u = -self.K @ x_error

        # Add to nominal control
        u = self.u_nominal + delta_u

        # Clamp to valid ranges
        batch_size, poll_interval = self._clamp_control(u[0], u[1])

        logger.debug(f"State: {x}, Error: {x_error}, Delta_u: {delta_u}, Control: ({batch_size}, {poll_interval})")

        return ControlOutput(
            batch_size=batch_size,
            poll_interval_ms=poll_interval,
            mode='lqr'
        )

    def update_matrices(self, A: np.ndarray, B: np.ndarray):
        """
        Update system matrices (e.g., after system identification).
        Recomputes the optimal gain matrix.
        """
        self.A = A
        self.B = B
        self.K = self._solve_dare()
        logger.info("System matrices updated, gain recomputed")

    def update_cost_matrices(self, Q: np.ndarray = None, R: np.ndarray = None):
        """
        Update cost matrices to adjust controller behavior.
        """
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        self.K = self._solve_dare()
        logger.info("Cost matrices updated, gain recomputed")


def create_controller(mode: str = 'static', **kwargs) -> BaseController:
    """
    Factory function to create appropriate controller based on mode.

    Args:
        mode: 'static' or 'lqr'
        **kwargs: Controller-specific parameters

    Returns:
        Controller instance
    """
    mode = mode.lower()

    # Get bounds from environment or kwargs
    bounds = {
        'min_batch_size': int(os.getenv('MIN_BATCH_SIZE', kwargs.get('min_batch_size', 1))),
        'max_batch_size': int(os.getenv('MAX_BATCH_SIZE', kwargs.get('max_batch_size', 1000))),
        'min_poll_interval_ms': int(os.getenv('MIN_POLL_INTERVAL_MS', kwargs.get('min_poll_interval_ms', 100))),
        'max_poll_interval_ms': int(os.getenv('MAX_POLL_INTERVAL_MS', kwargs.get('max_poll_interval_ms', 10000))),
    }

    if mode == 'static':
        return StaticController(
            batch_size=int(os.getenv('DEFAULT_BATCH_SIZE', kwargs.get('batch_size', 100))),
            poll_interval_ms=int(os.getenv('DEFAULT_POLL_INTERVAL_MS', kwargs.get('poll_interval_ms', 1000))),
            **bounds
        )
    elif mode == 'lqr':
        return LQRController(**bounds, **kwargs)
    else:
        raise ValueError(f"Unknown controller mode: {mode}. Use 'static' or 'lqr'.")


# Standalone test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print("Testing Static Controller:")
    static = create_controller('static', batch_size=50, poll_interval_ms=500)
    state = {'queue_length': 100, 'cpu_util': 80, 'mem_util': 60, 'io_ops': 1e6}
    output = static.compute_control(state)
    print(f"  State: {state}")
    print(f"  Output: batch_size={output.batch_size}, poll_interval={output.poll_interval_ms}ms")

    print("\nTesting LQR Controller:")
    lqr = create_controller('lqr')

    # Test with different states
    test_states = [
        {'queue_length': 0, 'cpu_util': 50, 'mem_util': 50, 'io_ops': 1e6},    # At target
        {'queue_length': 1000, 'cpu_util': 90, 'mem_util': 80, 'io_ops': 5e6}, # High load
        {'queue_length': 0, 'cpu_util': 10, 'mem_util': 20, 'io_ops': 1e5},    # Low load
    ]

    for state in test_states:
        output = lqr.compute_control(state)
        print(f"  State: queue={state['queue_length']}, cpu={state['cpu_util']}%, mem={state['mem_util']}%")
        print(f"  Output: batch_size={output.batch_size}, poll_interval={output.poll_interval_ms}ms")
        print()
