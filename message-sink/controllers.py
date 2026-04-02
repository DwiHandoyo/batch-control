"""
Controllers for CQRS synchronization parameter control.

Five control strategies for adaptively adjusting:
- batch_size: Number of messages to process per batch
- poll_interval: Time between polling Kafka (in milliseconds)

Based on the system state:
- queue_length: Kafka consumer lag
- cpu_util: Elasticsearch CPU utilization
- mem_util: Elasticsearch memory utilization
- io_write_ops: Elasticsearch disk I/O write operations per second

Controllers:
- StaticController: Fixed parameters (baseline)
- RuleBasedController: Threshold-based incremental adjustments
- PIDController: SISO feedback on queue_length error
- LQRController: Optimal discrete-time linear quadratic regulator
- ANNController: Neural network (numpy-only inference)
"""

import os
import json
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger('controllers')

# Q matrix presets for LQR (thesis §III.5.5, 5-dim state)
# State: [queue_length, cpu_util, mem_util, io_write_ops, indexing_time_rate]
Q_PRESETS = {
    'Q1': np.diag([10.0, 1.0, 1.0, 1.0, 1.0]),    # backlog priority
    'Q2': np.diag([1.0, 10.0, 10.0, 10.0, 1.0]),   # resource priority
    'Q3': np.diag([1.0, 1.0, 1.0, 1.0, 10.0]),     # latency priority
    'Q4': np.diag([1.0, 1.0, 1.0, 1.0, 1.0]),      # balanced (identity)
}


@dataclass
class ControlOutput:
    """Output from the controller."""
    batch_size: int
    poll_interval_ms: int
    mode: str  # 'static', 'rule_based', 'pid', 'lqr', or 'ann'


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


class RuleBasedController(BaseController):
    """
    Threshold-based controller with incremental adjustments.

    Rules:
    - If queue_length > backlog_high: increase batch_size by delta_b
    - If cpu_util > cpu_high: decrease batch_size by delta_b
    - Poll interval remains at default (or adjusted by separate rule)

    Thresholds can be derived from open-loop data distribution (mean + std).
    """

    def __init__(
        self,
        backlog_high: float = 5000.0,
        cpu_high: float = 80.0,
        delta_b: int = 50,
        default_batch_size: int = 100,
        default_poll_interval_ms: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backlog_high = backlog_high
        self.cpu_high = cpu_high
        self.delta_b = delta_b
        self.prev_batch = default_batch_size
        self.prev_poll = default_poll_interval_ms
        logger.info(
            f"RuleBasedController initialized: backlog_high={backlog_high}, "
            f"cpu_high={cpu_high}, delta_b={delta_b}"
        )

    @classmethod
    def from_sysid_json(cls, json_path: str, **kwargs) -> 'RuleBasedController':
        """
        Create RuleBasedController with thresholds derived from sysid normalization stats.
        backlog_high = mean + 1*std of queue_length
        cpu_high = mean + 1*std of cpu_util
        """
        with open(json_path) as f:
            data = json.load(f)
        norm = data.get('normalization', {})
        if 'queue_length' in norm:
            kwargs.setdefault('backlog_high', norm['queue_length']['mean'] + norm['queue_length']['std'])
        if 'cpu_util' in norm:
            kwargs.setdefault('cpu_high', norm['cpu_util']['mean'] + norm['cpu_util']['std'])
        logger.info(f"RuleBasedController: loading thresholds from {json_path}")
        return cls(**kwargs)

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute control using threshold rules."""
        b = self.prev_batch
        queue_length = state.get('queue_length', 0)
        cpu_util = state.get('cpu_util', 0)

        if queue_length > self.backlog_high:
            b += self.delta_b
        if cpu_util > self.cpu_high:
            b -= self.delta_b

        b, p = self._clamp_control(b, self.prev_poll)
        self.prev_batch = b
        self.prev_poll = p

        logger.debug(f"RuleBased: queue={queue_length}, cpu={cpu_util} -> batch={b}, poll={p}")

        return ControlOutput(batch_size=b, poll_interval_ms=p, mode='rule_based')


class PIDController(BaseController):
    """
    SISO PID controller — controls batch_size based on queue_length error.
    Poll interval stays at a fixed default.

    error = queue_length - backlog_ref  (positive when backlog is high)
    batch_size = Kp*e + Ki*integral(e) + Kd*derivative(e)

    Higher queue → larger batch_size (process more to drain the queue).
    Anti-windup via integral clamping.
    """

    def __init__(
        self,
        Kp: float = 0.1,
        Ki: float = 0.01,
        Kd: float = 0.05,
        dt: float = 5.0,
        backlog_ref: float = 0.0,
        default_poll_interval_ms: int = 1000,
        integral_max: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.backlog_ref = backlog_ref
        self.default_poll_interval_ms = default_poll_interval_ms
        self.integral_max = integral_max
        self.integral = 0.0
        self.prev_error = 0.0
        logger.info(
            f"PIDController initialized: Kp={Kp}, Ki={Ki}, Kd={Kd}, dt={dt}, "
            f"backlog_ref={backlog_ref}"
        )

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute batch_size via PID based on queue_length error."""
        queue_length = state.get('queue_length', 0)
        error = queue_length - self.backlog_ref

        # Integral with anti-windup
        self.integral = float(np.clip(
            self.integral + error * self.dt,
            -self.integral_max, self.integral_max
        ))

        # Derivative
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        self.prev_error = error

        # PID output
        b = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        b, p = self._clamp_control(b, self.default_poll_interval_ms)

        logger.debug(f"PID: queue={queue_length}, error={error:.1f}, batch={b}, poll={p}")

        return ControlOutput(batch_size=b, poll_interval_ms=p, mode='pid')

    def reset(self):
        """Reset PID integrator and derivative state."""
        self.integral = 0.0
        self.prev_error = 0.0


class LQRController(BaseController):
    """
    Linear Quadratic Regulator controller for optimal synchronization control.

    State vector x = [queue_length, cpu_util, mem_util, io_write_ops, indexing_time_rate]
    Control vector u = [batch_size, poll_interval]

    The optimal control law is: u = -K @ (x - x_target)
    Where K is the optimal gain matrix computed from the Riccati equation.

    When normalization is active (from sysid), raw state values are normalized
    before computing the control law: x_norm = (x_raw - mean) / std
    """

    STATE_KEYS = ['queue_length', 'cpu_util', 'mem_util', 'io_write_ops', 'avg_latency_ms']

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        u_nominal: Optional[np.ndarray] = None,
        normalization: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # System dimensions
        self.n_states = 5   # queue_length, cpu_util, mem_util, io_write_ops, indexing_time_rate
        self.n_controls = 2  # batch_size, poll_interval

        # Normalization scales from sysid (None = operate on raw values)
        self.normalization = normalization

        # Target state (desired equilibrium)
        # Default: low queue, moderate CPU/mem usage, moderate I/O
        self.x_target = x_target if x_target is not None else np.array([
            0.0,    # queue_length: target 0 (no lag)
            50.0,   # cpu_util: target 50%
            50.0,   # mem_util: target 50%
            30.0,   # io_write_ops: target ~30 ops/s (median from data)
            0.0,    # indexing_time_rate: target 0 (low latency)
        ])

        # Nominal control (operating point)
        # u = [batch_size, inv_poll_interval] where inv_poll = 1000/poll_ms
        self.u_nominal = u_nominal if u_nominal is not None else np.array([
            100.0,   # batch_size
            1.0,     # inv_poll_interval (1000/1000ms = 1.0 polls/s)
        ])

        # System matrices (to be identified from experiments)
        # These are placeholders - actual values should come from system identification
        self.A = A if A is not None else np.eye(self.n_states)
        self.B = B if B is not None else np.zeros((self.n_states, self.n_controls))

        # Cost matrices
        # Q: penalize state deviations (higher = more aggressive tracking)
        self.Q = Q if Q is not None else np.eye(self.n_states)

        # R: penalize control effort (higher = smoother control)
        self.R = R if R is not None else np.diag([
            0.1,    # batch_size change weight
            0.1,    # poll_interval change weight
        ])

        # Compute optimal gain matrix
        self.K = self._solve_dare()

        logger.info("LQRController initialized")
        logger.info(f"Target state: {self.x_target}")
        logger.info(f"Nominal control: {self.u_nominal}")
        logger.info(f"Normalization: {'active' if normalization else 'off'}")
        logger.info(f"Gain matrix K:\n{self.K}")

    @classmethod
    def from_sysid_json(cls, json_path: str, **kwargs) -> 'LQRController':
        """
        Create LQRController from system identification JSON output.

        Args:
            json_path: Path to sysid_matrices_*.json
            **kwargs: Additional arguments passed to constructor (Q, R, x_target, etc.)

        Returns:
            LQRController with identified A, B matrices and normalization scales
        """
        with open(json_path) as f:
            data = json.load(f)

        A = np.array(data['A'])
        B = np.array(data['B'])
        normalization = data.get('normalization', None)

        logger.info(f"Loading sysid from {json_path}")
        logger.info(f"  Fit R²: {data.get('fit_metrics', {}).get('r_squared', 'N/A')}")

        return cls(A=A, B=B, normalization=normalization, **kwargs)

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

    def _normalize_state(self, x_raw: np.ndarray) -> np.ndarray:
        """Normalize raw state vector using sysid normalization scales."""
        if self.normalization is None:
            return x_raw
        x_norm = np.zeros_like(x_raw)
        for i, key in enumerate(self.STATE_KEYS):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                x_norm[i] = (x_raw[i] - mean) / std
            else:
                x_norm[i] = x_raw[i]
        return x_norm

    def _normalize_target(self) -> np.ndarray:
        """Normalize target state using sysid normalization scales."""
        return self._normalize_state(self.x_target)

    def _normalize_control(self, u_raw: np.ndarray) -> np.ndarray:
        """Normalize raw control vector using sysid normalization scales."""
        if self.normalization is None:
            return u_raw
        control_keys = ['batch_size', 'inv_poll_interval']
        u_norm = np.zeros_like(u_raw)
        for i, key in enumerate(control_keys):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_norm[i] = (u_raw[i] - mean) / std
            else:
                u_norm[i] = u_raw[i]
        return u_norm

    def _denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        """Denormalize control vector back to raw units."""
        if self.normalization is None:
            return u_norm
        control_keys = ['batch_size', 'inv_poll_interval']
        u_raw = np.zeros_like(u_norm)
        for i, key in enumerate(control_keys):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_raw[i] = u_norm[i] * std + mean
            else:
                u_raw[i] = u_norm[i]
        return u_raw

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """
        Compute optimal control given current state.

        Args:
            state: Dictionary with keys 'queue_length', 'cpu_util', 'mem_util', 'io_write_ops'

        Returns:
            ControlOutput with batch_size and poll_interval_ms
        """
        # Convert state dict to vector
        x_raw = np.array([state.get(k, 0) for k in self.STATE_KEYS])

        if self.normalization:
            # Work in normalized space: x_norm, u_norm
            x_norm = self._normalize_state(x_raw)
            x_target_norm = self._normalize_target()
            x_error = x_norm - x_target_norm

            # One-sided penalty: only penalize resource overshoot
            # indices: 0=queue, 1=cpu, 2=mem, 3=io, 4=latency
            for i in [1, 2, 3]:  # cpu, mem, io
                x_error[i] = max(0.0, x_error[i])

            # Compute optimal control deviation in normalized space
            delta_u_norm = -self.K @ x_error

            # Add to normalized nominal control and denormalize
            u_nominal_norm = self._normalize_control(self.u_nominal)
            u_norm = u_nominal_norm + delta_u_norm
            u_raw = self._denormalize_control(u_norm)
        else:
            # Work in raw space
            x_error = x_raw - self.x_target
            for i in [1, 2, 3]:  # cpu, mem, io: one-sided penalty
                x_error[i] = max(0.0, x_error[i])
            delta_u = -self.K @ x_error
            u_raw = self.u_nominal + delta_u

        # u_raw = [batch_size, inv_poll_interval]
        # Convert inv_poll back to poll_interval_ms
        inv_poll = max(u_raw[1], 0.001)  # avoid division by zero
        poll_ms = 1000.0 / inv_poll

        # Clamp to valid ranges
        batch_size, poll_interval = self._clamp_control(u_raw[0], poll_ms)

        logger.debug(f"State: {x_raw}, Error: {x_error}, Control: ({batch_size}, {poll_interval})")

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


class ANNController(BaseController):
    """
    Neural network controller — numpy-only inference (no PyTorch at runtime).

    Loads a pre-trained model exported as JSON with weight matrices and biases.
    Architecture: state(4) → hidden(64) → hidden(64) → control(2), ReLU activations.

    Training is done offline via experiments/train_ann.py using PyTorch.
    The exported JSON contains numpy-compatible weights for fast inference.
    """

    STATE_KEYS = ['queue_length', 'cpu_util', 'mem_util', 'io_write_ops', 'avg_latency_ms']
    CONTROL_KEYS = ['batch_size', 'inv_poll_interval']

    def __init__(
        self,
        layers: list,
        normalization: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layers = layers  # list of {'weight': ndarray, 'bias': ndarray}
        self.normalization = normalization
        logger.info(
            f"ANNController initialized: {len(layers)} layers, "
            f"normalization={'active' if normalization else 'off'}"
        )

    @classmethod
    def from_model_json(cls, json_path: str, **kwargs) -> 'ANNController':
        """Load ANN controller from exported model JSON."""
        with open(json_path) as f:
            data = json.load(f)
        layers = [
            {'weight': np.array(l['weight']), 'bias': np.array(l['bias'])}
            for l in data['layers']
        ]
        normalization = data.get('normalization', None)
        logger.info(f"Loading ANN model from {json_path}")
        return cls(layers=layers, normalization=normalization, **kwargs)

    def _normalize_state(self, x_raw: np.ndarray) -> np.ndarray:
        """Normalize raw state vector using stored normalization scales."""
        if self.normalization is None:
            return x_raw
        x_norm = np.zeros_like(x_raw, dtype=np.float64)
        for i, key in enumerate(self.STATE_KEYS):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                x_norm[i] = (x_raw[i] - mean) / std if std > 0 else 0.0
            else:
                x_norm[i] = x_raw[i]
        return x_norm

    def _denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        """Denormalize control vector back to raw units."""
        if self.normalization is None:
            return u_norm
        u_raw = np.zeros_like(u_norm)
        for i, key in enumerate(self.CONTROL_KEYS):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_raw[i] = u_norm[i] * std + mean
            else:
                u_raw[i] = u_norm[i]
        return u_raw

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network using numpy."""
        for i, layer in enumerate(self.layers):
            x = layer['weight'] @ x + layer['bias']
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)  # ReLU on hidden layers
        return x

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute control via neural network forward pass."""
        x_raw = np.array([state.get(k, 0) for k in self.STATE_KEYS], dtype=np.float64)
        x_norm = self._normalize_state(x_raw)
        u_norm = self._forward(x_norm)
        u_raw = self._denormalize_control(u_norm)

        # u_raw = [batch_size, inv_poll_interval]
        inv_poll = max(u_raw[1], 0.001)
        poll_ms = 1000.0 / inv_poll

        batch_size, poll_interval = self._clamp_control(u_raw[0], poll_ms)

        logger.debug(f"ANN: state={x_raw} -> batch={batch_size}, poll={poll_interval}")

        return ControlOutput(batch_size=batch_size, poll_interval_ms=poll_interval, mode='ann')


def create_controller(mode: str = 'static', **kwargs) -> BaseController:
    """
    Factory function to create appropriate controller based on mode.

    Args:
        mode: 'static', 'rule_based', 'pid', 'lqr', or 'ann'
        **kwargs: Controller-specific parameters

    Returns:
        Controller instance
    """
    mode = mode.lower()

    # Parse Q-variant modes: "lqr_q1" → base_mode="lqr", Q=Q_PRESETS["Q1"]
    # Also: "ann_q1" → base_mode="ann", load ANN_MODEL_Q1 env var
    q_variant = None
    if '_q' in mode:
        parts = mode.rsplit('_', 1)
        q_key = parts[1].upper()  # "q1" → "Q1"
        if q_key in Q_PRESETS:
            q_variant = q_key
            mode = parts[0]  # "lqr_q1" → "lqr"
            if mode == 'lqr':
                kwargs['Q'] = Q_PRESETS[q_key]
            elif mode == 'ann':
                # Look for Q-specific model: ANN_MODEL_Q1, ANN_MODEL_Q2, etc.
                q_model = os.getenv(f'ANN_MODEL_{q_key}')
                if q_model:
                    kwargs['ann_model_json'] = q_model

    # Get bounds from environment or kwargs
    bounds = {
        'min_batch_size': int(os.getenv('MIN_BATCH_SIZE', kwargs.get('min_batch_size', 1))),
        'max_batch_size': int(os.getenv('MAX_BATCH_SIZE', kwargs.get('max_batch_size', 1000))),
        'min_poll_interval_ms': int(os.getenv('MIN_POLL_INTERVAL_MS', kwargs.get('min_poll_interval_ms', 100))),
        'max_poll_interval_ms': int(os.getenv('MAX_POLL_INTERVAL_MS', kwargs.get('max_poll_interval_ms', 10000))),
    }

    default_batch = int(os.getenv('DEFAULT_BATCH_SIZE', kwargs.pop('batch_size', 100)))
    default_poll = int(os.getenv('DEFAULT_POLL_INTERVAL_MS', kwargs.pop('poll_interval_ms', 1000)))

    if mode == 'static':
        return StaticController(
            batch_size=default_batch,
            poll_interval_ms=default_poll,
            **bounds
        )
    elif mode == 'rule_based':
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        rb_kwargs = {
            'default_batch_size': default_batch,
            'default_poll_interval_ms': default_poll,
        }
        # Only set thresholds if explicitly provided (env var or kwarg)
        # Otherwise let from_sysid_json derive them from data
        env_bh = os.getenv('RULE_BASED_BACKLOG_HIGH')
        env_ch = os.getenv('RULE_BASED_CPU_HIGH')
        env_db = os.getenv('RULE_BASED_DELTA_B')
        if env_bh or 'backlog_high' in kwargs:
            rb_kwargs['backlog_high'] = float(env_bh or kwargs.pop('backlog_high', 5000))
        if env_ch or 'cpu_high' in kwargs:
            rb_kwargs['cpu_high'] = float(env_ch or kwargs.pop('cpu_high', 80.0))
        if env_db or 'delta_b' in kwargs:
            rb_kwargs['delta_b'] = int(env_db or kwargs.pop('delta_b', 50))
        if sysid_json:
            return RuleBasedController.from_sysid_json(sysid_json, **bounds, **rb_kwargs, **kwargs)
        # No sysid — use defaults for any unset thresholds
        rb_kwargs.setdefault('backlog_high', 5000.0)
        rb_kwargs.setdefault('cpu_high', 80.0)
        rb_kwargs.setdefault('delta_b', 50)
        return RuleBasedController(**bounds, **rb_kwargs, **kwargs)
    elif mode == 'pid':
        return PIDController(
            Kp=float(os.getenv('PID_KP', kwargs.pop('Kp', 0.1))),
            Ki=float(os.getenv('PID_KI', kwargs.pop('Ki', 0.01))),
            Kd=float(os.getenv('PID_KD', kwargs.pop('Kd', 0.05))),
            dt=float(os.getenv('PID_DT', kwargs.pop('dt', 5.0))),
            backlog_ref=float(os.getenv('PID_BACKLOG_REF', kwargs.pop('backlog_ref', 0.0))),
            default_poll_interval_ms=default_poll,
            **bounds, **kwargs
        )
    elif mode == 'lqr':
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        if sysid_json:
            return LQRController.from_sysid_json(sysid_json, **bounds, **kwargs)
        return LQRController(**bounds, **kwargs)
    elif mode == 'ann':
        ann_model = os.getenv('ANN_MODEL_JSON', kwargs.pop('ann_model_json', None))
        if ann_model:
            return ANNController.from_model_json(ann_model, **bounds, **kwargs)
        raise ValueError("ANN mode requires ANN_MODEL_JSON env var or ann_model_json kwarg.")
    else:
        raise ValueError(
            f"Unknown controller mode: {mode}. "
            f"Use 'static', 'rule_based', 'pid', 'lqr', or 'ann'."
        )


# Standalone test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    test_states = [
        {'queue_length': 0, 'cpu_util': 50, 'mem_util': 50, 'io_write_ops': 30},      # At target
        {'queue_length': 10000, 'cpu_util': 90, 'mem_util': 80, 'io_write_ops': 200},  # High load
        {'queue_length': 0, 'cpu_util': 10, 'mem_util': 20, 'io_write_ops': 5},        # Low load
    ]

    def test_controller(name, ctrl):
        print(f"\n{'=' * 60}")
        print(f"Testing {name} (mode={ctrl.compute_control(test_states[0]).mode})")
        print('=' * 60)
        for state in test_states:
            output = ctrl.compute_control(state)
            print(f"  queue={state['queue_length']:>6}, cpu={state['cpu_util']:>3}%, "
                  f"mem={state['mem_util']:>3}% -> "
                  f"batch={output.batch_size:>4}, poll={output.poll_interval_ms:>5}ms")

    # 1. Static
    test_controller("Static", create_controller('static', batch_size=50, poll_interval_ms=500))

    # 2. Rule-Based
    test_controller("Rule-Based", create_controller('rule_based', backlog_high=5000, cpu_high=80))

    # 3. PID
    test_controller("PID", create_controller('pid', Kp=0.1, Ki=0.01, Kd=0.05))

    # 4. LQR
    test_controller("LQR", create_controller('lqr'))

    # 5. ANN (mock — create a simple 2-layer network with random weights)
    print(f"\n{'=' * 60}")
    print("Testing ANN (mock random weights)")
    print('=' * 60)
    np.random.seed(42)
    mock_layers = [
        {'weight': np.random.randn(64, 4) * 0.1, 'bias': np.zeros(64)},
        {'weight': np.random.randn(64, 64) * 0.1, 'bias': np.zeros(64)},
        {'weight': np.random.randn(2, 64) * 0.1, 'bias': np.array([100.0, 1000.0])},
    ]
    ann = ANNController(layers=mock_layers)
    for state in test_states:
        output = ann.compute_control(state)
        print(f"  queue={state['queue_length']:>6}, cpu={state['cpu_util']:>3}%, "
              f"mem={state['mem_util']:>3}% -> "
              f"batch={output.batch_size:>4}, poll={output.poll_interval_ms:>5}ms")
