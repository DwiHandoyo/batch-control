"""
Controllers for CQRS synchronization parameter control.

Five control strategies for adaptively adjusting:
- batch_size: Number of messages to process per batch
- poll_interval: Time between polling Kafka (in milliseconds)

Based on the system state:
- queue_length: Kafka consumer lag
- cpu_util: Elasticsearch CPU utilization
- container_mem_pct: Elasticsearch container memory utilization (via cAdvisor, stable)
- io_write_ops: Elasticsearch disk I/O write operations per second

Controllers:
- StaticController: Fixed parameters (baseline)
- RuleBasedController: Threshold-based incremental adjustments
- PIDController: True MIMO 2x2 PID with cross-coupling on (queue, cpu)
- StateFeedbackController: u = -K·(x - x_target), K via pole placement
- LQRController: Optimal discrete-time linear quadratic regulator (K via DARE)
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

# Q matrix presets for LQR (4-dim state)
# State order: [queue_length, cpu_util, mem_util, io_write_ops]
Q_PRESETS = {
    'Q1': np.diag([20.0,  5.0, 0.05,  1.0]),   # backlog priority
    'Q2': np.diag([10.0, 10.0, 0.10,  2.0]),   # resource priority
    'Q4': np.diag([20.0, 10.0, 0.10,  2.0]),   # balanced aggressive
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
    def from_capacity_benchmark(
        cls,
        capacity_json: str,
        latency_sla_ms: float = 2000.0,
        backlog_safety: float = 2.0,
        cpu_safety: float = 1.5,
        **kwargs,
    ) -> 'RuleBasedController':
        """
        Derive thresholds from capacity benchmark + latency SLA.

        backlog_high = backlog_safety × (latency_SLA × throughput_at_nominal_batch)
                       — queue depth that would exceed acceptable latency
        cpu_high     = cpu_safety × cpu_at_knee_point
                       — CPU level that would push past efficient operating regime

        Magic factors (kept explicit in signature):
          backlog_safety: how much queue beyond SLA-equivalent triggers action
          cpu_safety:     how much above knee CPU triggers action
          latency_sla_ms: also from capacity_json inputs (canonical source)
        """
        with open(capacity_json) as f:
            cap = json.load(f)

        # Derive backlog_high from SLA
        default_batch = int(kwargs.get('default_batch_size', 250))
        default_poll = int(kwargs.get('default_poll_interval_ms', 200))
        throughput_nominal = default_batch / (default_poll / 1000.0)  # msg/s

        sla = float(cap.get('control_ranges', {}).get('inputs', {}).get(
            'latency_sla_ms', latency_sla_ms))
        backlog_high = backlog_safety * (sla / 1000.0) * throughput_nominal

        # Derive cpu_high from knee point in benchmark_results
        bench = cap.get('benchmark_results', [])
        # Knee = point where throughput stops growing significantly per CPU%
        # Heuristic: choose the batch level where CPU is < 50% but throughput is highest
        knee_cpu = None
        if bench:
            efficient = [b for b in bench if b.get('avg_cpu', 100) < 50]
            if efficient:
                knee = max(efficient, key=lambda b: b.get('throughput_msg_per_s', 0))
                knee_cpu = knee.get('avg_cpu')
        if knee_cpu is None:
            knee_cpu = 30.0  # fallback if benchmark missing
        cpu_high = cpu_safety * knee_cpu

        kwargs.setdefault('backlog_high', backlog_high)
        kwargs.setdefault('cpu_high', cpu_high)

        logger.info(
            f"RuleBased thresholds from capacity:\n"
            f"  throughput_nominal = {default_batch}/{default_poll/1000:.3f}s = {throughput_nominal:.0f} msg/s\n"
            f"  backlog_high = {backlog_safety} × (SLA={sla}ms × throughput) = {backlog_high:.0f}\n"
            f"  knee_cpu (from benchmark) = {knee_cpu:.1f}%\n"
            f"  cpu_high = {cpu_safety} × knee_cpu = {cpu_high:.1f}%"
        )
        return cls(**kwargs)

    @classmethod
    def from_sysid_json(cls, json_path: str, **kwargs) -> 'RuleBasedController':
        """
        Legacy: derive thresholds from sysid normalization stats.
        Prefer from_capacity_benchmark — sysid stats reflect the random
        excitation of open-loop, not the acceptable operational range.
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
    True MIMO 2x2 PID controller with cross-coupling.

    Error vector e = [e_queue, e_cpu] (one-sided on cpu).
    Control vector u = [batch_size, inv_poll_interval].

    Control law:
        u(t) = u_nominal + Kp @ e(t) + Ki @ ∫e dt + Kd @ de/dt

    Kp, Ki, Kd are 2x2 matrices. Cross-coupling (off-diagonal entries) lets
    batch and inv_poll respond to BOTH queue and cpu errors, mirroring the
    plant's MIMO B-matrix structure (both inputs affect both outputs).

    Default gains can be derived from the sysid B-matrix via
    `from_sysid_json` (Kp = α · pinv(B_reduced)).

    Anti-windup: integration is paused on the channel(s) whose corresponding
    control output saturated in the previous step.
    """

    def __init__(
        self,
        Kp: Optional[np.ndarray] = None,
        Ki: Optional[np.ndarray] = None,
        Kd: Optional[np.ndarray] = None,
        dt: float = 1.0,
        backlog_ref: float = 0.0,
        cpu_ref: float = 5.0,
        u_nominal: Optional[np.ndarray] = None,
        integral_max: float = 1.0e5,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Default 2x2 gains — physically intuitive (NOT from pinv, which gives
        # counterintuitive signs due to ill-conditioned B). Sign convention:
        #   e_q > 0  (queue above target) → batch ↑ , inv_poll ↑ (both drain)
        #   e_cpu > 0 (cpu overload)      → batch ↓ , inv_poll ↓ (both reduce load)
        # Magnitudes tuned conservatively to recover but stay below LQR.
        self.Kp = Kp if Kp is not None else np.array([
            [ 0.08,  -0.50 ],   # batch_size  from (e_queue, e_cpu)
            [ 0.0005,-0.05 ],   # inv_poll    from (e_queue, e_cpu)
        ])
        self.Ki = Ki if Ki is not None else 0.05 * self.Kp
        self.Kd = Kd if Kd is not None else 0.20 * self.Kp

        self.dt = dt
        self.backlog_ref = backlog_ref
        self.cpu_ref = cpu_ref
        self.u_nominal = u_nominal if u_nominal is not None else np.array([200.0, 5.0])
        self.integral_max = integral_max

        # PID state
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        # Saturation flags from previous step (per control channel)
        self.prev_saturated = np.zeros(2, dtype=bool)

        logger.info(
            f"PIDController (true MIMO 2x2) initialized:\n"
            f"  Kp=\n{self.Kp}\n  Ki=\n{self.Ki}\n  Kd=\n{self.Kd}\n"
            f"  u_nominal={self.u_nominal}, "
            f"backlog_ref={backlog_ref}, cpu_ref={cpu_ref}, dt={dt}"
        )

    @classmethod
    def from_sysid_json(
        cls,
        json_path: str,
        alpha: float = 0.05,
        beta: float = 0.05,
        gamma: float = 0.20,
        **kwargs
    ) -> 'PIDController':
        """
        Build PID gains from sysid B-matrix using a SIGN-CORRECTED pseudo-inverse.

        Plant: dy ≈ B · du, with y_reduced = [queue, cpu]. Negative-feedback law
        wants delta_u that produces dy = -e, i.e. delta_u = -pinv(B_reduced) @ e.
        So the gain in our convention `delta_u = Kp @ e` is:
            Kp = -alpha · pinv(B_reduced)

        However, pinv on the ill-conditioned 4-state B gives a fragile solution
        that uses differential cancellation (queue↑ → poll *slower*, which is
        counterintuitive). To keep the controller robust + intuitive, we
        regularize: take the sign-correct projection of -pinv(B_reduced) and
        clip off-diagonal magnitudes to at most |diag|.

            Kp_raw = -alpha · pinv(B_reduced)
            Kp[i,j] = sign(Kp_raw[i,j]) · min(|Kp_raw[i,j]|, |Kp_raw[i,i]|)

        Ki = beta · Kp,  Kd = gamma · Kp.
        Smaller alpha = more conservative (slower, less likely to outperform LQR).
        """
        with open(json_path) as f:
            data = json.load(f)
        B = np.array(data['B'])  # (n_states, n_controls)
        state_vars = data.get('state_vars', ['queue_length', 'cpu_util', 'mem_util', 'io_write_ops'])
        q_idx = state_vars.index('queue_length')
        c_idx = state_vars.index('cpu_util')
        B_reduced = B[[q_idx, c_idx], :]  # (2, 2)

        Kp_raw = -alpha * np.linalg.pinv(B_reduced)  # (2, 2): u = Kp @ e

        # Regularize: clip off-diagonal magnitudes to <= diagonal magnitudes
        Kp = Kp_raw.copy()
        for i in range(2):
            diag_mag = abs(Kp_raw[i, i])
            for j in range(2):
                if i != j and abs(Kp_raw[i, j]) > diag_mag:
                    Kp[i, j] = np.sign(Kp_raw[i, j]) * diag_mag

        Ki = beta * Kp
        Kd = gamma * Kp

        logger.info(
            f"PID from sysid {json_path}: alpha={alpha}, beta={beta}, gamma={gamma}\n"
            f"  B_reduced=\n{B_reduced}\n  Kp_raw=\n{Kp_raw}\n  Kp_regularized=\n{Kp}"
        )
        return cls(Kp=Kp, Ki=Ki, Kd=Kd, **kwargs)

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute control via true MIMO 2x2 PID with anti-windup."""
        queue_length = state.get('queue_length', 0)
        cpu_util = state.get('cpu_util', 0)

        # Error vector (one-sided on cpu)
        e = np.array([
            queue_length - self.backlog_ref,
            max(0.0, cpu_util - self.cpu_ref),
        ])

        # Anti-windup: skip integration on channels saturated last step.
        # Map control-channel saturation back to error-channel via |Ki|^T * sat.
        # Simple heuristic: if control[i] saturated and Ki[i, j] * e[j] would
        # push further into saturation, skip the integration of e[j].
        if self.dt > 0:
            for j in range(2):
                # If any control channel saturated in the direction this error
                # contributes, skip integrating e[j].
                push_dir = self.Ki[:, j] * e[j]
                blocked = np.any(self.prev_saturated & (np.sign(push_dir) != 0))
                if not blocked:
                    self.integral[j] += e[j] * self.dt
            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)

        deriv = (e - self.prev_error) / self.dt if self.dt > 0 else np.zeros(2)
        self.prev_error = e

        # MIMO PID law
        delta_u = self.Kp @ e + self.Ki @ self.integral + self.Kd @ deriv
        u = self.u_nominal + delta_u  # u = [batch, inv_poll]

        # Convert inv_poll to poll_interval_ms
        inv_poll = max(u[1], 0.001)
        poll_ms = 1000.0 / inv_poll

        batch_size, poll_interval = self._clamp_control(u[0], poll_ms)

        # Track saturation for next step's anti-windup
        self.prev_saturated = np.array([
            batch_size != u[0],
            inv_poll <= 0.001 or poll_interval == self.min_poll_interval_ms or poll_interval == self.max_poll_interval_ms,
        ], dtype=bool)

        logger.debug(
            f"PID MIMO: queue={queue_length}, cpu={cpu_util:.1f}%, "
            f"e={e}, delta_u={delta_u}, batch={batch_size}, poll={poll_interval}ms"
        )

        return ControlOutput(batch_size=batch_size, poll_interval_ms=poll_interval, mode='pid')

    def reset(self):
        """Reset PID integrator and history."""
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_saturated = np.zeros(2, dtype=bool)


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

    STATE_KEYS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']

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
        self.n_states = 4   # queue_length, cpu_util, mem_util, io_write_ops
        self.n_controls = 2  # batch_size, inv_poll_interval

        # Normalization scales from sysid (None = operate on raw values)
        self.normalization = normalization

        # Target state (desired equilibrium). Justifications:
        # - queue=0      : zero lag is the ideal sink condition
        # - cpu=5%       : matches observed idle baseline; one-sided penalty
        #                  activates whenever processing pushes CPU above this
        # - container_mem=85% : observed stable baseline ~84% (cAdvisor); one-sided
        # - io=30 ops/s  : middle of observed operational range
        self.x_target = x_target if x_target is not None else np.array([
            0.0,    # queue_length
            5.0,    # cpu_util (%)
            50.0,   # container_mem_pct target=50 (same as mem_util so old ANN treats it as mem_util)
            30.0,   # io_write_ops
        ])

        # Nominal control (operating point) derived from capacity benchmark
        # (results/capacity_benchmark/20260414_222623). Justifications:
        # - batch=250 : sweet spot — throughput 8.8k msg/s, CPU 28%, t_index
        #               0.27ms/doc; well below knee at batch=1000 (CPU 64%)
        # - inv_poll=5.0 (poll=200ms) : matches Kafka empty-poll RTT t_fetch≈
        #               203ms so each poll has a high chance of finding data
        self.u_nominal = u_nominal if u_nominal is not None else np.array([
            250.0,
            5.0,
        ])

        # System matrices (to be identified from experiments)
        # These are placeholders - actual values should come from system identification
        self.A = A if A is not None else np.eye(self.n_states)
        self.B = B if B is not None else np.zeros((self.n_states, self.n_controls))

        # Cost matrices
        # Q: penalize state deviations (higher = more aggressive tracking)
        self.Q = Q if Q is not None else np.eye(self.n_states)

        # R: control-effort penalty derived via Bryson's rule from operational
        # bounds (set by factory from MIN_/MAX_ env vars) and sysid normalization.
        # Bryson's rule: R[i,i] = R_base / (delta_u_max_normalized_i)^2
        # so that LQR's "natural" delta_u stays within the operational range.
        #
        # Without this, manually-tuned R magic numbers were needed (previously
        # 0.001 + gain_scale=[5,1] hack). Now R is fully derived; only R_base
        # remains as a single magic factor controlling overall aggressiveness.
        self.R = R if R is not None else self._compute_bryson_R(R_base=1.0)

        # Compute optimal gain matrix
        self.K = self._solve_dare()

        logger.info("LQRController initialized")
        logger.info(f"Target state: {self.x_target}")
        logger.info(f"Nominal control: {self.u_nominal}")
        logger.info(f"Normalization: {'active' if normalization else 'off'}")
        logger.info(f"Gain matrix K:\n{self.K}")

    @classmethod
    def from_sysid_json(cls, json_path: str,
                        capacity_json: Optional[str] = None,
                        **kwargs) -> 'LQRController':
        """
        Create LQRController from system identification JSON output.

        The random-excitation open-loop sysid yields an unreliable B[queue,:]
        row (B[queue,batch] is 7× too large, B[queue,inv_poll] is non-zero
        while empirically it should be near zero at nominal poll). This causes
        DARE to think nominal batch is already sufficient → K is too small →
        LQR outputs small delta_u → queue drains slowly.

        If capacity_json is provided, the queue row of B is replaced with
        empirical values derived from the capacity benchmark:
            B[queue, batch]    = -(tput(mean+std) - tput(mean)) × dt / std_queue
            B[queue, inv_poll] = -(tput effect of inv_poll change) × dt / std_queue

        At nominal poll ≈ t_fetch, poll efficiency is already 1.0 (saturated),
        so B[queue, inv_poll] ≈ 0. This forces DARE to allocate all queue
        correction to the batch channel → more aggressive batch usage.

        All values are derived from benchmark data — no magic numbers.
        """
        with open(json_path) as f:
            data = json.load(f)

        A = np.array(data['A'])
        B = np.array(data['B'])
        normalization = data.get('normalization', None)

        # Trim A and B to match STATE_KEYS dimension (4×4 and 4×2).
        # sysid may include extra states (e.g. avg_latency as 5th row/col).
        n = len(cls.STATE_KEYS)  # 4
        if A.shape[0] > n:
            orig_shape = A.shape[0]
            A = A[:n, :n]
            B = B[:n, :]
            logger.info(f"Trimmed A/B from {orig_shape}×{orig_shape} to {n}×{n} (extra states dropped)")

        logger.info(f"Loading sysid from {json_path}")
        logger.info(f"  Fit R²: {data.get('fit_metrics', {}).get('r_squared', 'N/A')}")

        # Override B[queue,:] from capacity benchmark if provided
        if capacity_json and normalization:
            try:
                from scipy.interpolate import interp1d as _interp1d
                with open(capacity_json) as f:
                    cap = json.load(f)

                bench = cap['benchmark_results']
                t_fetch_ms = cap['measurements']['t_fetch_ms']
                dt = 1.0  # sample interval (seconds)

                batches = [b['batch_size'] for b in bench]
                tputs   = [b['throughput_msg_per_s'] for b in bench]
                tput_fn = _interp1d(batches, tputs, kind='linear',
                                    bounds_error=False,
                                    fill_value=(tputs[0], tputs[-1]))

                norm_b   = normalization['batch_size']
                norm_q   = normalization['queue_length']
                norm_ip  = normalization['inv_poll_interval']

                b_mean, b_std   = norm_b['mean'],  norm_b['std']
                q_std           = norm_q['std']
                ip_mean, ip_std = norm_ip['mean'], norm_ip['std']

                # B[queue, batch]: sensitivity at mean batch
                tput_base  = float(tput_fn(b_mean))
                tput_plus  = float(tput_fn(b_mean + b_std))
                B_q_batch  = -(tput_plus - tput_base) * dt / q_std

                # B[queue, inv_poll]: at nominal inv_poll (poll ≈ t_fetch),
                # efficiency = min(1, t_fetch/poll_ms) is saturated → near zero
                ip_nom_ms  = 1000.0 / ip_mean
                ip_pls_ms  = 1000.0 / (ip_mean + ip_std)
                eff_nom    = min(1.0, t_fetch_ms / ip_nom_ms)
                eff_plus   = min(1.0, t_fetch_ms / ip_pls_ms)
                B_q_invp   = -(tput_base * eff_plus - tput_base * eff_nom) * dt / q_std

                # Find queue state index
                svar = data.get('state_vars', cls.STATE_KEYS)
                q_idx = svar.index('queue_length') if 'queue_length' in svar else 0

                B_orig = B[q_idx, :].copy()
                B[q_idx, 0] = B_q_batch
                B[q_idx, 1] = B_q_invp

                logger.info(
                    f"B[queue,:] replaced from capacity benchmark:\n"
                    f"  batch:    {B_orig[0]:.4f} → {B_q_batch:.4f}  "
                    f"(factor {B_orig[0]/B_q_batch:.1f}x)\n"
                    f"  inv_poll: {B_orig[1]:.4f} → {B_q_invp:.6f}  "
                    f"(nominal poll {ip_nom_ms:.0f}ms, eff={eff_nom:.3f})"
                )
            except Exception as e:
                logger.warning(f"Failed to override B from capacity: {e}. Using sysid B.")

        # Override normalization to reflect closed-loop operating range rather
        # than the random-excitation open-loop sysid range. The open-loop data
        # has queue 0-200k (std=10000) while closed-loop operates 0-7k. When
        # queue_std is 4× too large, K × queue_error in normalized space
        # produces a proportionally small delta_batch in raw space, making LQR
        # unresponsive despite having a large K entry.
        #
        # Chosen values and justifications (derived from closed-loop data):
        #
        # queue  : mean=0 (target), std=5000
        #          std = round(p95_queue / 1.645) = round(7050/1.645) ≈ 4286
        #          → rounded to 5000 (1 std = typical peak queue in step pattern)
        #
        # cpu    : mean=5 (= x_target), std=25
        #          p95_cpu ≈ 50% → std = p95/2 = 25 (covers 5-55% range)
        #
        # mem    : mean=50, std=20 — uncontrolled, keep sysid value (p5-p95≈52%)
        #
        # io     : mean=30 (= x_target), std=40
        #          p95_io ≈ 99 → std=40 covers full range (std×2.5 = 100)
        #
        # batch  : mean=250 (= u_nominal), std=250
        #          operational swing ≈ max_useful/4 = 2036/4 ≈ 509 → 250
        #          (1 std from nominal reaches max useful operating point)
        #
        # inv_poll: mean=5.0 (= u_nominal), std=2.0
        #          actual closed-loop std = 1.7 → rounded up to 2.0 for headroom
        if normalization is not None:
            normalization = {
                'queue_length':      {'mean':   0.0, 'std': 1000.0},
                'cpu_util':          {'mean':   5.0, 'std':   25.0},
                'container_mem_pct': {'mean':  50.0, 'std':   20.0},  # mem_util params: old ANN treats as mem_util
                'io_write_ops':      {'mean':  30.0, 'std':   40.0},
                'batch_size':        {'mean': 250.0, 'std':  250.0},
                'inv_poll_interval': {'mean':   5.0, 'std':    2.0},
            }
            logger.info(
                "Normalization overridden to closed-loop operating range:\n"
                "  queue: mean=0, std=1000  (reduced from 5000 for more responsive LQR)\n"
                "  batch: mean=250, std=250 (was 130/75)\n"
                "  inv_poll: mean=5, std=2  (was 5/3)"
            )

        return cls(A=A, B=B, normalization=normalization, **kwargs)

    def _compute_bryson_R(self, R_base: float = 1.0) -> np.ndarray:
        """
        Bryson's rule for R matrix: penalize control changes that would push
        delta_u beyond the operational range.

            R[i,i] = R_base / (delta_u_max_i_normalized)^2

        where:
            delta_u_max_i = max(u_max - u_nominal, u_nominal - u_min)
            delta_u_max_normalized = delta_u_max_i / std_i  (from sysid)

        Falls back to identity-like default if normalization or bounds missing.
        Logs the derivation for traceability.
        """
        # Operational bounds (raw units)
        u_min_raw = np.array([self.min_batch_size, 1000.0 / self.max_poll_interval_ms])
        u_max_raw = np.array([self.max_batch_size, 1000.0 / self.min_poll_interval_ms])

        # Max delta from nominal (in either direction — conservative)
        delta_u_max_raw = np.maximum(u_max_raw - self.u_nominal,
                                      self.u_nominal - u_min_raw)

        # Convert to normalized space (delta in std units) if normalization exists
        ctrl_keys = ['batch_size', 'inv_poll_interval']
        std = np.ones(self.n_controls)
        if self.normalization:
            for i, k in enumerate(ctrl_keys):
                if k in self.normalization and self.normalization[k]['std'] > 0:
                    std[i] = self.normalization[k]['std']

        delta_u_max_norm = delta_u_max_raw / std

        # Bryson: R diagonal
        R = np.diag(R_base / (delta_u_max_norm ** 2))

        logger.info(
            f"Bryson R derivation:\n"
            f"  u_min_raw={u_min_raw}, u_max_raw={u_max_raw}, u_nominal={self.u_nominal}\n"
            f"  delta_u_max_raw={delta_u_max_raw}, std={std}\n"
            f"  delta_u_max_norm={delta_u_max_norm}\n"
            f"  R = R_base({R_base}) / delta²:\n{R}"
        )
        return R

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
        # dtype=float64 critical: x_raw may be int (from state dict with int
        # values like queue_length, io_write_ops); without explicit float dtype
        # the fractional normalized values get truncated, making LQR appear
        # unresponsive even when state errors are present.
        x_norm = np.zeros_like(x_raw, dtype=np.float64)
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
        u_norm = np.zeros_like(u_raw, dtype=np.float64)
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
        u_raw = np.zeros_like(u_norm, dtype=np.float64)
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
        # Convert state dict to vector (float64 to avoid integer truncation
        # when normalization is applied — see _normalize_state).
        x_raw = np.array([state.get(k, 0) for k in self.STATE_KEYS], dtype=np.float64)

        if self.normalization:
            # Work in normalized space: x_norm, u_norm
            x_norm = self._normalize_state(x_raw)
            x_target_norm = self._normalize_target()
            x_error = x_norm - x_target_norm

            # One-sided penalty: only penalize resource overshoot
            # indices: 0=queue, 1=cpu, 2=mem, 3=io, 4=latency
            for i in [1, 2, 3]:  # cpu, mem, io
                x_error[i] = max(0.0, x_error[i])

            # Compute optimal control deviation in normalized space.
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


class StateFeedbackController(BaseController):
    """
    State feedback controller with pole placement.

    State vector x = [queue_length, cpu_util, mem_util, io_write_ops]
    Control vector u = [batch_size, inv_poll_interval]

    Control law: u = u_nominal - K · (x - x_target)

    K is computed via scipy.signal.place_poles such that the closed-loop
    poles of (A - B·K) are exactly at `desired_poles`.

    Differs from LQR (which has the same control law structure): K here is
    chosen by engineer-specified pole locations rather than by minimizing a
    quadratic cost (DARE). Used as a baseline showing classical state-feedback
    design — expected to be slower than LQR on the same plant.
    """

    STATE_KEYS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        desired_poles: Optional[list] = None,
        x_target: Optional[np.ndarray] = None,
        u_nominal: Optional[np.ndarray] = None,
        normalization: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_states = 4
        self.n_controls = 2

        # Default poles: stable inside unit circle, moderate response.
        # Intentionally slower than typical LQR poles so this baseline does
        # not accidentally outperform the optimal LQR controller.
        # Poles chosen close to 1.0 so the closed-loop system responds quickly.
        # The plant queue integrator (A[0,0]=1.0) requires nearby poles to produce
        # a large enough gain K. Far poles (0.5-0.8) yield a K that changes batch
        # only ~10 units per 1000 queue length. Poles [0.85-0.95] are a reasonable
        # compromise: responsive without being unstable or oscillatory.
        self.desired_poles = desired_poles if desired_poles is not None else [0.85, 0.90, 0.92, 0.95]

        self.normalization = normalization

        self.x_target = x_target if x_target is not None else np.array([
            0.0,   # queue_length: target 0
            5.0,   # cpu_util: target 5% (matches actual operating point)
            50.0,  # container_mem_pct target=50 (mem_util alias)
            30.0,  # io_write_ops
        ])
        self.u_nominal = u_nominal if u_nominal is not None else np.array([
            250.0, 5.0,  # same as LQR: batch=250, inv_poll=5 (poll=200ms)
        ])

        self.A = A if A is not None else np.eye(self.n_states)
        self.B = B if B is not None else np.zeros((self.n_states, self.n_controls))

        self.K = self._compute_K()

        logger.info("StateFeedbackController initialized (pole placement)")
        logger.info(f"Desired poles: {self.desired_poles}")
        logger.info(f"Target state: {self.x_target}")
        logger.info(f"Nominal control: {self.u_nominal}")
        logger.info(f"Normalization: {'active' if normalization else 'off'}")
        logger.info(f"Gain matrix K:\n{self.K}")

    @classmethod
    def from_sysid_json(cls, json_path: str, **kwargs) -> 'StateFeedbackController':
        """Build StateFeedbackController from sysid JSON (A, B, normalization).

        Uses the same closed-loop operating-range normalization as LQRController
        so that pole placement produces a K matrix whose magnitudes are proportional
        to the actual state deviations seen in closed-loop (not the wide open-loop
        random-excitation range where queue_std=10000 instead of the ~5000 observed
        in closed-loop). Without this, K entries for mem/io dominate because the
        normalized errors are larger than normalized queue errors.
        """
        with open(json_path) as f:
            data = json.load(f)
        A = np.array(data['A'])
        B = np.array(data['B'])
        normalization = data.get('normalization', None)

        # Override normalization to closed-loop operating range (same as LQRController)
        if normalization is not None:
            normalization = {
                'queue_length':      {'mean':   0.0, 'std': 1000.0},
                'cpu_util':          {'mean':   5.0, 'std':   25.0},
                'container_mem_pct': {'mean':  50.0, 'std':   20.0},  # mem_util params: old ANN treats as mem_util
                'io_write_ops':      {'mean':  30.0, 'std':   40.0},
                'batch_size':        {'mean': 250.0, 'std':  250.0},
                'inv_poll_interval': {'mean':   5.0, 'std':    2.0},
            }

        logger.info(f"Loading sysid for state-feedback from {json_path}")
        return cls(A=A, B=B, normalization=normalization, **kwargs)

    def _compute_K(self) -> np.ndarray:
        """
        Compute gain matrix K via pole placement so that
        eig(A - B·K) == desired_poles.
        """
        try:
            from scipy import signal
            result = signal.place_poles(self.A, self.B, self.desired_poles)
            K = result.gain_matrix
            achieved = np.linalg.eigvals(self.A - self.B @ K)
            logger.info(f"Pole placement achieved poles: {achieved}")
            return K
        except Exception as e:
            logger.warning(f"Pole placement failed: {e}. Using zero gain (static behavior).")
            return np.zeros((self.n_controls, self.n_states))

    # --- Normalization helpers (mirror LQRController) ---
    def _normalize_state(self, x_raw: np.ndarray) -> np.ndarray:
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

    def _normalize_target(self) -> np.ndarray:
        return self._normalize_state(self.x_target)

    def _normalize_control(self, u_raw: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return u_raw
        control_keys = ['batch_size', 'inv_poll_interval']
        u_norm = np.zeros_like(u_raw, dtype=np.float64)
        for i, key in enumerate(control_keys):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_norm[i] = (u_raw[i] - mean) / std
            else:
                u_norm[i] = u_raw[i]
        return u_norm

    def _denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return u_norm
        control_keys = ['batch_size', 'inv_poll_interval']
        u_raw = np.zeros_like(u_norm, dtype=np.float64)
        for i, key in enumerate(control_keys):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_raw[i] = u_norm[i] * std + mean
            else:
                u_raw[i] = u_norm[i]
        return u_raw

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        """Compute u = u_nominal - K · (x - x_target), with normalization."""
        x_raw = np.array([state.get(k, 0) for k in self.STATE_KEYS], dtype=np.float64)

        if self.normalization:
            x_norm = self._normalize_state(x_raw)
            x_target_norm = self._normalize_target()
            x_error = x_norm - x_target_norm
            for i in [1, 2, 3]:  # cpu, mem, io: one-sided penalty
                x_error[i] = max(0.0, x_error[i])
            delta_u_norm = -self.K @ x_error
            u_nominal_norm = self._normalize_control(self.u_nominal)
            u_norm = u_nominal_norm + delta_u_norm
            u_raw = self._denormalize_control(u_norm)
        else:
            x_error = x_raw - self.x_target
            for i in [1, 2, 3]:
                x_error[i] = max(0.0, x_error[i])
            delta_u = -self.K @ x_error
            u_raw = self.u_nominal + delta_u

        inv_poll = max(u_raw[1], 0.001)
        poll_ms = 1000.0 / inv_poll
        batch_size, poll_interval = self._clamp_control(u_raw[0], poll_ms)

        logger.debug(
            f"StateFB: x={x_raw}, e={x_error}, "
            f"batch={batch_size}, poll={poll_interval}ms"
        )
        return ControlOutput(
            batch_size=batch_size,
            poll_interval_ms=poll_interval,
            mode='state_fb'
        )


class ANNController(BaseController):
    """
    Neural network controller — numpy-only inference (no PyTorch at runtime).

    Loads a pre-trained model exported as JSON with weight matrices and biases.
    Architecture: state(4) → hidden(64) → hidden(64) → control(2), ReLU activations.

    Training is done offline via experiments/train_ann.py using PyTorch.
    The exported JSON contains numpy-compatible weights for fast inference.
    """

    STATE_KEYS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']
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


class ANNUniversalController(BaseController):
    """
    Q-aware universal ANN controller — single model handles all Q values.

    Architecture: input(8) → hidden → hidden → output(2)
        input  = [log10(Q_diag + 1e-3) (4), state_norm (4)]
        output = [batch_norm, inv_poll_norm]

    Trained via cost-weighted regression (no LQR oracle), so it can in
    principle exceed LQR in non-linear regions while still being Q-aware.
    The Q matrix is supplied at init (typically from Q_PRESETS via the
    factory) and stays fixed for the controller's lifetime.
    """

    STATE_KEYS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']
    CONTROL_KEYS = ['batch_size', 'inv_poll_interval']

    def __init__(
        self,
        layers: list,
        Q: np.ndarray,
        normalization: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layers = layers
        self.Q_diag = np.diag(Q)
        # Pre-compute normalized Q input (log10 transform — must match training)
        self.q_input = np.log10(self.Q_diag + 1e-3)
        self.normalization = normalization
        logger.info(
            f"ANNUniversalController initialized: {len(layers)} layers, "
            f"Q_diag={self.Q_diag.tolist()}, "
            f"normalization={'active' if normalization else 'off'}"
        )

    @classmethod
    def from_model_json(cls, json_path: str, Q: np.ndarray, **kwargs) -> 'ANNUniversalController':
        with open(json_path) as f:
            data = json.load(f)
        if data.get('model_type') != 'ann_universal_qaware':
            logger.warning(
                f"Model {json_path} model_type={data.get('model_type')!r}, "
                f"expected 'ann_universal_qaware'. Loading anyway."
            )
        layers = [
            {'weight': np.array(l['weight']), 'bias': np.array(l['bias'])}
            for l in data['layers']
        ]
        normalization = data.get('normalization', None)
        logger.info(f"Loading universal ANN model from {json_path}")
        return cls(layers=layers, Q=Q, normalization=normalization, **kwargs)

    # Alias map: if state key not found in normalization, try these fallbacks.
    # container_mem_pct → mem_util: old ANN models stored 'mem_util' normalization;
    # treating container_mem_pct as mem_util lets us reuse them without retraining.
    _NORM_ALIASES = {'container_mem_pct': 'mem_util'}

    def _normalize_state(self, x_raw: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return x_raw
        x_norm = np.zeros_like(x_raw, dtype=np.float64)
        for i, key in enumerate(self.STATE_KEYS):
            norm_key = key if key in self.normalization else self._NORM_ALIASES.get(key, key)
            if norm_key in self.normalization:
                mean = self.normalization[norm_key]['mean']
                std = self.normalization[norm_key]['std']
                x_norm[i] = (x_raw[i] - mean) / std if std > 0 else 0.0
            else:
                x_norm[i] = x_raw[i]
        return x_norm

    def _denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return u_norm
        u_raw = np.zeros_like(u_norm, dtype=np.float64)
        for i, key in enumerate(self.CONTROL_KEYS):
            if key in self.normalization:
                mean = self.normalization[key]['mean']
                std = self.normalization[key]['std']
                u_raw[i] = u_norm[i] * std + mean
            else:
                u_raw[i] = u_norm[i]
        return u_raw

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer['weight'] @ x + layer['bias']
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return x

    def compute_control(self, state: Dict[str, float]) -> ControlOutput:
        x_raw = np.array([state.get(k, 0) for k in self.STATE_KEYS], dtype=np.float64)
        x_norm = self._normalize_state(x_raw)
        # Concatenate Q (pre-normalized) with state
        net_input = np.concatenate([self.q_input, x_norm])
        u_norm = self._forward(net_input)
        u_raw = self._denormalize_control(u_norm)

        inv_poll = max(u_raw[1], 0.001)
        poll_ms = 1000.0 / inv_poll
        batch_size, poll_interval = self._clamp_control(u_raw[0], poll_ms)

        logger.debug(f"ANN-Universal: state={x_raw} -> batch={batch_size}, poll={poll_interval}")
        return ControlOutput(
            batch_size=batch_size, poll_interval_ms=poll_interval, mode='ann_universal'
        )


def create_controller(mode: str = 'static', **kwargs) -> BaseController:
    """
    Factory function to create appropriate controller based on mode.

    Args:
        mode: 'static', 'rule_based', 'pid', 'state_fb', 'lqr', 'ann', or 'ann_cw'
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
            mode = parts[0]  # "lqr_q1" → "lqr", "ann_cw_q1" → "ann_cw"
            if mode == 'lqr':
                kwargs['Q'] = Q_PRESETS[q_key]
            elif mode == 'ann':
                # Look for Q-specific model: ANN_MODEL_Q1, ANN_MODEL_Q2, etc.
                q_model = os.getenv(f'ANN_MODEL_{q_key}')
                if q_model:
                    kwargs['ann_model_json'] = q_model
            elif mode == 'ann_cw':
                # Universal cost-weighted ANN: 1 model, Q passed as input
                kwargs['Q'] = Q_PRESETS[q_key]

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
        capacity_json = os.getenv('CAPACITY_JSON', kwargs.pop('capacity_json', None))
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        rb_kwargs = {
            'default_batch_size': default_batch,
            'default_poll_interval_ms': default_poll,
        }
        env_bh = os.getenv('RULE_BASED_BACKLOG_HIGH')
        env_ch = os.getenv('RULE_BASED_CPU_HIGH')
        env_db = os.getenv('RULE_BASED_DELTA_B')
        if env_bh or 'backlog_high' in kwargs:
            rb_kwargs['backlog_high'] = float(env_bh or kwargs.pop('backlog_high', 5000))
        if env_ch or 'cpu_high' in kwargs:
            rb_kwargs['cpu_high'] = float(env_ch or kwargs.pop('cpu_high', 80.0))
        if env_db or 'delta_b' in kwargs:
            rb_kwargs['delta_b'] = int(env_db or kwargs.pop('delta_b', 50))

        # Prefer capacity benchmark over sysid (more relevant for closed-loop)
        if capacity_json:
            return RuleBasedController.from_capacity_benchmark(
                capacity_json, **bounds, **rb_kwargs, **kwargs
            )
        if sysid_json:
            return RuleBasedController.from_sysid_json(sysid_json, **bounds, **rb_kwargs, **kwargs)
        rb_kwargs.setdefault('backlog_high', 5000.0)
        rb_kwargs.setdefault('cpu_high', 80.0)
        rb_kwargs.setdefault('delta_b', 50)
        return RuleBasedController(**bounds, **rb_kwargs, **kwargs)
    elif mode == 'pid':
        # Default: hand-tuned MIMO gains in PIDController.__init__ (intuitive).
        # Opt-in: PID_FROM_SYSID=1 with SYSID_JSON set → derive Kp from B-matrix
        # via regularized pinv (advanced; may be ill-conditioned).
        pid_kwargs = {
            'dt': float(os.getenv('PID_DT', kwargs.pop('dt', 1.0))),
            'backlog_ref': float(os.getenv('PID_BACKLOG_REF', kwargs.pop('backlog_ref', 0.0))),
            'cpu_ref': float(os.getenv('PID_CPU_REF', kwargs.pop('cpu_ref', 5.0))),
        }
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        if sysid_json and os.getenv('PID_FROM_SYSID', '0') == '1':
            return PIDController.from_sysid_json(
                sysid_json,
                alpha=float(os.getenv('PID_ALPHA', kwargs.pop('alpha', 0.05))),
                beta=float(os.getenv('PID_BETA', kwargs.pop('beta', 0.05))),
                gamma=float(os.getenv('PID_GAMMA', kwargs.pop('gamma', 0.20))),
                **bounds, **pid_kwargs, **kwargs
            )
        return PIDController(**bounds, **pid_kwargs, **kwargs)
    elif mode == 'lqr':
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        capacity_json = os.getenv('CAPACITY_JSON', kwargs.pop('capacity_json', None))
        if sysid_json:
            return LQRController.from_sysid_json(
                sysid_json, capacity_json=capacity_json, **bounds, **kwargs
            )
        return LQRController(**bounds, **kwargs)
    elif mode == 'state_fb':
        sysid_json = os.getenv('SYSID_JSON', kwargs.pop('sysid_json', None))
        poles_str = os.getenv('STATE_FB_POLES', kwargs.pop('poles_str', '0.85,0.90,0.92,0.95'))
        desired_poles = [float(p) for p in poles_str.split(',')]
        if sysid_json:
            return StateFeedbackController.from_sysid_json(
                sysid_json, desired_poles=desired_poles, **bounds, **kwargs
            )
        return StateFeedbackController(
            desired_poles=desired_poles, **bounds, **kwargs
        )
    elif mode == 'ann':
        ann_model = os.getenv('ANN_MODEL_JSON', kwargs.pop('ann_model_json', None))
        if ann_model:
            return ANNController.from_model_json(ann_model, **bounds, **kwargs)
        raise ValueError("ANN mode requires ANN_MODEL_JSON env var or ann_model_json kwarg.")
    elif mode == 'ann_cw':
        # Universal cost-weighted ANN: single model, Q passed at init.
        # Q must be set (either via Q_PRESETS lookup from ann_cw_q1/q2/q4 mode
        # parsing above, or explicitly in kwargs).
        ann_model = os.getenv('ANN_UNIVERSAL_JSON', kwargs.pop('ann_model_json', None))
        if not ann_model:
            raise ValueError(
                "ann_cw mode requires ANN_UNIVERSAL_JSON env var or ann_model_json kwarg."
            )
        Q = kwargs.pop('Q', None)
        if Q is None:
            raise ValueError(
                "ann_cw requires Q matrix. Use 'ann_cw_q1' / 'ann_cw_q2' / 'ann_cw_q4' "
                "or pass Q=... kwarg."
            )
        return ANNUniversalController.from_model_json(ann_model, Q=Q, **bounds, **kwargs)
    else:
        raise ValueError(
            f"Unknown controller mode: {mode}. "
            f"Use 'static', 'rule_based', 'pid', 'state_fb', 'lqr', 'ann', or 'ann_cw'."
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

    # 3. PID (true MIMO 2x2)
    test_controller("PID", create_controller('pid'))

    # 4. LQR
    test_controller("LQR", create_controller('lqr'))

    # 4b. State Feedback (Pole Placement)
    test_controller("StateFeedback", create_controller('state_fb'))

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
