"""
System Identification Analysis

Estimates system matrices A and B from experiment data using Least Squares regression.

Model: x[k+1] = A·x[k] + B·u[k]

The least squares formulation:
    X_next = [A | B] · [X_curr]
                       [U_curr]

    Let Theta = [A | B], Z = [X_curr; U_curr]
    Then: X_next = Theta · Z
    Solution: Theta = X_next · Z' · (Z · Z')^(-1)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sysid-analysis')

# State and control variable names
STATE_VARS = ['queue_length', 'cpu_util', 'mem_util', 'io_ops']
CONTROL_VARS = ['batch_size', 'poll_interval']
N_STATES = len(STATE_VARS)
N_CONTROLS = len(CONTROL_VARS)


def load_data(filepath: str) -> pd.DataFrame:
    """Load experiment data from CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {filepath}")
    logger.info(f"Phases: {df['phase'].unique()}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def normalize_data(df: pd.DataFrame) -> tuple:
    """
    Normalize data for better numerical conditioning.
    Returns normalized dataframe and scaling parameters.
    """
    scales = {}
    df_norm = df.copy()

    for var in STATE_VARS + CONTROL_VARS:
        mean = df[var].mean()
        std = df[var].std()
        if std == 0:
            std = 1.0
        df_norm[var] = (df[var] - mean) / std
        scales[var] = {'mean': mean, 'std': std}

    return df_norm, scales


def build_regression_matrices(df: pd.DataFrame, phases: list = None) -> tuple:
    """
    Build regression matrices for least squares estimation.

    X_next = [A | B] · [X_curr; U_curr]

    Returns:
        X_next: (n_states, N-1) matrix of next states
        Z: (n_states + n_controls, N-1) matrix of [current_state; control]
    """
    if phases:
        df = df[df['phase'].isin(phases)].reset_index(drop=True)

    N = len(df) - 1  # number of transitions

    # Build state matrix X: each column is a state vector
    X = df[STATE_VARS].values  # (N+1, n_states)

    # Build control matrix U
    U = df[CONTROL_VARS].values  # (N+1, n_controls)

    # Current states and controls
    X_curr = X[:-1, :].T  # (n_states, N)
    U_curr = U[:-1, :].T  # (n_controls, N)

    # Next states
    X_next = X[1:, :].T   # (n_states, N)

    # Combined [X_curr; U_curr]
    Z = np.vstack([X_curr, U_curr])  # (n_states + n_controls, N)

    logger.info(f"Regression matrices: X_next={X_next.shape}, Z={Z.shape}, transitions={N}")
    return X_next, Z


def least_squares_identification(X_next: np.ndarray, Z: np.ndarray) -> tuple:
    """
    Perform Least Squares system identification.

    Theta = X_next · Z^T · (Z · Z^T)^(-1)

    Returns:
        A: (n_states, n_states) state transition matrix
        B: (n_states, n_controls) input matrix
        residuals: fitting residuals
    """
    # Solve using numpy least squares (more numerically stable than direct inverse)
    # X_next = Theta @ Z
    # X_next.T = Z.T @ Theta.T
    # Solve: Z.T @ Theta.T = X_next.T for Theta.T

    Theta_T, residuals, rank, sv = np.linalg.lstsq(Z.T, X_next.T, rcond=None)
    Theta = Theta_T.T  # (n_states, n_states + n_controls)

    A = Theta[:, :N_STATES]
    B = Theta[:, N_STATES:]

    # Compute fit quality
    X_pred = Theta @ Z
    error = X_next - X_pred
    rmse = np.sqrt(np.mean(error ** 2, axis=1))

    # R-squared for each state variable
    ss_res = np.sum(error ** 2, axis=1)
    ss_tot = np.sum((X_next - X_next.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r_squared = 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1)

    logger.info("Least Squares Identification Results:")
    logger.info(f"  Matrix rank: {rank}")
    logger.info(f"  Singular values: {sv}")
    for i, var in enumerate(STATE_VARS):
        logger.info(f"  {var}: RMSE={rmse[i]:.6f}, R²={r_squared[i]:.4f}")

    return A, B, {'rmse': rmse, 'r_squared': r_squared, 'rank': rank}


def validate_model(A, B, df_val: pd.DataFrame) -> dict:
    """
    Validate identified model against validation data.
    Simulates the model forward and compares to actual data.
    """
    X = df_val[STATE_VARS].values
    U = df_val[CONTROL_VARS].values
    N = len(X)

    # Simulate forward
    X_sim = np.zeros_like(X)
    X_sim[0] = X[0]  # start from actual initial state

    for k in range(N - 1):
        x_k = X_sim[k]
        u_k = U[k]
        X_sim[k + 1] = A @ x_k + B @ u_k

    # Compute validation metrics
    error = X - X_sim
    rmse = np.sqrt(np.mean(error ** 2, axis=0))

    ss_res = np.sum(error ** 2, axis=0)
    ss_tot = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    r_squared = 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1)

    logger.info("Validation Results (forward simulation):")
    for i, var in enumerate(STATE_VARS):
        logger.info(f"  {var}: RMSE={rmse[i]:.6f}, R²={r_squared[i]:.4f}")

    return {
        'rmse': rmse,
        'r_squared': r_squared,
        'X_actual': X,
        'X_simulated': X_sim,
    }


def check_stability(A: np.ndarray) -> dict:
    """Check system stability via eigenvalue analysis."""
    eigenvalues = np.linalg.eigvals(A)
    magnitudes = np.abs(eigenvalues)
    is_stable = np.all(magnitudes < 1.0)  # discrete-time stability

    logger.info("Stability Analysis:")
    logger.info(f"  Eigenvalues: {eigenvalues}")
    logger.info(f"  Magnitudes: {magnitudes}")
    logger.info(f"  System is {'STABLE' if is_stable else 'UNSTABLE'}")

    return {
        'eigenvalues': eigenvalues,
        'magnitudes': magnitudes,
        'is_stable': is_stable,
    }


def check_controllability(A: np.ndarray, B: np.ndarray) -> dict:
    """Check system controllability."""
    n = A.shape[0]
    # Controllability matrix: [B, AB, A²B, ..., A^(n-1)B]
    C = B.copy()
    Ak = A.copy()
    for i in range(1, n):
        C = np.hstack([C, Ak @ B])
        Ak = Ak @ A

    rank = np.linalg.matrix_rank(C)
    is_controllable = rank == n

    logger.info("Controllability Analysis:")
    logger.info(f"  Controllability matrix rank: {rank}/{n}")
    logger.info(f"  System is {'CONTROLLABLE' if is_controllable else 'NOT CONTROLLABLE'}")

    return {
        'rank': rank,
        'n_states': n,
        'is_controllable': is_controllable,
    }


def save_results(A, B, metrics, output_dir, scales=None):
    """Save identified matrices and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save matrices as numpy
    np.save(os.path.join(output_dir, f'A_matrix_{timestamp}.npy'), A)
    np.save(os.path.join(output_dir, f'B_matrix_{timestamp}.npy'), B)

    # Save as readable text
    report_path = os.path.join(output_dir, f'sysid_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("System Identification Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model: x[k+1] = A·x[k] + B·u[k]\n\n")
        f.write(f"State variables: {STATE_VARS}\n")
        f.write(f"Control variables: {CONTROL_VARS}\n\n")

        f.write("A matrix (state transition):\n")
        f.write(np.array2string(A, precision=6, suppress_small=True) + "\n\n")

        f.write("B matrix (input):\n")
        f.write(np.array2string(B, precision=6, suppress_small=True) + "\n\n")

        f.write("Fit Quality (R²):\n")
        for i, var in enumerate(STATE_VARS):
            f.write(f"  {var}: R²={metrics['r_squared'][i]:.4f}, RMSE={metrics['rmse'][i]:.6f}\n")

        if scales:
            f.write("\nNormalization Scales:\n")
            for var, s in scales.items():
                f.write(f"  {var}: mean={s['mean']:.4f}, std={s['std']:.4f}\n")

    # Save as JSON for loading into LQR controller
    json_path = os.path.join(output_dir, f'sysid_matrices_{timestamp}.json')
    result = {
        'A': A.tolist(),
        'B': B.tolist(),
        'state_vars': STATE_VARS,
        'control_vars': CONTROL_VARS,
        'fit_metrics': {
            'rmse': metrics['rmse'].tolist(),
            'r_squared': metrics['r_squared'].tolist(),
        },
        'timestamp': datetime.now().isoformat(),
    }
    if scales:
        result['normalization'] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in scales.items()}

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Matrices (JSON): {json_path}")

    return report_path, json_path


def main():
    parser = argparse.ArgumentParser(description='System Identification Analysis')
    parser.add_argument('data_file', help='Path to experiment data CSV')
    parser.add_argument('--output-dir', default='./results')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize data before identification')
    parser.add_argument('--train-phases', nargs='+',
                        default=['vary_batch', 'vary_poll'],
                        help='Phases to use for training')
    parser.add_argument('--val-phases', nargs='+',
                        default=['vary_both'],
                        help='Phases to use for validation')

    args = parser.parse_args()

    # Load data
    df = load_data(args.data_file)

    # Optionally normalize
    scales = None
    if args.normalize:
        df, scales = normalize_data(df)
        logger.info("Data normalized")

    # Build regression matrices from training phases
    train_phases = [p for p in args.train_phases if p in df['phase'].unique()]
    if not train_phases:
        logger.warning("No matching training phases found, using all data")
        train_phases = df['phase'].unique().tolist()

    X_next, Z = build_regression_matrices(df, phases=train_phases)

    # Identify system
    A, B, metrics = least_squares_identification(X_next, Z)

    print("\n" + "=" * 60)
    print("IDENTIFIED SYSTEM MATRICES")
    print("=" * 60)
    print(f"\nA matrix ({N_STATES}x{N_STATES}) - State Transition:")
    print(f"  State: {STATE_VARS}")
    print(np.array2string(A, precision=4, suppress_small=True))

    print(f"\nB matrix ({N_STATES}x{N_CONTROLS}) - Control Input:")
    print(f"  Control: {CONTROL_VARS}")
    print(np.array2string(B, precision=6, suppress_small=True))

    # Stability and controllability analysis
    print("\n" + "=" * 60)
    stability = check_stability(A)
    controllability = check_controllability(A, B)

    # Validate if validation data exists
    val_phases = [p for p in args.val_phases if p in df['phase'].unique()]
    if val_phases:
        print("\n" + "=" * 60)
        df_val = df[df['phase'].isin(val_phases)].reset_index(drop=True)
        if len(df_val) > 1:
            validation = validate_model(A, B, df_val)

    # Save results
    print("\n" + "=" * 60)
    save_results(A, B, metrics, args.output_dir, scales)

    # Print usage hint
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("Load matrices into LQR controller:")
    print("  import numpy as np")
    print(f"  A = np.array({A.tolist()})")
    print(f"  B = np.array({B.tolist()})")
    print("  controller = LQRController(A=A, B=B)")


# Need json for save_results
import json

if __name__ == '__main__':
    main()
