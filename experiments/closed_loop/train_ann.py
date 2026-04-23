"""
ANN Controller Training — Train a neural network to approximate LQR control.

Uses open-loop experiment data with LQR controller as oracle to generate
optimal control targets u* for supervised learning.

Usage:
    python train_ann.py --sysid-json <path> --run <run_id> [options]
    python train_ann.py --sysid-json <path> --runs <id1> <id2> [options]

Output:
    ann_model_<timestamp>.json — exported model weights (numpy-compatible)
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add paths for importing message-sink and shared experiment modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'message-sink'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # for select_run

from select_run import get_run_data, merge_run_data, RUNS_DIR
from controllers import LQRController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train-ann')

STATE_VARS = ['queue_length', 'cpu_util', 'mem_util', 'io_write_ops']
CONTROL_VARS = ['batch_size', 'inv_poll_interval']


def load_and_prepare_data(
    run_ids: List[str],
    runs_dir: str = RUNS_DIR,
    filter_exhausted: bool = True,
    exclude_settle: bool = True,
) -> pd.DataFrame:
    """Load experiment data, filter unusable samples."""
    if len(run_ids) == 1:
        df = get_run_data(run_ids[0], runs_dir)
    else:
        df = merge_run_data(run_ids, runs_dir)

    n_before = len(df)

    if filter_exhausted and 'queue_exhausted' in df.columns:
        df = df[df['queue_exhausted'].astype(str) != 'True']
        logger.info(f"Filtered queue_exhausted: {n_before} -> {len(df)}")

    if exclude_settle and 'phase' in df.columns:
        df = df[~df['phase'].str.contains('settle', na=False)]
        logger.info(f"Excluded settle phases: -> {len(df)}")

    # Verify required columns exist
    missing = [c for c in STATE_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing state columns: {missing}")

    logger.info(f"Prepared {len(df)} samples from {len(run_ids)} run(s)")
    return df


def generate_lqr_targets(
    df: pd.DataFrame,
    lqr: LQRController,
) -> np.ndarray:
    """
    Generate optimal control targets using LQR controller as oracle.

    For each state in the dataset, compute u* = LQR.compute_control(state).
    Returns array of shape (n_samples, 2) with [batch_size, poll_interval].
    """
    targets = np.zeros((len(df), 2))
    for i, (_, row) in enumerate(df.iterrows()):
        state = {k: float(row[k]) for k in STATE_VARS}
        output = lqr.compute_control(state)
        targets[i, 0] = output.batch_size
        targets[i, 1] = output.poll_interval_ms

    logger.info(
        f"Generated LQR targets: batch=[{targets[:,0].min():.0f}, {targets[:,0].max():.0f}], "
        f"poll=[{targets[:,1].min():.0f}, {targets[:,1].max():.0f}]"
    )
    return targets


def normalize_data(
    states: np.ndarray,
    targets: np.ndarray,
    normalization: Dict[str, Dict[str, float]],
) -> tuple:
    """Normalize states and targets using sysid normalization scales."""
    states_norm = np.zeros_like(states)
    for i, key in enumerate(STATE_VARS):
        if key in normalization:
            mean, std = normalization[key]['mean'], normalization[key]['std']
            states_norm[:, i] = (states[:, i] - mean) / std if std > 0 else 0.0
        else:
            states_norm[:, i] = states[:, i]

    targets_norm = np.zeros_like(targets)
    for i, key in enumerate(CONTROL_VARS):
        if key in normalization:
            mean, std = normalization[key]['mean'], normalization[key]['std']
            targets_norm[:, i] = (targets[:, i] - mean) / std if std > 0 else 0.0
        else:
            targets_norm[:, i] = targets[:, i]

    return states_norm, targets_norm


def train_model(
    states: np.ndarray,
    targets: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Train a feedforward neural network using PyTorch.

    Architecture: state_dim -> hidden -> hidden -> control_dim (ReLU activations)

    Returns dict with 'layers' (list of weight/bias arrays) and training info.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = states.shape[1]
    control_dim = targets.shape[1]

    # Train/val split
    n = len(states)
    n_val = max(1, int(n * val_ratio))
    indices = np.random.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train = torch.FloatTensor(states[train_idx])
    y_train = torch.FloatTensor(targets[train_idx])
    X_val = torch.FloatTensor(states[val_idx])
    y_val = torch.FloatTensor(targets[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Build model
    model = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, control_dim),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Training ANN: {state_dim}->{hidden_dim}->{hidden_dim}->{control_dim}")
    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Epochs: {epochs}")

    best_val_loss = float('inf')
    best_state_dict = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>4d}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Restore best model
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    # Extract weights as numpy arrays
    model.eval()
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layers.append({'weight': param.detach().numpy().tolist()})
        elif 'bias' in name:
            layers[-1]['bias'] = param.detach().numpy().tolist()

    logger.info(f"Training complete. Best val_loss={best_val_loss:.6f}")

    return {
        'layers': layers,
        'training_info': {
            'epochs': epochs,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(history['train_loss'][-1]),
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
        },
    }


def export_model(
    model_data: dict,
    normalization: Optional[Dict],
    output_dir: str,
    suffix: Optional[str] = None,
) -> str:
    """Export trained model as JSON for numpy-only inference."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_parts = ['ann_model']
    if suffix:
        name_parts.append(suffix)
    name_parts.append(timestamp)
    output_path = os.path.join(output_dir, f'{"_".join(name_parts)}.json')

    export = {
        'layers': model_data['layers'],
        'normalization': normalization,
        'state_vars': STATE_VARS,
        'control_vars': CONTROL_VARS,
        'architecture': f"{len(STATE_VARS)}-{model_data['training_info']['hidden_dim']}-"
                        f"{model_data['training_info']['hidden_dim']}-{len(CONTROL_VARS)}",
        'training_info': model_data['training_info'],
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)

    logger.info(f"Model exported to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Train ANN controller from open-loop experiment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ann.py --sysid-json runs/run_20260303_151634/results/sysid_output/sysid_matrices_20260323_182944.json --run 20260303_151634
  python train_ann.py --sysid-json <path> --runs 20260216_070737 20260303_151634 --epochs 300 --lr 0.0005
        """
    )
    parser.add_argument('--sysid-json', required=True,
                        help='Path to sysid_matrices_*.json (for LQR oracle + normalization)')
    parser.add_argument('--run', default=None,
                        help='Single run ID for training data')
    parser.add_argument('--runs', nargs='+', default=None,
                        help='Multiple run IDs to merge for training data')
    parser.add_argument('--runs-dir', default=RUNS_DIR,
                        help='Runs directory')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for model JSON (default: alongside sysid JSON)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--filter-exhausted', action='store_true', default=True)
    parser.add_argument('--no-filter-exhausted', dest='filter_exhausted', action='store_false')
    parser.add_argument('--exclude-settle', action='store_true', default=True)
    parser.add_argument('--no-exclude-settle', dest='exclude_settle', action='store_false')

    # LQR Q/R tuning (optional — affects what the ANN learns)
    parser.add_argument('--q-diag', nargs='+', type=float, default=None,
                        help='Q diagonal weights (4 or 5 values)')
    parser.add_argument('--r-diag', nargs=2, type=float, default=None,
                        help='R diagonal weights [batch, poll]')
    parser.add_argument('--output-suffix', default=None,
                        help='Suffix for output filename, e.g. "Q1" → ann_model_Q1_<ts>.json')

    args = parser.parse_args()

    run_ids = args.runs if args.runs else ([args.run] if args.run else None)
    if not run_ids:
        parser.error("Specify --run <id> or --runs <id1> <id2> ...")

    # 1. Load sysid JSON and create LQR oracle
    logger.info(f"Loading sysid from {args.sysid_json}")
    with open(args.sysid_json) as f:
        sysid_data = json.load(f)
    normalization = sysid_data.get('normalization', None)

    lqr_kwargs = {}
    if args.q_diag:
        lqr_kwargs['Q'] = np.diag(args.q_diag)
    if args.r_diag:
        lqr_kwargs['R'] = np.diag(args.r_diag)

    lqr = LQRController.from_sysid_json(args.sysid_json, **lqr_kwargs)

    # 2. Load and prepare training data
    df = load_and_prepare_data(
        run_ids, args.runs_dir,
        filter_exhausted=args.filter_exhausted,
        exclude_settle=args.exclude_settle,
    )

    # 3. Generate LQR targets
    states = df[STATE_VARS].values.astype(np.float64)
    targets = generate_lqr_targets(df, lqr)

    # 4. Normalize
    if normalization:
        states_norm, targets_norm = normalize_data(states, targets, normalization)
        logger.info("Data normalized using sysid scales")
    else:
        states_norm, targets_norm = states, targets
        logger.warning("No normalization scales — training on raw data")

    # 5. Train
    model_data = train_model(
        states_norm, targets_norm,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 6. Export
    output_dir = args.output_dir or os.path.dirname(args.sysid_json)
    model_path = export_model(model_data, normalization, output_dir, suffix=args.output_suffix)

    print(f"\nTraining complete!")
    print(f"  Best val loss: {model_data['training_info']['best_val_loss']:.6f}")
    print(f"  Model saved: {model_path}")


if __name__ == '__main__':
    main()
