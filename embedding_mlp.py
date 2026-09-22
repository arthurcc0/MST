import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mst.data.datasets.dataset_3d_penn import PENN_Dataset3D

class MLP(nn.Module):
    def __init__(self, input_dim=384, output_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=7, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter (default=2)
        alpha: weights for classes, e.g. a list or tensor of shape [num_classes]
        reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (not probabilities), shape (batch_size, num_classes)
        targets: ground truth labels, shape (batch_size), long/int dtype
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def _embedding_vec_from_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as z:
        if 'pred' in z.files:
            arr = np.asarray(z['pred'], dtype=np.float32)
        elif 'pred.npy' in z.files:
            arr = np.asarray(z['pred.npy'], dtype=np.float32)
        else:
            raise KeyError(f"{npz_path}: expected key 'pred', got {list(z.files)}")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr[:1] if arr.shape[0] > 1 else arr
    raise ValueError(f"Unexpected embedding shape {arr.shape} in {npz_path}")


def _resolve_predict_paths(
    predict_run: str | Path | None,
    embeddings_dir: str | Path | None,
    labels_path: str | Path | None,
    results_subdir: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    if predict_run is not None:
        run = Path(predict_run)
        if not run.is_absolute():
            run = Path(output_dir) / results_subdir / 'runs' / 'PENN' / run
        emb = run / 'embeddings_fused'
        labels = run / 'results.csv'
        if not labels.is_file():
            index_csv = emb / 'index.csv'
            if index_csv.is_file():
                labels = index_csv
        return emb, labels
    if embeddings_dir is None or labels_path is None:
        raise ValueError("Provide --predict_run or both --embeddings_dir and --labels_path.")
    return Path(embeddings_dir), Path(labels_path)


def load_embedding_npz_dir(embeddings_dir: Path, to_exclude: list[str]) -> pd.DataFrame:
    """Load all fused embedding vectors from embeddings_fused/."""
    rows = []
    embeddings_dir = Path(embeddings_dir)
    for emb_path in os.listdir(embeddings_dir):
        if not emb_path.endswith('.npz'):
            continue
        uid = emb_path.replace('_pred.npz', '')
        if uid in to_exclude:
            continue
        vec = _embedding_vec_from_npz(embeddings_dir / emb_path)
        row = pd.DataFrame(vec)
        row.insert(0, 'uid', uid)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No .npz embeddings found under {embeddings_dir}")
    embs_df = pd.concat(rows, ignore_index=True)
    embs_df['base_id'] = embs_df['uid'].str.replace(r'_(left|right)$', '', regex=True)
    return embs_df


def _apply_one_side(embs_df: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.merge(embs_df, labels_df, left_on='uid', right_on='UID', how='inner')
    merged['base_id'] = merged['uid'].str.replace(r'_(left|right)$', '', regex=True)
    benign = merged[merged['GT'] == 0].copy()
    malignant = merged[merged['GT'] == 1].copy()
    benign_one_side = (
        benign.groupby('base_id')
        .apply(lambda x: x.sample(n=1, random_state=SEED))
        .reset_index(drop=True)
    )
    merged_filtered = pd.concat([malignant, benign_one_side], ignore_index=True)
    return (
        merged_filtered[embs_df.columns].reset_index(drop=True),
        merged_filtered[labels_df.columns].reset_index(drop=True),
    )


def load_embeddings(embeddings_dir, labels_path, to_exclude, one_side=False):
    embs_df = load_embedding_npz_dir(Path(embeddings_dir), to_exclude)
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[~labels_df['UID'].isin(to_exclude)]
    labels_df['base_id'] = labels_df['UID'].str.replace(r'_(left|right)$', '', regex=True)

    if one_side:
        return _apply_one_side(embs_df, labels_df)

    merged = pd.merge(embs_df, labels_df, left_on='uid', right_on='UID', how='inner')
    return (
        merged[embs_df.columns].reset_index(drop=True),
        merged[labels_df.columns].reset_index(drop=True),
    )


def load_embeddings_for_penn_split(
    embeddings_dir: Path,
    penn_split_csv: str | Path,
    fold: int,
    to_exclude: list[str],
    one_side: bool = False,
    high_risk_policy: str = "exclude",
    dcis_policy: str = "malignant",
) -> pd.DataFrame:
    """Join embeddings with new_penn_datasplit (UID, Split, Malignant) for one MST fold."""
    embs_df = load_embedding_npz_dir(embeddings_dir, to_exclude)
    split_path = PENN_Dataset3D.resolve_split_csv_path(penn_split_csv)
    df_split = PENN_Dataset3D.load_split(
        split_path,
        fold=fold,
        split=None,
        high_risk_policy=high_risk_policy,
        dcis_policy=dcis_policy,
    )
    df_split = df_split[['UID', 'Split', 'Malignant']].copy()
    df_split['GT'] = df_split['Malignant'].astype(int)

    merged = pd.merge(embs_df, df_split, left_on='uid', right_on='UID', how='inner')
    merged['base_id'] = merged['uid'].astype(str).str.replace(r'_(left|right)$', '', regex=True)
    if one_side:
        benign = merged[merged['GT'] == 0].copy()
        malignant = merged[merged['GT'] == 1].copy()
        benign_one_side = (
            benign.groupby('base_id')
            .apply(lambda x: x.sample(n=1, random_state=SEED))
            .reset_index(drop=True)
        )
        merged = pd.concat([malignant, benign_one_side], ignore_index=True)

    for split_name in ('train', 'val', 'test'):
        n = int((merged['Split'] == split_name).sum())
        print(f"Penn split fold {fold} / {split_name}: {n} cases with embeddings")
    missing = df_split[~df_split['UID'].isin(merged['UID'])]
    if len(missing):
        print(
            f"Warning: {len(missing)} rows in {split_path.name} (fold {fold}) "
            f"have no embedding under {embeddings_dir}"
        )
    return merged

SEED = 42
USE_SMOTE = True
ONE_SIDE = True

LR = 1e-6
PATIENCE = 5
BATCH_SIZE = 32
LOSS_FUNC = 'focal'


def _feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop = {'uid', 'base_id', 'UID', 'Split', 'Malignant', 'GT', 'results_csv'}
    return df.drop(columns=[c for c in df.columns if c in drop], errors='ignore')


def _train_mlp_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    device: torch.device,
    results_dir: Path,
    tag: str = '',
) -> tuple[MLP, StandardScaler, list[float], list[float]]:
    """Train one MLP with early stopping on val; return model, scaler, loss curves."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    if USE_SMOTE:
        X_train, y_train = BorderlineSMOTE(random_state=SEED).fit_resample(X_train, y_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False
    )

    model = MLP(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    if LOSS_FUNC == 'crossentropy':
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    else:
        criterion = FocalLoss(gamma=2, alpha=weights_tensor, reduction='mean')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    train_losses, val_losses = [], []
    for epoch in range(500):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        batch_probs, batch_preds, batch_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                batch_probs.extend(probs[:, 1].cpu().numpy().tolist())
                batch_preds.extend(preds.cpu().numpy().tolist())
                batch_labels.extend(yb.cpu().numpy().tolist())

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f'{tag}epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        early_stopping.step(val_loss, model)
        if early_stopping.early_stop:
            print(f'{tag}early stop')
            break

    suffix = f'_{tag.strip()}' if tag else ''
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and validation loss{suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / f'train_val_loss{suffix}.png', dpi=400)
    plt.close()

    return model, scaler, train_losses, val_losses


def _eval_mlp(
    model: MLP,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    X_s = scaler.transform(X)
    x_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return y, probs[:, 1].cpu().numpy(), preds.cpu().numpy()


def run_penn_split_eval(
    merged: pd.DataFrame,
    fold: int,
    results_dir: Path,
    device: torch.device,
) -> None:
    """Train MLP on train embeddings, early-stop on val, evaluate on held-out test."""
    train_df = merged[merged['Split'] == 'train']
    val_df = merged[merged['Split'] == 'val']
    test_df = merged[merged['Split'] == 'test']
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "penn_split mode requires train and test embeddings. "
            "Run main_predict.py with --save_embeddings for --split train, val, and test."
        )
    if len(val_df) == 0:
        raise ValueError("penn_split mode requires val embeddings for early stopping.")

    feat = _feature_cols(merged).columns.tolist()
    y_train = train_df['GT'].to_numpy()
    y_val = val_df['GT'].to_numpy()
    y_test = test_df['GT'].to_numpy()

    print(
        f"Penn-split MLP (MST fold {fold}): train n={len(y_train)} (pos={int(y_train.sum())}), "
        f"val n={len(y_val)} (pos={int(y_val.sum())}), test n={len(y_test)} (pos={int(y_test.sum())})"
    )

    model, scaler, _, _ = _train_mlp_once(
        train_df[feat].to_numpy(),
        y_train,
        val_df[feat].to_numpy(),
        y_val,
        len(feat),
        device,
        results_dir,
        tag='',
    )
    _, test_probs, test_preds = _eval_mlp(
        model, scaler, test_df[feat].to_numpy(), y_test, device
    )
    test_auc = roc_auc_score(y_test, test_probs)
    if test_auc < 0.5:
        test_probs = 1.0 - test_probs
        test_auc = roc_auc_score(y_test, test_probs)

    cm = confusion_matrix(y_test, test_preds)
    print(f"Test AUC (penn_split, MST fold {fold}): {test_auc:.4f}")
    print(f"Test confusion matrix:\n{cm}")

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        test_probs,
        name=f"Test ROC (AUC={test_auc:.2f})",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.75)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"MLP on MST embeddings — penn_split fold {fold}")
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(results_dir / f'roc_mlp_test_fold{fold}_{SEED}.png', dpi=400)
    plt.close(fig)

    out = test_df[['uid', 'UID', 'Split', 'GT']].copy()
    out['NN_pred'] = test_probs
    out['NN'] = test_preds
    out.to_csv(results_dir / f'test_predictions_fold{fold}.csv', index=False)


def run_internal_cv(
    embs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    results_dir: Path,
    device: torch.device,
) -> None:
    """Legacy: random 5-fold CV on loaded embeddings (not new_penn_datasplit)."""
    print(
        "WARNING: internal_cv re-splits embeddings with StratifiedGroupKFold — "
        "NOT new_penn_datasplit.csv. With test-only embeddings this retrains on "
        "portions of the MST test set. Prefer --eval_mode penn_split."
    )
    y = labels_df['GT']
    X = embs_df.drop(columns=['uid', 'base_id'])
    print(y.value_counts())

    num_classes = int(y.nunique())
    tprs: list[list[np.ndarray]] = [[] for _ in range(num_classes)]
    aucs: list[list[float]] = [[] for _ in range(num_classes)]
    cms = np.zeros((num_classes, num_classes), dtype=int)
    fig, ax = plt.subplots()
    target_names = ['Benign', 'Malignant']
    mean_fpr = np.linspace(0, 1, 100)
    pooled_preds, pooled_probs, pooled_labels = [], [], []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_losses_per_fold, val_losses_per_fold = [], []

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=labels_df['base_id'])):
        print(f'Internal CV fold {i+1}: train n = {len(train_idx)}, val n = {len(val_idx)}')
        model, scaler, train_losses, val_losses = _train_mlp_once(
            X.iloc[train_idx].values,
            y.iloc[train_idx].values,
            X.iloc[val_idx].values,
            y.iloc[val_idx].values,
            X.shape[1],
            device,
            results_dir,
            tag=f'fold{i+1}_',
        )
        train_losses_per_fold.append(train_losses)
        val_losses_per_fold.append(val_losses)
        batch_labels, batch_probs, batch_preds = _eval_mlp(
            model,
            scaler,
            X.iloc[val_idx].values,
            y.iloc[val_idx].values,
            device,
        )
        cm = confusion_matrix(batch_labels, batch_preds)
        cms += cm
        auc_check = roc_auc_score(batch_labels, batch_probs)
        if auc_check < 0.5:
            batch_probs = 1.0 - batch_probs
            auc_check = roc_auc_score(batch_labels, batch_probs)
        viz = RocCurveDisplay.from_predictions(
            batch_labels,
            batch_probs,
            name=f"ROC - Fold {i+1}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs[1].append(interp_tpr)
        aucs[1].append(viz.roc_auc)
        pooled_preds.append(batch_preds.tolist())
        pooled_probs.append(batch_probs.tolist())
        pooled_labels.append(batch_labels.tolist())
        print(f'Fold {i+1} AUC = {auc_check:.4f}')

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("tab10", n_colors=len(train_losses_per_fold))
    for i in range(len(train_losses_per_fold)):
        epochs_train = range(1, len(train_losses_per_fold[i]) + 1)
        epochs_val = range(1, len(val_losses_per_fold[i]) + 1)
        base_color = palette[i]
        plt.plot(
            epochs_train, train_losses_per_fold[i], linestyle='-',
            label=f'Fold {i+1} train loss', color=mcolors.to_rgba(base_color, alpha=1.0),
        )
        plt.plot(
            epochs_val, val_losses_per_fold[i], linestyle='--',
            label=f'Fold {i+1} val loss', color=mcolors.to_rgba(base_color, alpha=0.5),
        )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss for all folds')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / 'train_val_loss.png', dpi=400)
    plt.close()

    pooled_probs_flat = [x for xs in pooled_probs for x in xs]
    pooled_labels_flat = [x for xs in pooled_labels for x in xs]
    mean_tpr = np.mean(tprs[1], axis=0)
    mean_tpr[-1] = 1.0
    std_auc = np.std(aucs[1])
    pooled_auc = roc_auc_score(pooled_labels_flat, pooled_probs_flat)
    if pooled_auc < 0.5:
        pooled_probs_flat = [1 - p for p in pooled_probs_flat]
        pooled_auc = roc_auc_score(pooled_labels_flat, pooled_probs_flat)
    print(f'Binary AUC (internal CV pooled): {pooled_auc:.4f} +- {std_auc:.4f}')
    RocCurveDisplay.from_predictions(
        pooled_labels_flat, pooled_probs_flat, color='b', name='Pooled ROC', lw=2, alpha=0.8, ax=ax
    )
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.75)
    std_tpr = np.std(tprs[1], axis=0)
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.',
    )
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Cross-validated ROC - MLP')
    ax.legend(loc='lower right')
    fig.savefig(results_dir / f'roc_mlp_{SEED}.png', dpi=400)
    plt.close(fig)

    cms_norm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    combined = np.array([
        [f"{a} ({b:.2f})" for a, b in zip(cms_row, cms_norm_row)]
        for cms_row, cms_norm_row in zip(cms, cms_norm)
    ])
    fig, ax = plt.subplots()
    sns.heatmap(
        cms, annot=combined, cbar=False, cmap='Reds',
        xticklabels=target_names, yticklabels=target_names, fmt='',
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Pooled Confusion Matrix - MLP')
    plt.savefig(results_dir / f'norm_cm_mlp_{SEED}.png')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train MLP on fused embeddings from main_predict.py (--save_embeddings).'
    )
    parser.add_argument('--predict_run', type=str, default=None)
    parser.add_argument('--embeddings_dir', type=str, default=None)
    parser.add_argument('--labels_path', type=str, default=None)
    parser.add_argument('--results_subdir', type=str, default='results-duke-penn-training')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument(
        '--eval_mode',
        choices=['penn_split', 'internal_cv'],
        default='penn_split',
        help=(
            "penn_split: train/val/test from new_penn_datasplit (recommended). "
            "internal_cv: legacy random 5-fold on loaded embeddings only."
        ),
    )
    parser.add_argument(
        '--penn_split_csv',
        type=str,
        default='new_penn_datasplit.csv',
        help='Split CSV for penn_split mode (UID, Fold, Split, Malignant).',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='MST CV fold index (must match the fine-tuned checkpoint used for embeddings).',
    )
    parser.add_argument(
        '--high-risk-policy',
        dest='high_risk_policy',
        type=str,
        default='exclude',
        choices=('malignant', 'benign', 'exclude'),
        help="Treat datasplit 'high risk' as malignant, benign, or drop. Default: exclude.",
    )
    parser.add_argument(
        '--dcis-policy',
        dest='dcis_policy',
        type=str,
        default='malignant',
        choices=('malignant', 'benign', 'exclude'),
        help="Treat datasplit 'dcis' as malignant, benign, or drop. Default: malignant.",
    )
    cli = parser.parse_args()

    embeddings_dir, labels_path = _resolve_predict_paths(
        cli.predict_run,
        cli.embeddings_dir,
        cli.labels_path,
        cli.results_subdir,
        cli.output_dir,
    )
    print(f"Embeddings: {embeddings_dir}")
    print(f"Eval mode:  {cli.eval_mode}")

    mst_model_name = Path(embeddings_dir).parent.name
    run_suffix = f'{LOSS_FUNC}_bs{BATCH_SIZE}_lr{LR}_es{PATIENCE}'
    if USE_SMOTE:
        run_suffix += '_SMOTE'
    run_suffix += f'_{cli.eval_mode}_fold{cli.fold}_{SEED}'
    results_dir = Path(f'./results_mlp/{mst_model_name}/{run_suffix}')
    results_dir.mkdir(parents=True, exist_ok=True)

    to_exclude_path = Path('./to_exclude.txt')
    if to_exclude_path.is_file():
        with open(to_exclude_path, encoding='utf-8') as f:
            to_exclude = [e.split('_')[0].strip() for e in f if e.strip()]
    else:
        to_exclude = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cli.eval_mode == 'penn_split':
        merged = load_embeddings_for_penn_split(
            embeddings_dir,
            cli.penn_split_csv,
            cli.fold,
            to_exclude,
            one_side=ONE_SIDE,
            high_risk_policy=cli.high_risk_policy,
            dcis_policy=cli.dcis_policy,
        )
        run_penn_split_eval(merged, cli.fold, results_dir, device)
    else:
        embs_df, labels_df = load_embeddings(
            embeddings_dir, labels_path, to_exclude, one_side=ONE_SIDE
        )
        run_internal_cv(embs_df, labels_df, results_dir, device)


if __name__ == '__main__':
    main()
