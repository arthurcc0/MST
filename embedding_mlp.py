import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, auc
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def load_embeddings(embeddings_dir, labels_path, to_exclude, one_side=False):
    embs_df = pd.DataFrame()
    labels_df = pd.read_csv(labels_path)

    for emb_path in os.listdir(embeddings_dir):
        emb_npz = np.load(f'{embeddings_dir}/{emb_path}')
        emb_df = pd.DataFrame(emb_npz['pred.npy'][0]).T
        emb_df.insert(loc=0, column='uid', value=emb_path.split('_pred.npz')[0])
        embs_df = pd.concat([embs_df, emb_df], ignore_index=True)
    
    embs_df = embs_df[~embs_df['uid'].isin(to_exclude)]
    labels_df = labels_df[~labels_df['UID'].isin(to_exclude)]

    embs_df['base_id'] = embs_df['uid'].str.replace(r'_(left|right)$', '', regex=True)
    labels_df['base_id'] = labels_df['UID'].str.replace(r'_(left|right)$', '', regex=True)

    if one_side:
        merged = pd.merge(embs_df, labels_df, left_on='uid', right_on='UID')
        merged['base_id'] = merged['uid'].str.replace(r'_(left|right)$', '', regex=True)

        benign = merged[merged['GT'] == 0].copy()
        malignant = merged[merged['GT'] == 1].copy()

        benign_one_side = (
            benign.groupby('base_id')
            .apply(lambda x: x.sample(n=1, random_state=SEED))
            .reset_index(drop=True)
        )

        merged_filtered = pd.concat([malignant, benign_one_side], ignore_index=True)
        embs_df = merged_filtered[embs_df.columns]
        labels_df = merged_filtered[labels_df.columns]

    return embs_df, labels_df

SEED = 42
USE_SMOTE = False
ONE_SIDE = True

LR = 1e-6
PATIENCE = 5
BATCH_SIZE = 32
LOSS_FUNC = 'focal'


if __name__ == '__main__':
    embeddings_dir = './embeddings'
    labels_path = './results.csv'

    to_exclude_path = './to_exclude.txt'
    with open(to_exclude_path, 'r') as f:
        exclude_list = f.readlines()
    
    to_exclude = [e.split('_')[0] for e in exclude_list]

    embs_df, labels_df = load_embeddings(embeddings_dir, labels_path, to_exclude, one_side=ONE_SIDE)

    y = labels_df['GT']

    uids = embs_df['uid']
    X = embs_df.drop(columns=['uid', 'base_id'])

    print(y.value_counts())

    num_classes = y.value_counts().shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs = []
    aucs = [[] for i in range(num_classes)]
    tprs = [[] for i in range(num_classes)]
    cms = np.zeros((num_classes, num_classes), dtype=int)

    fig, ax = plt.subplots()
    target_names = ['Benign', 'Malignant']

    mean_fpr = np.linspace(0, 1, 100)

    pooled_preds = []
    pooled_probs = []
    pooled_labels = []

    num_splits = 5
    #cv = StratifiedKFold(num_splits, shuffle=True, random_state=SEED)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    train_losses_per_fold = []
    val_losses_per_fold = []

    if USE_SMOTE:
        results_dir = f'./results_{LOSS_FUNC}_bs{BATCH_SIZE}_lr{str(LR)}_es{PATIENCE}_SMOTE_{SEED}'
    else:
        results_dir = f'./results_{LOSS_FUNC}_bs{BATCH_SIZE}_lr{str(LR)}_es{PATIENCE}_{SEED}'
    os.makedirs(results_dir, exist_ok=True)

    #for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=labels_df['base_id'])):
        print(f'Fold {i+1}: training n = {len(train_idx)}, val n = {len(val_idx)}')

        X_train, y_train = X.iloc[train_idx].values, y.iloc[train_idx].values
        X_val, y_val = X.iloc[val_idx].values, y.iloc[val_idx].values

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

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = MLP().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        
        if LOSS_FUNC == 'crossentropy':
            criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
        elif LOSS_FUNC == 'focal':
            criterion = FocalLoss(gamma=2, alpha=weights_tensor, reduction='mean')

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

        train_losses = []
        val_losses = []

        for epoch in range(500):
            model.train()
            train_loss = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0

            batch_probs = []
            batch_preds = []
            batch_labels = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    batch_probs.extend(probs[:, 1].cpu().numpy().tolist()) 
                    batch_preds.extend(preds.cpu().numpy().tolist())
                    batch_labels.extend(yb.cpu().numpy().tolist())

                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f'epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            early_stopping.step(val_loss, model)

            if early_stopping.early_stop:
                print('early stop')
                break

        train_losses_per_fold.append(train_losses)
        val_losses_per_fold.append(val_losses)
        
        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and validation loss - Fold {i+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{results_dir}/train_val_loss_fold_{i+1}.png', dpi=400)
        plt.close()

        cm = confusion_matrix(batch_labels, batch_preds)
        cms = cms + cm

        auc_check = roc_auc_score(batch_labels, batch_probs)
        if auc_check < 0.5:
            batch_probs = [1 - p for p in batch_probs]
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

        pooled_preds.append(batch_preds)
        pooled_probs.append(batch_probs)
        pooled_labels.append(batch_labels)

        print(f'Fold {i+1} AUC = {auc_check:.4f}')

    plt.figure(figsize=(12,8))

    palette = sns.color_palette("tab10", n_colors=len(train_losses_per_fold))

    for i in range(len(train_losses_per_fold)):
        epochs_train = range(1, len(train_losses_per_fold[i]) + 1)
        epochs_val = range(1, len(val_losses_per_fold[i]) + 1)

        base_color = palette[i]
        train_color = mcolors.to_rgba(base_color, alpha=1.0)  # darker
        val_color = mcolors.to_rgba(base_color, alpha=0.5)    # lighter

        plt.plot(epochs_train, train_losses_per_fold[i], linestyle='-', label=f'Fold {i+1} train loss', color=train_color)
        plt.plot(epochs_val, val_losses_per_fold[i], linestyle='--', label=f'Fold {i+1} val loss', color=val_color)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss for all folds')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True)
    plt.tight_layout()

    combined_loss_path = f'{results_dir}/train_val_loss.png'
    plt.savefig(combined_loss_path, dpi=400)
    plt.close()

    pooled_preds_flat = [x for xs in pooled_preds for x in xs]
    pooled_probs_flat = [x for xs in pooled_probs for x in xs]
    pooled_labels_flat = [x for xs in pooled_labels for x in xs]
    pooled_df = pd.DataFrame(pooled_probs_flat)
    pooled_onehot = label_binarize(pooled_labels_flat, classes=range(num_classes))

    pooled_df['label'] = pooled_labels_flat

    # ROC
    mean_tpr = np.mean(tprs[1], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[1])
    print('Binary AUC: {} +- {}'.format(mean_auc, std_auc))
    auc_check = roc_auc_score(pooled_labels_flat, pooled_df[0])
    if auc_check < 0.5:
        pooled_df[0] = 1 - pooled_df[0]
    viz = RocCurveDisplay.from_predictions(
        pooled_labels_flat,
        pooled_df[0],
        color='b',
        name=f"Pooled ROC",
        lw=2,
        alpha=0.8,
        ax=ax
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    std_tpr = np.std(tprs[1], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Cross-validated ROC - MLP \n(Benign vs. Malignant)",
    )
    ax.legend(loc="lower right")

    fig.savefig(f'./{results_dir}/roc_mlp_{SEED}.png', dpi=400)
    plt.close(fig)

    # CM
    fig, ax = plt.subplots()
    cms_norm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    avg_score = np.mean(accs)
    std_score = np.std(accs)
    combined = np.array([[f"{cms_val} ({cms_norm_val:.2f})" for cms_val, cms_norm_val in zip(cms_row, cms_norm_row)] for cms_row, cms_norm_row in zip(cms, cms_norm)])

    sns.heatmap(cms, annot=combined, cbar=False, cmap='Reds', xticklabels=target_names, yticklabels=target_names, fmt='')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Pooled Confusion Matrix - MLP')

    plt.savefig(f'./{results_dir}/norm_cm_mlp_{SEED}.png')
