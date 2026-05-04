import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, auc
from sklearn.utils.class_weight import compute_class_weight
from kan import *
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_embeddings(embeddings_dir):
    embs_df = pd.DataFrame()
    for emb_path in os.listdir(embeddings_dir):
        emb_npz = np.load(f'{embeddings_dir}/{emb_path}')
        emb_df = pd.DataFrame(emb_npz['pred.npy'][0]).T
        emb_df.insert(loc=0, column='uid', value=emb_path.split('_pred.npz')[0])
        embs_df = pd.concat([embs_df, emb_df], ignore_index=True)
    
    return embs_df

class EmbeddingClassifierSmall(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )    #nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.net(x)

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=384, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # returns logits (for BCEWithLogitsLoss)

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened size after conv layers
        # Input is (batch, 1, input_dim)
        # After first pool: (batch, 16, input_dim / 2)
        # After second pool: (batch, 32, input_dim / 4)
        flattened_size = 32 * (input_dim // 4)

        self.fc_net = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Reshape input from (batch_size, embedding_dim) to (batch_size, 1, embedding_dim)
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_net(x)
        return x.squeeze(1) # returns logits (for BCEWithLogitsLoss)


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

SEED = 42
USE_SMOTE = True

if __name__ == '__main__':
    embeddings_dir = Path(r'D:\Users\UFPB\gabriel ayres\MST\results-fulldataset\runs\PENN\ResNetSliceTrans_2025_06_06_192329_multi\embeddings-correct-fulldata')
    embs_df = load_embeddings(embeddings_dir)

    to_exclude_path = Path(r'D:\Users\UFPB\gabriel ayres\MST\scripts\preprocessing\penn-preprocessed\to_exclude.txt') 
    with open(to_exclude_path, 'r') as f:
        exclude_list = f.readlines()
    
    to_exclude = [e.split(' -')[0] for e in exclude_list]
    embs_df = embs_df[~embs_df['uid'].isin(to_exclude)]

    labels_path = Path(r'D:\Users\UFPB\gabriel ayres\MST\results-fulldataset\runs\PENN\ResNetSliceTrans_2025_06_06_192329_multi\results.csv')
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[~labels_df['UID'].isin(to_exclude)]
    y = labels_df['GT']

    uids = embs_df['uid']
    X = embs_df.drop(columns=['uid'])

    num_classes = y.value_counts().shape[0]

    device = torch.device("cuda")

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
    cv = StratifiedKFold(num_splits, shuffle=True, random_state=SEED)

    train_losses_per_fold = []
    val_losses_per_fold = []

    loss_export_dir = './loss_plots_fulldataset_kan'
    os.makedirs(loss_export_dir, exist_ok=True)

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')

        best_val_loss = float('inf')
        patience = 5
        epochs_no_improve = 0
        early_stop=False

        X_train, y_train = X.iloc[train_idx].values, y.iloc[train_idx].values
        X_test, y_test = X.iloc[test_idx].values, y.iloc[test_idx].values

        if USE_SMOTE:
            X_train, y_train = BorderlineSMOTE(random_state=SEED).fit_resample(X_train, y_train)
            
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = CNNClassifier(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=5)

        train_losses = []
        val_losses = []

        for epoch in range(500):
            model.train()
            train_loss = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb.to(device).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0

            batch_probs = []
            batch_preds = []
            batch_labels = []

            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb)
                    loss = criterion(logits, yb.to(device).float())
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long()

                    # Get probabilities for the positive class (class 1)
                    batch_probs.extend(probs.cpu().numpy().tolist())
                    batch_preds.extend(preds.cpu().numpy().tolist())
                    batch_labels.extend(yb.cpu().numpy().tolist())
                    val_loss += loss.item()
            
            print(f'epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            early_stopping.step(val_loss)

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
        plt.savefig(f'{loss_export_dir}/train_val_loss_fold_{i+1}.png', dpi=400)
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

        #plt.plot(len(train_losses)-1, train_losses[-1], 'o', color=train_color)
        #plt.plot(len(val_losses)-1, val_losses[-1], 's', color=val_color)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss for all folds')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True)
    plt.tight_layout()

    combined_loss_path = f'{loss_export_dir}/train_val_loss.png'
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
        title=f"Cross-validated ROC - KAN \n(Benign vs. Malignant)",
    )
    ax.legend(loc="lower right")

    fig.savefig(f'{loss_export_dir}/roc_kan_{SEED}.png', dpi=400)
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
    plt.title(f'Pooled Confusion Matrix - KAN')

    plt.savefig(f'{loss_export_dir}/norm_cm_kan_{SEED}.png')
