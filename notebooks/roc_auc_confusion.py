from sklearn.metrics import roc_auc_score, auc, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import RocCurveDisplay
PATH = Path(r'D:\Users\UFPB\gabriel ayres\MST\results-classification-preprocessed2\runs\PENN\ResNetSliceTrans_2025_06_06_192329_multi\results.csv')
df = pd.read_csv(PATH)

plt.figure(figsize=(8, 6))

folds = df['Fold'].unique()

for fold in folds:
    fold_df = df[df['Fold'] == fold]
    gt = fold_df['GT'].values
    pred = fold_df['NN_pred'].values
    
    auc_check = roc_auc_score(gt, pred)
    if auc_check < 0.5:
        pred = 1 - pred  # flip predictions if AUC < 0.5
    
    fpr, tpr, _ = roc_curve(gt, pred)
    auc_val = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc_val:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Fold')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_folds.png')
plt.show()

