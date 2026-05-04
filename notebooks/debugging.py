import numpy as np
from pathlib import Path 
import pandas as pd 
import ast
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import xgboost as xgb
from imblearn.over_sampling import BorderlineSMOTE

import matplotlib
import matplotlib.pyplot as plt
np.random.seed(42)

EMBED_DIR = Path(r'D:\Users\UFPB\gabriel ayres\MST\results-classification-preprocessed2\runs\PENN\ResNetSliceTrans_2025_06_06_192329_multi\embeddings-correct')
split_file = Path(r'D:\Users\UFPB\gabriel ayres\MST\notebooks\penn-preprocessed_datasplit.csv')
to_exclude_path = Path(r"D:\Users\UFPB\gabriel ayres\MST\scripts\preprocessing\penn-preprocessed\to_exclude.txt") 

df_splits = pd.read_csv(split_file)
embeds = [np.load(embed)['pred'] for embed in EMBED_DIR.glob('*.npz')]
print(embeds[0].shape)
embeds = np.stack(embeds)
labels = df_splits['Malignant'].values

# Flatten the entire dataset
embeds_flat = embeds.reshape(embeds.shape[0], -1)
print(f"Flattened embeddings shape: {embeds_flat.shape}")



def cm2acc(cm):
    # [[TN, FP], [FN, TP]] 
    tn, fp, fn, tp = cm.ravel()
    return (tn+tp)/(tn+tp+fn+fp)

def safe_div(x,y):
    if y == 0:
        return float('nan') 
    return x / y

def cm2x(cm):
    tn, fp, fn, tp = cm.ravel()
    pp = tp + fp  # predicted positive 
    pn = fn + tn  # predicted negative
    p = tp + fn   # actual positive
    n = fp + tn   # actual negative  

    ppv = safe_div(tp,pp)  # positive predictive value 
    npv = safe_div(tn,pn)  # negative predictive value 
    tpr = safe_div(tp,p)   # true positive rate (sensitivity, recall)
    tnr = safe_div(tn,n)   # true negative rate (specificity)
    # Note: other values are 1-x eg. fdr=1-ppv, for=1-npv, ....
    return ppv, npv, tpr, tnr

def auc_bootstrapping(y_true, y_score, bootstrapping=1000, drop_intermediate=True):
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 1000) # Use 1000 points for a smooth average

    for _ in range(bootstrapping):
        # Bootstrap sample
        y_true_set, y_score_set = resample(y_true, y_score, replace=True, n_samples=len(y_true))

        # Ensure both classes are present
        if len(np.unique(y_true_set)) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)
        optimal_idx = np.argmax(tpr - fpr) # Youden's J statistic

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
        thrs.append(thresholds[optimal_idx])

    return tprs, aucs, thrs, mean_fpr



def plot_roc_curve(y_true, y_score, axis, bootstrapping=1000, drop_intermediate=False, fontdict={}, name='ROC', color='b', show_wp=True):
    # ----------- Bootstrapping ------------
    auc_check = roc_auc_score(y_true, y_score)
    if auc_check < 0.5:
        y_score = 1-y_score
    
    tprs_boot, aucs_boot, thrs_boot, mean_fpr = auc_bootstrapping(y_true, y_score, bootstrapping, drop_intermediate)

    mean_tpr = np.mean(tprs_boot, axis=0)
    mean_tpr[-1] = 1.0        
    std_tpr = np.std(tprs_boot, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # ------ Averaged based on bootspraping ------
    mean_auc = np.mean(aucs_boot)
    std_auc = np.std(aucs_boot, ddof=1)
    print(f"Std Dev of AUCs: {std_auc:.4f}")
 

    # --------- Specific Case (the main curve) -------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs)

    # Interpolate the main ROC curve onto the same grid as the bootstrapped CIs
    interp_tpr = np.interp(mean_fpr, fprs, tprs)

    # Find optimal point on the interpolated curve
    opt_idx = np.argmax(interp_tpr - mean_fpr)
    opt_tpr = interp_tpr[opt_idx]
    opt_fpr = mean_fpr[opt_idx]
    
    # Find the threshold corresponding to the optimal point
    # We can find the original threshold that gives the closest (fpr, tpr) to our optimal one.
    opt_thr_idx = (np.abs(thrs - np.interp(opt_fpr, fprs, thrs))).argmin()
    opt_thr = thrs[opt_thr_idx]

    y_scores_bin = y_score >= opt_thr
    conf_matrix = confusion_matrix(y_true, y_scores_bin)
    
    # --- Plotting ---
    axis.plot(mean_fpr, interp_tpr, color=color, label=rf"{name} (AUC = {auc_val:.2f} $\pm$ {std_auc:.2f})",
                lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    if show_wp:
        axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color='g', linestyle='--')
        axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color='g', linestyle='--')
    axis.plot(opt_fpr, opt_tpr, color=color, marker='o') 
    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])
    
    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)
    
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')
 
    return interp_tpr, mean_fpr, auc_val, thrs, opt_idx, conf_matrix


def analyze_and_plot_models_cv():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42, max_iter=1000),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    fontdict = {'fontsize': 12}
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results for all models
    all_results = {}

    for name, model_instance in models.items():
        print(f"--- Analyzing Model: {name} with {n_splits}-Fold CV ---")
        
        fold_aucs, fold_accs, fold_sens, fold_specs = [], [], [], []
        tprs_per_fold = []
        mean_fpr = np.linspace(0, 1, 1000)

        for fold, (train_index, val_index) in enumerate(skf.split(embeds_flat, labels)):
            print(f"  -- Fold {fold + 1}/{n_splits} --")
            
            x_train, x_val = embeds_flat[train_index], embeds_flat[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            # Apply SMOTE only to the training data of this fold
            smote = BorderlineSMOTE(random_state=42)
            x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

            # Create and fit the pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_instance)
            ])
            pipeline.fit(x_train_resampled, y_train_resampled)

            # Evaluate on the validation set
            y_prob = pipeline.predict_proba(x_val)[:, 1]
            
            # --- Collect metrics for this fold ---
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            tprs_per_fold.append(np.interp(mean_fpr, fpr, tpr))
            
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)

            # Get other metrics using the optimal threshold from this fold's ROC
            opt_idx = np.argmax(tpr - fpr)
            opt_thr = thresholds[opt_idx]
            y_pred_bin = y_prob >= opt_thr
            cm = confusion_matrix(y_val, y_pred_bin)
            
            acc = cm2acc(cm)
            _, _, sens, spec = cm2x(cm)
            
            fold_accs.append(acc)
            fold_sens.append(sens)
            fold_specs.append(spec)

        # --- Aggregate and store results for the model ---
        all_results[name] = {
            'auc': (np.mean(fold_aucs), np.std(fold_aucs)),
            'acc': (np.mean(fold_accs), np.std(fold_accs)),
            'sens': (np.mean(fold_sens), np.std(fold_sens)),
            'spec': (np.mean(fold_specs), np.std(fold_specs))
        }

        # --- Plot the average ROC curve for the model ---
        fig_roc, ax_roc = plt.subplots(1, 1, figsize=(6, 6))
        
        mean_tpr = np.mean(tprs_per_fold, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs_per_fold, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        mean_auc_val, std_auc_val = all_results[name]['auc']

        ax_roc.plot(mean_fpr, mean_tpr, color='b',
                    label=rf'{name} (AUC = {mean_auc_val:.2f} $\pm$ {std_auc_val:.2f})',
                    lw=2, alpha=.8)
        ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"Mean ROC for {name} ({n_splits}-Fold CV)")
        ax_roc.legend(loc="lower right")
        ax_roc.set_xlabel('1 - Specificity', fontdict=fontdict)
        ax_roc.set_ylabel('Sensitivity', fontdict=fontdict)
        plt.show()

    # --- Print final summary table ---
    print("\n--- Overall Model Comparison (5-Fold CV) ---")
    summary_df = pd.DataFrame.from_dict(all_results, orient='index', columns=['auc', 'acc', 'sens', 'spec'])
    summary_df = summary_df.applymap(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}")
    summary_df.columns = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    print(summary_df)
    print("---------------------------------------------")


# Run the analysis
print("Running 5-Fold Cross-Validation Analysis...")
analyze_and_plot_models_cv()
