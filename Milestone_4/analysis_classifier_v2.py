"""
CIND820 Milestone 4 — Classification Model v2
Two-Model Comparison: Logistic Regression vs Random Forest

Predicting MNAR status from observable schema attributes.

Dependent variable: MNAR status (binary 0=OBSERVED, 1=MNAR/PARTIAL)
Independent variables: Actor Type, MAS Function, Valuation Type,
                       IATA Subject Area

Models:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (non-linear, feature importance)

Evaluation: Stratified 5-fold CV + held-out test split (80/20)
Metrics: Accuracy, F1, ROC-AUC, Confusion Matrix

Author: Marie-Louise Thurton, 500314210, TMU Chang School
"""
import os, sys

def get_base_path():
    if 'google.colab' in sys.modules or os.path.exists('/content'):
        if not os.path.exists('/content/Capstone_Cind820'):
            os.system('git clone https://github.com/ThurtonTMU/'
                      'Capstone_Cind820 /content/Capstone_Cind820')
        return '/content/Capstone_Cind820/Milestone_4'
    if os.path.exists('/mnt/user-data/outputs'):
        return '/mnt/user-data/outputs'
    return os.path.dirname(os.path.abspath(__file__))

BASE = get_base_path()
OUTS = BASE
EDA  = os.path.join(BASE, 'eda_outputs')
os.makedirs(EDA, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, accuracy_score, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUTS, "full_corpus_L1.csv"))
df['Target'] = df['Data Access'].map({'OBSERVED':0,'MNAR':1,'PARTIAL':1})
df['Val_Simple'] = df['Valuation Type'].apply(lambda x: x.split('/')[0].strip())

FEATURES = ['Actor Type','MAS Function','Val_Simple','Subject_Area']
X = df[FEATURES]
y = df['Target']

print("="*60)
print("CIND820 M4 — TWO-MODEL MNAR CLASSIFICATION")
print("="*60)
print(f"\nCorpus: {len(df)} variables")
print(f"Target distribution:")
print(f"  OBSERVED (0): {(y==0).sum()} ({(y==0).mean():.1%})")
print(f"  MNAR/PARTIAL (1): {(y==1).sum()} ({(y==1).mean():.1%})")

# ── TRAIN/TEST SPLIT ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Train MNAR rate: {y_train.mean():.1%}")
print(f"Test MNAR rate:  {y_test.mean():.1%}")

# ── PREPROCESSING ─────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), FEATURES)
])

# ── MODELS ────────────────────────────────────────────────────
MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, C=1.0),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced'),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in MODELS.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    cv_scores = cross_validate(pipe, X_train, y_train, cv=cv,
        scoring=['accuracy','f1','roc_auc'], return_train_score=True)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    results[name] = {
        'pipe': pipe, 'y_pred': y_pred, 'y_prob': y_prob,
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_prob),
        'cv_f1_mean': cv_scores['test_f1'].mean(),
        'cv_f1_std': cv_scores['test_f1'].std(),
        'cv_auc_mean': cv_scores['test_roc_auc'].mean(),
        'cv_auc_std': cv_scores['test_roc_auc'].std(),
        'train_f1': cv_scores['train_f1'].mean(),
        'cv_f1_folds': cv_scores['test_f1'],
    }
    print(f"\n{name}:")
    print(f"  Test Accuracy: {results[name]['test_acc']:.3f}")
    print(f"  Test F1:       {results[name]['test_f1']:.3f}")
    print(f"  Test ROC-AUC:  {results[name]['test_auc']:.3f}")
    print(f"  CV F1:         {results[name]['cv_f1_mean']:.3f} "
          f"± {results[name]['cv_f1_std']:.3f}")
    print(f"  Train F1:      {results[name]['train_f1']:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
        target_names=['OBSERVED','MNAR/PARTIAL']))

# ── VISUALIZATIONS ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Two-Model Comparison: Logistic Regression vs Random Forest\n"
    "MNAR Classification | CIND820 M4 | "
    "Marie-Louise Thurton | 500314210",
    fontsize=12, fontweight='bold')

NAVY="1F3864"; BLUE="2E75B6"; GREEN="70AD47"
RED="C00000"; LGRAY="F2F2F2"

# Panel 1: Confusion Matrix LR
ax = axes[0,0]
cm_lr = confusion_matrix(y_test, results['Logistic Regression']['y_pred'])
ConfusionMatrixDisplay(confusion_matrix=cm_lr,
    display_labels=['OBSERVED','MNAR/PARTIAL']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix\nLogistic Regression",
             fontsize=10, fontweight='bold')

# Panel 2: Confusion Matrix RF
ax = axes[0,1]
cm_rf = confusion_matrix(y_test, results['Random Forest']['y_pred'])
ConfusionMatrixDisplay(confusion_matrix=cm_rf,
    display_labels=['OBSERVED','MNAR/PARTIAL']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix\nRandom Forest",
             fontsize=10, fontweight='bold')

# Panel 3: Metric comparison
ax = axes[0,2]
metrics = ['Test\nAccuracy','Test\nF1','Test\nROC-AUC',
           'CV F1\n(mean)','CV AUC\n(mean)']
lr_vals = [results['Logistic Regression']['test_acc'],
           results['Logistic Regression']['test_f1'],
           results['Logistic Regression']['test_auc'],
           results['Logistic Regression']['cv_f1_mean'],
           results['Logistic Regression']['cv_auc_mean']]
rf_vals = [results['Random Forest']['test_acc'],
           results['Random Forest']['test_f1'],
           results['Random Forest']['test_auc'],
           results['Random Forest']['cv_f1_mean'],
           results['Random Forest']['cv_auc_mean']]
x = np.arange(len(metrics))
w = 0.35
b1 = ax.bar(x-w/2, lr_vals, width=w, label='Logistic Regression',
            color=f'#{NAVY}', edgecolor='white')
b2 = ax.bar(x+w/2, rf_vals, width=w, label='Random Forest',
            color=f'#{BLUE}', edgecolor='white')
for bar, val in zip(list(b1)+list(b2), lr_vals+rf_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("Performance Comparison\n(Test Set + CV)",
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 4: CV stability
ax = axes[1,0]
folds = range(1,6)
ax.plot(folds, results['Logistic Regression']['cv_f1_folds'],
        'o-', color=f'#{NAVY}', linewidth=2, markersize=8,
        label=f"LR (σ={results['Logistic Regression']['cv_f1_std']:.3f})")
ax.plot(folds, results['Random Forest']['cv_f1_folds'],
        's-', color=f'#{BLUE}', linewidth=2, markersize=8,
        label=f"RF (σ={results['Random Forest']['cv_f1_std']:.3f})")
ax.set_xlabel("Fold", fontsize=9)
ax.set_ylabel("F1 Score", fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.set_xticks(folds)
ax.set_title("CV Stability: F1 per Fold\n(5-fold stratified)",
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 5: RF Feature importance
ax = axes[1,1]
ohe = results['Random Forest']['pipe']['pre'].named_transformers_['ohe']
feature_names = ohe.get_feature_names_out(FEATURES)
importances = results['Random Forest']['pipe']['clf'].feature_importances_
imp_df = pd.DataFrame({'Feature':feature_names,'Importance':importances})
imp_df['Category'] = imp_df['Feature'].apply(lambda x: x.split('_')[0])
cat_imp = imp_df.groupby('Category')['Importance'].sum().sort_values(ascending=True)
colors_imp = [f'#{RED}' if 'Actor' in c else f'#{NAVY}' if 'MAS' in c
              else f'#{GREEN}' if 'Val' in c else '#FFC000'
              for c in cat_imp.index]
ax.barh(range(len(cat_imp)), cat_imp.values,
        color=colors_imp, edgecolor='white')
ax.set_yticks(range(len(cat_imp)))
ax.set_yticklabels(cat_imp.index, fontsize=9)
for i, val in enumerate(cat_imp.values):
    ax.text(val+0.003, i, f'{val:.3f}', va='center', fontsize=9)
ax.set_title("Feature Importance by Category\n(Random Forest)",
             fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 6: Summary
ax = axes[1,2]
ax.axis('off')
lr = results['Logistic Regression']
rf = results['Random Forest']
summary = (
    f"MODEL SELECTION SUMMARY\n\n"
    f"{'Metric':<18} {'LR':>7} {'RF':>7}\n"
    f"{'-'*34}\n"
    f"{'Test Accuracy':<18} {lr['test_acc']:>7.3f} {rf['test_acc']:>7.3f}\n"
    f"{'Test F1':<18} {lr['test_f1']:>7.3f} {rf['test_f1']:>7.3f}\n"
    f"{'Test ROC-AUC':<18} {lr['test_auc']:>7.3f} {rf['test_auc']:>7.3f}\n"
    f"{'CV F1 Mean':<18} {lr['cv_f1_mean']:>7.3f} {rf['cv_f1_mean']:>7.3f}\n"
    f"{'CV F1 Std':<18} {lr['cv_f1_std']:>7.4f} {rf['cv_f1_std']:>7.4f}\n"
    f"{'Train F1':<18} {lr['train_f1']:>7.3f} {rf['train_f1']:>7.3f}\n"
    f"{'-'*34}\n\n"
    f"Dependent variable:\n"
    f"  MNAR status (binary 0/1)\n\n"
    f"Independent variables:\n"
    f"  Actor Type\n"
    f"  MAS Function\n"
    f"  Valuation Type\n"
    f"  IATA Subject Area\n\n"
    f"Split: 80/20 stratified\n"
    f"CV: 5-fold stratified\n\n"
    f"Limitation:\n"
    f"Cross-source F1 = 0.37-0.42\n"
    f"Inter-rater reliability\n"
    f"testing required."
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=f'#{LGRAY}',
                  edgecolor=f'#{NAVY}', linewidth=2))

plt.tight_layout()
out_path = os.path.join(EDA, "two_model_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

# Save CSV
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test_Accuracy': [results[m]['test_acc'] for m in results],
    'Test_F1': [results[m]['test_f1'] for m in results],
    'Test_ROC_AUC': [results[m]['test_auc'] for m in results],
    'CV_F1_Mean': [results[m]['cv_f1_mean'] for m in results],
    'CV_F1_Std': [results[m]['cv_f1_std'] for m in results],
    'CV_AUC_Mean': [results[m]['cv_auc_mean'] for m in results],
    'Train_F1': [results[m]['train_f1'] for m in results],
})
csv_path = os.path.join(EDA, "two_model_comparison.csv")
comparison_df.to_csv(csv_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Saved: {csv_path}")
