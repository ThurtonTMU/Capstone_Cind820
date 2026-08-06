"""
CIND820 Milestone 4 — Random Forest Feature Importance Analysis
Shows which feature values contribute most to MNAR prediction.

Note: Random Forest shows MAGNITUDE only — not direction.
It tells us which features matter, not whether they push
toward MNAR or OBSERVED. Use LR coefficients for direction.

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUTS, "full_corpus_L1.csv"))
df['Target'] = df['Data Access'].map({'OBSERVED':0,'MNAR':1,'PARTIAL':1})
df['Val_Simple'] = df['Valuation Type'].apply(
    lambda x: x.split('/')[0].strip())

FEATURES = ['Actor Type','MAS Function','Val_Simple','Subject_Area']
X = df[FEATURES]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore',
                           sparse_output=False), FEATURES)
])

pipe = Pipeline([('pre', preprocessor),
                 ('clf', RandomForestClassifier(
                     n_estimators=200, max_depth=8,
                     random_state=42, class_weight='balanced'))])
pipe.fit(X_train, y_train)

# ── EXTRACT IMPORTANCE ────────────────────────────────────────
ohe = pipe['pre'].named_transformers_['ohe']
feature_names = ohe.get_feature_names_out(FEATURES)
importances = pipe['clf'].feature_importances_

imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
}).sort_values('Importance', ascending=False)

# Category totals
imp_df['Category'] = imp_df['Feature'].apply(
    lambda x: x.split('_')[0])
cat_imp = imp_df.groupby('Category')['Importance'].sum(
    ).sort_values(ascending=False)

# ── PRINT RESULTS ─────────────────────────────────────────────
print("RANDOM FOREST FEATURE IMPORTANCE")
print("Higher = more predictive of MNAR status")
print("Note: magnitude only — no direction information")
print("="*60)

print(f"\nBy category (total importance):")
for cat, val in cat_imp.items():
    print(f"  {cat:<20} {val:.3f} ({val*100:.1f}%)")

CATEGORIES = {
    'Actor Type': 'Actor',
    'MAS Function': 'MAS',
    'Valuation Type': 'Val',
    'Subject Area': 'Subject',
}

for label, prefix in CATEGORIES.items():
    sub = imp_df[imp_df['Feature'].str.startswith(prefix)]
    print(f"\n{label} (top features):")
    print(f"  {'Feature':<40} {'Importance':>10}")
    print(f"  {'-'*52}")
    for _, row in sub.head(10).iterrows():
        clean = row['Feature'].replace(f"{prefix}_","")
        print(f"  {clean:<40} {row['Importance']:>10.4f}")

print("""
KEY DISTINCTION FROM LOGISTIC REGRESSION:
  Random Forest importance tells you WHICH features matter.
  It does NOT tell you whether a feature pushes toward
  MNAR or OBSERVED — only that it contributes to the split.

  For direction: see analysis_lr_coefficients.py
  For magnitude: this file

  Together they give the complete picture.
""")

# ── VISUALIZATION ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Random Forest Feature Importance — MNAR Prediction\n"
    "Magnitude Only (no direction) — use LR for direction\n"
    "CIND820 M4 | Marie-Louise Thurton | 500314210",
    fontsize=11, fontweight='bold')

NAVY="1F3864"; BLUE="2E75B6"; GREEN="70AD47"
RED="C00000"; GOLD="FFC000"

CAT_COLORS = {
    'Actor': f'#{RED}',
    'MAS': f'#{NAVY}',
    'Val': f'#{GREEN}',
    'Subject': f'#{GOLD}',
}

def imp_plot(ax, prefix, title, color):
    sub = imp_df[imp_df['Feature'].str.startswith(
        prefix)].sort_values('Importance', ascending=True)
    labels = [r['Feature'].replace(f"{prefix}_","")
              for _, r in sub.iterrows()]
    vals = sub['Importance'].values
    bars = ax.barh(range(len(vals)), vals,
                   color=color, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    for bar, val in zip(bars, vals):
        if val > 0.001:
            ax.text(bar.get_width()+0.001,
                    bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Feature Importance", fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

imp_plot(axes[0,0], 'Actor', 'Actor Type Importance',
         CAT_COLORS['Actor'])
imp_plot(axes[0,1], 'MAS', 'MAS Function Importance',
         CAT_COLORS['MAS'])
imp_plot(axes[1,0], 'Val', 'Valuation Type Importance',
         CAT_COLORS['Val'])
imp_plot(axes[1,1], 'Subject', 'Subject Area Importance',
         CAT_COLORS['Subject'])

plt.tight_layout()
out_path = os.path.join(EDA, "rf_importance.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

imp_df.to_csv(os.path.join(EDA, "rf_importance.csv"), index=False)
print(f"Saved: {out_path}")
print(f"Saved: {os.path.join(EDA, 'rf_importance.csv')}")
