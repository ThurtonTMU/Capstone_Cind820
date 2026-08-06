"""
CIND820 Milestone 4 — Logistic Regression Coefficient Analysis
Extracts and interprets coefficients by feature category.

Shows direction and magnitude of each feature value's
contribution to MNAR prediction.

Positive coefficient = pushes toward MNAR
Negative coefficient = pushes toward OBSERVED

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
from sklearn.linear_model import LogisticRegression
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
                 ('clf', LogisticRegression(
                     max_iter=1000, random_state=42, C=1.0))])
pipe.fit(X_train, y_train)

# ── EXTRACT COEFFICIENTS ──────────────────────────────────────
ohe = pipe['pre'].named_transformers_['ohe']
feature_names = ohe.get_feature_names_out(FEATURES)
coefficients = pipe['clf'].coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Direction': ['→ MNAR' if c > 0 else '→ OBSERVED'
                  for c in coefficients],
    'Magnitude': np.abs(coefficients)
}).sort_values('Coefficient', ascending=False)

# ── PRINT BY CATEGORY ─────────────────────────────────────────
print("LOGISTIC REGRESSION COEFFICIENTS")
print("Positive = pushes toward MNAR")
print("Negative = pushes toward OBSERVED")
print("="*60)

CATEGORIES = {
    'Actor Type': 'Actor',
    'MAS Function': 'MAS',
    'Valuation Type': 'Val',
    'Subject Area': 'Subject',
}

for label, prefix in CATEGORIES.items():
    sub = coef_df[coef_df['Feature'].str.startswith(prefix)]
    print(f"\n{label}:")
    print(f"  {'Feature':<40} {'Coeff':>8} {'Direction'}")
    print(f"  {'-'*60}")
    for _, row in sub.iterrows():
        clean = row['Feature'].replace(f"{prefix}_","")
        print(f"  {clean:<40} {row['Coefficient']:>8.3f}"
              f"  {row['Direction']}")

# ── GOVERNANCE INTERPRETATION ─────────────────────────────────
print("\n" + "="*60)
print("GOVERNANCE INTERPRETATION")
print("="*60)
print("""
ACTOR TYPE:
  Vendor (+1.872) — strongest MNAR predictor. Vendor-controlled
  variables are most likely to be structurally withheld.
  Airport (-1.036) and Regulator (-0.969) — strongest OBSERVED
  predictors. Regulated actors produce observable variables.

MAS FUNCTION:
  Fraud/Payment (+1.231) and Workforce Management (+1.016) —
  most opaque functions. Commercial transaction functions
  concentrate MNAR variables.
  Environmental Boundary (-1.739) — most transparent. Physical
  and party identification variables must be observable for
  aviation operations to function.

VALUATION TYPE:
  Income (+0.907) — variables that generate revenue are more
  likely to be withheld. Market (-0.515) and Cost (-0.198)
  push toward OBSERVED.

SUBJECT AREA:
  Shopping Criteria, Tickets, Orders, Offers — strongly MNAR.
  Commercial transaction subjects are opaque.
  Flights (-2.243), Baggage (-1.598), Aircraft Technical
  Details (-1.309) — strongly OBSERVED. Physical operational
  subjects are transparent.

GOVERNANCE FINDING:
  Variables are most likely MNAR when:
  - Controlled by a Vendor
  - In a commercial transaction function (Fraud/Payment, GDS)
  - Producing Income valuation
  - Concerning Shopping, Tickets, or Orders subjects

  This is the architecture of the evidentiary bootstrap problem
  made statistically explicit by the logistic regression model.
""")

# ── VISUALIZATION ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Logistic Regression Coefficients — MNAR Prediction\n"
    "Direction and Magnitude by Feature Category\n"
    "CIND820 M4 | Marie-Louise Thurton | 500314210",
    fontsize=11, fontweight='bold')

NAVY="1F3864"; RED="C00000"; GREEN="70AD47"

def coef_plot(ax, prefix, title):
    sub = coef_df[coef_df['Feature'].str.startswith(
        prefix)].sort_values('Coefficient')
    labels = [r['Feature'].replace(f"{prefix}_","")
              for _, r in sub.iterrows()]
    vals = sub['Coefficient'].values
    colors = [f'#{RED}' if v > 0 else f'#{GREEN}' for v in vals]
    bars = ax.barh(range(len(vals)), vals,
                   color=colors, edgecolor='white')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, vals):
        x_pos = val + 0.03 if val >= 0 else val - 0.03
        ax.text(x_pos,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8,
                ha='left' if val >= 0 else 'right')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Coefficient (+ = MNAR, - = OBSERVED)",
                  fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

coef_plot(axes[0,0], 'Actor', 'Actor Type Coefficients')
coef_plot(axes[0,1], 'MAS', 'MAS Function Coefficients')
coef_plot(axes[1,0], 'Val', 'Valuation Type Coefficients')
coef_plot(axes[1,1], 'Subject', 'Subject Area Coefficients')

plt.tight_layout()
out_path = os.path.join(EDA, "lr_coefficients.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

coef_df.to_csv(os.path.join(EDA, "lr_coefficients.csv"), index=False)
print(f"Saved: {out_path}")
print(f"Saved: {os.path.join(EDA, 'lr_coefficients.csv')}")
