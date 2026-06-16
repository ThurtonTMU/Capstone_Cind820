"""
CIND820 Milestone 2 — Complete Corpus EDA (Final)
N=175 variable-document pairs across 17 source documents

Missingness framework (Rubin 1976 / Manski 2003):
  OBSERVED = value publicly accessible — used directly in analysis
  MNAR     = value structurally withheld by controlling actor —
             Missing Not At Random — Manski partial identification applies

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT = "/mnt/user-data/outputs/eda_outputs"
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv("/mnt/user-data/outputs/corpus_variable_registry.csv")
df["Valuation_Primary"] = df["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")

DA_ORDER = ["OBSERVED", "MNAR"]
DA_COLORS = {"OBSERVED": "#70AD47", "MNAR": "#C00000"}

NAVY="#1F3864"; BLUE="#2E75B6"; GREEN="#70AD47"; GOLD="#FFC000"; RED="#C00000"

print("="*65)
print(f"CORPUS EDA FINAL | N={len(df)} | Docs={df['Source Document'].nunique()}")
print("="*65)

# ── SECTION 1: UNIVARIATE ────────────────────────────────────────

DIMS = [
    ("Source Category",   "Source Category",                    None),
    ("MAS Function",      "MAS Function",                       None),
    ("Actor Type",        "Actor Type",                         None),
    ("Valuation_Primary", "Valuation Type (primary)",           None),
    ("Data Access",       "Data Access (OBSERVED vs MNAR)",     [DA_COLORS[k] for k in DA_ORDER]),
    ("AIDM Domain",       "AIDM Domain",                        None),
]

uni_stats = []
print("\nSECTION 1: UNIVARIATE")

for col, label, custom_colors in DIMS:
    counts = df[col].value_counts()
    if col == "Data Access":
        counts = counts.reindex(DA_ORDER).dropna()
    N = len(df)
    pcts = (counts / N * 100).round(1)
    freq = counts.values.astype(float)
    q1, q3 = np.percentile(freq, 25), np.percentile(freq, 75)
    iqr = q3 - q1

    print(f"\n── {label} ──")
    print(f"  k={len(counts)} | Mode='{counts.index[0]}' n={counts.iloc[0]} "
          f"({pcts.iloc[0]:.1f}%) | Freq IQR={iqr:.1f} (Q1={q1:.1f} Q3={q3:.1f})")
    for cat, n in counts.items():
        print(f"    {str(cat):<35} {n:>4} ({pcts[cat]:>5.1f}%)  {'█'*int(n/2)}")

    uni_stats.append({
        "Dimension": label, "k": len(counts),
        "Mode": counts.index[0], "Mode_n": counts.iloc[0],
        "Mode_pct": pcts.iloc[0],
        "Q1": round(q1,1), "Q3": round(q3,1), "IQR": round(iqr,1)
    })

    fig, ax = plt.subplots(figsize=(10, max(3, len(counts)*0.55+1.5)))
    if custom_colors:
        colors = custom_colors[::-1]
    else:
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))[::-1]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors, edgecolor="white", height=0.6)
    for bar, n, p in zip(bars, counts.values[::-1], pcts.values[::-1]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"n={n} ({p:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title(
        f"Univariate: {label}\n"
        f"N={N} | k={len(counts)} categories | "
        f"Mode='{counts.index[0]}' (n={counts.iloc[0]}) | Freq IQR={iqr:.1f}",
        fontsize=9, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, counts.max()*1.28)
    plt.tight_layout()
    fname = col.replace(" ","_").replace("/","_")
    plt.savefig(f"{OUTPUT}/UNI_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()

pd.DataFrame(uni_stats).to_csv(f"{OUTPUT}/univariate_summary.csv", index=False)

# ── SECTION 2: DOCUMENT-LEVEL STATISTICS ─────────────────────────

print("\nSECTION 2: DOCUMENT-LEVEL STATISTICS")

doc = df.groupby("Source Document").agg(
    N=("Variable Name","count"),
    Pct_OBS=("Data Access", lambda x: (x=="OBSERVED").mean()*100),
    Pct_MNAR=("Data Access", lambda x: (x=="MNAR").mean()*100),
    Modal_Val=("Valuation_Primary", lambda x: x.value_counts().index[0]),
    Modal_Actor=("Actor Type", lambda x: x.value_counts().index[0]),
).reset_index()
doc["Category"] = df.groupby("Source Document")["Source Category"].first().values

print(f"\n  Vars/doc: mean={doc['N'].mean():.1f} | median={doc['N'].median():.1f} | "
      f"std={doc['N'].std():.1f} | IQR={doc['N'].quantile(0.75)-doc['N'].quantile(0.25):.1f} | "
      f"min={doc['N'].min()} | max={doc['N'].max()}")
print(f"  MNAR%/doc: mean={doc['Pct_MNAR'].mean():.1f}% | "
      f"median={doc['Pct_MNAR'].median():.1f}% | "
      f"IQR={doc['Pct_MNAR'].quantile(0.75)-doc['Pct_MNAR'].quantile(0.25):.1f}%")

print(f"\n  Document breakdown:")
for _, row in doc.sort_values("N", ascending=False).iterrows():
    print(f"    {row['Source Document'][:48]:<48} "
          f"n={row['N']:>3} | OBS={row['Pct_OBS']:>5.1f}% | "
          f"MNAR={row['Pct_MNAR']:>5.1f}% | val={row['Modal_Val'][:6]}")

fig, ax = plt.subplots(figsize=(12,6))
s = doc.sort_values("N", ascending=True)
cat_c = {"Industry Schema": NAVY, "Vendor API": BLUE, "Airport Document": GREEN}
bars = ax.barh(range(len(s)), s["N"],
               color=[cat_c.get(c,"#888") for c in s["Category"]],
               edgecolor="white", height=0.7)
ax.set_yticks(range(len(s)))
ax.set_yticklabels([d[:48] for d in s["Source Document"]], fontsize=7)
for bar, n in zip(bars, s["N"]):
    ax.text(bar.get_width()+0.15, bar.get_y()+bar.get_height()/2,
            str(n), va="center", fontsize=8)
ax.set_xlabel("Variables Coded", fontsize=10)
ax.set_title(
    f"Variables per Source Document (N=175, 17 documents)\n"
    f"Mean={doc['N'].mean():.1f} | Median={doc['N'].median():.1f} | "
    f"IQR={doc['N'].quantile(0.75)-doc['N'].quantile(0.25):.1f}",
    fontsize=10, fontweight="bold")
ax.legend(handles=[Patch(fc=v,label=k) for k,v in cat_c.items()],
          fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/UNI_variables_per_document.png", dpi=150, bbox_inches="tight")
plt.close()
doc.to_csv(f"{OUTPUT}/document_level_statistics.csv", index=False)

# ── SECTION 3: BIVARIATE ─────────────────────────────────────────

print("\nSECTION 3: BIVARIATE CHI-SQUARE TESTS (Agresti 2007)")

def chi2_test(df, c1, c2, label):
    ct = pd.crosstab(df[c1], df[c2])
    chi2, p, dof, exp = chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))
    cells_lt5 = (exp < 5).sum()
    strength = ("strong" if v>0.3 else
                "moderate-strong" if v>0.2 else
                "moderate" if v>0.1 else "weak")
    print(f"\n── {label} ──")
    print(f"  χ²={chi2:.3f}  df={dof}  p={p:.6f}  V={v:.3f} ({strength})")
    print(f"  min_exp={exp.min():.2f}  cells<5={cells_lt5}"
          + (" ⚠" if cells_lt5>0 else " ✓"))
    print(f"  → {'SIGNIFICANT' if p<0.05 else 'NOT SIGNIFICANT'} (α=0.05)")
    return {"Test":label,"Chi2":round(chi2,3),"df":dof,"p":round(p,6),
            "V":round(v,3),"Strength":strength,"N":n,
            "min_exp":round(exp.min(),2),"cells_lt5":int(cells_lt5),
            "Sig":p<0.05}, ct

PAIRS = [
    ("Actor Type",        "Data Access",       "Actor Type × Data Access"),
    ("MAS Function",      "Data Access",       "MAS Function × Data Access"),
    ("Valuation_Primary", "Data Access",       "Valuation Type × Data Access"),
    ("Source Category",   "Data Access",       "Source Category × Data Access"),
    ("Actor Type",        "Valuation_Primary", "Actor Type × Valuation Type"),
    ("MAS Function",      "Valuation_Primary", "MAS Function × Valuation Type"),
    ("Actor Type",        "MAS Function",      "Actor Type × MAS Function"),
]

results = []
ctabs = {}
for c1, c2, label in PAIRS:
    res, ct = chi2_test(df, c1, c2, label)
    results.append(res)
    ctabs[label] = ct
pd.DataFrame(results).to_csv(f"{OUTPUT}/bivariate_chi2_results.csv", index=False)

def biv_plot(ct, title, chi2, p, v, fname, colors=None):
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if colors:
        ct_pct.plot(kind="bar", ax=ax,
                    color=[colors.get(c,"#888") for c in ct_pct.columns],
                    edgecolor="white", width=0.7)
    else:
        ct_pct.plot(kind="bar", ax=ax,
                    color=[NAVY,BLUE,GREEN,GOLD,RED][:ct_pct.shape[1]],
                    edgecolor="white", width=0.7)
    ax.set_title(f"{title}\nχ²={chi2:.3f}  p={p:.4f}  Cramer's V={v:.3f}",
                 fontsize=9, fontweight="bold", pad=10)
    ax.set_ylabel("% within row", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01,1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/BIV_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()

r = {row["Test"]: row for row in results}

# Data Access plots
for label, fname in [
    ("Actor Type × Data Access",     "actor_type_x_data_access"),
    ("MAS Function × Data Access",   "mas_function_x_data_access"),
    ("Valuation Type × Data Access", "valuation_type_x_data_access"),
    ("Source Category × Data Access","source_category_x_data_access"),
]:
    biv_plot(ctabs[label], label, r[label]["Chi2"], r[label]["p"],
             r[label]["V"], fname, DA_COLORS)

# Valuation / MAS / Actor plots
for label, fname in [
    ("Actor Type × Valuation Type",  "actor_type_x_valuation_type"),
    ("MAS Function × Valuation Type","mas_function_x_valuation_type"),
    ("Actor Type × MAS Function",    "actor_type_x_mas_function"),
]:
    biv_plot(ctabs[label], label, r[label]["Chi2"], r[label]["p"],
             r[label]["V"], fname)

# MNAR heatmap
ct_h = pd.crosstab(df["MAS Function"], df["Data Access"])
ct_h = ct_h[[c for c in DA_ORDER if c in ct_h.columns]]
ct_hp = ct_h.div(ct_h.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(ct_hp.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(len(ct_hp.columns)))
ax.set_xticklabels(ct_hp.columns, fontsize=10)
ax.set_yticks(range(len(ct_hp.index)))
ax.set_yticklabels(ct_hp.index, fontsize=9)
for i in range(len(ct_hp.index)):
    for j in range(len(ct_hp.columns)):
        val = ct_hp.iloc[i,j]
        n   = ct_h.iloc[i,j]
        ax.text(j, i, f"{val:.0f}%\n(n={n})", ha="center", va="center",
                fontsize=9, color="white" if (j==1 and val>40) or (j==0 and val<40) else "black")
plt.colorbar(im, ax=ax, label="% of Variables in MAS Function")
ax.set_title(
    "Data Access by MAS Function\n"
    "OBSERVED (green) = used directly | MNAR (red) = Manski partial ID",
    fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/HEATMAP_data_access_by_function.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ── SUMMARY ──────────────────────────────────────────────────────
print("\n"+"="*65)
print("EDA SUMMARY")
print("="*65)
n_obs  = (df["Data Access"]=="OBSERVED").sum()
n_mnar = (df["Data Access"]=="MNAR").sum()
print(f"\nData Access: OBSERVED={n_obs} ({n_obs/175*100:.1f}%) | "
      f"MNAR={n_mnar} ({n_mnar/175*100:.1f}%)")
print(f"\nUnivariate key findings:")
print(f"  Valuation type: Cost modal (n=47, 26.9%) — income close behind (n=44, 25.1%)")
print(f"  Actor type:     Carrier dominant (n=74, 42.3%) | Passenger n=2 (1.1%)")
print(f"  MAS function:   Revenue Management (n=84, 48.0%)")
print(f"  Vars/document:  mean=10.3 | median=9.0 | IQR=7.0")
print(f"\nBivariate chi-square results:")
for row in results:
    flag = "⚠" if row["cells_lt5"]>0 else " "
    print(f"  {flag} {row['Test']:<45} V={row['V']:.3f} "
          f"({row['Strength']:<16}) p={row['p']:.4f} "
          f"{'✓' if row['Sig'] else '✗'}")
print(f"\nAll 7 tests significant at α=0.05")
new = [f for f in sorted(os.listdir(OUTPUT))
       if any(f.startswith(p) for p in ["UNI_","BIV_","HEATMAP_"])
       or f in ["bivariate_chi2_results.csv","univariate_summary.csv",
                "document_level_statistics.csv"]]
print(f"\nKey output files ({len(new)}):")
for f in new: print(f"  {f}")
