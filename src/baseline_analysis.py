"""
CIND820 Milestone 2 — Baseline Method
Keyword-Based Content Analysis of Schema Corpus

The baseline method: count the frequency of valuation and governance
keywords across schema documents. This establishes what a simple
term-frequency approach can and cannot reveal about the corpus.

The planned method (bipartite actor-variable network) is compared
against this baseline to show what the structural approach adds.

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

import re
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = "/mnt/user-data/outputs/eda_outputs"
os.makedirs(OUTPUT, exist_ok=True)

# ── KEYWORD LISTS ────────────────────────────────────────────────
# Valuation terms
VALUATION_TERMS = [
    "fare", "settlement", "fee", "charge", "price", "cost", "revenue",
    "income", "payment", "refund", "penalty", "surcharge", "tax",
    "proration", "commission", "yield", "rate", "amount", "total",
    "billing", "invoice", "loyalty", "miles", "points", "discount",
    "markup", "margin", "profit", "tariff", "premium"
]

# Governance terms
GOVERNANCE_TERMS = [
    "liability", "audit", "dispute", "risk", "fraud", "compliance",
    "obligation", "restriction", "prohibition", "consent", "enforce",
    "penalty", "violation", "breach", "delay", "cancellation",
    "compensation", "accountability", "transparency", "disclosure",
    "regulation", "standard", "requirement", "condition", "limit"
]

ALL_TERMS = list(set(VALUATION_TERMS + GOVERNANCE_TERMS))

# ── DOCUMENT SOURCES ─────────────────────────────────────────────
# Text-accessible files in project
DOCS = {
    "PCI DSS v4.0.1":            "/mnt/project/PCI-DSS-v4_0_1.pdf",
    "SIS IS-XML Handbook":        "/mnt/project/sis_implementation_handbook_for_airports.pdf",
    "ATPCO Composite":            "/mnt/project/atpco_composite_table.txt",
    "AIDM Glossary (RP1008)":     "/mnt/project/iatapassengerglossaryofterms__Sheet1.csv",
}

# ── READ DOCUMENTS ───────────────────────────────────────────────
def read_doc(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().lower()
    except Exception as e:
        return ""

doc_texts = {}
for name, path in DOCS.items():
    text = read_doc(path)
    doc_texts[name] = text
    print(f"Read: {name:<35} {len(text):>8,} chars")

# ── COUNT KEYWORDS ───────────────────────────────────────────────
results = []
for doc_name, text in doc_texts.items():
    row = {"Document": doc_name, "Total_chars": len(text)}
    total_hits = 0
    for term in ALL_TERMS:
        count = len(re.findall(r'\b' + re.escape(term) + r'\b', text))
        row[term] = count
        total_hits += count
    row["Total_keyword_hits"] = total_hits
    # Normalized per 1000 characters
    row["Hits_per_1k_chars"] = round(total_hits / max(len(text), 1) * 1000, 2)
    results.append(row)

df = pd.DataFrame(results)

# ── SUMMARY TABLE ────────────────────────────────────────────────
print("\n" + "="*65)
print("BASELINE: KEYWORD FREQUENCY CONTENT ANALYSIS")
print("="*65)

print(f"\nDocument-level summary:")
for _, row in df.iterrows():
    print(f"  {row['Document']:<35} "
          f"chars={row['Total_chars']:>8,} | "
          f"hits={row['Total_keyword_hits']:>5} | "
          f"per_1k={row['Hits_per_1k_chars']:>5.2f}")

# Top terms per document
print(f"\nTop 10 terms per document:")
term_cols = ALL_TERMS
for _, row in df.iterrows():
    term_counts = {t: row[t] for t in term_cols}
    top = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  {row['Document']}:")
    for term, count in top:
        bar = "█" * min(int(count/5), 30)
        print(f"    {term:<20} {count:>5}  {bar}")

# Valuation vs governance split
print(f"\nValuation vs Governance term split:")
for _, row in df.iterrows():
    val_total = sum(row[t] for t in VALUATION_TERMS if t in row)
    gov_total = sum(row[t] for t in GOVERNANCE_TERMS if t in row)
    print(f"  {row['Document']:<35} "
          f"valuation={val_total:>4} | governance={gov_total:>4}")

# ── WHAT THE BASELINE CANNOT SHOW ────────────────────────────────
print(f"\n" + "="*65)
print("WHAT THE BASELINE REVEALS VS WHAT IT CANNOT SHOW")
print("="*65)
print("""
REVEALS:
  - Which documents contain valuation and governance language
  - Relative frequency of key terms across documents
  - Whether a document is valuation-heavy vs governance-heavy
  - Simple cross-document comparison of term presence

CANNOT SHOW:
  - Which ACTOR contributes which variable
  - At which AIDM TRANSACTION POINT variables appear
  - How variables from different actors INTERACT
  - Whether the same term means the same thing across actors
    (e.g. 'penalty' in ATPCO = fare rule restriction;
          'penalty' in PCI DSS = regulatory fine;
          'penalty' in SGHA = liability cap)
  - The DIRECTIONALITY of value flows between actors
  - Which transaction points have MNAR data access

CONCLUSION:
  Keyword frequency confirms that valuation and governance concepts
  are present in the corpus. It cannot answer the research questions
  because it treats documents as independent term bags rather than
  as schemas defining actor-variable relationships at specific
  transaction points. The planned bipartite network analysis is
  required to model the structural relationships that determine
  governance stress points.
""")

# ── PLOTS ────────────────────────────────────────────────────────

# 1. Total keyword hits per document
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#1F3864", "#2E75B6", "#70AD47", "#FFC000"]
bars = ax.bar(df["Document"], df["Total_keyword_hits"],
              color=colors, edgecolor="white", width=0.6)
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_title("Baseline: Total Keyword Hits per Document\n"
             "Valuation + Governance terms combined",
             fontsize=10, fontweight="bold")
ax.set_ylabel("Keyword Frequency Count", fontsize=10)
ax.tick_params(axis="x", rotation=20, labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/BASELINE_keyword_hits_total.png",
            dpi=150, bbox_inches="tight")
plt.close()

# 2. Normalized hits per 1k chars
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(df["Document"], df["Hits_per_1k_chars"],
              color=colors, edgecolor="white", width=0.6)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set_title("Baseline: Keyword Density (hits per 1,000 characters)\n"
             "Normalised for document length",
             fontsize=10, fontweight="bold")
ax.set_ylabel("Hits per 1,000 chars", fontsize=10)
ax.tick_params(axis="x", rotation=20, labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/BASELINE_keyword_density.png",
            dpi=150, bbox_inches="tight")
plt.close()

# 3. Valuation vs Governance split — stacked bar
val_totals = [sum(row[t] for t in VALUATION_TERMS if t in row)
              for _, row in df.iterrows()]
gov_totals = [sum(row[t] for t in GOVERNANCE_TERMS if t in row)
              for _, row in df.iterrows()]

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(df))
b1 = ax.bar(x, val_totals, color="#2E75B6", label="Valuation terms",
            edgecolor="white", width=0.6)
b2 = ax.bar(x, gov_totals, bottom=val_totals, color="#C00000",
            label="Governance terms", edgecolor="white", width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(df["Document"], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Keyword Count", fontsize=10)
ax.set_title("Baseline: Valuation vs Governance Term Distribution\n"
             "Shows presence but not structure or actor relationships",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/BASELINE_valuation_vs_governance.png",
            dpi=150, bbox_inches="tight")
plt.close()

# 4. Heatmap — top 15 terms across documents
top_terms = (df[term_cols].sum().sort_values(ascending=False).head(15).index.tolist())
heat_data = df.set_index("Document")[top_terms]

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heat_data.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(top_terms)))
ax.set_xticklabels(top_terms, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["Document"], fontsize=8)
for i in range(len(df)):
    for j in range(len(top_terms)):
        val = heat_data.values[i, j]
        ax.text(j, i, str(int(val)), ha="center", va="center",
                fontsize=7, color="white" if val > heat_data.values.max()*0.5 else "black")
plt.colorbar(im, ax=ax, label="Term frequency")
ax.set_title("Baseline: Top 15 Term Frequencies Across Documents\n"
             "Same term — different actors, different meanings — baseline cannot distinguish",
             fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/BASELINE_term_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Save results
df.to_csv(f"{OUTPUT}/baseline_keyword_frequencies.csv", index=False)
print(f"\nFiles saved:")
for f in ["BASELINE_keyword_hits_total.png", "BASELINE_keyword_density.png",
          "BASELINE_valuation_vs_governance.png", "BASELINE_term_heatmap.png",
          "baseline_keyword_frequencies.csv"]:
    print(f"  {OUTPUT}/{f}")
