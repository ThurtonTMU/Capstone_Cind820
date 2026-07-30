"""
CIND820 Milestone 4 — Analysis Level 2 (L2)
Formula Feature Engineering on Full Corpus Property Graph

Extends L1 (full AIDM corpus, property graph) with engineered
features from authoritative economics texts:
  - ICAO Doc 9562 Airport Economics Manual (2013)
  - IATA/ICAO standard airline KPIs
  - Barnhart et al. (2003) Operations Research
  - Borenstein & Rose (1994) American Economic Review
  - Xu, Wandelt & Sun (2024) Oxford Academic

Three analytical dimensions:
  A. Feature valuation type distribution across MAS functions
  B. MNAR dependency at each interaction edge (formula layer)
  C. Jurisdictional asymmetry in formula outcomes (RQ2 evidence)

Note: 15 features at PoC scale — directional findings only.
Full implementation (USRA) will scale to full AIDM formula scope.

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

# ── PATH RESOLVER ─────────────────────────────────────────────
# Works in Google Colab, local environment, and Claude sandbox
import os, sys

def get_base_path():
    """Detect environment and return base project path."""
    # Colab
    if 'google.colab' in sys.modules or os.path.exists('/content'):
        # Clone repo if not already present
        if not os.path.exists('/content/Capstone_Cind820'):
            os.system('git clone https://github.com/ThurtonTMU/'
                      'Capstone_Cind820 /content/Capstone_Cind820')
        return '/content/Capstone_Cind820/Milestone_4'
    # Claude sandbox
    if os.path.exists('/mnt/user-data/outputs'):
        return '/mnt/user-data/outputs'
    # Local — use script directory
    return os.path.dirname(os.path.abspath(__file__))

BASE = get_base_path()
DATA = BASE   # data files live alongside scripts in repo
OUTS = BASE   # outputs go to same directory
EDA  = os.path.join(BASE, 'eda_outputs')
os.makedirs(EDA, exist_ok=True)
# ── END PATH RESOLVER ─────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT = os.path.join(EDA,"level2")
os.makedirs(OUTPUT, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUTS,"full_corpus_L1.csv"))
df["Val_Primary"] = df["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")

feat_df = pd.read_csv(os.path.join(OUTS,"feature_definitions.csv"))
edges_df = pd.read_csv(
    os.path.join(EDA,"valid_interaction_edges.csv"))
agents_df = pd.read_csv(
    os.path.join(EDA,"agent_nodes.csv"))
stress_df = pd.read_csv(
    os.path.join(EDA,"level1/L1_governance_stress_full.csv"))

print("="*65)
print("L2: FORMULA FEATURE ENGINEERING ON FULL CORPUS")
print(f"Corpus: {len(df)} variables | Features: {len(feat_df)}")
print("="*65)

# ── COLOURS ───────────────────────────────────────────────────
FUNC_COLORS = {
    "Revenue Management":    "#1F3864",
    "Distribution / GDS":    "#0F6E56",
    "Disruption Management": "#534AB7",
    "Fraud / Payment":       "#C00000",
    "Workforce Management":  "#7030A0",
    "Environmental Boundary":"#888888",
}
VAL_COLORS = {
    "Income": "#1F3864", "Cost": "#C00000",
    "Risk":   "#ED7D31", "Market": "#2E75B6",
    "Option": "#70AD47",
}
JURIS_COLORS = {
    "YYZ": "#1F3864", "EWR": "#C00000", "LHR": "#0F6E56"
}
MNAR_COLORS = {
    "YES": "#C00000", "NO": "#70AD47", "PARTIAL": "#FFC000"
}

# ── SECTION A: FEATURE VALUATION PROFILE ──────────────────────
print("\n--- SECTION A: FEATURE VALUATION TYPE PROFILE ---")

# Parse output type for each feature
def parse_output_val(output_type):
    """Extract primary valuation type from output type string."""
    val_map = {
        "Income": "Income", "Cost": "Cost",
        "Risk": "Risk", "Market": "Market", "Option": "Option"
    }
    for val, label in val_map.items():
        if val in output_type:
            return label
    return "Unknown"

feat_df["Primary_Output_Val"] = feat_df["Output_Type"].apply(
    parse_output_val)
feat_df["MNAR_Primary"] = feat_df["MNAR_Dependency"].apply(
    lambda x: x.split(" — ")[0])
feat_df["Juris_Primary"] = feat_df["Jurisdictional_Variant"].apply(
    lambda x: x.split(" — ")[0])

print(f"\nFeature output valuation types:")
print(feat_df["Primary_Output_Val"].value_counts().to_string())
print(f"\nMNAR dependency:")
print(feat_df["MNAR_Primary"].value_counts().to_string())
print(f"\nJurisdictional variant:")
print(feat_df["Juris_Primary"].value_counts().to_string())

# Actor-dependent valuation shifts across features
print(f"\nFeatures with actor-dependent output (different val for "
      f"different actors):")
actor_dep = feat_df[feat_df["Output_Type"].str.contains("/")]
for _, row in actor_dep.iterrows():
    print(f"  {row['Feature_ID']}: {row['Feature_Name']}")
    print(f"    {row['Output_Type']}")

# ── SECTION B: FEATURE → EDGE MAPPING ─────────────────────────
print("\n--- SECTION B: FEATURE TO INTERACTION EDGE MAPPING ---")

# Map each feature to its corresponding valid interaction edge
FEATURE_EDGE_MAP = {
    "FE-01": ("ATPCO_Filing_Agent",   "RMS_Agent"),
    "FE-02": ("Airport_Charge_Agent", "RMS_Agent"),
    "FE-03": ("Airport_Charge_Agent", "RMS_Agent"),
    "FE-04": ("RMS_Agent",            "Execution_Engine"),
    "FE-05": ("RMS_Agent",            "Execution_Engine"),
    "FE-06": ("RMS_Agent",            "Execution_Engine"),
    "FE-07": ("Crew_Scheduling_Agent","Crew_Recovery_Agent"),
    "FE-08": ("RMS_Agent",            "Loyalty_Agent"),
    "FE-09": ("ATPCO_Filing_Agent",   "RMS_Agent"),
    "FE-10": ("GDS_Agent",            "Interline_Settlement_Agent"),
    "FE-11": ("Crew_Recovery_Agent",  "Passenger_Recovery_Agent"),
    "FE-12": ("CDM_Airport_Agent",    "Aircraft_Recovery_Agent"),
    "FE-13": ("Payment_Clearance_Agent","Interline_Settlement_Agent"),
    "FE-14": ("Ground_Handler_Agent", "Crew_Scheduling_Agent"),
    "FE-15": ("Schedule_Agent",       "Crew_Scheduling_Agent"),
}

feat_df["Edge_Source"] = feat_df["Feature_ID"].map(
    {k: v[0] for k, v in FEATURE_EDGE_MAP.items()})
feat_df["Edge_Target"] = feat_df["Feature_ID"].map(
    {k: v[1] for k, v in FEATURE_EDGE_MAP.items()})

# Join with edge stress scores
edge_stress = stress_df[["Source","Target","Governance_Stress_Score",
                          "Hammond_Risk","Data_Access"]].copy()
feat_with_stress = feat_df.merge(
    edge_stress,
    left_on=["Edge_Source","Edge_Target"],
    right_on=["Source","Target"],
    how="left"
)

print(f"\nFeature to edge mapping:")
print(f"{'FID':<8} {'Feature':<32} {'Edge':<45} "
      f"{'Stress':>7} {'Hammond'}")
print("-"*100)
for _, row in feat_with_stress.iterrows():
    edge = f"{str(row.get('Edge_Source',''))[:20]}→{str(row.get('Edge_Target',''))[:20]}"
    print(f"  {row['Feature_ID']:<8} {row['Feature_Name'][:30]:<32} "
          f"{edge:<45} "
          f"{str(row.get('Governance_Stress_Score',''))[:7]:>7} "
          f"{str(row.get('Hammond_Risk',''))}")

# ── SECTION C: JURISDICTIONAL ASYMMETRY ANALYSIS ──────────────
print("\n--- SECTION C: JURISDICTIONAL ASYMMETRY (RQ2) ---")

# Parse computed values for each jurisdiction
def extract_val(text):
    """Extract first numeric or key phrase from computed value."""
    if pd.isna(text) or "Not available" in str(text) or "N/A" in str(text):
        return None
    return str(text)[:80]

feat_df["YYZ_Value"] = feat_df["Computed_Value_YYZ"].apply(extract_val)
feat_df["EWR_Value"] = feat_df["Computed_Value_EWR"].apply(extract_val)
feat_df["LHR_Value"] = feat_df["Computed_Value_LHR"].apply(extract_val)

# Count observable values per jurisdiction
yyz_obs = feat_df["YYZ_Value"].notna().sum()
ewr_obs = feat_df["EWR_Value"].notna().sum()
lhr_obs = feat_df["LHR_Value"].notna().sum()

print(f"\nObservable formula outcomes by jurisdiction:")
print(f"  YYZ (Canada):  {yyz_obs}/{len(feat_df)} features "
      f"({yyz_obs/len(feat_df)*100:.0f}%)")
print(f"  EWR (USA):     {ewr_obs}/{len(feat_df)} features "
      f"({ewr_obs/len(feat_df)*100:.0f}%)")
print(f"  LHR (EU/UK):   {lhr_obs}/{len(feat_df)} features "
      f"({lhr_obs/len(feat_df)*100:.0f}%)")

# Features with asymmetric observability
print(f"\nFeatures with jurisdictional observability gaps:")
asym_features = feat_df[
    (feat_df["YYZ_Value"].notna() != feat_df["EWR_Value"].notna()) |
    (feat_df["YYZ_Value"].notna() != feat_df["LHR_Value"].notna()) |
    (feat_df["EWR_Value"].notna() != feat_df["LHR_Value"].notna())
]
for _, row in asym_features.iterrows():
    yyz_status = "✓" if row["YYZ_Value"] else "✗"
    ewr_status = "✓" if row["EWR_Value"] else "✗"
    lhr_status = "✓" if row["LHR_Value"] else "✗"
    print(f"  {row['Feature_ID']}: {row['Feature_Name'][:35]:<35} "
          f"YYZ:{yyz_status} EWR:{ewr_status} LHR:{lhr_status}")

# ── SECTION D: FORMULA MNAR PROPAGATION ───────────────────────
print("\n--- SECTION D: MNAR PROPAGATION THROUGH FORMULA LAYER ---")
print("Features where MNAR inputs propagate to make output MNAR:")

mnar_prop = feat_df[feat_df["MNAR_Primary"].isin(["YES","PARTIAL"])]
print(f"\n{len(mnar_prop)} of {len(feat_df)} features have MNAR inputs")
print(f"\nMNAR propagation by edge stress level:")
mnar_prop2 = feat_with_stress[
    feat_with_stress["MNAR_Primary"].isin(["YES","PARTIAL"])].copy()
for _, row in mnar_prop2.sort_values(
        "Governance_Stress_Score", ascending=False,
        na_position="last").iterrows():
    stress = row.get("Governance_Stress_Score","?")
    hammond = row.get("Hammond_Risk","?")
    print(f"  [{row['MNAR_Primary']:<8}] {row['Feature_ID']}: "
          f"{row['Feature_Name'][:35]:<35} "
          f"stress={stress} [{hammond}]")

# ── SECTION E: L2 GOVERNANCE STRESS ENRICHMENT ────────────────
print("\n--- SECTION E: L2 ENRICHED GOVERNANCE STRESS ---")
print("Adding formula layer to M3 stress scores:")
print("L2 stress = L1 stress + formula MNAR weight + juris asymmetry")

# Enrich stress scores with feature layer
enriched_rows = []
for _, edge in edges_df.iterrows():
    l1_stress = stress_df[
        (stress_df["Source"]==edge["source"]) &
        (stress_df["Target"]==edge["target"])
    ]["Governance_Stress_Score"].values

    l1_score = l1_stress[0] if len(l1_stress) > 0 else 0

    # Features mapped to this edge
    edge_features = feat_with_stress[
        (feat_with_stress["Edge_Source"]==edge["source"]) &
        (feat_with_stress["Edge_Target"]==edge["target"])
    ]

    n_features = len(edge_features)
    n_mnar_features = (edge_features["MNAR_Primary"]=="YES").sum()
    n_juris_asym = (edge_features["Juris_Primary"]=="YES").sum()

    # L2 enrichment score (additive)
    l2_enrichment = (n_mnar_features * 0.5) + (n_juris_asym * 0.25)
    l2_score = round(l1_score + l2_enrichment, 2)

    enriched_rows.append({
        "Source": edge["source"],
        "Target": edge["target"],
        "Edge_Type": edge["edge_type"],
        "Hammond_Risk": edge["hammond_risk"],
        "L1_Stress_Score": l1_score,
        "N_Features_at_Edge": n_features,
        "N_MNAR_Features": n_mnar_features,
        "N_Juris_Asym_Features": n_juris_asym,
        "L2_Enrichment": round(l2_enrichment, 2),
        "L2_Stress_Score": l2_score,
        "Features": "; ".join(edge_features["Feature_ID"].tolist())
                    if n_features > 0 else "None",
    })

enriched_df = pd.DataFrame(enriched_rows).sort_values(
    "L2_Stress_Score", ascending=False)

print(f"\n{'Edge':<55} {'L1':>4} {'L2':>5} {'Δ':>5} {'Feat':>5}")
print("-"*75)
for _, row in enriched_df.iterrows():
    edge = (f"{row['Source'].replace('_Agent','')[:22]}→"
            f"{row['Target'].replace('_Agent','')[:22]}")
    delta = row["L2_Stress_Score"] - row["L1_Stress_Score"]
    print(f"  {edge:<55} {row['L1_Stress_Score']:>4} "
          f"{row['L2_Stress_Score']:>5.2f} "
          f"{delta:>5.2f} {row['N_Features_at_Edge']:>5}")

# ── VISUALIZATIONS ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Feature output valuation types
ax1 = fig.add_subplot(gs[0, 0])
val_counts = feat_df["Primary_Output_Val"].value_counts()
colors_v = [VAL_COLORS.get(v,"#888") for v in val_counts.index]
bars1 = ax1.bar(range(len(val_counts)), val_counts.values,
                color=colors_v, edgecolor="white")
ax1.set_xticks(range(len(val_counts)))
ax1.set_xticklabels(val_counts.index, rotation=30, ha="right", fontsize=9)
for bar, val in zip(bars1, val_counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             str(val), ha="center", fontsize=9)
ax1.set_ylabel("Feature Count", fontsize=9)
ax1.set_title("Engineered Feature\nOutput Valuation Types",
              fontsize=9, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel 2: MNAR dependency pie
ax2 = fig.add_subplot(gs[0, 1])
mnar_counts = feat_df["MNAR_Primary"].value_counts()
colors_m = [MNAR_COLORS.get(m,"#888") for m in mnar_counts.index]
wedges, texts, autotexts = ax2.pie(
    mnar_counts.values, labels=mnar_counts.index,
    colors=colors_m, autopct="%1.0f%%",
    startangle=90, textprops={"fontsize":8})
ax2.set_title("Feature MNAR\nDependency",
              fontsize=9, fontweight="bold")

# Panel 3: Jurisdictional observability
ax3 = fig.add_subplot(gs[0, 2])
categories = ["YYZ\n(Canada)", "EWR\n(USA)", "LHR\n(EU/UK)"]
obs_counts = [yyz_obs, ewr_obs, lhr_obs]
not_obs = [len(feat_df)-v for v in obs_counts]
ax3.bar(range(3), obs_counts,
        color=[JURIS_COLORS[k] for k in ["YYZ","EWR","LHR"]],
        edgecolor="white", label="Observable")
ax3.bar(range(3), not_obs, bottom=obs_counts,
        color="#CCCCCC", edgecolor="white", label="Not available")
ax3.set_xticks(range(3))
ax3.set_xticklabels(categories, fontsize=9)
ax3.set_ylabel("Feature Count", fontsize=9)
ax3.set_title("Formula Observability\nby Jurisdiction",
              fontsize=9, fontweight="bold")
ax3.legend(fontsize=7)
for i, (obs, tot) in enumerate(zip(obs_counts,
                                    [len(feat_df)]*3)):
    ax3.text(i, obs/2, f"{obs}/{tot}",
             ha="center", va="center",
             fontsize=9, fontweight="bold", color="white")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Panel 4: L1 vs L2 stress scores
ax4 = fig.add_subplot(gs[1, :2])
sorted_e = enriched_df.sort_values("L2_Stress_Score", ascending=True)
labels_e = [f"{r['Source'].replace('_Agent','')[:18]}→\n"
            f"{r['Target'].replace('_Agent','')[:18]}"
            for _, r in sorted_e.iterrows()]
y = range(len(sorted_e))
ax4.barh(y, sorted_e["L1_Stress_Score"],
         color="#2E75B6", label="L1 Stress Score",
         edgecolor="white", height=0.7)
ax4.barh(y, sorted_e["L2_Enrichment"],
         left=sorted_e["L1_Stress_Score"],
         color="#FFC000", label="L2 Formula Enrichment",
         edgecolor="white", height=0.7)
ax4.set_yticks(y)
ax4.set_yticklabels(labels_e, fontsize=6.5)
ax4.set_xlabel("Governance Stress Score", fontsize=9)
ax4.set_title("L1 vs L2 Governance Stress Scores\n"
              "Blue=L1 property graph | Gold=L2 formula enrichment",
              fontsize=9, fontweight="bold")
ax4.legend(fontsize=8, loc="lower right")
ax4.axvline(3, color="gray", linestyle="--",
            alpha=0.5, linewidth=0.8)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Panel 5: Feature distribution by MAS function
ax5 = fig.add_subplot(gs[1, 2])
func_feat = feat_df["MAS_Function"].value_counts()
colors_ff = [FUNC_COLORS.get(f,"#888") for f in func_feat.index]
bars5 = ax5.barh(range(len(func_feat)), func_feat.values,
                  color=colors_ff, edgecolor="white")
ax5.set_yticks(range(len(func_feat)))
ax5.set_yticklabels([f.replace("Revenue Management","RevMgmt")
                      .replace("Distribution / GDS","Dist/GDS")
                      .replace("Disruption Management","Disrupt")
                      .replace("Fraud / Payment","Fraud")
                      .replace("Workforce Management","Workforce")
                     for f in func_feat.index], fontsize=8)
for bar, val in zip(bars5, func_feat.values):
    ax5.text(bar.get_width()+0.05,
             bar.get_y()+bar.get_height()/2,
             str(val), va="center", fontsize=9)
ax5.set_title("Features by\nMAS Function",
              fontsize=9, fontweight="bold")
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# Panel 6: Jurisdictional asymmetry detail — FE-11 delay compensation
ax6 = fig.add_subplot(gs[2, :])
delay_scenarios = [
    ("3h delay\n<1500km", 0, 400, 250),
    ("3h delay\n1500-3500km", 0, 700, 400),
    ("3h delay\n>3500km", 0, 1000, 600),
    ("6h delay\n(any dist)", 0, 700, 600),
    ("9h delay\n(any dist)", 0, 1000, 600),
]
labels_d = [s[0] for s in delay_scenarios]
us_vals = [s[1] for s in delay_scenarios]
ca_vals = [s[2] for s in delay_scenarios]
eu_vals = [s[3] for s in delay_scenarios]

x6 = np.arange(len(labels_d))
w6 = 0.25
ax6.bar(x6-w6, us_vals, width=w6,
        color=JURIS_COLORS["EWR"], label="USA (EWR) — USD",
        edgecolor="white")
ax6.bar(x6, ca_vals, width=w6,
        color=JURIS_COLORS["YYZ"], label="Canada (YYZ) — CAD (APPR)",
        edgecolor="white")
ax6.bar(x6+w6, eu_vals, width=w6,
        color=JURIS_COLORS["LHR"], label="EU/UK (LHR) — EUR (EC 261)",
        edgecolor="white")

ax6.set_xticks(x6)
ax6.set_xticklabels(labels_d, fontsize=9)
ax6.set_ylabel("Compensation Amount (local currency)", fontsize=9)
ax6.set_title("FE-11: Delay Compensation Liability — Jurisdictional Asymmetry\n"
              "Identical input variable (Delay Code) → three different "
              "formula outcomes by jurisdiction. RQ2 direct evidence.",
              fontsize=9, fontweight="bold")
ax6.legend(fontsize=8)
ax6.axhline(0, color="gray", linewidth=0.5)
ax6.annotate("US: $0\n(no federal\nmandate)",
             xy=(0, 0), xytext=(0.3, 150),
             fontsize=7.5, color=JURIS_COLORS["EWR"],
             arrowprops=dict(arrowstyle="->",
                             color=JURIS_COLORS["EWR"]))
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

fig.suptitle(
    "L2 Analysis: Formula Feature Engineering on Full Corpus\n"
    f"15 features | 9 revenue mgmt | 2 disruption | 2 workforce | "
    f"1 distribution | 1 fraud | "
    f"13/15 show jurisdictional asymmetry",
    fontsize=11, fontweight="bold", y=1.01)

plt.savefig(f"{OUTPUT}/L2_formula_features.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ── SAVE OUTPUTS ──────────────────────────────────────────────
enriched_df.to_csv(f"{OUTPUT}/L2_enriched_stress_scores.csv", index=False)
feat_with_stress[[
    "Feature_ID","Feature_Name","MAS_Function","Primary_Output_Val",
    "MNAR_Primary","Juris_Primary","Edge_Source","Edge_Target",
    "Governance_Stress_Score","Hammond_Risk"
]].to_csv(f"{OUTPUT}/L2_feature_edge_mapping.csv", index=False)

# ── L2 SUMMARY ────────────────────────────────────────────────
print("\n" + "="*65)
print("L2 ANALYSIS SUMMARY")
print("="*65)

top_edge = enriched_df.iloc[0]
most_enriched = enriched_df.sort_values("L2_Enrichment",
                                         ascending=False).iloc[0]

print(f"""
FEATURE ENGINEERING (PoC scale — 15 features):
  Revenue Management:   9 features (airport economics + airline KPIs)
  Disruption Mgmt:      2 features (delay compensation + CDM compliance)
  Workforce Management: 2 features (ground handling cost + crew cost/BH)
  Distribution/GDS:     1 feature  (interline proration proxy)
  Fraud/Payment:        1 feature  (fraud loss rate)

MNAR DEPENDENCY THROUGH FORMULA LAYER:
  YES (fully MNAR inputs):     5 features
  PARTIAL (some MNAR inputs):  4 features
  NO (all OBSERVED inputs):    6 features
  
  Key finding: MNAR propagates through formulas.
  When a formula depends on MNAR inputs, the formula output
  is also effectively MNAR even if the formula itself is public.
  Break-even load factor (FE-08): both CASK and Yield are MNAR
  at carrier level — the most strategically sensitive formula
  in airline economics is completely opaque.

JURISDICTIONAL ASYMMETRY (RQ2 evidence):
  13 of 15 features show different outcomes across jurisdictions.
  Most significant: FE-11 Delay Compensation Liability.
  Same Delay Code variable → USD 0 (EWR) vs CAD 400-1000 (YYZ)
  vs EUR 250-600 (LHR). The formula is jurisdiction-specific
  even when the input variable is not.
  
  Observability asymmetry: LHR most observable (WACC published,
  EC 261 amounts fixed). YYZ partially observable (GTAA fees
  published, APPR amounts fixed). EWR least observable in
  disruption layer (no federal compensation formula).

L2 ENRICHMENT OF GOVERNANCE STRESS:
  Highest L2 score: {top_edge['Source'].replace('_Agent','')} → 
    {top_edge['Target'].replace('_Agent','')} (score: {top_edge['L2_Stress_Score']:.2f})
  Most enriched by formula layer: {most_enriched['Source'].replace('_Agent','')} →
    {most_enriched['Target'].replace('_Agent','')} 
    (L2 enrichment: +{most_enriched['L2_Enrichment']:.2f})

METHODOLOGICAL NOTE (PoC scope):
  15 features at PoC scale produce directional findings only.
  The USRA full implementation will scale to:
  - Complete ICAO Doc 9562 charge formula taxonomy
  - Full ISO 20022 payment message formula set
  - BSP/ICH clearing formula layer
  - ANSP route charge formulas (NAV CANADA/FAA/EUROCONTROL)
  - Complete IFRS 15/IAS 37/IFRS 9 accounting formula layer
  At full scale the formula interaction model becomes a
  genuinely novel analytical instrument for aviation
  MAS governance research.
""")

print(f"Outputs saved to {OUTPUT}")
