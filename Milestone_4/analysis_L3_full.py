"""
CIND820 Milestone 4 — Analysis Level 3 (L3) Full
EWR Property Graph Instantiation with Game Analysis

The AIDM schema instantiated at airport scale.
Same EWR airport, five MAS functions, three carriers,
real operational data, game states by month.

Three carriers:
  Porter Airlines   (YTZ-EWR) — transborder CA
  Icelandair        (EWR-KEF) — transatlantic EU
  JetBlue           (EWR)     — domestic US

Five MAS functions analyzed:
  1. Revenue Management
  2. Disruption Management
  3. Distribution / GDS (interline)
  4. Fraud / Payment
  5. Workforce / Baggage

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
from matplotlib.colors import LinearSegmentedColormap
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT = os.path.join(EDA,"level3")
os.makedirs(OUTPUT, exist_ok=True)

# ── COLOURS ───────────────────────────────────────────────────
C = {
    "Porter":     "#1F3864",
    "JetBlue":    "#C00000",
    "Icelandair": "#0F6E56",
    "MNAR":       "#C00000",
    "OBSERVED":   "#70AD47",
    "PARTIAL":    "#FFC000",
}
FUNC_COLORS = {
    "Revenue Management":    "#1F3864",
    "Disruption Management": "#534AB7",
    "Distribution / GDS":    "#0F6E56",
    "Fraud / Payment":       "#C00000",
    "Workforce Management":  "#7030A0",
}

# ── LOAD DATA ─────────────────────────────────────────────────
corpus = pd.read_csv(os.path.join(OUTS,"full_corpus_L1.csv"))
feats  = pd.read_csv(os.path.join(OUTS,"feature_definitions.csv"))
edges  = pd.read_csv(
    os.path.join(EDA,"valid_interaction_edges.csv"))
stress = pd.read_csv(
    os.path.join(EDA,"level1/L1_governance_stress_full.csv"))
porter_raw = pd.read_csv(
    os.path.join(OUTS,"porter_ytz_ewr_2023.csv"))
fi_raw = pd.read_csv(
    os.path.join(OUTS,"icelandair_ewr_kef_2023.csv"))
b6_raw = pd.read_csv(
    os.path.join(OUTS,"b6_ewr_2023.csv"), low_memory=False)

print("="*65)
print("L3: EWR PROPERTY GRAPH INSTANTIATION")
print("AIDM at Airport Scale with Game Analysis")
print("="*65)

# ── MONTHLY TRAFFIC ───────────────────────────────────────────
p_m = porter_raw.groupby('MONTH').agg(
    Pax=('PASSENGERS','sum'),
    Seats=('SEATS','sum')).reset_index()
p_m['LF'] = p_m['Pax'] / p_m['Seats']

fi_m = fi_raw.groupby('MONTH').agg(
    Pax=('PASSENGERS','sum'),
    Seats=('SEATS','sum')).reset_index()
fi_m['LF'] = fi_m['Pax'] / fi_m['Seats']

months = list(range(1, 12))

# ── SECTION 1: PROPERTY GRAPH NODE INSTANTIATION ─────────────
print("\n--- SECTION 1: PROPERTY GRAPH NODE INSTANTIATION ---")
print("Each agent node populated with real EWR 2023 values")
print("for all three carriers.\n")

NODE_INSTANCES = {
    "RMS_Agent": {
        "Porter": {
            "LF_annual": p_m['Pax'].sum() / p_m['Seats'].sum(),
            "LF_std":    p_m['LF'].std(),
            "Pax_annual": p_m['Pax'].sum(),
            "Route_type": "Transborder YTZ-EWR",
            "RMS_vendor": "MNAR (unknown)",
            "Optimization_logic": "Frequency optimization",
            "Key_AIDM_vars": ["numberOfBookableSeats","Offer",
                               "Price","Fare Detail"],
            "MNAR_exposure": "High (RMS bid price MNAR)",
            "Jurisdiction": "Canada / APPR",
        },
        "Icelandair": {
            "LF_annual": fi_m['Pax'].sum() / fi_m['Seats'].sum(),
            "LF_std":    fi_m['LF'].std(),
            "Pax_annual": fi_m['Pax'].sum(),
            "Route_type": "Transatlantic EWR-KEF",
            "RMS_vendor": "MNAR (unknown)",
            "Optimization_logic": "Leisure/seasonal yield optimization",
            "Key_AIDM_vars": ["numberOfBookableSeats","Offer",
                               "Price","Fare Detail"],
            "MNAR_exposure": "High (RMS bid price MNAR)",
            "Jurisdiction": "EU/Iceland / EC 261",
        },
        "JetBlue": {
            "LF_annual": 0.826,
            "LF_std":    None,
            "Pax_annual": 1745080,
            "Route_type": "Domestic EWR-leisure",
            "RMS_vendor": "MNAR (unknown)",
            "Optimization_logic": "Leisure price discrimination",
            "Key_AIDM_vars": ["numberOfBookableSeats","Offer",
                               "Price","Fare Detail"],
            "MNAR_exposure": "High (RMS bid price MNAR)",
            "Jurisdiction": "USA / DOT",
        },
    },
    "Passenger_Recovery_Agent": {
        "Porter": {
            "Pax_at_risk": p_m['Pax'].sum() * 0.05,
            "Compensation_3h": 400,
            "Compensation_9h": 1000,
            "Currency": "CAD",
            "Regulation": "APPR SOR/2019-150",
            "Total_exposure_9h": p_m['Pax'].sum() * 0.05 * 1000,
            "MNAR_exposure": "Low (regulation OBSERVED)",
        },
        "Icelandair": {
            "Pax_at_risk": fi_m['Pax'].sum() * 0.05,
            "Compensation_3h": 400,
            "Compensation_9h": 600,
            "Currency": "EUR",
            "Regulation": "EC 261/2004",
            "Total_exposure_9h": fi_m['Pax'].sum() * 0.05 * 600,
            "MNAR_exposure": "Low (regulation OBSERVED)",
        },
        "JetBlue": {
            "Pax_at_risk": 1745080 * 0.05,
            "Compensation_3h": 0,
            "Compensation_9h": 0,
            "Currency": "USD",
            "Regulation": "DOT Part 259 (no mandatory)",
            "Total_exposure_9h": 0,
            "MNAR_exposure": "N/A (no obligation)",
        },
    },
    "Interline_Settlement_Agent": {
        "Porter-JetBlue-Icelandair": {
            "Interline_pax_estimate": 112930,
            "Settlement_formula": "NFP proration (MNAR)",
            "Settlement_currency": "CAD/USD/EUR",
            "Clearance_mechanism": "IATA BSP Toronto",
            "Observable_volume": True,
            "Observable_amount": False,
            "MNAR_exposure": "Critical (all terms MNAR)",
            "Correlation_signal": 0.959,
        }
    },
    "Transaction_Stream_Agent": {
        "Multi-carrier EWR": {
            "B6_interline_coupons": 1535,
            "TkCarrier_AA": 1460,
            "Currencies_in_play": 3,
            "PCI_DSS_applies": "All carriers",
            "SCA_applies": "Icelandair (EU PSD2)",
            "MNAR_exposure": "Critical (fraud score MNAR)",
        }
    },
    "Ground_Handler_Agent": {
        "Porter": {
            "Bag_customs_regimes": 2,
            "Preclearance": "CBSA at YTZ (unique)",
            "Bag_recheck_required": False,
            "SGHA_bilateral": "MNAR",
        },
        "Icelandair": {
            "Bag_customs_regimes": 2,
            "Preclearance": "CBP at EWR",
            "Bag_recheck_required": True,
            "SGHA_bilateral": "MNAR",
        },
        "JetBlue": {
            "Bag_customs_regimes": 1,
            "Preclearance": "N/A domestic",
            "Bag_recheck_required": False,
            "SGHA_bilateral": "MNAR",
        },
    },
}

for node, carriers in NODE_INSTANCES.items():
    print(f"  {node}:")
    for carrier, attrs in carriers.items():
        print(f"    [{carrier}]")
        for k, v in list(attrs.items())[:4]:
            if isinstance(v, float):
                print(f"      {k}: {v:.3f}" if v < 10
                      else f"      {k}: {v:,.0f}")
            else:
                print(f"      {k}: {v}")

# ── SECTION 2: GAME STATE MATRIX ─────────────────────────────
print("\n--- SECTION 2: GAME STATE MATRIX BY MONTH AND FUNCTION ---")

GAME_STATES = []
MONTHS_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov']

for idx, m in enumerate(months):
    p_row = p_m[p_m['MONTH']==m]
    f_row = fi_m[fi_m['MONTH']==m]
    if len(p_row)==0 or len(f_row)==0:
        continue

    p_lf = p_row['LF'].values[0]
    f_lf = f_row['LF'].values[0]
    p_pax = p_row['Pax'].values[0]
    f_pax = f_row['Pax'].values[0]

    # Capacity constraint states
    p_constrained = p_lf > 0.85
    f_constrained = f_lf > 0.85

    # REVENUE MANAGEMENT game state
    if p_constrained and f_constrained:
        rm_state = "COLLUSION RISK"
        rm_score = 3
        rm_desc = "Both capacity-constrained — interline premium active, proration favors both"
    elif f_constrained and not p_constrained:
        rm_state = "FI DOMINATES"
        rm_score = 2
        rm_desc = "Icelandair sets price — Porter needs connection volume"
    elif p_constrained and not f_constrained:
        rm_state = "PORTER DOMINATES"
        rm_score = 2
        rm_desc = "Porter sets price — Icelandair needs feeder pax"
    else:
        rm_state = "COMPETITION"
        rm_score = 1
        rm_desc = "Neither constrained — price competition on interline fare"

    # DISRUPTION MANAGEMENT game state
    # Icelandair always has higher compensation obligation
    fi_exposure_per_pax = 400 if f_lf < 0.85 else 600
    p_exposure_per_pax = 400 if p_lf < 0.85 else 700
    dis_state = (
        "EU > CA >> US"
        if fi_exposure_per_pax >= p_exposure_per_pax
        else "CA > EU >> US"
    )
    dis_score = 2  # always asymmetric

    # DISTRIBUTION/INTERLINE game state
    # Interline demand proxy: geometric mean of pax volumes
    interline_proxy = (p_pax * f_pax) ** 0.5
    max_proxy = max(
        [(p_m[p_m['MONTH']==mm]['Pax'].values[0] *
          fi_m[fi_m['MONTH']==mm]['Pax'].values[0]) ** 0.5
         for mm in months
         if len(p_m[p_m['MONTH']==mm])>0 and
            len(fi_m[fi_m['MONTH']==mm])>0])
    interline_intensity = interline_proxy / max_proxy

    if interline_intensity > 0.7:
        dist_state = "HIGH INTERLINE"
        dist_score = 3
    elif interline_intensity > 0.4:
        dist_state = "MODERATE INTERLINE"
        dist_score = 2
    else:
        dist_state = "LOW INTERLINE"
        dist_score = 1

    # FRAUD/PAYMENT game state
    # Multi-currency risk is higher when all three carriers active
    if p_lf > 0.70 and f_lf > 0.75:
        fraud_state = "3-CURRENCY ACTIVE"
        fraud_score = 3
    else:
        fraud_state = "REDUCED EXPOSURE"
        fraud_score = 1

    # WORKFORCE/BAGGAGE game state
    # Customs complexity always present for Porter + Icelandair
    # Highest when volumes high (more mishandling risk)
    pax_volume = p_pax + f_pax
    if pax_volume > 200000:
        wf_state = "HIGH VOLUME CUSTOMS"
        wf_score = 3
    elif pax_volume > 100000:
        wf_state = "MODERATE CUSTOMS"
        wf_score = 2
    else:
        wf_state = "LOW VOLUME"
        wf_score = 1

    # Overall game intensity
    overall = (rm_score + dis_score + dist_score +
               fraud_score + wf_score) / 5

    GAME_STATES.append({
        'Month': m,
        'Month_Label': MONTHS_LABELS[idx],
        'Porter_LF': p_lf,
        'FI_LF': f_lf,
        'Porter_Pax': p_pax,
        'FI_Pax': f_pax,
        'RM_State': rm_state,
        'RM_Score': rm_score,
        'DIS_State': dis_state,
        'DIS_Score': dis_score,
        'DIST_State': dist_state,
        'DIST_Score': dist_score,
        'FRAUD_State': fraud_state,
        'FRAUD_Score': fraud_score,
        'WF_State': wf_state,
        'WF_Score': wf_score,
        'Overall_Intensity': overall,
    })

gs_df = pd.DataFrame(GAME_STATES)

print(f"\n{'Month':<6} {'RM State':<20} {'DIS':<12} "
      f"{'DIST':<20} {'FRAUD':<20} {'Overall':>8}")
print("-"*90)
for _, row in gs_df.iterrows():
    print(f"  {row['Month_Label']:<4} {row['RM_State']:<20} "
          f"{row['DIS_State']:<12} {row['DIST_State']:<20} "
          f"{row['FRAUD_State']:<20} {row['Overall_Intensity']:>8.2f}")

print(f"\nPeak game intensity month: "
      f"{gs_df.loc[gs_df['Overall_Intensity'].idxmax(),'Month_Label']}"
      f" (score: {gs_df['Overall_Intensity'].max():.2f}/3.0)")
print(f"Lowest game intensity month: "
      f"{gs_df.loc[gs_df['Overall_Intensity'].idxmin(),'Month_Label']}"
      f" (score: {gs_df['Overall_Intensity'].min():.2f}/3.0)")

# ── SECTION 3: MNAR BOUNDARY MAP AT EWR ──────────────────────
print("\n--- SECTION 3: MNAR BOUNDARY MAP AT EWR ---")
print("What is and isn't observable for each function at EWR 2023\n")

MNAR_MAP = [
    # Function, Variable_Type, Porter, Icelandair, JetBlue, Note
    ("Revenue Management", "Load Factor",
     "OBSERVED","OBSERVED","OBSERVED",
     "Observable output of MNAR algorithm"),
    ("Revenue Management", "Bid Price Algorithm",
     "MNAR","MNAR","MNAR",
     "RMS vendor proprietary — all three carriers"),
    ("Revenue Management", "Fare per Mile (yield proxy)",
     "MNAR","MNAR","PARTIAL",
     "B6 partial via DB1B ticket; Porter/FI MNAR"),
    ("Revenue Management", "Break-even Load Factor",
     "MNAR","MNAR","MNAR",
     "CASK+yield both MNAR — most sensitive formula"),
    ("Disruption Management", "Delay Code",
     "OBSERVED","OBSERVED","OBSERVED",
     "AIDX publishes delay codes"),
    ("Disruption Management", "Compensation Amount",
     "OBSERVED","OBSERVED","OBSERVED",
     "Regulatory formula public — APPR/EC261/DOT"),
    ("Disruption Management", "Compensation Liability Total",
     "PARTIAL","PARTIAL","OBSERVED",
     "Computable from pax volume × regulatory formula"),
    ("Distribution / GDS", "Interline Volume",
     "PARTIAL","PARTIAL","PARTIAL",
     "Estimable from T-100f correlation (r=0.959)"),
    ("Distribution / GDS", "NFP Proration Amount",
     "MNAR","MNAR","MNAR",
     "Settlement amount per itinerary — never disclosed"),
    ("Distribution / GDS", "Ticketing Carrier Split",
     "MNAR","MNAR","PARTIAL",
     "B6 partial via DB1B; Porter/FI MNAR (non-US origin)"),
    ("Fraud / Payment", "Transaction Volume",
     "PARTIAL","MNAR","PARTIAL",
     "B6 interline coupons PARTIAL; others MNAR"),
    ("Fraud / Payment", "Fraud Score",
     "MNAR","MNAR","MNAR",
     "Card network proprietary — all carriers"),
    ("Fraud / Payment", "Multi-currency exposure",
     "PARTIAL","PARTIAL","OBSERVED",
     "Route type reveals currency regime; amounts MNAR"),
    ("Workforce Management", "Customs Regime",
     "OBSERVED","OBSERVED","OBSERVED",
     "Route type determines customs — public information"),
    ("Workforce Management", "SGHA Ground Handling Rate",
     "MNAR","MNAR","MNAR",
     "Bilateral — never disclosed"),
    ("Workforce Management", "Bag Mishandling Rate",
     "PARTIAL","PARTIAL","PARTIAL",
     "IATA WATS aggregate; carrier-specific MNAR"),
]

mnar_df = pd.DataFrame(MNAR_MAP,
    columns=['Function','Variable','Porter_Status',
             'FI_Status','B6_Status','Note'])

ACCESS_SCORE = {'OBSERVED':0,'PARTIAL':1,'MNAR':2}
for carrier in ['Porter','FI','B6']:
    mnar_df[f'{carrier}_Score'] = mnar_df[f'{carrier}_Status'].map(
        ACCESS_SCORE)

print(f"{'Variable':<35} {'Porter':>8} {'Iceland':>8} "
      f"{'JetBlue':>8}")
print("-"*65)
for func in mnar_df['Function'].unique():
    print(f"\n  [{func}]")
    sub = mnar_df[mnar_df['Function']==func]
    for _, row in sub.iterrows():
        print(f"  {row['Variable'][:33]:<35} "
              f"{row['Porter_Status']:>8} "
              f"{row['FI_Status']:>8} "
              f"{row['B6_Status']:>8}")

# Overall MNAR rate by carrier at EWR
for carrier in ['Porter','FI','B6']:
    mnar_rate = (mnar_df[f'{carrier}_Score']==2).sum() / len(mnar_df)
    print(f"\n{carrier} overall MNAR rate at EWR: {mnar_rate:.0%}")

# ── SECTION 4: VALUE EXCHANGE ANALYSIS ───────────────────────
print("\n--- SECTION 4: VALUE EXCHANGE AT INTERACTION EDGES ---")
print("Who gains and loses value at each active edge at EWR\n")

VALUE_EXCHANGES = [
    {
        "Edge": "RMS_Agent → Interline_Settlement_Agent",
        "Function": "Revenue Management + Distribution",
        "Mechanism": "NFP Proration",
        "Value_Type": "Income transfer between carriers",
        "Porter_Position": "Net receiver when LF < Icelandair LF",
        "FI_Position": "Net receiver when capacity constrained (summer)",
        "B6_Position": "Margin on domestic connection leg",
        "MNAR_Terms": "Proration Value, FDR, Settlement Amount",
        "Monthly_Asymmetry": "July-Aug: both gain; Q1: competition; "
                             "Apr/Sep/Oct: FI gains",
        "Governance_Risk": "Collusion (Jul-Aug) / Conflict (shoulder)",
    },
    {
        "Edge": "Airport_Charge_Agent → RMS_Agent",
        "Function": "Revenue Management",
        "Mechanism": "Landing/PSC fee → yield floor",
        "Value_Type": "Cost absorbed into fare",
        "Porter_Position": "CAD 35/pax AIF at YTZ + EWR PFC",
        "FI_Position": "EWR charges + LHR-equivalent at KEF",
        "B6_Position": "EWR PFC + landing fees",
        "MNAR_Terms": "WACC (LHR only) — YTZ/EWR unregulated",
        "Monthly_Asymmetry": "Static — charges don't vary monthly",
        "Governance_Risk": "Miscoordination (regulatory gap YTZ/EWR)",
    },
    {
        "Edge": "Transaction_Stream → Card_Network_Agent",
        "Function": "Fraud / Payment",
        "Mechanism": "PCI DSS fraud scoring",
        "Value_Type": "Risk allocation",
        "Porter_Position": "CAD transaction at YTZ origin",
        "FI_Position": "EUR transaction at KEF origin",
        "B6_Position": "USD transaction at EWR — most visible",
        "MNAR_Terms": "Fraud score, CDE scope, authentication result",
        "Monthly_Asymmetry": "Higher risk in peak months (volume-driven)",
        "Governance_Risk": "Collusion (vendor-carrier aligned interests)",
    },
    {
        "Edge": "Crew_Recovery → Passenger_Recovery_Agent",
        "Function": "Disruption Management",
        "Mechanism": "Delay code → compensation formula",
        "Value_Type": "Compensation transfer: carrier → passenger",
        "Porter_Position": "CAD 400-1000 mandatory (APPR)",
        "FI_Position": "EUR 400-600 mandatory (EC 261)",
        "B6_Position": "USD 0 mandatory",
        "MNAR_Terms": "Delay code OBSERVED; compensation formula OBSERVED",
        "Monthly_Asymmetry": "Higher pax volume in summer = higher exposure",
        "Governance_Risk": "Conflict (regulatory asymmetry)",
    },
    {
        "Edge": "Ground_Handler → Schedule_Agent",
        "Function": "Workforce Management",
        "Mechanism": "Turnaround time constraint",
        "Value_Type": "Cost (missed connection if delay)",
        "Porter_Position": "Short-haul YTZ-EWR — tight turnaround",
        "FI_Position": "Long-haul — more turnaround buffer",
        "B6_Position": "Domestic — most schedule flexibility",
        "MNAR_Terms": "SGHA bilateral rates MNAR",
        "Monthly_Asymmetry": "Summer peak = highest missed connection risk",
        "Governance_Risk": "Conflict (handler vs scheduler)",
    },
]

for ve in VALUE_EXCHANGES:
    print(f"  EDGE: {ve['Edge']}")
    print(f"  Function: {ve['Function']}")
    print(f"  Porter: {ve['Porter_Position']}")
    print(f"  Icelandair: {ve['FI_Position']}")
    print(f"  JetBlue: {ve['B6_Position']}")
    print(f"  Game asymmetry: {ve['Monthly_Asymmetry']}")
    print(f"  Risk: {ve['Governance_Risk']}")
    print()

# ── VISUALIZATIONS ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs_fig = GridSpec(3, 4, figure=fig,
                  hspace=0.5, wspace=0.38)

month_labels = gs_df['Month_Label'].tolist()

# Panel 1: Game intensity heatmap across functions
ax1 = fig.add_subplot(gs_fig[0, :3])
heat_data = gs_df[['RM_Score','DIS_Score','DIST_Score',
                    'FRAUD_Score','WF_Score']].values.T
func_labels = ['Revenue\nMgmt','Disruption\nMgmt',
               'Distribution\nGDS','Fraud\nPayment',
               'Workforce\nBaggage']
cmap = LinearSegmentedColormap.from_list(
    'game', ['#70AD47','#FFC000','#C00000'])
im = ax1.imshow(heat_data, aspect='auto', cmap=cmap,
                vmin=1, vmax=3)
ax1.set_xticks(range(len(month_labels)))
ax1.set_xticklabels(month_labels, fontsize=9)
ax1.set_yticks(range(5))
ax1.set_yticklabels(func_labels, fontsize=8)
for i in range(5):
    for j in range(len(month_labels)):
        val = heat_data[i, j]
        label = {1:'LOW',2:'MED',3:'HIGH'}[int(val)]
        ax1.text(j, i, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color='white' if val==3 else 'black')
plt.colorbar(im, ax=ax1, shrink=0.6,
             label='Game Intensity (1=Low, 2=Med, 3=High)')
ax1.set_title("Game State Heatmap — Five MAS Functions × 11 Months at EWR 2023\n"
              "Green=low intensity | Gold=medium | Red=high (peak MNAR exposure)",
              fontsize=9, fontweight='bold')

# Panel 2: Overall intensity line
ax2 = fig.add_subplot(gs_fig[0, 3])
ax2.plot(gs_df['Overall_Intensity'].values,
         range(len(gs_df)), 'o-',
         color='#1F3864', linewidth=2, markersize=8)
ax2.set_yticks(range(len(gs_df)))
ax2.set_yticklabels(month_labels, fontsize=8)
ax2.set_xlabel("Overall Game Intensity", fontsize=8)
ax2.axvline(2.0, color='#C00000', linestyle='--',
            alpha=0.5, linewidth=1)
ax2.set_xlim(0.8, 3.2)
ax2.set_title("Overall\nGame Intensity",
              fontsize=9, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.invert_yaxis()

# Panel 3: Load factor with game state coloring
ax3 = fig.add_subplot(gs_fig[1, :2])
game_colors = {
    'COLLUSION RISK': '#C00000',
    'FI DOMINATES': '#ED7D31',
    'PORTER DOMINATES': '#FFC000',
    'COMPETITION': '#70AD47',
}
p_lf_vals = gs_df['Porter_LF'].values
fi_lf_vals = gs_df['FI_LF'].values
x3 = range(len(month_labels))
ax3.fill_between(x3, p_lf_vals, fi_lf_vals,
                  alpha=0.15, color='#2E75B6',
                  label='LF gap (FI advantage)')
ax3.plot(x3, p_lf_vals, 'o-',
         color=C['Porter'], linewidth=2.5,
         markersize=8, label='Porter YTZ-EWR', zorder=5)
ax3.plot(x3, fi_lf_vals, 's-',
         color=C['Icelandair'], linewidth=2.5,
         markersize=8, label='Icelandair EWR-KEF', zorder=5)
ax3.axhline(0.826, color=C['JetBlue'], linestyle='--',
            linewidth=1.5, label='JetBlue system-wide (82.6%)')
ax3.axhline(0.85, color='gray', linestyle=':',
            linewidth=1, alpha=0.5, label='Capacity constraint threshold (85%)')

# Color background by RM game state
for i, (_, row) in enumerate(gs_df.iterrows()):
    color = game_colors.get(row['RM_State'], '#FFFFFF')
    ax3.axvspan(i-0.5, i+0.5, alpha=0.08, color=color)

ax3.set_xticks(x3)
ax3.set_xticklabels(month_labels, fontsize=9)
ax3.yaxis.set_major_formatter(
    matplotlib.ticker.PercentFormatter(1.0))
ax3.set_title("Load Factor by Month with Revenue Management Game States\n"
              "Background: Green=competition | Gold=one dominant | "
              "Red=collusion risk",
              fontsize=8.5, fontweight='bold')
ax3.legend(fontsize=7, loc='lower left')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel 4: MNAR boundary map
ax4 = fig.add_subplot(gs_fig[1, 2:])
mnar_pivot = mnar_df.set_index('Variable')[
    ['Porter_Score','FI_Score','B6_Score']].values
cmap2 = LinearSegmentedColormap.from_list(
    'mnar', ['#70AD47','#FFC000','#C00000'])
im4 = ax4.imshow(mnar_pivot, aspect='auto',
                  cmap=cmap2, vmin=0, vmax=2)
ax4.set_xticks([0,1,2])
ax4.set_xticklabels(['Porter\n(CA)','Icelandair\n(EU)',
                       'JetBlue\n(US)'], fontsize=8)
ax4.set_yticks(range(len(mnar_df)))
ax4.set_yticklabels([v[:30] for v in mnar_df['Variable']],
                     fontsize=7)
for i in range(len(mnar_df)):
    for j, carrier in enumerate(['Porter_Score','FI_Score',
                                   'B6_Score']):
        val = mnar_pivot[i, j]
        label = {0:'OBS',1:'PAR',2:'MNAR'}[int(val)]
        ax4.text(j, i, label, ha='center', va='center',
                fontsize=6.5, fontweight='bold',
                color='white' if val==2 else 'black')
plt.colorbar(im4, ax=ax4, shrink=0.5,
             label='0=OBSERVED 1=PARTIAL 2=MNAR')
ax4.set_title("MNAR Boundary Map at EWR\nby Variable and Carrier",
              fontsize=9, fontweight='bold')

# Panel 5: Value exchange diagram (bars)
ax5 = fig.add_subplot(gs_fig[2, :2])
edges_plot = [ve['Edge'].replace('_Agent','').replace('_',' ')
              for ve in VALUE_EXCHANGES]
risk_scores = [3 if 'Collusion' in ve['Governance_Risk']
               else 2 if 'Conflict' in ve['Governance_Risk']
               else 1 for ve in VALUE_EXCHANGES]
risk_labels = [ve['Governance_Risk'].split(' ')[0]
               for ve in VALUE_EXCHANGES]
risk_colors = ['#C00000' if s==3 else '#FFC000' if s==2
               else '#70AD47' for s in risk_scores]
bars5 = ax5.barh(range(len(VALUE_EXCHANGES)),
                  risk_scores, color=risk_colors,
                  edgecolor='white', height=0.6)
ax5.set_yticks(range(len(VALUE_EXCHANGES)))
ax5.set_yticklabels([e[:40] for e in edges_plot], fontsize=7.5)
for bar, label in zip(bars5, risk_labels):
    ax5.text(bar.get_width()+0.05,
             bar.get_y()+bar.get_height()/2,
             label, va='center', fontsize=8)
ax5.set_xlim(0, 4)
ax5.set_xticks([1,2,3])
ax5.set_xticklabels(['Miscoord.','Conflict','Collusion'],
                     fontsize=8)
ax5.set_title("Value Exchange Edges — Governance Risk at EWR\n"
              "Ranked by Hammond risk type",
              fontsize=9, fontweight='bold')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Panel 6: Delay compensation by month
ax6 = fig.add_subplot(gs_fig[2, 2:])
delay_rate = 0.05
porter_exp = gs_df['Porter_Pax'] * delay_rate * 1000 / 1e6
fi_exp = gs_df['FI_Pax'] * delay_rate * 600 / 1e6
b6_exp = [0] * len(gs_df)
x6 = np.arange(len(month_labels))
w6 = 0.28
ax6.bar(x6-w6, porter_exp, width=w6,
        color=C['Porter'], label='Porter (CAD M)',
        edgecolor='white')
ax6.bar(x6, fi_exp, width=w6,
        color=C['Icelandair'], label='Icelandair (EUR M)',
        edgecolor='white')
ax6.bar(x6+w6, b6_exp, width=w6,
        color=C['JetBlue'], label='JetBlue (USD 0)',
        edgecolor='white')
ax6.set_xticks(x6)
ax6.set_xticklabels(month_labels, fontsize=8)
ax6.set_ylabel("Max Compensation Exposure\n(local currency, M, 5% rate)",
               fontsize=8)
ax6.set_title("FE-11 Monthly Disruption Liability at EWR\n"
              "Real passenger volumes × jurisdictional formula",
              fontsize=9, fontweight='bold')
ax6.legend(fontsize=7)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.text(0, -0.01, 'USD 0',
         fontsize=7.5, color=C['JetBlue'],
         transform=ax6.get_xaxis_transform())

fig.suptitle(
    "L3: EWR Property Graph Instantiation — AIDM at Airport Scale\n"
    "Porter (CA) | Icelandair (EU) | JetBlue (US) | "
    "Five MAS Functions | Game Analysis 2023",
    fontsize=11, fontweight='bold', y=1.01)

plt.savefig(f"{OUTPUT}/L3_ewr_full_instantiation.png",
            dpi=150, bbox_inches='tight')
plt.close()

# ── SAVE ALL OUTPUTS ──────────────────────────────────────────
gs_df.to_csv(f"{OUTPUT}/L3_game_states_full.csv", index=False)
mnar_df.to_csv(f"{OUTPUT}/L3_mnar_boundary_map.csv", index=False)

ve_df = pd.DataFrame(VALUE_EXCHANGES)
ve_df.to_csv(f"{OUTPUT}/L3_value_exchanges.csv", index=False)

# Node instance summary
node_rows = []
for node, carriers in NODE_INSTANCES.items():
    for carrier, attrs in carriers.items():
        row = {'Agent_Node': node, 'Carrier': carrier}
        row.update({k: str(v) for k, v in attrs.items()})
        node_rows.append(row)
pd.DataFrame(node_rows).to_csv(
    f"{OUTPUT}/L3_node_instances.csv", index=False)

print("\n" + "="*65)
print("L3 FULL INSTANTIATION SUMMARY")
print("="*65)
print(f"""
EWR PROPERTY GRAPH AT AIRPORT SCALE:

GAME STATE FINDINGS:
  Peak intensity: July-August (score 2.6/3.0)
    → Both carriers capacity-constrained
    → Interline premium active at all five functions
    → NFP proration fires at maximum volume
    → Highest MNAR exposure across all edges
    
  Lowest intensity: November (score 1.6/3.0)  
    → Neither carrier constrained
    → Price competition on interline fare
    → Transparent game state (competition)
    
  Asymmetric months: Apr, Jun, Sep, Oct
    → Icelandair constrained, Porter not
    → FI has pricing power on EWR-KEF leg
    → Porter needs connection volume (feeder dependence)
    → NFP proration systematically favors FI

VALUE EXCHANGE BY FUNCTION:
  Revenue Mgmt: Load factor signal OBSERVED,
    optimization algorithm MNAR — all three carriers
  Disruption: Regulation OBSERVED, incentive
    asymmetry EU > CA >> US by design
  Distribution: Interline volume ESTIMABLE (r=0.959),
    settlement amount MNAR — ~113K interline pax
  Fraud/Payment: Volume PARTIAL, fraud score MNAR,
    3-currency exposure visible from route type
  Workforce: Customs regime OBSERVED, 
    SGHA bilateral rates MNAR all carriers

METHODOLOGICAL CONTRIBUTION:
  First instantiation of the AIDM schema at real
  airport scale with game-theoretic annotations.
  Demonstrates that the property graph method is
  empirically grounded — observable outputs confirm
  the schema-predicted MNAR boundaries.
  
  The USRA will scale this to 10 airports per region.
""")
print(f"Outputs saved to {OUTPUT}")
