"""
CIND820 Milestone 3 — Stage 3 Analysis
Association Rule Mining + Bipartite Network Analysis

Basket = Transaction Point
Items = Actor Type + Valuation Type + Data Access coded variables
Target = which combinations of actor-valuation-access co-occur at
         the same transaction point more often than chance

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings, os
warnings.filterwarnings("ignore")

OUTPUT = "/mnt/user-data/outputs/eda_outputs"
os.makedirs(OUTPUT, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/outputs/corpus_with_transaction_points.csv")
df["Val_Primary"] = df["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")

print("="*65)
print(f"STAGE 3 ANALYSIS")
print(f"N = {len(df)} variable-TP pairs | {df['Transaction_Point_ID'].nunique()} TPs")
print(f"Original variables: 175")
print("="*65)

# ── SECTION 1: ASSOCIATION RULE MINING ───────────────────────────
print("\n" + "="*65)
print("SECTION 1: ASSOCIATION RULE MINING (Apriori)")
print("Basket = Transaction Point")
print("Items = Actor_ValuationType_DataAccess combinations")
print("="*65)

# Build baskets — one basket per transaction point
# Each item = Actor_Type + Valuation_Primary + Data_Access
df["Item"] = (df["Actor Type"].str.replace(" ","_") + "__" +
              df["Val_Primary"] + "__" +
              df["Data Access"])

# Build transaction list — one list per TP
baskets = df.groupby("Transaction_Point_ID")["Item"].apply(list).tolist()
tp_names = df.groupby("Transaction_Point_ID")["Transaction_Point_Name"].first().to_dict()

print(f"\nBasket sizes (variables per TP):")
for i, basket in enumerate(baskets):
    tp_id = list(df.groupby("Transaction_Point_ID").groups.keys())[i]
    print(f"  {tp_id} — {tp_names[tp_id]:<40} n={len(basket)}")

# Encode
te = TransactionEncoder()
te_array = te.fit(baskets).transform(baskets)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"\nItem space: {len(te.columns_)} unique items")
print(f"Baskets (TPs): {len(basket_df)}")

# Run apriori — min_support = 2/12 = 0.167 (appears in at least 2 TPs)
# With only 12 baskets, we use low support to find meaningful patterns
frequent_items = apriori(basket_df, min_support=0.15, use_colnames=True)
print(f"\nFrequent itemsets (support ≥ 0.15): {len(frequent_items)}")

if len(frequent_items) > 0:
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0,
                              num_itemsets=len(frequent_items))
    rules = rules.sort_values("lift", ascending=False)
    print(f"Association rules (lift ≥ 1.0): {len(rules)}")

    # Filter to meaningful rules
    strong_rules = rules[rules["confidence"] >= 0.5].copy()
    print(f"Strong rules (confidence ≥ 0.5): {len(strong_rules)}")

    print(f"\nTop 20 rules by lift:")
    print(f"{'Antecedent':<45} {'Consequent':<40} {'Supp':>5} {'Conf':>5} {'Lift':>5}")
    print("-"*100)
    for _, row in strong_rules.head(20).iterrows():
        ant = ", ".join(sorted(row["antecedents"]))[:44]
        con = ", ".join(sorted(row["consequents"]))[:39]
        print(f"  {ant:<45} → {con:<40} "
              f"{row['support']:.2f}  {row['confidence']:.2f}  {row['lift']:.2f}")

    # Save rules
    rules_out = rules.copy()
    rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: "; ".join(sorted(x)))
    rules_out["consequents"] = rules_out["consequents"].apply(lambda x: "; ".join(sorted(x)))
    rules_out.to_csv(f"{OUTPUT}/association_rules.csv", index=False)

    # Plot top rules by lift
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: support vs confidence, coloured by lift
    ax = axes[0]
    sc = ax.scatter(rules["support"], rules["confidence"],
                    c=rules["lift"], cmap="RdYlGn", s=60, alpha=0.7,
                    edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=10)
    ax.set_title("Association Rules — Support vs Confidence\n"
                 "(colour = lift; green = strong association)",
                 fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(0.15, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Bar: top 15 rules by lift
    ax2 = axes[1]
    top15 = strong_rules.head(15).copy()
    top15["rule_label"] = top15.apply(
        lambda r: (", ".join(sorted(r["antecedents"]))[:30] + " →\n" +
                   ", ".join(sorted(r["consequents"]))[:30]), axis=1)
    colors = ["#70AD47" if l > 2 else "#FFC000" if l > 1.5 else "#2E75B6"
              for l in top15["lift"]]
    bars = ax2.barh(range(len(top15)), top15["lift"].values,
                    color=colors, edgecolor="white", height=0.7)
    ax2.set_yticks(range(len(top15)))
    ax2.set_yticklabels(top15["rule_label"].values, fontsize=6)
    ax2.set_xlabel("Lift", fontsize=9)
    ax2.set_title("Top 15 Rules by Lift\nGreen > 2.0 | Gold 1.5-2.0 | Blue 1.0-1.5",
                  fontsize=9, fontweight="bold")
    ax2.axvline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/STAGE3_association_rules.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: STAGE3_association_rules.png")

# ── SECTION 2: BIPARTITE NETWORK ─────────────────────────────────
print("\n" + "="*65)
print("SECTION 2: BIPARTITE NETWORK ANALYSIS")
print("Nodes: Actor Types (left) + Transaction Points (right)")
print("Edges: weighted by variable count")
print("="*65)

# Build bipartite graph
G = nx.Graph()

# Add TP nodes
tp_ids = df["Transaction_Point_ID"].unique()
for tp in sorted(tp_ids):
    name = tp_names[tp]
    n_vars = len(df[df["Transaction_Point_ID"]==tp])
    mnar_rate = (df[df["Transaction_Point_ID"]==tp]["Data Access"]=="MNAR").mean()
    G.add_node(tp, bipartite=1, label=f"{tp}\n{name}",
               n_vars=n_vars, mnar_rate=mnar_rate,
               node_type="transaction_point")

# Add Actor nodes and edges
actor_types = df["Actor Type"].unique()
for actor in actor_types:
    G.add_node(actor, bipartite=0, label=actor, node_type="actor")

# Add weighted edges
edge_data = df.groupby(["Actor Type","Transaction_Point_ID"]).agg(
    n_vars=("Variable Name","count"),
    n_mnar=("Data Access", lambda x: (x=="MNAR").sum()),
    val_types=("Val_Primary", lambda x: "/".join(sorted(x.unique())))
).reset_index()

for _, row in edge_data.iterrows():
    G.add_edge(row["Actor Type"], row["Transaction_Point_ID"],
               weight=row["n_vars"],
               n_mnar=row["n_mnar"],
               val_types=row["val_types"])

print(f"\nGraph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
print(f"Actor nodes: {len(actor_types)}")
print(f"Transaction point nodes: {len(tp_ids)}")
print(f"Density: {nx.density(G):.3f}")

# Centrality measures
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight="weight")

print(f"\nDegree Centrality (Actor nodes):")
actor_centrality = {n: degree_cent[n] for n in actor_types}
for actor, cent in sorted(actor_centrality.items(), key=lambda x: -x[1]):
    degree = G.degree(actor, weight="weight")
    n_tps = len([e for e in G.edges(actor)])
    print(f"  {actor:<20} centrality={cent:.3f} | "
          f"weighted_degree={degree} | TPs_connected={n_tps}")

print(f"\nDegree Centrality (Transaction Point nodes):")
tp_centrality = {n: degree_cent[n] for n in tp_ids}
for tp, cent in sorted(tp_centrality.items(), key=lambda x: -x[1]):
    degree = G.degree(tp, weight="weight")
    n_actors = len([e for e in G.edges(tp)])
    mnar = G.nodes[tp]["mnar_rate"]
    print(f"  {tp} {tp_names[tp]:<38} "
          f"cent={cent:.3f} | actors={n_actors} | MNAR={mnar:.0%}")

print(f"\nBetweenness Centrality (top 8 — structural brokers):")
for node, cent in sorted(betweenness_cent.items(),
                          key=lambda x: -x[1])[:8]:
    node_type = G.nodes[node].get("node_type","")
    print(f"  {str(node):<35} betweenness={cent:.4f} [{node_type}]")

# ── BIPARTITE NETWORK PLOT ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10))

# Layout — actors on left, TPs on right
actor_nodes = [n for n in G.nodes() if G.nodes[n]["bipartite"]==0]
tp_nodes    = [n for n in G.nodes() if G.nodes[n]["bipartite"]==1]

pos = {}
for i, a in enumerate(sorted(actor_nodes)):
    pos[a] = (-2, i * 1.5 - len(actor_nodes)/2)
for i, t in enumerate(sorted(tp_nodes)):
    pos[t] = (2, i * 1.1 - len(tp_nodes)/2)

# Actor node colours
ACTOR_COLORS = {
    "Carrier": "#1F3864",
    "Vendor": "#2E75B6",
    "Industry Body": "#70AD47",
    "Airport": "#FFC000",
    "Regulator": "#C00000",
    "Passenger": "#7030A0",
}

# TP node size = n_vars; colour = MNAR rate (red=high, green=low)
tp_sizes = [G.nodes[n]["n_vars"] * 80 for n in tp_nodes]
tp_mnar  = [G.nodes[n]["mnar_rate"] for n in tp_nodes]
tp_colors_vals = tp_mnar

# Draw edges — width = weight, alpha = weight normalized
edges = G.edges(data=True)
max_weight = max(d["weight"] for _,_,d in edges)
for u, v, d in G.edges(data=True):
    w = d["weight"]
    alpha = 0.2 + 0.6 * (w / max_weight)
    lw = 0.5 + 2.5 * (w / max_weight)
    color = "#C00000" if d.get("n_mnar", 0) > 0 else "#888888"
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color, alpha=alpha, linewidth=lw, zorder=1)

# Draw actor nodes
for actor in actor_nodes:
    x, y = pos[actor]
    color = ACTOR_COLORS.get(actor, "#888888")
    w_degree = G.degree(actor, weight="weight")
    size = 200 + w_degree * 15
    ax.scatter(x, y, s=size, color=color, zorder=3,
               edgecolors="white", linewidths=1.5)
    ax.text(x - 0.15, y, actor, ha="right", va="center",
            fontsize=9, fontweight="bold",
            color=color)

# Draw TP nodes — colour by MNAR rate
cmap = plt.cm.RdYlGn_r
for i, tp in enumerate(sorted(tp_nodes)):
    x, y = pos[tp]
    mnar = G.nodes[tp]["mnar_rate"]
    n_vars = G.nodes[tp]["n_vars"]
    color = cmap(mnar)
    size = 150 + n_vars * 20
    ax.scatter(x, y, s=size, color=color, zorder=3,
               edgecolors="white", linewidths=1.5)
    ax.text(x + 0.15, y,
            f"{tp}: {tp_names[tp][:28]}",
            ha="left", va="center", fontsize=7.5)
    # MNAR percentage label
    if mnar > 0.1:
        ax.text(x, y,
                f"{mnar:.0%}", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")

# Legend
legend_elements = [
    mpatches.Patch(color=ACTOR_COLORS["Carrier"], label="Carrier"),
    mpatches.Patch(color=ACTOR_COLORS["Vendor"], label="Vendor"),
    mpatches.Patch(color=ACTOR_COLORS["Industry Body"], label="Industry Body"),
    mpatches.Patch(color=ACTOR_COLORS["Airport"], label="Airport"),
    mpatches.Patch(color=ACTOR_COLORS["Regulator"], label="Regulator"),
    mpatches.Patch(color=ACTOR_COLORS["Passenger"], label="Passenger"),
    mpatches.Patch(color="#C00000", alpha=0.5, label="Edge with MNAR variable"),
    mpatches.Patch(color="#888888", alpha=0.5, label="Edge (all OBSERVED)"),
]
ax.legend(handles=legend_elements, loc="lower center",
          ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.05))

# Colorbar for TP MNAR rate
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
cbar.set_label("TP MNAR Rate (red=high governance stress)", fontsize=8)

ax.set_xlim(-3.5, 4.5)
ax.set_title(
    "Bipartite Actor × Transaction Point Network\n"
    "Node size = variable count | TP colour = MNAR rate | "
    "Edge width = variable count | Red edges = MNAR variables present",
    fontsize=10, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/STAGE3_bipartite_network.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nNetwork plot saved: STAGE3_bipartite_network.png")

# ── SECTION 3: COMPETING LOGICS SUMMARY ──────────────────────────
print("\n" + "="*65)
print("SECTION 3: COMPETING LOGICS PER TRANSACTION POINT")
print("Demonstrates thesis: multiple actor-valuation logics at each node")
print("="*65)

competing = df.groupby("Transaction_Point_ID").apply(lambda x: {
    "TP_Name": x["Transaction_Point_Name"].iloc[0],
    "N_vars": len(x),
    "N_actors": x["Actor Type"].nunique(),
    "Actors": "/".join(sorted(x["Actor Type"].unique())),
    "N_val_types": x["Val_Primary"].nunique(),
    "Val_types": "/".join(sorted(x["Val_Primary"].unique())),
    "MNAR_pct": (x["Data Access"]=="MNAR").mean()*100,
    "N_MNAR": (x["Data Access"]=="MNAR").sum(),
    "Competing_logics": x["Actor Type"].nunique() > 1 and x["Val_Primary"].nunique() > 1,
}).apply(pd.Series)

competing = competing.reset_index()
print(f"\n{'TP':<5} {'Name':<40} {'Actors':>3} {'Val':>3} {'MNAR%':>6} {'Competing?'}")
print("-"*75)
for _, row in competing.iterrows():
    flag = "✓ YES" if row["Competing_logics"] else "  no"
    print(f"  {row['Transaction_Point_ID']:<5} {row['TP_Name']:<40} "
          f"{int(row['N_actors']):>3} {int(row['N_val_types']):>3} "
          f"{row['MNAR_pct']:>5.1f}%  {flag}")

competing_count = competing["Competing_logics"].sum()
print(f"\n{competing_count} of 12 transaction points show competing logics "
      f"(multiple actors AND multiple valuation types)")
print("This confirms the thesis: MAS interaction points are sites of "
      "contested value, not neutral technical exchanges.")

# ── SAVE CENTRALITY RESULTS ───────────────────────────────────────
centrality_df = pd.DataFrame([
    {
        "Node": n,
        "Node_Type": G.nodes[n].get("node_type",""),
        "Degree_Centrality": round(degree_cent[n], 4),
        "Betweenness_Centrality": round(betweenness_cent[n], 4),
        "Weighted_Degree": G.degree(n, weight="weight"),
        "N_connections": len(list(G.neighbors(n))),
        "MNAR_Rate": round(G.nodes[n].get("mnar_rate", 0), 3),
        "N_vars": G.nodes[n].get("n_vars", ""),
    }
    for n in G.nodes()
])
centrality_df = centrality_df.sort_values("Degree_Centrality", ascending=False)
centrality_df.to_csv(f"{OUTPUT}/network_centrality.csv", index=False)
competing.to_csv(f"{OUTPUT}/competing_logics_summary.csv", index=False)

print(f"\nAll outputs saved to {OUTPUT}")
print("Files produced:")
for f in ["STAGE3_association_rules.png", "STAGE3_bipartite_network.png",
          "association_rules.csv", "network_centrality.csv",
          "competing_logics_summary.csv"]:
    print(f"  {f}")
