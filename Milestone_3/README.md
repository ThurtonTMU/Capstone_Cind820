# CIND820 — Milestone 3: Proof of Concept Execution
**Valuation-Layer Analysis of Multi-Agent Governance in Civil Aviation**

Marie-Louise Thurton | Toronto Metropolitan University | July 2026
Supervisor: Dr. Ceni Babaoglu

---

## Project Overview

This repository contains the Milestone 3 proof of concept for CIND820 Big Data Analytics Capstone. The research examines how value is structured and captured across airline multi-agent system (MAS) transaction points — the specific nodes where variables contributed by different actors interact to produce valuation outcomes.

**Central thesis:** MAS interaction points are sites of contested value, not neutral technical exchanges. Governance stress concentrates where actor-contributed variables are structurally withheld (MNAR) from outside scrutiny.

**Three research questions:**
- RQ1: What valuation signals are detectable in publicly available aviation schema data?
- RQ2: Do those signals differ across jurisdictions in ways consistent with the enforceability-profitability governance framework?
- RQ3: Which methodological approach most effectively reveals optimization patterns and their governance implications?

---

## Repository Contents

### Data
| File | Description |
|------|-------------|
| `corpus_with_transaction_points.csv` | 266 variable-transaction point pairs (175 variables × 12 TPs) — primary analytical dataset |
| `bipartite_edge_list.csv` | Actor × transaction point edge list weighted by variable count and valuation type |
| `transaction_point_definitions.csv` | 12 transaction point definitions with authoritative ERD sources |

### Code
| File | Description |
|------|-------------|
| `map_transaction_points.py` | Assigns all 175 corpus variables to transaction points; produces expanded corpus and edge list |
| `stage3_analysis.py` | Stage 3: association rule mining (apriori) and bipartite network analysis |

### Outputs
| File | Description |
|------|-------------|
| `STAGE3_bipartite_network.png` | Bipartite Actor × Transaction Point network — key analytical output |
| `STAGE3_association_rules.png` | Association rules — support vs confidence and top rules by lift |
| `association_rules.csv` | Full association rule set (16,680 rules) |
| `network_centrality.csv` | Degree and betweenness centrality for all network nodes |
| `competing_logics_summary.csv` | Competing logics analysis — actor and valuation type counts per TP |

---

## How to Run

### Environment
```
Python 3.10+
pandas
numpy
matplotlib
scipy
mlxtend
networkx
```

Install dependencies:
```bash
pip install pandas numpy matplotlib scipy mlxtend networkx
```

### Pipeline

**Step 1 — Transaction point mapping:**
```bash
python map_transaction_points.py
```
Reads `corpus_variable_registry.csv` (from M2), produces:
- `corpus_with_transaction_points.csv`
- `bipartite_edge_list.csv`
- `transaction_point_definitions.csv`

**Step 2 — Stage 3 analysis:**
```bash
python stage3_analysis.py
```
Reads `corpus_with_transaction_points.csv`, produces all Stage 3 outputs.

---

## Key Findings

**All 12 of 12 transaction points show competing logics** — multiple actor types and multiple valuation types present simultaneously. The thesis is confirmed at the structural level.

**Vendor betweenness centrality = 0.491** — the highest of any actor, despite carriers contributing the most variables (n=125). Vendors are the structural brokers of the MAS. They connect otherwise-disconnected transaction points and operate with the second-highest MNAR concentration in the corpus.

**MNAR governance stress ranking:**
- TP01 Fare Construction — 78.6% MNAR (highest)
- TP09 Payment Authorisation — 66.7% MNAR
- TP10 Payment Clearance — 46.2% MNAR
- TP05 Interline Settlement — 44.4% MNAR
- TP12 Airport Charge Assessment — 3.8% MNAR (governance benchmark)

**TP07 Delay Event** directly demonstrates RQ2 — the same AIDM delay code variable produces three different financial outcomes: EC 261 compensation (EU), APPR compensation (Canada), no federal obligation (US).

---

## Methodological Notes

**Missingness framework:** Data access is coded as OBSERVED (n=125, 71.4%) or MNAR (n=50, 28.6%) under Rubin's (1976) missing data taxonomy. MNAR variables are structurally withheld by the controlling actor. Manski's (2003) partial identification framework is applied — the schema definition provides the structural bound.

**Association rule scale constraint:** With 12 transaction point baskets, lift scores ceiling at 6.0 — an artifact of fixed item composition within each basket rather than genuine cross-basket discovery. The pipeline is validated as functional; rule quality scales with corpus expansion in the full implementation. Future work will restructure baskets around the MAS interaction sequences documented in the literature (MASDIMA directed agent chain, Bondoux et al. RMS execution sequence) so that rules surface patterns consistent with the causal architecture of each MAS function.

**Semi-structured design:** Transaction points are defined theoretically from authoritative process models (IATA ONE Order lifecycle, AIDX CDM milestones, PCI DSS payment flow, IATA SGHA), then populated empirically from the coded corpus. Variables may appear at multiple transaction points, reflecting reality.

---

## AI Use Declaration

Generative AI (Claude, Anthropic) was used to support implementation of human-designed research architecture, including Python code scaffolding, literature search assistance, schema field extraction, and document preparation. All research design, analytical framing, variable selection, codebook decisions, transaction point definitions, and interpretive judgments were made by the author. All AI-assisted outputs were reviewed, revised, and critically integrated before inclusion.

---

## References

- Agrawal, R. & Srikant, R. (1994). Fast algorithms for mining association rules. *Proceedings of VLDB*, 487–499.
- Alqithami, S. (2025). Adaptive accountability in networked multi-agent systems. *AAAI/ACM AIES 2025*.
- Bondoux, N., Nguyen, A., Fiig, T. & Acuna-Agost, R. (2020). Reinforcement learning applied to airline revenue management. *Journal of Revenue and Pricing Management*, 19, 332–348.
- Castro, A.J.M. & Oliveira, E. (2011–2024). MASDIMA: Multi-agent system for airline disruption management. University of Porto LIACC.
- Hammond, L. et al. (2025). Multi-agent risks from advanced AI. Technical Report #1. Cooperative AI Foundation.
- Manski, C.F. (2003). *Partial Identification of Probability Distributions*. Springer.
- Newman, M.E.J. (2010). *Networks: An Introduction*. Oxford University Press.
- Rodrigues, F. et al. (2025). Disruption management in airline operations. arXiv:2510.26831.
- Rubin, D.B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
