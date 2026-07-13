"""
CIND820 Milestone 3 — Stage 3 Revised
Property Graph / HIN Analysis with Valid Interaction Edges
from Published MAS Function Schemas

Agent nodes drawn from:
  - Revenue Management: Bondoux et al. (2020) RMS agent architecture
  - Distribution/GDS: IATA ONE Order Transition Study (2021); Castro & Oliveira MASDIMA
  - Fraud/Payment: IMF Note 2026/004; arXiv:2606.17555 (Walusimbi & Ssentongo)
  - Workforce Mgmt: ScienceDirect 2025 crew pairing graph; IATA SGHA 2023

Corpus variables mapped to agent nodes and valid interaction edges.
Valuation type and MNAR status encoded as edge attributes.
Hammond et al. (2025) risk taxonomy annotated at each valid edge.

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT = "/mnt/user-data/outputs/eda_outputs"
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv("/mnt/user-data/outputs/corpus_with_transaction_points.csv")
df["Val_Primary"] = df["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")

# ── GAP STATEMENT ─────────────────────────────────────────────
print("="*65)
print("GAP STATEMENT (for M3 report Section 6.2)")
print("="*65)
print("""
The corpus contains no variables from the Passenger actor type in 
the Distribution/GDS, Fraud/Payment, or Workforce Management 
functions, and no Regulator actor variables in the 
Distribution/GDS, Fraud/Payment, or Workforce Management 
functions. This absence is a structural governance finding, not a 
data collection failure: the schema documents that define these 
functions — IATA SIS IS-XML, PCI DSS, IATA SGHA, and IATA SSIM — 
are written entirely from the perspective of institutional actors 
(carriers, vendors, industry bodies, and airports). The passenger 
and regulator do not contribute variable-defining power at the 
schema layer in these functions. In the full implementation, the 
passenger actor will be populated through a complaints and 
enforcement data layer (CTA adjudications, DOT consumer complaint 
database, EC 261 enforcement decisions, APPR determinations), and 
the regulatory actor will be populated through jurisdictional 
governance sources (Canada Labour Code enforcement, FLSA 
enforcement, EU-OPS compliance records). Additionally, Market and 
Option valuation types are absent from the Fraud/Payment function, 
and Income and Option types are absent from Workforce Management — 
reflecting the cost-and-risk-dominant schema architecture of those 
functions, where neither function creates a market mechanism or 
grants actors a contingent claim at the schema level. These gaps 
are named here and will be addressed systematically in Milestone 4 
through targeted corpus expansion and the introduction of the 
complaints enforcement data layer.
""")

# ── AGENT NODE DEFINITIONS FROM PUBLISHED SCHEMAS ──────────────
print("="*65)
print("AGENT NODE DEFINITIONS FROM PUBLISHED MAS SCHEMAS")
print("="*65)

AGENT_NODES = {

    # ── REVENUE MANAGEMENT (Bondoux et al. 2020) ──────────────
    "RMS_Agent": {
        "function": "Revenue Management",
        "schema_source": "Bondoux et al. 2020 (JRPM)",
        "actor_types": ["Carrier", "Vendor"],
        "description": "Demand forecast + bid price optimization agent. "
                       "Produces bid price as boundary output to distribution layer. "
                       "Internal algorithm is MNAR.",
        "primary_valuation": "Income/Option",
    },
    "Execution_Engine": {
        "function": "Revenue Management",
        "schema_source": "Bondoux et al. 2020 (JRPM)",
        "actor_types": ["Carrier", "Vendor"],
        "description": "Translates RMS bid price to booking class availability. "
                       "Interfaces between RMS and distribution channel.",
        "primary_valuation": "Market/Option",
    },
    "ATPCO_Filing_Agent": {
        "function": "Revenue Management",
        "schema_source": "ATPCO Composite; 14 CFR 221",
        "actor_types": ["Carrier", "Industry Body", "Regulator"],
        "description": "Fare filing and rule application agent. "
                       "Carrier files fares; ATPCO enforces taxonomy; "
                       "DOT regulates public disclosure floor.",
        "primary_valuation": "Market/Cost",
    },
    "Loyalty_Agent": {
        "function": "Revenue Management",
        "schema_source": "IATA IAWG IFRS 15 Loyalty Guide",
        "actor_types": ["Carrier", "Passenger"],
        "description": "Loyalty point issuance, redemption, and breakage agent. "
                       "Carrier controls breakage estimate (MNAR); "
                       "Passenger holds redemption option.",
        "primary_valuation": "Option/Income",
    },
    "Airport_Charge_Agent": {
        "function": "Revenue Management",
        "schema_source": "GTAA 2025; PANYNJ 2024; Heathrow CoU 2025/26",
        "actor_types": ["Airport", "Regulator", "Carrier"],
        "description": "Airport charge assessment agent. "
                       "Airport levies charges; Regulator caps (LHR/CAA H7); "
                       "Carrier pays.",
        "primary_valuation": "Cost/Market",
    },

    # ── DISTRIBUTION / GDS (IATA ONE Order Transition Study 2021) ─
    "Responsible_Airline_Agent": {
        "function": "Distribution / GDS",
        "schema_source": "IATA ONE Order Transition Study 2021",
        "actor_types": ["Carrier", "Industry Body"],
        "description": "Holds master order record. "
                       "Controls record in ONE Order; pushes to interline partners. "
                       "Actor control transfer point.",
        "primary_valuation": "Income/Market",
    },
    "GDS_Agent": {
        "function": "Distribution / GDS",
        "schema_source": "IATA ONE Order Transition Study 2021; Amadeus OpenAPI",
        "actor_types": ["Vendor", "Industry Body"],
        "description": "Distribution channel agent. "
                       "Holds master record in agency sales. "
                       "Translates carrier bid price to passenger-facing offer.",
        "primary_valuation": "Market/Income",
    },
    "Interline_Settlement_Agent": {
        "function": "Distribution / GDS",
        "schema_source": "IATA SIS IS-XML; IATA Interline Checklist",
        "actor_types": ["Industry Body", "Vendor", "Carrier"],
        "description": "Proration and clearing agent. "
                       "IATA defines NFP formula; Amadeus/SIS executes clearing; "
                       "Carriers receive settlement shares.",
        "primary_valuation": "Income/Cost",
    },
    "CDM_Airport_Agent": {
        "function": "Distribution / GDS",
        "schema_source": "AIDX v22.1; IATA SSIM",
        "actor_types": ["Airport", "Carrier", "Industry Body"],
        "description": "CDM milestone coordination agent. "
                       "Airport assigns TSAT; carrier requests TOBT; "
                       "Industry body defines CDM protocol.",
        "primary_valuation": "Cost/Option",
    },

    # ── DISRUPTION MANAGEMENT (Castro & Oliveira MASDIMA 2011-2024)
    "Aircraft_Recovery_Agent": {
        "function": "Disruption Management",
        "schema_source": "Castro & Oliveira MASDIMA (2011-2024)",
        "actor_types": ["Carrier", "Airport"],
        "description": "Manages aircraft routing recovery during IROPs. "
                       "Proposes recovery options subject to maintenance, "
                       "slot, and CDM constraints.",
        "primary_valuation": "Cost/Risk",
    },
    "Crew_Recovery_Agent": {
        "function": "Disruption Management",
        "schema_source": "Castro & Oliveira MASDIMA (2011-2024)",
        "actor_types": ["Carrier", "Vendor"],
        "description": "Manages crew legality and reassignment during IROPs. "
                       "Constrained by regulatory duty/rest limits "
                       "(injected as hard external constraints).",
        "primary_valuation": "Cost/Risk",
    },
    "Passenger_Recovery_Agent": {
        "function": "Disruption Management",
        "schema_source": "Castro & Oliveira MASDIMA (2011-2024)",
        "actor_types": ["Carrier", "Industry Body"],
        "description": "Manages passenger reprotection and compensation. "
                       "EC 261/APPR/DOT obligations define boundary. "
                       "Passenger actor absent from schema layer "
                       "(addressed in full implementation).",
        "primary_valuation": "Risk/Cost",
    },
    "Coordinator_Agent": {
        "function": "Disruption Management",
        "schema_source": "Castro & Oliveira MASDIMA (2011-2024)",
        "actor_types": ["Carrier"],
        "description": "Integrates aircraft/crew/passenger recovery proposals "
                       "via Generic Q-Negotiation (GQN) protocol. "
                       "Human expert approves final solution.",
        "primary_valuation": "Cost/Risk",
    },

    # ── FRAUD / PAYMENT (IMF 2026; arXiv:2606.17555) ──────────
    "Transaction_Stream_Agent": {
        "function": "Fraud / Payment",
        "schema_source": "IMF Note 2026/004; arXiv:2606.17555",
        "actor_types": ["Vendor", "Carrier"],
        "description": "Real-time card fraud and AML detection agent. "
                       "Operates on transaction stream (card fraud, ACH/wire, AML). "
                       "Fraud model weights are MNAR.",
        "primary_valuation": "Risk",
    },
    "Card_Network_Agent": {
        "function": "Fraud / Payment",
        "schema_source": "PCI DSS v4.0.1; IMF Note 2026/004",
        "actor_types": ["Vendor"],
        "description": "External card network authorization agent (Visa/MC). "
                       "Produces accept/decline decision. "
                       "Internal scoring algorithm MNAR to airline. "
                       "Environmental boundary node.",
        "primary_valuation": "Risk",
    },
    "Payment_Clearance_Agent": {
        "function": "Fraud / Payment",
        "schema_source": "PCI DSS v4.0.1; IATA SIS IS-XML",
        "actor_types": ["Vendor", "Carrier"],
        "description": "Manages settlement clearance and dispute resolution. "
                       "Connects payment authorization to interline settlement. "
                       "Chargeback/dispute handling.",
        "primary_valuation": "Income/Risk",
    },

    # ── WORKFORCE MANAGEMENT (ScienceDirect 2025; IATA SGHA 2023)
    "Crew_Scheduling_Agent": {
        "function": "Workforce Management",
        "schema_source": "ScienceDirect 2025 crew pairing graph; "
                         "Xu, Wandelt & Sun 2024",
        "actor_types": ["Carrier", "Vendor"],
        "description": "Solves Set Partitioning Problem for crew pairing. "
                       "Directed acyclic pairing graph: duty nodes, base nodes, "
                       "sequential feasibility edges. "
                       "Optimization algorithm (Jeppesen/CAE) is MNAR.",
        "primary_valuation": "Cost/Market",
    },
    "Ground_Handler_Agent": {
        "function": "Workforce Management",
        "schema_source": "IATA SGHA 2023",
        "actor_types": ["Vendor", "Carrier"],
        "description": "Ground handling service delivery agent. "
                       "Service charge rates bilaterally negotiated (MNAR). "
                       "Liability limit and minimum wage adjustment "
                       "define regulatory boundary.",
        "primary_valuation": "Cost/Risk",
    },
    "Schedule_Agent": {
        "function": "Workforce Management",
        "schema_source": "IATA SSIM Chapter 7",
        "actor_types": ["Carrier", "Industry Body"],
        "description": "Schedule publication and slot coordination agent. "
                       "SSIM defines message format; carrier publishes schedule; "
                       "Feeds crew scheduling as primary hard constraint.",
        "primary_valuation": "Cost/Market",
    },
}

print(f"\nAgent nodes defined: {len(AGENT_NODES)}")
for node, attrs in AGENT_NODES.items():
    print(f"  {node:<35} [{attrs['function']}] "
          f"actors={attrs['actor_types']}")

# ── VALID INTERACTION EDGES FROM PUBLISHED SCHEMAS ─────────────
print("\n" + "="*65)
print("VALID INTERACTION EDGES FROM PUBLISHED SCHEMAS")
print("="*65)

VALID_EDGES = [
    # ── REVENUE MANAGEMENT ────────────────────────────────────
    {
        "source": "ATPCO_Filing_Agent",
        "target": "RMS_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "Bondoux et al. 2020; ATPCO composite",
        "description": "Filed fares and fare rules constrain RMS bid price optimization. "
                       "Carrier cannot price below or outside filed fare boundaries.",
        "valuation_payload": "Market/Cost",
        "data_access": "PARTIAL",
        "hammond_risk": "Miscoordination",
        "hammond_note": "ATPCO Cat 25 algorithm MNAR — RMS cannot fully "
                        "observe the fare rule engine it must comply with",
    },
    {
        "source": "RMS_Agent",
        "target": "Execution_Engine",
        "edge_type": "PASSES_TO",
        "schema_source": "Bondoux et al. 2020 (JRPM)",
        "description": "RMS produces bid price; execution engine translates "
                       "to booking class availability.",
        "valuation_payload": "Income/Option",
        "data_access": "MNAR",
        "hammond_risk": "Conflict",
        "hammond_note": "Bid price algorithm MNAR to distribution layer. "
                        "Carrier income optimization vs vendor market price mechanism.",
    },
    {
        "source": "Execution_Engine",
        "target": "GDS_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "Bondoux et al. 2020; IATA ONE Order 2021",
        "description": "Execution engine sends booking class availability to GDS. "
                       "GDS constructs passenger-facing offer from availability signal.",
        "valuation_payload": "Market/Income",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Booking class availability is OBSERVED. "
                        "The translation from bid price to availability is the "
                        "information asymmetry boundary.",
    },
    {
        "source": "RMS_Agent",
        "target": "Loyalty_Agent",
        "edge_type": "TRIGGERS",
        "schema_source": "IATA IAWG IFRS 15 Loyalty Guide",
        "description": "Booking triggers loyalty point issuance at SSP rate. "
                       "Deferred revenue liability created on balance sheet.",
        "valuation_payload": "Option/Cost",
        "data_access": "MNAR",
        "hammond_risk": "Collusion",
        "hammond_note": "Carrier controls breakage estimate (MNAR). "
                        "Carrier income from breakage is aligned with vendor "
                        "distribution income — both benefit from passenger "
                        "point non-redemption. Passenger has no visibility.",
    },
    {
        "source": "GDS_Agent",
        "target": "Interline_Settlement_Agent",
        "edge_type": "TRIGGERS",
        "schema_source": "IATA SIS IS-XML; IATA ONE Order 2021",
        "description": "Completed booking triggers interline proration calculation "
                       "and SIS invoice creation.",
        "valuation_payload": "Income/Cost",
        "data_access": "MNAR",
        "hammond_risk": "Conflict",
        "hammond_note": "Each carrier maximizes proration share. "
                        "NFP formula MNAR — neither carrier sees other's weights.",
    },
    {
        "source": "Airport_Charge_Agent",
        "target": "RMS_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "GTAA 2025; PANYNJ 2024; Heathrow CoU 2025/26",
        "description": "Airport charges are external cost constraints on "
                       "carrier revenue optimization. LHR MAY cap is "
                       "regulatory price ceiling.",
        "valuation_payload": "Cost/Market",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Airport charges OBSERVED — governance benchmark. "
                        "CAA H7 price regulation produces contestability.",
    },

    # ── DISTRIBUTION / GDS ────────────────────────────────────
    {
        "source": "GDS_Agent",
        "target": "Responsible_Airline_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "IATA ONE Order Transition Study 2021",
        "description": "GDS passes confirmed booking to Responsible Airline "
                       "for order creation. Control transfer: GDS → Carrier.",
        "valuation_payload": "Income/Market",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Control transfer is documented in ONE Order. "
                        "OBSERVED — the order record is the public interface.",
    },
    {
        "source": "Responsible_Airline_Agent",
        "target": "Interline_Settlement_Agent",
        "edge_type": "TRIGGERS",
        "schema_source": "IATA SIS IS-XML; IATA Interline Checklist",
        "description": "Interline itinerary triggers proration and SIS billing. "
                       "Operating carrier and ticketing carrier proration split.",
        "valuation_payload": "Income/Cost",
        "data_access": "MNAR",
        "hammond_risk": "Conflict",
        "hammond_note": "Proration formula (NFP) MNAR. Competing income claims "
                        "between operating and ticketing carriers at settlement.",
    },
    {
        "source": "CDM_Airport_Agent",
        "target": "Aircraft_Recovery_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "AIDX v22.1; EUROCONTROL CDM concept",
        "description": "CDM milestones (TOBT/TSAT/AOBT) are external constraints "
                       "on disruption management recovery. Airport assigns TSAT; "
                       "carrier cannot depart before slot.",
        "valuation_payload": "Cost/Option",
        "data_access": "OBSERVED",
        "hammond_risk": "Miscoordination",
        "hammond_note": "CDM milestones OBSERVED but carrier TOBT request "
                        "may not reflect true readiness — strategic gaming "
                        "of slot coordination is documented in CDM literature.",
    },

    # ── DISRUPTION MANAGEMENT ─────────────────────────────────
    {
        "source": "Aircraft_Recovery_Agent",
        "target": "Crew_Recovery_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "Castro & Oliveira MASDIMA; Rodrigues AIRS 2025",
        "description": "Aircraft recovery solution constrains crew recovery "
                       "feasibility. Crew must be certified for reassigned aircraft "
                       "and legal under duty/rest rules.",
        "valuation_payload": "Cost/Risk",
        "data_access": "OBSERVED",
        "hammond_risk": "Miscoordination",
        "hammond_note": "Flow balance constraints couple aircraft and crew agents. "
                        "Separation produces feasibility failures — documented in "
                        "MASDIMA and AIRS literature.",
    },
    {
        "source": "Crew_Recovery_Agent",
        "target": "Passenger_Recovery_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "Castro & Oliveira MASDIMA",
        "description": "Crew recovery solution constrains passenger reprotection "
                       "options. Passenger rebooking depends on crew availability "
                       "for reprotection flight.",
        "valuation_payload": "Risk/Cost",
        "data_access": "PARTIAL",
        "hammond_risk": "Conflict",
        "hammond_note": "Carrier minimizes crew cost; passenger bears risk of "
                        "inadequate reprotection. Regulatory obligation "
                        "(EC 261/APPR) is boundary constraint — "
                        "OBSERVED but jurisdictionally asymmetric.",
    },
    {
        "source": "Aircraft_Recovery_Agent",
        "target": "Coordinator_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "Castro & Oliveira MASDIMA",
        "description": "Aircraft agent proposes recovery solution to Coordinator "
                       "via GQN negotiation protocol.",
        "valuation_payload": "Cost/Risk",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Negotiation protocol (GQN) is the defined interaction "
                        "mechanism. Human expert approval required at L4 maturity.",
    },
    {
        "source": "Crew_Recovery_Agent",
        "target": "Coordinator_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "Castro & Oliveira MASDIMA",
        "description": "Crew agent proposes recovery solution to Coordinator "
                       "via GQN negotiation protocol.",
        "valuation_payload": "Cost/Risk",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Same negotiation protocol. Crew legality is "
                        "OBSERVED boundary constraint.",
    },
    {
        "source": "Passenger_Recovery_Agent",
        "target": "Coordinator_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "Castro & Oliveira MASDIMA",
        "description": "Passenger agent proposes reprotection solution. "
                       "Coordinator integrates all three recovery proposals.",
        "valuation_payload": "Risk/Cost",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Passenger agent schema-absent in corpus — "
                        "addressed in full implementation.",
    },

    # ── FRAUD / PAYMENT ───────────────────────────────────────
    {
        "source": "Transaction_Stream_Agent",
        "target": "Card_Network_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "PCI DSS v4.0.1; IMF Note 2026/004",
        "description": "Transaction data sent to card network for authorization. "
                       "PAN and cardholder data cross CDE boundary.",
        "valuation_payload": "Risk",
        "data_access": "MNAR",
        "hammond_risk": "Conflict",
        "hammond_note": "Card network fraud scoring MNAR to airline. "
                        "Airline sees accept/decline only — not the scoring basis. "
                        "Non-deterministic: same input → different output.",
    },
    {
        "source": "Card_Network_Agent",
        "target": "Payment_Clearance_Agent",
        "edge_type": "PASSES_TO",
        "schema_source": "PCI DSS v4.0.1; ISO 20022 (pending M3)",
        "description": "Authorized transaction passes to clearance and settlement. "
                       "Card network accept/decline crosses back to airline CDE.",
        "valuation_payload": "Income/Risk",
        "data_access": "MNAR",
        "hammond_risk": "Miscoordination",
        "hammond_note": "Clearance amount and status MNAR. "
                        "Airline cannot observe card network fee extraction "
                        "from settlement flow.",
    },
    {
        "source": "Payment_Clearance_Agent",
        "target": "Interline_Settlement_Agent",
        "edge_type": "TRIGGERS",
        "schema_source": "IATA SIS IS-XML; PCI DSS v4.0.1",
        "description": "Payment clearance triggers interline settlement "
                       "for multi-carrier bookings. "
                       "Payment rail crosses to SIS/ICH clearing.",
        "valuation_payload": "Income/Cost",
        "data_access": "MNAR",
        "hammond_risk": "Collusion",
        "hammond_note": "Carrier and vendor both extract income from clearance. "
                        "Passenger has no visibility into combined fee structure "
                        "across payment + settlement layers.",
    },

    # ── WORKFORCE MANAGEMENT ──────────────────────────────────
    {
        "source": "Schedule_Agent",
        "target": "Crew_Scheduling_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "IATA SSIM Ch.7; Xu, Wandelt & Sun 2024",
        "description": "Published schedule is the primary input to crew "
                       "pairing optimization. Schedule defines flight legs "
                       "that form the nodes of the pairing graph.",
        "valuation_payload": "Cost/Market",
        "data_access": "OBSERVED",
        "hammond_risk": "None",
        "hammond_note": "Schedule is OBSERVED. Downstream crew optimization "
                        "algorithm (Jeppesen/CAE) is MNAR — the dark zone "
                        "begins at the pairing graph, not the schedule.",
    },
    {
        "source": "Crew_Scheduling_Agent",
        "target": "Crew_Recovery_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "Xu, Wandelt & Sun 2024; Castro & Oliveira MASDIMA",
        "description": "Planned crew roster constrains disruption recovery options. "
                       "Recovery must produce legally compliant modifications "
                       "of the planned roster.",
        "valuation_payload": "Cost/Risk",
        "data_access": "MNAR",
        "hammond_risk": "Miscoordination",
        "hammond_note": "Crew optimization algorithm MNAR. "
                        "Recovery agent cannot fully observe the cost function "
                        "that generated the roster it must modify.",
    },
    {
        "source": "Ground_Handler_Agent",
        "target": "Crew_Scheduling_Agent",
        "edge_type": "CONSTRAINS",
        "schema_source": "IATA SGHA 2023",
        "description": "Ground handling service delivery times constrain "
                       "minimum turnaround in crew scheduling. "
                       "SGHA service rates are bilateral (MNAR).",
        "valuation_payload": "Cost/Risk",
        "data_access": "MNAR",
        "hammond_risk": "Conflict",
        "hammond_note": "Carrier minimizes ground handling cost (MNAR rates); "
                        "Ground handler maximizes service fee. "
                        "Conflict at turnaround time boundary.",
    },
]

print(f"\nValid interaction edges defined: {len(VALID_EDGES)}")
for edge in VALID_EDGES:
    print(f"  {edge['source']:<35} --[{edge['edge_type']}]--> "
          f"{edge['target']:<35} "
          f"val={edge['valuation_payload']:<15} "
          f"access={edge['data_access']:<8} "
          f"hammond={edge['hammond_risk']}")

# ── CORPUS VARIABLE → AGENT NODE MAPPING ──────────────────────
print("\n" + "="*65)
print("CORPUS VARIABLE → AGENT NODE MAPPING")
print("="*65)

VARIABLE_AGENT_MAP = {
    # Revenue Management
    "Fare Basis Code":                    ["ATPCO_Filing_Agent", "GDS_Agent"],
    "Rule Number":                        ["ATPCO_Filing_Agent"],
    "Tariff Type (public/private)":       ["ATPCO_Filing_Agent"],
    "Category 25 — Fare By Rule":         ["ATPCO_Filing_Agent"],
    "Category 35 — Negotiated Fare":      ["ATPCO_Filing_Agent", "GDS_Agent"],
    "Category 16 — Penalties (display text)": ["ATPCO_Filing_Agent"],
    "Category 31 — Voluntary Changes":    ["ATPCO_Filing_Agent", "Passenger_Recovery_Agent"],
    "Category 33 — Voluntary Refunds":    ["ATPCO_Filing_Agent", "Passenger_Recovery_Agent"],
    "YQ/YR Surcharge":                    ["ATPCO_Filing_Agent", "RMS_Agent"],
    "HIP (Highest Intermediate Point)":   ["ATPCO_Filing_Agent", "Interline_Settlement_Agent"],
    "MPM (Maximum Permitted Mileage)":    ["ATPCO_Filing_Agent", "Interline_Settlement_Agent"],
    "Add-On / Proportional Fare":         ["ATPCO_Filing_Agent", "Interline_Settlement_Agent"],
    "Advance Purchase Requirement":       ["ATPCO_Filing_Agent", "RMS_Agent"],
    "Combinability Indicator":            ["ATPCO_Filing_Agent", "GDS_Agent"],
    "Offer":                              ["RMS_Agent", "Execution_Engine"],
    "Price (Base Amount)":                ["RMS_Agent", "Execution_Engine"],
    "Price (Total Amount)":               ["Execution_Engine", "GDS_Agent"],
    "grandTotal":                         ["GDS_Agent", "Execution_Engine"],
    "base (price)":                       ["RMS_Agent", "Execution_Engine"],
    "fees (SUPPLIER / TICKETING)":        ["GDS_Agent", "Execution_Engine"],
    "currency":                           ["GDS_Agent", "Interline_Settlement_Agent"],
    "numberOfBookableSeats":              ["RMS_Agent", "Execution_Engine"],
    "lastTicketingDate":                  ["Execution_Engine", "GDS_Agent"],
    "fareDetailsBySegment (cabin)":       ["GDS_Agent"],
    "fareDetailsBySegment (fareBasis)":   ["GDS_Agent", "Interline_Settlement_Agent"],
    "fareDetailsBySegment (brandedFare)": ["GDS_Agent"],
    "fareDetailsBySegment (class)":       ["GDS_Agent"],
    "RBD (Booking Class Code)":           ["Execution_Engine", "GDS_Agent"],
    "additionalServices (type/amount)":   ["GDS_Agent", "Execution_Engine"],
    "chargeableSeatNumber":               ["GDS_Agent"],
    "instantTicketingRequired":           ["GDS_Agent", "Execution_Engine"],
    "Cancel Restrictions (Allowed Modification Indicator)": ["RMS_Agent", "GDS_Agent"],
    "Change Restrictions (Allowed Modification Indicator)": ["RMS_Agent", "GDS_Agent"],
    "Transaction Price (IFRS 15 via IATA IAWG)": ["RMS_Agent", "Loyalty_Agent"],
    "Fare Owner":                         ["ATPCO_Filing_Agent", "Interline_Settlement_Agent"],
    "Fare Type":                          ["ATPCO_Filing_Agent", "GDS_Agent"],
    "Class of Service":                   ["ATPCO_Filing_Agent", "Interline_Settlement_Agent"],
    "Order":                              ["Responsible_Airline_Agent"],
    "Order Item (Grand Total Amount)":    ["Responsible_Airline_Agent", "Interline_Settlement_Agent"],
    "Price Guarantee Time Limit":         ["RMS_Agent", "GDS_Agent"],
    "Payment Time Limit":                 ["Responsible_Airline_Agent", "Payment_Clearance_Agent"],
    "EMD (Electronic Miscellaneous Document)": ["Responsible_Airline_Agent", "Interline_Settlement_Agent"],
    "Tax (Filed Amount)":                 ["Airport_Charge_Agent", "GDS_Agent"],
    "Tax Code":                           ["Airport_Charge_Agent", "GDS_Agent"],
    "Tax (Refund Indicator)":             ["Responsible_Airline_Agent", "Payment_Clearance_Agent"],
    "Penalty (Amount)":                   ["RMS_Agent", "Responsible_Airline_Agent"],
    "Fee (Amount)":                       ["RMS_Agent", "Responsible_Airline_Agent"],
    "Interline Settlement Information / Settlement Amount": ["Interline_Settlement_Agent"],
    "Interline Settlement Information / Method Code": ["Interline_Settlement_Agent"],
    "SSP — Points Component":             ["Loyalty_Agent"],
    "Deferred Revenue Liability":         ["Loyalty_Agent", "Payment_Clearance_Agent"],
    "Breakage Estimate":                  ["Loyalty_Agent"],
    "Loyalty Program Account Identifier": ["Loyalty_Agent"],
    "Loyalty Program Account Tier Code":  ["Loyalty_Agent"],
    "Loyalty Redemption (Loyalty Currency Amount)": ["Loyalty_Agent", "Transaction_Stream_Agent"],
    "Qualifying Balance Amount":          ["Loyalty_Agent"],
    "Partner Airline Agency Fee":         ["Interline_Settlement_Agent", "Loyalty_Agent"],
    # Airport charges
    "Landing Fee (CAD/tonne MTOW)":       ["Airport_Charge_Agent"],
    "Airport Improvement Fee (AIF) — Departing": ["Airport_Charge_Agent"],
    "Airport Improvement Fee (AIF) — Connecting": ["Airport_Charge_Agent"],
    "AIF in lieu (per seat)":             ["Airport_Charge_Agent"],
    "Inflation Adjustment Rate (YYZ)":    ["Airport_Charge_Agent"],
    "Public Landing Area Charge (EWR)":   ["Airport_Charge_Agent"],
    "Ramp and Apron Charge (EWR)":        ["Airport_Charge_Agent"],
    "Aircraft Parking / Storage Charge (EWR)": ["Airport_Charge_Agent"],
    "Federal Inspection Space Charge (EWR)": ["Airport_Charge_Agent"],
    "Passenger Facility Charge (PFC — EWR)": ["Airport_Charge_Agent"],
    "Non-Signatory Rate Premium (EWR)":   ["Airport_Charge_Agent"],
    "Maximum Allowable Yield (MAY/pax — LHR)": ["Airport_Charge_Agent"],
    "Airport Charge per Passenger (LHR)": ["Airport_Charge_Agent"],
    "K Factor (LHR)":                     ["Airport_Charge_Agent"],
    "CPI Adjustment (LHR)":               ["Airport_Charge_Agent"],
    "WACC (LHR — 4.01%)":                 ["Airport_Charge_Agent"],
    "H7 Regulatory Period (LHR)":         ["Airport_Charge_Agent"],
    "Aircraft Parking Fee (YYZ)":         ["Airport_Charge_Agent"],
    "Apron Fee (YYZ)":                    ["Airport_Charge_Agent"],
    "Slot Holding Fee (YYZ)":             ["Airport_Charge_Agent", "CDM_Airport_Agent"],
    "Airfield Charge per ATM (LHR)":      ["Airport_Charge_Agent"],
    "SAF Incentive (LHR)":                ["Airport_Charge_Agent"],
    # Distribution / settlement
    "Settlement Amount":                  ["Interline_Settlement_Agent"],
    "Proration Value":                    ["Interline_Settlement_Agent"],
    "Five Day Rate (FDR)":                ["Interline_Settlement_Agent"],
    "Billing Period":                     ["Interline_Settlement_Agent"],
    "Charge Category Code":               ["Interline_Settlement_Agent"],
    "Charge Code":                        ["Interline_Settlement_Agent"],
    "F12 File Total":                     ["Interline_Settlement_Agent"],
    "Reject Amount":                      ["Interline_Settlement_Agent", "Payment_Clearance_Agent"],
    "Dispute Amount":                     ["Interline_Settlement_Agent", "Payment_Clearance_Agent"],
    "Credit Note Value":                  ["Interline_Settlement_Agent", "Payment_Clearance_Agent"],
    "VAT Amount":                         ["Interline_Settlement_Agent"],
    "Auto-Billing Uplift":                ["Interline_Settlement_Agent"],
    "NFP (Neutral Fare Proration)":       ["Interline_Settlement_Agent"],
    "BilledMember":                       ["Interline_Settlement_Agent"],
    "InvoiceNumber":                      ["Interline_Settlement_Agent"],
    "BillingCategory":                    ["Interline_Settlement_Agent"],
    "LineItem/ChargeCode":                ["Interline_Settlement_Agent"],
    "LineItem/ChargeAmount":              ["Interline_Settlement_Agent"],
    "LineItem/TotalNetAmount":            ["Interline_Settlement_Agent"],
    "InvoiceSummary/TotalLineItemAmount": ["Interline_Settlement_Agent"],
    "InvoiceHeader/PaymentTerms/CurrencyCode": ["Interline_Settlement_Agent"],
    "Interline Billing Basis":            ["Interline_Settlement_Agent"],
    "Amount to be Prorated":              ["Interline_Settlement_Agent"],
    "Settlement Method":                  ["Interline_Settlement_Agent"],
    "Commission (Amount)":                ["GDS_Agent", "Interline_Settlement_Agent"],
    "Sector Type":                        ["Interline_Settlement_Agent", "Schedule_Agent"],
    # CDM / Flight
    "TOBT (Target Off-Block Time)":       ["CDM_Airport_Agent", "Aircraft_Recovery_Agent"],
    "TSAT (Target Start Up Approval Time)": ["CDM_Airport_Agent"],
    "AOBT (Actual Off-Block Time)":       ["CDM_Airport_Agent", "Aircraft_Recovery_Agent"],
    "ELDT (Estimated Landing Time)":      ["CDM_Airport_Agent"],
    "Gate Assignment":                    ["CDM_Airport_Agent", "Aircraft_Recovery_Agent"],
    "Flight Leg (Arrival Date Time)":     ["CDM_Airport_Agent", "Schedule_Agent"],
    "Flight Leg (Departure Date Time)":   ["CDM_Airport_Agent", "Schedule_Agent"],
    "Transport Service Leg (Status Code)":["CDM_Airport_Agent", "Aircraft_Recovery_Agent"],
    "FlightLegNotifRQ (message)":         ["CDM_Airport_Agent"],
    "Flight Leg Status (AIDX via SITA)":  ["CDM_Airport_Agent", "Aircraft_Recovery_Agent"],
    "Outbound / Inbound Flight Info":     ["CDM_Airport_Agent", "Crew_Recovery_Agent"],
    # Disruption
    "Delay Code (IATA AHM730)":           ["Aircraft_Recovery_Agent", "Crew_Recovery_Agent", "Passenger_Recovery_Agent"],
    "Delay Duration (minutes)":           ["Aircraft_Recovery_Agent", "Passenger_Recovery_Agent"],
    "Scheduled Time of Departure (STD)":  ["Schedule_Agent", "Aircraft_Recovery_Agent"],
    "Actual Time of Departure (ATD)":     ["Aircraft_Recovery_Agent", "CDM_Airport_Agent"],
    "Actual Time of Arrival (ATA)":       ["Aircraft_Recovery_Agent", "Ground_Handler_Agent"],
    "TPA_Extension (IROPs)":              ["Aircraft_Recovery_Agent", "Coordinator_Agent"],
    "Passenger Count (ops)":              ["Passenger_Recovery_Agent", "Ground_Handler_Agent"],
    # Fraud / Payment
    "PAN (Primary Account Number)":       ["Transaction_Stream_Agent", "Card_Network_Agent"],
    "Cardholder Name":                    ["Transaction_Stream_Agent"],
    "Expiration Date":                    ["Transaction_Stream_Agent"],
    "Service Code":                       ["Transaction_Stream_Agent"],
    "Full Track Data":                    ["Transaction_Stream_Agent", "Card_Network_Agent"],
    "Card Verification Code":             ["Transaction_Stream_Agent", "Card_Network_Agent"],
    "PIN / PIN Block":                    ["Transaction_Stream_Agent"],
    "Cardholder Data Environment (CDE)":  ["Transaction_Stream_Agent"],
    "Network Security Controls":          ["Transaction_Stream_Agent"],
    "Multi-Factor Authentication":        ["Transaction_Stream_Agent"],
    "Encryption Requirement":             ["Transaction_Stream_Agent"],
    "Payment Card (Card Number)":         ["Transaction_Stream_Agent", "Card_Network_Agent"],
    "Payment Information (Payment Status Code)": ["Payment_Clearance_Agent"],
    "Settlement Data (Net Clearance Amount)": ["Payment_Clearance_Agent"],
    "Clearance (Status Code)":            ["Payment_Clearance_Agent"],
    "Audit Log":                          ["Transaction_Stream_Agent", "Payment_Clearance_Agent"],
    "Incident Response Plan":             ["Transaction_Stream_Agent", "Payment_Clearance_Agent"],
    "PNR (Passenger Name Record)":        ["Responsible_Airline_Agent", "Transaction_Stream_Agent"],
    # Workforce
    "Flight Number (SSIM Chapter 7)":     ["Schedule_Agent", "Crew_Scheduling_Agent"],
    "Service Type Code (SSIM)":           ["Schedule_Agent", "Crew_Scheduling_Agent"],
    "Aircraft Type Code (SSIM)":          ["Schedule_Agent", "Crew_Scheduling_Agent"],
    "Period of Operation (SSIM)":         ["Schedule_Agent"],
    "Days of Operation (SSIM)":           ["Schedule_Agent"],
    "Departure Time / Arrival Time (SSIM)": ["Schedule_Agent", "Crew_Scheduling_Agent"],
    "Codeshare Indicator (SSIM)":         ["Schedule_Agent", "GDS_Agent"],
    "Release Sell Date (SSIM)":           ["Schedule_Agent", "GDS_Agent"],
    "Minimum Connect Time (SSIM)":        ["Schedule_Agent", "CDM_Airport_Agent"],
    "Service Charge Rate (SGHA)":         ["Ground_Handler_Agent"],
    "Payment Terms — 30 days (SGHA)":     ["Ground_Handler_Agent", "Interline_Settlement_Agent"],
    "Liability Limit (SGHA)":             ["Ground_Handler_Agent"],
    "Minimum Wage Adjustment (SGHA)":     ["Ground_Handler_Agent"],
    "Audit Right (SGHA)":                 ["Ground_Handler_Agent"],
    "Section 2 Passenger Services Fee (SGHA)": ["Ground_Handler_Agent", "Airport_Charge_Agent"],
    "Baggage Handling Rate (SGHA)":       ["Ground_Handler_Agent"],
    "Bag Tag Number":                     ["Ground_Handler_Agent"],
    "Bag Status (CHECKED_IN / LOADED / MISHANDLED)": ["Ground_Handler_Agent"],
    "Bag Event Code":                     ["Ground_Handler_Agent"],
    "frequent_flyer (bag event)":         ["Ground_Handler_Agent", "Loyalty_Agent"],
    # EMD
    "EMD Number":                         ["Responsible_Airline_Agent"],
    "Service Code (EMD)":                 ["Responsible_Airline_Agent"],
    "Ancillary Charge Amount":            ["Responsible_Airline_Agent", "GDS_Agent"],
    "EMD-A (Associated)":                 ["Responsible_Airline_Agent"],
    "EMD-S (Standalone)":                 ["Responsible_Airline_Agent"],
    "Change Fee (EMD)":                   ["Responsible_Airline_Agent", "Payment_Clearance_Agent"],
    "Upgrade Fee (EMD)":                  ["Responsible_Airline_Agent", "GDS_Agent"],
    "Refund Amount (EMD)":                ["Responsible_Airline_Agent", "Payment_Clearance_Agent"],
    "Coupon Status (EMD)":                ["Responsible_Airline_Agent"],
    "Excess Baggage Proration Method":    ["Interline_Settlement_Agent", "Ground_Handler_Agent"],
    "includedCheckedBags (quantity)":     ["GDS_Agent", "Ground_Handler_Agent"],
    "source (GDS)":                       ["GDS_Agent"],
}

# ── BUILD PROPERTY GRAPH ───────────────────────────────────────
print("\n" + "="*65)
print("BUILDING PROPERTY GRAPH")
print("="*65)

G = nx.DiGraph()

# Add agent nodes
for node_id, attrs in AGENT_NODES.items():
    G.add_node(node_id,
               node_type="agent",
               function=attrs["function"],
               schema_source=attrs["schema_source"],
               actor_types="; ".join(attrs["actor_types"]),
               primary_valuation=attrs["primary_valuation"],
               description=attrs["description"])

# Add valid interaction edges
for edge in VALID_EDGES:
    G.add_edge(
        edge["source"], edge["target"],
        edge_type=edge["edge_type"],
        schema_source=edge["schema_source"],
        valuation_payload=edge["valuation_payload"],
        data_access=edge["data_access"],
        hammond_risk=edge["hammond_risk"],
        hammond_note=edge["hammond_note"],
        description=edge["description"]
    )

# Add variable nodes and map to agents
mapped = 0
unmapped = []
for _, var_row in df.drop_duplicates("Variable Name").iterrows():
    vname = var_row["Variable Name"]
    agents = VARIABLE_AGENT_MAP.get(vname, [])
    if not agents:
        unmapped.append(vname)
        continue
    # Add variable node
    G.add_node(vname,
               node_type="variable",
               actor_type=var_row["Actor Type"],
               valuation_type=var_row["Val_Primary"] if "Val_Primary" in var_row else var_row["Valuation Type"],
               data_access=var_row["Data Access"],
               aidm_domain=var_row["AIDM Domain"],
               mas_function=var_row["MAS Function"])
    # Connect variable to each agent node
    for agent in agents:
        if agent in G.nodes:
            G.add_edge(vname, agent,
                       edge_type="BELONGS_TO",
                       valuation_payload=var_row["Val_Primary"] if "Val_Primary" in var_row else "",
                       data_access=var_row["Data Access"],
                       hammond_risk="")
            mapped += 1

print(f"\nGraph summary:")
print(f"  Total nodes: {G.number_of_nodes()}")
print(f"  Agent nodes: {len(AGENT_NODES)}")
print(f"  Variable nodes: {G.number_of_nodes() - len(AGENT_NODES)}")
print(f"  Total edges: {G.number_of_edges()}")
print(f"  Valid interaction edges: {len(VALID_EDGES)}")
print(f"  Variable→Agent mappings: {mapped}")
print(f"  Unmapped variables: {len(unmapped)}")
if unmapped:
    print(f"  Unmapped list: {unmapped[:10]}{'...' if len(unmapped)>10 else ''}")

# ── GOVERNANCE STRESS ANALYSIS ─────────────────────────────────
print("\n" + "="*65)
print("GOVERNANCE STRESS AT VALID INTERACTION EDGES")
print("="*65)

print(f"\n{'Source':<35} {'Target':<35} {'Val Payload':<18} "
      f"{'Access':<10} {'Hammond Risk'}")
print("-"*115)
for edge in VALID_EDGES:
    print(f"  {edge['source']:<35} → {edge['target']:<35} "
          f"{edge['valuation_payload']:<18} "
          f"{edge['data_access']:<10} "
          f"{edge['hammond_risk']}")

# Summarize by Hammond risk
print(f"\nHammond Risk Distribution at Valid Interaction Edges:")
risks = [e["hammond_risk"] for e in VALID_EDGES]
for risk in ["Conflict","Miscoordination","Collusion","None"]:
    count = risks.count(risk)
    pct = count/len(risks)*100
    print(f"  {risk:<20} n={count} ({pct:.0f}%)")

# MNAR at interaction edges
print(f"\nData Access at Valid Interaction Edges:")
access = [e["data_access"] for e in VALID_EDGES]
for a in ["OBSERVED","PARTIAL","MNAR"]:
    count = access.count(a)
    print(f"  {a:<15} n={count} ({count/len(access)*100:.0f}%)")

# Valuation conflicts at edges
print(f"\nCompeting Valuation Logic at Edges (source vs target primary valuation):")
for edge in VALID_EDGES:
    src_val = AGENT_NODES.get(edge["source"],{}).get("primary_valuation","")
    tgt_val = AGENT_NODES.get(edge["target"],{}).get("primary_valuation","")
    if src_val and tgt_val:
        src_primary = src_val.split("/")[0]
        tgt_primary = tgt_val.split("/")[0]
        competing = src_primary != tgt_primary
        if competing:
            print(f"  {edge['source'][:30]:<30} [{src_primary}] → "
                  f"{edge['target'][:30]:<30} [{tgt_primary}] ← COMPETING")

# Save outputs
edge_df = pd.DataFrame(VALID_EDGES)
edge_df.to_csv(f"{OUTPUT}/valid_interaction_edges.csv", index=False)

agent_df = pd.DataFrame([
    {"Agent": k, **{kk:vv for kk,vv in v.items() if kk != "description"}}
    for k,v in AGENT_NODES.items()
])
agent_df.to_csv(f"{OUTPUT}/agent_nodes.csv", index=False)

var_agent_df = pd.DataFrame([
    {"Variable": var, "Agent": agent}
    for var, agents in VARIABLE_AGENT_MAP.items()
    for agent in agents
])
var_agent_df.to_csv(f"{OUTPUT}/variable_agent_mapping.csv", index=False)

print(f"\nFiles saved:")
print(f"  valid_interaction_edges.csv ({len(edge_df)} edges)")
print(f"  agent_nodes.csv ({len(agent_df)} agents)")
print(f"  variable_agent_mapping.csv ({len(var_agent_df)} variable-agent pairs)")

# ── NETWORK VISUALIZATION ──────────────────────────────────────
FUNC_COLORS = {
    "Revenue Management":    "#1F3864",
    "Distribution / GDS":    "#0F6E56",
    "Disruption Management": "#534AB7",
    "Fraud / Payment":       "#C00000",
    "Workforce Management":  "#7030A0",
}
EDGE_COLORS = {
    "Conflict":        "#C00000",
    "Miscoordination": "#FFC000",
    "Collusion":       "#ED7D31",
    "None":            "#AAAAAA",
}
EDGE_STYLES = {
    "CONSTRAINS": "dashed",
    "PASSES_TO":  "solid",
    "TRIGGERS":   "dotted",
}

fig, ax = plt.subplots(figsize=(20, 14))

# Use spring layout on agent nodes only
agent_g = G.subgraph(list(AGENT_NODES.keys()))
pos = nx.spring_layout(agent_g, k=3.5, seed=42)

# Draw edges with Hammond risk colour
for edge in VALID_EDGES:
    src, tgt = edge["source"], edge["target"]
    if src in pos and tgt in pos:
        color = EDGE_COLORS.get(edge["hammond_risk"], "#AAAAAA")
        style = EDGE_STYLES.get(edge["edge_type"], "solid")
        ax.annotate("",
            xy=pos[tgt], xycoords='data',
            xytext=pos[src], textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>",
                color=color, lw=1.8,
                linestyle=style,
                connectionstyle="arc3,rad=0.08",
            )
        )
        # Edge label (valuation payload)
        mid = ((pos[src][0]+pos[tgt][0])/2,
               (pos[src][1]+pos[tgt][1])/2)
        ax.text(mid[0], mid[1],
                edge["valuation_payload"],
                fontsize=5.5, ha="center", va="center",
                color=color, alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.1",
                          fc="white", ec="none", alpha=0.7))

# Draw agent nodes
for node in AGENT_NODES:
    if node not in pos:
        continue
    func = AGENT_NODES[node]["function"]
    color = FUNC_COLORS.get(func, "#888888")
    x, y = pos[node]
    ax.scatter(x, y, s=1200, color=color, zorder=3,
               edgecolors="white", linewidths=1.5)
    label = node.replace("_Agent","").replace("_"," ")
    ax.text(x, y, label, ha="center", va="center",
            fontsize=6.5, fontweight="bold",
            color="white", zorder=4)

# Legend
legend_elements = [
    mpatches.Patch(color=FUNC_COLORS["Revenue Management"],
                   label="Revenue Management"),
    mpatches.Patch(color=FUNC_COLORS["Distribution / GDS"],
                   label="Distribution / GDS"),
    mpatches.Patch(color=FUNC_COLORS["Disruption Management"],
                   label="Disruption Management"),
    mpatches.Patch(color=FUNC_COLORS["Fraud / Payment"],
                   label="Fraud / Payment"),
    mpatches.Patch(color=FUNC_COLORS["Workforce Management"],
                   label="Workforce Management"),
    mpatches.Patch(color="#AAAAAA", label="No risk"),
    mpatches.Patch(color=EDGE_COLORS["Conflict"], label="Conflict risk"),
    mpatches.Patch(color=EDGE_COLORS["Miscoordination"],
                   label="Miscoordination risk"),
    mpatches.Patch(color=EDGE_COLORS["Collusion"], label="Collusion risk"),
]
ax.legend(handles=legend_elements, fontsize=8,
          loc="lower left", ncol=3,
          bbox_to_anchor=(0, -0.06))

ax.set_title(
    "Property Graph — MAS Agent Nodes and Valid Interaction Edges\n"
    "Node colour = MAS function | Edge colour = Hammond (2025) risk type | "
    "Edge label = valuation payload\n"
    "Solid = PASSES_TO | Dashed = CONSTRAINS | Dotted = TRIGGERS",
    fontsize=10, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/STAGE3_property_graph.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nProperty graph visualization saved: STAGE3_property_graph.png")

print("\n" + "="*65)
print("WHAT IS WORKING AND WHAT IS MISSING")
print("="*65)
print("""
WORKING:
  Revenue Management — all 6 actor types represented at some node.
    All 5 valuation types present. RMS→Execution→GDS chain complete.
    Airport charge governance benchmark clear.
    Loyalty collusion risk identifiable at RMS→Loyalty edge.

  Distribution / GDS — industry body and vendor dominant.
    Interline settlement chain complete (GDS→Responsible→Settlement).
    CDM constraint chain complete (CDM→Aircraft Recovery).
    MISSING: Regulator and Passenger actors — named gap,
    addressed in full implementation.

  Disruption Management — MASDIMA agent structure complete.
    All 4 MASDIMA agents represented.
    Aircraft→Crew→Passenger→Coordinator chain fully defined.
    CDM constraint from airport layer fully connected.
    Regulatory boundary (EC 261/APPR) identified as external
    constraint — OBSERVED but jurisdictionally asymmetric.

  Fraud / Payment — two-stream architecture (IMF 2026) represented.
    Transaction stream and card network boundary defined.
    70% MNAR rate — highest of any function — is the primary finding.
    MISSING: Industry Body, Airport, Regulator actors.
    MISSING: Market and Option valuation types.
    All gaps are governance findings, not data failures.

MISSING / GAPS FOR MILESTONE 4:
  1. ISO 20022 payment message layer (pacs.008/camt.029) —
     connects card network boundary to bank settlement rail.
     Adds Income and Market valuation types to Fraud/Payment.

  2. EC 261/APPR/DOT compensation obligation flows —
     adds Regulator actor to Disruption Management with
     jurisdictional asymmetry as explicit edge attribute.

  3. BSP/ICH clearing layer —
     connects Interline Settlement Agent to payment rail.
     Closes the financial perimeter.

  4. Complaints/enforcement data layer (CTA, DOT, EC 261) —
     adds Passenger actor with empirical data on
     fines, complaints upheld, and compensation outcomes.
     This is the full implementation passenger layer.

  5. NAV CANADA / FAA / EUROCONTROL CRCO ANSP charges —
     adds environmental boundary nodes for route charges.
     Connects Schedule Agent to ANSP cost layer.
""")
