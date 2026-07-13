"""
Transaction Point Mapping
Assigns all 175 corpus variables to MAS interaction nodes.
Variables can appear at multiple transaction points — this reflects reality.
The basket for association rule mining = the transaction point.
The edge list = actor × transaction point for bipartite network.

Transaction Points derived from:
  - IATA ONE Order lifecycle (TP01-TP05)
  - AIDX / CDM milestone sequence (TP06-TP08)
  - ISO 20022 / PCI DSS payment flow (TP09-TP10)
  - IATA SSIM / SGHA workforce flow (TP11-TP12)

Author: Marie-Louise Thurton, Toronto Metropolitan University
"""

import pandas as pd
import numpy as np

# ── TRANSACTION POINT DEFINITIONS ────────────────────────────────
TRANSACTION_POINTS = {
    # ONE Order lifecycle
    "TP01": {
        "name": "Fare Construction",
        "description": "ATPCO fare filing, rule application, fare basis code assignment. "
                       "Carrier constructs fares; ATPCO/industry body enforces taxonomy; "
                       "regulator sets disclosure floor (14 CFR 221). "
                       "Competing logics: carrier income optimization vs regulatory "
                       "market transparency obligation.",
        "mas_function": "Revenue Management",
        "authoritative_erd": "ATPCO Category Taxonomy; 14 CFR 221",
    },
    "TP02": {
        "name": "Offer Pricing & Display",
        "description": "RMS bid price → NDC/GDS offer construction → passenger-facing price. "
                       "Carrier RM agent produces bid price (MNAR boundary); "
                       "Amadeus/GDS translates to bookable offer (OBSERVED); "
                       "passenger sees grandTotal. Three actors, three logics: "
                       "carrier income, vendor market, passenger option.",
        "mas_function": "Revenue Management / Distribution",
        "authoritative_erd": "IATA ONE Order Offer lifecycle; Amadeus OpenAPI",
    },
    "TP03": {
        "name": "Order Confirmation & Ticketing",
        "description": "Passenger confirms offer → Order created → EMD issued for ancillaries → "
                       "Tax and fee obligations triggered. "
                       "Carrier income recognized; regulator cost imposed; "
                       "payment time limit option assigned.",
        "mas_function": "Revenue Management",
        "authoritative_erd": "IATA ONE Order Order lifecycle; IATA EMD Resolution 725",
    },
    "TP04": {
        "name": "Ancillary Service Charge",
        "description": "Separate performance obligation under IFRS 15. "
                       "EMD issued for upgrade, change fee, excess baggage. "
                       "Carrier income; passenger risk (cancellation/change). "
                       "Competing logics: carrier revenue maximization vs "
                       "passenger contestability of fee.",
        "mas_function": "Revenue Management",
        "authoritative_erd": "IATA EMD Resolution 725; IATA ONE Order",
    },
    "TP05": {
        "name": "Interline Proration & Settlement",
        "description": "Through-fare split between operating and ticketing carriers via NFP/SIS. "
                       "Invoice created, disputed, cleared through ICH/BSP. "
                       "Highest MNAR concentration in corpus. "
                       "Competing logics: each carrier maximizes proration share; "
                       "Amadeus/SIS controls the formula.",
        "mas_function": "Distribution / GDS",
        "authoritative_erd": "IATA SIS IS-XML Handbook; IATA Interline Checklist",
    },
    # CDM / Disruption Management
    "TP06": {
        "name": "Schedule Publication & Slot Coordination",
        "description": "SSIM schedule published → slot allocated → CDM milestones assigned. "
                       "Carrier publishes schedule (OBSERVED); airport assigns slot (OBSERVED); "
                       "crew scheduling ingests as hard constraint. "
                       "Competing logics: carrier operational preference (TOBT) vs "
                       "airport capacity management (TSAT).",
        "mas_function": "Distribution / GDS / Workforce Management",
        "authoritative_erd": "IATA SSIM Chapter 7; AIDX v22.1 CDM milestones",
    },
    "TP07": {
        "name": "Delay Event & IROPs Trigger",
        "description": "ATD deviates from STD → Delay code assigned (AHM730) → "
                       "IROPs rebooking triggered → Compensation obligation created. "
                       "Same AIDM variable (delay code) creates different regulatory "
                       "obligations across jurisdictions: EC 261 (EU), APPR (Canada), "
                       "none federally (US). "
                       "Competing logics: carrier risk minimization vs "
                       "passenger/regulator compensation right.",
        "mas_function": "Disruption Management",
        "authoritative_erd": "AIDX v22.1; EC 261/2004; APPR SOR/2019-150",
    },
    "TP08": {
        "name": "Ground Handling Service Delivery",
        "description": "SGHA activates at departure/arrival → ground handler performs services → "
                       "invoice raised → liability limit invoked if mishandling. "
                       "Competing logics: carrier cost minimization (SGHA bilateral rates MNAR) "
                       "vs handler revenue; minimum wage regulation as external constraint.",
        "mas_function": "Workforce Management",
        "authoritative_erd": "IATA SGHA 2023; IATA SSIM",
    },
    # Payment / Fraud
    "TP09": {
        "name": "Payment Authorisation & Fraud Scoring",
        "description": "Passenger initiates payment → PAN enters CDE → "
                       "carrier fraud model scores → card network authorises (MNAR to carrier). "
                       "Most MNAR-dense transaction point in corpus. "
                       "Competing logics: carrier fraud risk minimization vs "
                       "passenger payment right vs card network fee extraction.",
        "mas_function": "Fraud / Payment",
        "authoritative_erd": "PCI DSS v4.0.1; ISO 20022 pacs.008 (pending M3)",
    },
    "TP10": {
        "name": "Payment Clearance & Dispute",
        "description": "Settlement cleared through acquirer → chargeback/dispute if contested → "
                       "credit note or reject amount issued. "
                       "Competing logics: carrier income recognition vs "
                       "passenger chargeback right vs acquirer fee extraction.",
        "mas_function": "Fraud / Payment / Distribution",
        "authoritative_erd": "PCI DSS v4.0.1; IATA SIS IS-XML",
    },
    # Loyalty
    "TP11": {
        "name": "Loyalty Issuance & Redemption",
        "description": "Points issued at booking (SSP under IFRS 15) → "
                       "deferred revenue liability created → points redeemed or broken. "
                       "Competing logics: carrier breakage income optimization (MNAR actuarial) "
                       "vs passenger expectation of redeemable value.",
        "mas_function": "Revenue Management",
        "authoritative_erd": "IATA IAWG IFRS 15 Loyalty Guide",
    },
    # Airport Charges
    "TP12": {
        "name": "Airport Charge Assessment",
        "description": "Aircraft lands / passenger boards → airport charges assessed → "
                       "carrier invoiced. Three jurisdictions, three governance regimes: "
                       "GTAA (unregulated), PANYNJ (signatory/non-signatory differential), "
                       "LHR (CAA H7 price cap). "
                       "Competing logics: airport income maximization vs "
                       "carrier cost minimization vs regulator market price cap.",
        "mas_function": "Revenue Management",
        "authoritative_erd": "GTAA Fees 2025; PANYNJ 2024; Heathrow CoU 2025/26",
    },
}

# ── VARIABLE → TRANSACTION POINT MAPPING ────────────────────────
# Format: "Variable Name": ["TP01", "TP02", ...]
# Variables can map to multiple transaction points

VARIABLE_TP_MAP = {
    # ── FARE CONSTRUCTION (TP01) ──────────────────────────────
    "Fare Basis Code":                          ["TP01", "TP02", "TP05"],
    "Rule Number":                              ["TP01", "TP02"],
    "Tariff Type (public/private)":             ["TP01", "TP02"],
    "Category 25 — Fare By Rule":               ["TP01"],
    "Category 35 — Negotiated Fare":            ["TP01", "TP02"],
    "Category 16 — Penalties (display text)":   ["TP01", "TP03"],
    "Category 31 — Voluntary Changes":          ["TP01", "TP03", "TP04"],
    "Category 33 — Voluntary Refunds":          ["TP01", "TP03", "TP04"],
    "YQ/YR Surcharge":                          ["TP01", "TP02"],
    "HIP (Highest Intermediate Point)":         ["TP01", "TP05"],
    "MPM (Maximum Permitted Mileage)":          ["TP01", "TP05"],
    "Add-On / Proportional Fare":               ["TP01", "TP05"],
    "Advance Purchase Requirement":             ["TP01", "TP02"],
    "Combinability Indicator":                  ["TP01", "TP02"],

    # ── OFFER PRICING & DISPLAY (TP02) ───────────────────────
    "Offer":                                    ["TP02"],
    "Price (Base Amount)":                      ["TP02", "TP03"],
    "Price (Total Amount)":                     ["TP02", "TP03"],
    "grandTotal":                               ["TP02", "TP03"],
    "base (price)":                             ["TP02", "TP03"],
    "fees (SUPPLIER / TICKETING)":              ["TP02", "TP03"],
    "currency":                                 ["TP02", "TP03", "TP05"],
    "numberOfBookableSeats":                    ["TP02"],
    "lastTicketingDate":                        ["TP02", "TP03"],
    "fareDetailsBySegment (cabin)":             ["TP02"],
    "fareDetailsBySegment (fareBasis)":         ["TP02", "TP05"],
    "fareDetailsBySegment (brandedFare)":       ["TP02"],
    "fareDetailsBySegment (class)":             ["TP02"],
    "RBD (Booking Class Code)":                 ["TP02", "TP05"],
    "additionalServices (type/amount)":         ["TP02", "TP04"],
    "chargeableSeatNumber":                     ["TP02", "TP04"],
    "source (GDS)":                             ["TP02", "TP05"],
    "instantTicketingRequired":                 ["TP02", "TP03"],
    "Cancel Restrictions (Allowed Modification Indicator)": ["TP02", "TP03"],
    "Change Restrictions (Allowed Modification Indicator)": ["TP02", "TP03"],
    "Release Sell Date (SSIM)":                 ["TP02", "TP06"],
    "Transaction Price (IFRS 15 via IATA IAWG)":["TP02", "TP03"],
    "Fare Owner":                               ["TP02", "TP05"],
    "Fare Type":                                ["TP02", "TP05"],
    "Class of Service":                         ["TP02", "TP05"],

    # ── ORDER CONFIRMATION & TICKETING (TP03) ────────────────
    "Order":                                    ["TP03"],
    "Order Item (Grand Total Amount)":          ["TP03"],
    "Price Guarantee Time Limit":               ["TP02", "TP03"],
    "Payment Time Limit":                       ["TP03", "TP09"],
    "Ticket Type":                              ["TP03", "TP05"],
    "Journey Type":                             ["TP03", "TP05"],
    "EMD (Electronic Miscellaneous Document)":  ["TP03", "TP04"],
    "Tax (Filed Amount)":                       ["TP03", "TP12"],
    "Tax Code":                                 ["TP03", "TP12"],
    "Tax (Refund Indicator)":                   ["TP03", "TP10"],
    "Penalty (Amount)":                         ["TP03", "TP04"],
    "Fee (Amount)":                             ["TP03", "TP04"],
    "Interline Settlement Information / Settlement Amount": ["TP03", "TP05"],
    "Interline Settlement Information / Method Code":       ["TP03", "TP05"],

    # ── ANCILLARY SERVICE CHARGE (TP04) ──────────────────────
    "EMD Number":                               ["TP04"],
    "Service Code (EMD)":                       ["TP04"],
    "Ancillary Charge Amount":                  ["TP04"],
    "EMD-A (Associated)":                       ["TP04"],
    "EMD-S (Standalone)":                       ["TP04"],
    "Change Fee (EMD)":                         ["TP04", "TP10"],
    "Upgrade Fee (EMD)":                        ["TP04"],
    "Refund Amount (EMD)":                      ["TP04", "TP10"],
    "Coupon Status (EMD)":                      ["TP04", "TP10"],
    "includedCheckedBags (quantity)":           ["TP02", "TP04"],
    "Excess Baggage Proration Method":          ["TP04", "TP05"],

    # ── INTERLINE PRORATION & SETTLEMENT (TP05) ──────────────
    "Settlement Amount":                        ["TP05"],
    "Proration Value":                          ["TP05"],
    "Five Day Rate (FDR)":                      ["TP05"],
    "Billing Period":                           ["TP05"],
    "Charge Category Code":                     ["TP05"],
    "Charge Code":                              ["TP05"],
    "F12 File Total":                           ["TP05"],
    "Reject Amount":                            ["TP05", "TP10"],
    "Dispute Amount":                           ["TP05", "TP10"],
    "Credit Note Value":                        ["TP05", "TP10"],
    "VAT Amount":                               ["TP05"],
    "Auto-Billing Uplift":                      ["TP05"],
    "NFP (Neutral Fare Proration)":             ["TP05"],
    "BilledMember":                             ["TP05"],
    "InvoiceNumber":                            ["TP05"],
    "BillingCategory":                          ["TP05"],
    "LineItem/ChargeCode":                      ["TP05"],
    "LineItem/ChargeAmount":                    ["TP05"],
    "LineItem/TotalNetAmount":                  ["TP05"],
    "InvoiceSummary/TotalLineItemAmount":        ["TP05"],
    "InvoiceHeader/PaymentTerms/CurrencyCode":  ["TP05"],
    "Interline Billing Basis":                  ["TP05"],
    "Amount to be Prorated":                    ["TP05"],
    "Settlement Method":                        ["TP05"],
    "Commission (Amount)":                      ["TP05"],
    "Partner Airline Agency Fee":               ["TP05"],
    "Sector Type":                              ["TP05", "TP06"],

    # ── SCHEDULE PUBLICATION & SLOT (TP06) ───────────────────
    "Flight Number (SSIM Chapter 7)":           ["TP06", "TP07", "TP08"],
    "Service Type Code (SSIM)":                 ["TP06", "TP08"],
    "Aircraft Type Code (SSIM)":                ["TP06", "TP08"],
    "Period of Operation (SSIM)":               ["TP06"],
    "Days of Operation (SSIM)":                 ["TP06"],
    "Departure Time / Arrival Time (SSIM)":     ["TP06", "TP07"],
    "Codeshare Indicator (SSIM)":               ["TP06", "TP05"],
    "Minimum Connect Time (SSIM)":              ["TP06", "TP07"],
    "TOBT (Target Off-Block Time)":             ["TP06", "TP07"],
    "TSAT (Target Start Up Approval Time)":     ["TP06", "TP07"],
    "Gate Assignment":                          ["TP06", "TP07", "TP08"],
    "Slot Holding Fee (YYZ)":                   ["TP06", "TP12"],
    "Flight Leg (Arrival Date Time)":           ["TP06", "TP07"],
    "Flight Leg (Departure Date Time)":         ["TP06", "TP07"],
    "Transport Service Leg (Status Code)":      ["TP06", "TP07"],
    "FlightLegNotifRQ (message)":               ["TP06", "TP07"],
    "Flight Leg Status (AIDX via SITA)":        ["TP06", "TP07"],
    "Outbound / Inbound Flight Info":           ["TP06", "TP07"],
    "ELDT (Estimated Landing Time)":            ["TP06", "TP07"],

    # ── DELAY EVENT & IROPS (TP07) ───────────────────────────
    "Delay Code (IATA AHM730)":                 ["TP07"],
    "Delay Duration (minutes)":                 ["TP07"],
    "Scheduled Time of Departure (STD)":        ["TP06", "TP07"],
    "Actual Time of Departure (ATD)":           ["TP07"],
    "Actual Time of Arrival (ATA)":             ["TP07", "TP08"],
    "AOBT (Actual Off-Block Time)":             ["TP07", "TP08"],
    "TPA_Extension (IROPs)":                    ["TP07"],
    "Passenger Count (ops)":                    ["TP07", "TP08", "TP12"],

    # ── GROUND HANDLING SERVICE DELIVERY (TP08) ──────────────
    "Service Charge Rate (SGHA)":               ["TP08"],
    "Payment Terms — 30 days (SGHA)":           ["TP08", "TP05"],
    "Liability Limit (SGHA)":                   ["TP08"],
    "Minimum Wage Adjustment (SGHA)":           ["TP08"],
    "Audit Right (SGHA)":                       ["TP08"],
    "Section 2 Passenger Services Fee (SGHA)":  ["TP08", "TP12"],
    "Baggage Handling Rate (SGHA)":             ["TP08"],
    "Bag Tag Number":                           ["TP08"],
    "Bag Status (CHECKED_IN / LOADED / MISHANDLED)": ["TP08"],
    "Bag Event Code":                           ["TP08"],

    # ── PAYMENT AUTHORISATION & FRAUD (TP09) ─────────────────
    "PAN (Primary Account Number)":             ["TP09"],
    "Cardholder Name":                          ["TP09"],
    "Expiration Date":                          ["TP09"],
    "Service Code":                             ["TP09"],
    "Full Track Data":                          ["TP09"],
    "Card Verification Code":                   ["TP09"],
    "PIN / PIN Block":                          ["TP09"],
    "Cardholder Data Environment (CDE)":        ["TP09"],
    "Network Security Controls":               ["TP09"],
    "Multi-Factor Authentication":              ["TP09"],
    "Encryption Requirement":                   ["TP09"],
    "Payment Card (Card Number)":               ["TP09"],
    "Payment Information (Payment Status Code)":["TP09", "TP10"],

    # ── PAYMENT CLEARANCE & DISPUTE (TP10) ───────────────────
    "Settlement Data (Net Clearance Amount)":   ["TP10"],
    "Clearance (Status Code)":                  ["TP10"],
    "Audit Log":                                ["TP09", "TP10"],
    "Incident Response Plan":                   ["TP09", "TP10"],
    "Deferred Revenue Liability":               ["TP10", "TP11"],

    # ── LOYALTY ISSUANCE & REDEMPTION (TP11) ─────────────────
    "Loyalty Program Account Identifier":       ["TP11"],
    "Loyalty Program Account Tier Code":        ["TP11"],
    "Loyalty Redemption (Loyalty Currency Amount)": ["TP11", "TP09"],
    "Qualifying Balance Amount":                ["TP11"],
    "SSP — Points Component":                   ["TP11"],
    "Breakage Estimate":                        ["TP11"],
    "frequent_flyer (bag event)":               ["TP11", "TP08"],

    # ── AIRPORT CHARGE ASSESSMENT (TP12) ─────────────────────
    "Landing Fee (CAD/tonne MTOW)":             ["TP12"],
    "Airport Improvement Fee (AIF) — Departing":["TP12"],
    "Airport Improvement Fee (AIF) — Connecting":["TP12"],
    "AIF in lieu (per seat)":                   ["TP12"],
    "Inflation Adjustment Rate (YYZ)":          ["TP12"],
    "Federal Inspection Space Charge (EWR)":    ["TP12"],
    "Passenger Facility Charge (PFC — EWR)":    ["TP12"],
    "Non-Signatory Rate Premium (EWR)":         ["TP12"],
    "Maximum Allowable Yield (MAY/pax — LHR)":  ["TP12"],
    "Airport Charge per Passenger (LHR)":       ["TP12"],
    "K Factor (LHR)":                           ["TP12"],
    "CPI Adjustment (LHR)":                     ["TP12"],
    "WACC (LHR — 4.01%)":                       ["TP12"],
    "H7 Regulatory Period (LHR)":               ["TP12"],
    "Aircraft Parking Fee (YYZ)":               ["TP12"],
    "Apron Fee (YYZ)":                          ["TP12"],
    "Public Landing Area Charge (EWR)":         ["TP12"],
    "Ramp and Apron Charge (EWR)":              ["TP12"],
    "Aircraft Parking / Storage Charge (EWR)":  ["TP12"],
    "Airfield Charge per ATM (LHR)":            ["TP12"],
    "SAF Incentive (LHR)":                      ["TP12"],
    "PNR (Passenger Name Record)":              ["TP03", "TP09"],
}

# ── BUILD EXPANDED CORPUS ────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/outputs/corpus_variable_registry.csv")

# Check coverage
mapped = set(VARIABLE_TP_MAP.keys())
corpus_vars = set(df["Variable Name"].tolist())
unmapped = corpus_vars - mapped
print(f"Total corpus variables: {len(corpus_vars)}")
print(f"Mapped to transaction points: {len(mapped)}")
print(f"Unmapped: {len(unmapped)}")
if unmapped:
    print("UNMAPPED VARIABLES:")
    for v in sorted(unmapped):
        print(f"  - {v}")

# Build expanded dataframe — one row per variable × transaction point
rows = []
for _, var_row in df.iterrows():
    vname = var_row["Variable Name"]
    tps = VARIABLE_TP_MAP.get(vname, [])
    if not tps:
        # Assign to most likely TP based on AIDM domain if not explicitly mapped
        domain_default = {
            "Shopping": ["TP02"],
            "Order Management": ["TP03"],
            "Settlement with Orders": ["TP05"],
            "Flights": ["TP06"],
            "Payments": ["TP09"],
            "Baggage": ["TP08"],
            "Loyalty Accounts": ["TP11"],
            "Parties": ["TP12"],
            "Taxes": ["TP03"],
            "Settlements": ["TP05"],
        }
        tps = domain_default.get(var_row["AIDM Domain"], ["TP02"])
        print(f"  DEFAULT mapping: {vname} → {tps}")
    for tp in tps:
        row = var_row.to_dict()
        row["Transaction_Point_ID"] = tp
        row["Transaction_Point_Name"] = TRANSACTION_POINTS[tp]["name"]
        row["TP_MAS_Function"] = TRANSACTION_POINTS[tp]["mas_function"]
        row["TP_ERD_Source"] = TRANSACTION_POINTS[tp]["authoritative_erd"]
        rows.append(row)

df_expanded = pd.DataFrame(rows)

print(f"\nExpanded corpus: {len(df_expanded)} variable × transaction point pairs")
print(f"(from {len(df)} unique variables)")
print(f"Average TPs per variable: {len(df_expanded)/len(df):.2f}")

print(f"\nVariables per transaction point:")
tp_counts = df_expanded.groupby(
    ["Transaction_Point_ID","Transaction_Point_Name"]
).size().reset_index(name="N_variables")
for _, row in tp_counts.iterrows():
    print(f"  {row['Transaction_Point_ID']} — {row['Transaction_Point_Name']:<40} n={row['N_variables']}")

print(f"\nActor types per transaction point (bipartite graph nodes):")
actor_tp = df_expanded.groupby(
    ["Transaction_Point_ID","Actor Type"]
).size().unstack(fill_value=0)
print(actor_tp.to_string())

print(f"\nValuation type mix per transaction point (competing logics):")
df_expanded["Val_Primary"] = df_expanded["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")
val_tp = df_expanded.groupby(
    ["Transaction_Point_ID","Val_Primary"]
).size().unstack(fill_value=0)
print(val_tp.to_string())

print(f"\nMNAR rate per transaction point (governance stress indicator):")
mnar_tp = df_expanded.groupby("Transaction_Point_ID").apply(
    lambda x: (x["Data Access"]=="MNAR").sum() / len(x) * 100
).round(1)
for tp, rate in mnar_tp.items():
    name = TRANSACTION_POINTS[tp]["name"]
    print(f"  {tp} — {name:<40} MNAR={rate:.1f}%")

# Save expanded corpus
df_expanded.to_csv(
    "/mnt/user-data/outputs/corpus_with_transaction_points.csv",
    index=False
)
print(f"\nSaved: corpus_with_transaction_points.csv")

# Save edge list for bipartite network
edge_list = df_expanded.groupby(
    ["Transaction_Point_ID","Transaction_Point_Name",
     "Actor Type","Val_Primary","Data Access"]
).agg(
    N_variables=("Variable Name","count"),
    Variables=("Variable Name", lambda x: "; ".join(x.tolist()))
).reset_index()
edge_list.to_csv(
    "/mnt/user-data/outputs/bipartite_edge_list.csv",
    index=False
)
print(f"Saved: bipartite_edge_list.csv")

# Save TP summary
tp_summary = pd.DataFrame([
    {
        "TP_ID": k,
        "TP_Name": v["name"],
        "MAS_Function": v["mas_function"],
        "ERD_Source": v["authoritative_erd"],
        "Description": v["description"],
    }
    for k, v in TRANSACTION_POINTS.items()
])
tp_summary.to_csv(
    "/mnt/user-data/outputs/transaction_point_definitions.csv",
    index=False
)
print(f"Saved: transaction_point_definitions.csv")
