"""
CIND820 Milestone 4 — Full Corpus Construction
Systematic coding of AIDM ABIEs for L1 analysis

Extends the M3 PoC corpus (175 variables) to full AIDM scope
by coding all relevant ABIEs across four MAS functions
plus environmental boundary nodes.

Coding decisions at ABIE level:
- Actor Type: derived from ABIE subject area + definition
- Valuation Type: derived from ABIE semantics + ICAO/IATA formula layer
- Data Access: derived from schema source + public availability
- MAS Function: from subject area mapping
- Source: AIDM RP1008 (authoritative)

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
import os

OUTPUT = OUTS
os.makedirs(OUTPUT, exist_ok=True)

# ── LOAD AIDM EXTRACT ─────────────────────────────────────────
aidm = pd.read_csv(os.path.join(DATA,"26_1_AIDM_Extract_Subjects_Areas_xlsm__Page_1.csv"))
bbies = aidm[aidm["Stereotype"]=="IATA_BBIE"].copy()
abies_only = aidm[aidm["Stereotype"]=="IATA_ABIE"].copy()

print(f"AIDM BBIEs: {len(bbies)}")
print(f"AIDM ABIEs: {len(abies_only)}")

# ── SUBJECT AREA → MAS FUNCTION MAPPING ───────────────────────
AREA_FUNCTION_MAP = {
    "Offers":                              "Revenue Management",
    "Prices":                              "Revenue Management",
    "Taxes":                               "Revenue Management",
    "Loyalty Accounts":                    "Revenue Management",
    "Shopping Criteria":                   "Revenue Management",
    "Services & Products":                 "Revenue Management",
    "Settlements":                         "Distribution / GDS",
    "Tickets":                             "Distribution / GDS",
    "Orders":                              "Distribution / GDS",
    "Flights":                             "Disruption Management",
    "Processes":                           "Disruption Management",
    "Payments":                            "Fraud / Payment",
    "Bags":                                "Workforce Management",
    "Aircraft & Other Transport Vehicles": "Environmental Boundary",
    "Aircraft Technical Details":          "Environmental Boundary",
    "Aircraft Load Calculations":          "Environmental Boundary",
    "Parties":                             "Environmental Boundary",
    "Contacts":                            "Environmental Boundary",
}

# ── ABIE → ACTOR TYPE MAPPING ─────────────────────────────────
# Based on who defines/controls the ABIE in practice
ABIE_ACTOR_MAP = {
    # Revenue Management ABIEs
    "Tax":                      "Regulator",
    "Fare Detail":              "Carrier",
    "Price":                    "Carrier",
    "Fee":                      "Carrier",
    "Penalty":                  "Carrier",
    "Offer":                    "Carrier",
    "Offer Price":              "Carrier",
    "Offer Item":               "Carrier",
    "Offer Condition":          "Carrier",
    "Eligibility":              "Carrier",
    "Loyalty Account":          "Carrier",
    "Loyalty Transaction":      "Carrier",
    "Loyalty Program":          "Carrier",
    "Points Balance":           "Carrier",
    "Redemption":               "Carrier",
    "Tier":                     "Carrier",
    "Shopping Criteria":        "Passenger",
    "Search Criteria":          "Passenger",
    "Preference":               "Passenger",
    "Service Definition":       "Industry Body",
    "Service":                  "Vendor",
    "Ancillary Service":        "Carrier",
    "Surcharge":                "Carrier",
    "Discount":                 "Carrier",

    # Distribution / GDS ABIEs
    "Order":                    "Carrier",
    "Order Item":               "Carrier",
    "Coupon":                   "Industry Body",
    "Commission":               "Vendor",
    "Ticket":                   "Industry Body",
    "Passenger Segment":        "Carrier",
    "Itinerary":                "Carrier",
    "Booking":                  "Vendor",
    "Settlement":               "Industry Body",
    "Proration":                "Industry Body",
    "Invoice":                  "Vendor",
    "Clearance":                "Vendor",
    "Debit Memo":               "Industry Body",
    "Credit Memo":              "Industry Body",
    "Agency":                   "Vendor",
    "Travel Agency":            "Vendor",

    # Disruption Management ABIEs
    "Flight Leg":               "Carrier",
    "Transport Service Leg":    "Carrier",
    "Transport Service Segment":"Carrier",
    "Carrier":                  "Carrier",
    "Airline":                  "Carrier",
    "Transit Stop":             "Airport",
    "Bag Activity":             "Vendor",
    "Aircraft":                 "Carrier",
    "Clearance":                "Regulator",
    "Delay":                    "Carrier",
    "Flight":                   "Carrier",

    # Fraud / Payment ABIEs
    "Payment Card":             "Vendor",
    "Payment Transaction":      "Vendor",
    "Payment Information":      "Vendor",
    "Device":                   "Vendor",
    "Individual":               "Passenger",
    "Identity Document":        "Regulator",
    "Fraud":                    "Vendor",
    "Authentication":           "Vendor",

    # Workforce Management ABIEs
    "Bag Tag":                  "Industry Body",
    "Bag Physical Properties":  "Carrier",
    "Dated Operating Segment":  "Carrier",
    "Segment Security Controls":"Regulator",
    "Hold":                     "Carrier",
    "Cabin Component":          "Carrier",
    "Seat":                     "Carrier",
    "Passenger":                "Passenger",

    # Environmental Boundary ABIEs
    "Aircraft Group":           "Carrier",
    "Aircraft Group Configuration": "Carrier",
    "Balance Output Requirement":"Carrier",
    "Payment Card":             "Vendor",
    "Identity Document":        "Regulator",
    "Individual":               "Passenger",
}

DEFAULT_ACTOR = {
    "Revenue Management":    "Carrier",
    "Distribution / GDS":    "Industry Body",
    "Disruption Management": "Carrier",
    "Fraud / Payment":       "Vendor",
    "Workforce Management":  "Carrier",
    "Environmental Boundary":"Carrier",
}

# ── ABIE → VALUATION TYPE MAPPING ─────────────────────────────
ABIE_VAL_MAP = {
    # Income-generating ABIEs
    "Tax":                  "Cost",
    "Fare Detail":          "Income",
    "Price":                "Income",
    "Fee":                  "Income",
    "Penalty":              "Income",
    "Offer":                "Option",
    "Offer Price":          "Market",
    "Offer Item":           "Option",
    "Commission":           "Cost",
    "Settlement":           "Income",
    "Proration":            "Income",
    "Invoice":              "Income",
    "Clearance":            "Income",
    "Debit Memo":           "Cost",
    "Credit Memo":          "Income",
    "Coupon":               "Option",
    "Ticket":               "Income",
    "Surcharge":            "Cost",
    "Discount":             "Market",
    "Order":                "Income",
    "Order Item":           "Income",

    # Loyalty
    "Loyalty Account":      "Option",
    "Loyalty Transaction":  "Option",
    "Loyalty Program":      "Option",
    "Points Balance":       "Option",
    "Redemption":           "Option",
    "Tier":                 "Market",

    # Risk/Cost ABIEs
    "Payment Card":         "Risk",
    "Payment Transaction":  "Risk",
    "Payment Information":  "Risk",
    "Device":               "Risk",
    "Identity Document":    "Risk",
    "Segment Security Controls": "Risk",
    "Authentication":       "Risk",
    "Fraud":                "Risk",

    # Market ABIEs
    "Shopping Criteria":    "Market",
    "Search Criteria":      "Market",
    "Preference":           "Option",
    "Service":              "Market",
    "Service Definition":   "Market",
    "Ancillary Service":    "Income",

    # Cost ABIEs
    "Bag Tag":              "Cost",
    "Bag Physical Properties": "Cost",
    "Dated Operating Segment": "Cost",
    "Hold":                 "Cost",
    "Cabin Component":      "Cost",
    "Seat":                 "Market",
    "Passenger":            "Option",
    "Booking":              "Income",
    "Agency":               "Cost",
    "Travel Agency":        "Cost",

    # Operational ABIEs
    "Flight Leg":           "Cost",
    "Transport Service Leg":"Cost",
    "Transport Service Segment":"Cost",
    "Flight":               "Cost",
    "Delay":                "Risk",
    "Bag Activity":         "Cost",
    "Aircraft":             "Cost",
    "Transit Stop":         "Cost",

    # Environmental ABIEs
    "Aircraft Group":       "Cost",
    "Aircraft Group Configuration": "Cost",
    "Balance Output Requirement": "Cost",
    "Individual":           "Risk",
    "Carrier":              "Income",
    "Airline":              "Income",
}

DEFAULT_VAL = {
    "Revenue Management":    "Income",
    "Distribution / GDS":    "Income",
    "Disruption Management": "Risk",
    "Fraud / Payment":       "Risk",
    "Workforce Management":  "Cost",
    "Environmental Boundary":"Cost",
}

# ── DATA ACCESS CODING ─────────────────────────────────────────
# Based on whether the ABIE's values are publicly observable
# in any of the three jurisdictions
ABIE_ACCESS_MAP = {
    # OBSERVED — defined in public schemas, values accessible
    "Tax":                   "OBSERVED",
    "Flight Leg":            "OBSERVED",
    "Transport Service Leg": "OBSERVED",
    "Flight":                "OBSERVED",
    "Bag Tag":               "OBSERVED",
    "Bag Activity":          "OBSERVED",
    "Ticket":                "OBSERVED",
    "Coupon":                "OBSERVED",
    "Segment Security Controls": "OBSERVED",
    "Transit Stop":          "OBSERVED",
    "Aircraft Group":        "OBSERVED",
    "Carrier":               "OBSERVED",
    "Airline":               "OBSERVED",
    "Shopping Criteria":     "OBSERVED",
    "Search Criteria":       "OBSERVED",
    "Dated Operating Segment":"OBSERVED",
    "Identity Document":     "OBSERVED",
    "Delay":                 "OBSERVED",
    "Order":                 "OBSERVED",
    "Booking":               "OBSERVED",

    # MNAR — values exist but structurally withheld
    "Price":                 "MNAR",
    "Fare Detail":           "MNAR",
    "Fee":                   "MNAR",
    "Penalty":               "MNAR",
    "Commission":            "MNAR",
    "Settlement":            "MNAR",
    "Proration":             "MNAR",
    "Invoice":               "MNAR",
    "Clearance":             "MNAR",
    "Debit Memo":            "MNAR",
    "Credit Memo":           "MNAR",
    "Loyalty Account":       "MNAR",
    "Loyalty Transaction":   "MNAR",
    "Points Balance":        "MNAR",
    "Redemption":            "MNAR",
    "Tier":                  "MNAR",
    "Payment Card":          "MNAR",
    "Payment Transaction":   "MNAR",
    "Payment Information":   "MNAR",
    "Device":                "MNAR",
    "Fraud":                 "MNAR",
    "Authentication":        "MNAR",
    "Balance Output Requirement": "MNAR",
    "Hold":                  "MNAR",
    "Agency":                "MNAR",
    "Travel Agency":         "MNAR",

    # PARTIAL — some elements observable, some not
    "Offer":                 "PARTIAL",
    "Offer Price":           "PARTIAL",
    "Offer Item":            "PARTIAL",
    "Order Item":            "PARTIAL",
    "Service":               "PARTIAL",
    "Service Definition":    "PARTIAL",
    "Ancillary Service":     "PARTIAL",
    "Surcharge":             "PARTIAL",
    "Discount":              "PARTIAL",
    "Passenger Segment":     "PARTIAL",
    "Passenger":             "PARTIAL",
    "Individual":            "PARTIAL",
    "Bag Physical Properties":"PARTIAL",
    "Cabin Component":       "PARTIAL",
    "Seat":                  "PARTIAL",
    "Eligibility":           "PARTIAL",
    "Loyalty Program":       "PARTIAL",
    "Aircraft Group Configuration": "PARTIAL",
    "Preference":            "PARTIAL",
    "Itinerary":             "PARTIAL",
}

DEFAULT_ACCESS = {
    "Revenue Management":    "MNAR",
    "Distribution / GDS":    "MNAR",
    "Disruption Management": "OBSERVED",
    "Fraud / Payment":       "MNAR",
    "Workforce Management":  "PARTIAL",
    "Environmental Boundary":"OBSERVED",
}

# ── BUILD FULL CORPUS FROM AIDM ABIEs ─────────────────────────
print("\nBuilding full corpus from AIDM ABIEs...")

# Get unique ABIEs per subject area
abie_list = abies_only.copy()
abie_list["MAS_Function"] = abie_list["Subject Area Name"].map(
    AREA_FUNCTION_MAP)
abie_list = abie_list[abie_list["MAS_Function"].notna()].copy()

# Count BBIEs per ABIE (complexity indicator)
bbie_counts = bbies.groupby(
    ["Subject Area Name","ABIE Name"])["Element Name"].count().reset_index()
bbie_counts.columns = ["Subject Area Name","ABIE Name","N_Elements"]

abie_list = abie_list.merge(bbie_counts, on=["Subject Area Name","ABIE Name"],
                             how="left")
abie_list["N_Elements"] = abie_list["N_Elements"].fillna(1)

# Assign coding dimensions
rows = []
for _, row in abie_list.iterrows():
    func = row["MAS_Function"]
    abie = row["ABIE Name"]
    defn = str(row["Element Definition"])[:200] if pd.notna(
        row["Element Definition"]) else ""

    actor = ABIE_ACTOR_MAP.get(abie, DEFAULT_ACTOR.get(func, "Carrier"))
    val   = ABIE_VAL_MAP.get(abie, DEFAULT_VAL.get(func, "Cost"))
    access= ABIE_ACCESS_MAP.get(abie, DEFAULT_ACCESS.get(func, "MNAR"))

    # Actor type from actor
    actor_type_map = {
        "Carrier": "Carrier", "Airport": "Airport",
        "Regulator": "Regulator", "Vendor": "Vendor",
        "Industry Body": "Industry Body", "Passenger": "Passenger",
    }
    actor_type = actor_type_map.get(actor, "Carrier")

    rows.append({
        "Variable Name": abie,
        "Source Document": "AIDM RP1008",
        "Source Category": "Industry Schema",
        "MAS Function": func,
        "Subject Area": row["Subject Area Name"],
        "Actor": actor,
        "Actor Type": actor_type,
        "Valuation Type": val,
        "Data Access": access,
        "N_AIDM_Elements": int(row["N_Elements"]),
        "AIDM_Definition": defn,
        "Coding_Source": "AIDM RP1008 systematic coding — ABIE level",
    })

aidm_corpus = pd.DataFrame(rows).drop_duplicates(
    subset=["Variable Name","MAS Function"])

print(f"AIDM ABIEs coded: {len(aidm_corpus)}")
print(f"\nBy MAS Function:")
print(aidm_corpus["MAS Function"].value_counts().to_string())
print(f"\nData Access:")
print(aidm_corpus["Data Access"].value_counts().to_string())

# ── MERGE WITH EXISTING CORPUS ─────────────────────────────────
existing = pd.read_csv(os.path.join(OUTS,"corpus_variable_registry.csv"))
existing["Val_Primary"] = existing["Valuation Type"].apply(
    lambda v: v.split("/")[0].strip() if pd.notna(v) else "Unknown")
existing["Source"] = "M3_PoC_Schema"
existing["N_AIDM_Elements"] = 1
existing["Subject Area"] = existing["AIDM Domain"]
existing["AIDM_Definition"] = ""
existing["Coding_Source"] = "Manual schema coding (M2/M3)"

# Standardize columns
aidm_corpus["Source"] = "M4_AIDM_Full"
aidm_corpus["Val_Primary"] = aidm_corpus["Valuation Type"]
aidm_corpus["AIDM Domain"] = aidm_corpus["Subject Area"]
aidm_corpus["Data Type"] = "ABIE"
aidm_corpus["Notes"] = aidm_corpus["AIDM_Definition"]

# Combine — keep existing where variable names overlap
existing_vars = set(existing["Variable Name"].tolist())
aidm_new = aidm_corpus[~aidm_corpus["Variable Name"].isin(existing_vars)]

print(f"\nExisting corpus: {len(existing)} variables")
print(f"New AIDM ABIEs (not in existing): {len(aidm_new)}")
print(f"Overlap (already coded): {len(aidm_corpus) - len(aidm_new)}")

# Full corpus
cols = ["Variable Name","Source Document","Source Category","MAS Function",
        "Subject Area","Actor","Actor Type","Valuation Type","Val_Primary",
        "Data Access","N_AIDM_Elements","Coding_Source","Source"]

full_corpus = pd.concat([
    existing[["Variable Name","Source Document","Source Category","MAS Function",
              "Actor","Actor Type","Valuation Type","Val_Primary","Data Access",
              "Source"]].assign(
        Subject_Area=existing["AIDM Domain"],
        N_AIDM_Elements=1,
        Coding_Source="Manual schema coding (M2/M3)"
    ),
    aidm_new[["Variable Name","Source Document","Source Category","MAS Function",
              "Actor","Actor Type","Valuation Type","Val_Primary","Data Access",
              "Source"]].assign(
        Subject_Area=aidm_new["Subject Area"],
        N_AIDM_Elements=aidm_new["N_AIDM_Elements"],
        Coding_Source=aidm_new["Coding_Source"]
    )
], ignore_index=True)

print(f"\nFull L1 corpus: {len(full_corpus)} variables")
print(f"\nBy MAS Function:")
print(full_corpus["MAS Function"].value_counts().to_string())
print(f"\nBy Data Access:")
print(full_corpus["Data Access"].value_counts().to_string())
print(f"\nBy Actor Type:")
print(full_corpus["Actor Type"].value_counts().to_string())
print(f"\nBy Valuation Type:")
print(full_corpus["Val_Primary"].value_counts().to_string())

# ── ENVIRONMENTAL BOUNDARY VARIABLES ──────────────────────────
print("\n--- ENVIRONMENTAL BOUNDARY VARIABLES ---")
env = full_corpus[full_corpus["MAS Function"]=="Environmental Boundary"]
print(f"N = {len(env)}")
print("\nThese connect the MAS to the wider environment:")
for _, row in env.iterrows():
    print(f"  [{row['Data Access']:<8}] {row['Variable Name']:<35} "
          f"[{row['Actor Type']:<14}] [{row['Val_Primary']}]")

# ── SAVE ──────────────────────────────────────────────────────
full_corpus.to_csv(f"{OUTPUT}/full_corpus_L1.csv", index=False)
aidm_new.to_csv(f"{OUTPUT}/aidm_new_variables.csv", index=False)

print(f"\nSaved: full_corpus_L1.csv ({len(full_corpus)} variables)")
print(f"Saved: aidm_new_variables.csv ({len(aidm_new)} new variables)")

# ── COVERAGE ANALYSIS ─────────────────────────────────────────
print("\n--- AIDM COVERAGE ANALYSIS ---")
total_aidm_inscope = 2429
total_abies_inscope = len(aidm_corpus)
coded_vars = len(full_corpus)

print(f"Total AIDM in-scope BBIEs: {total_aidm_inscope}")
print(f"Total AIDM in-scope ABIEs: {total_abies_inscope}")
print(f"Full corpus variables: {coded_vars}")
print(f"ABIE-level coverage: {coded_vars/total_abies_inscope*100:.1f}%")
print(f"BBIE-level coverage: {coded_vars/total_aidm_inscope*100:.1f}%")
print(f"\nMethodological note: Coding is at the ABIE (entity) level.")
print(f"Each ABIE represents a business concept with multiple sub-elements.")
print(f"ABIE-level coding is analytically appropriate for property graph")
print(f"node assignment — individual BBIEs are element attributes,")
print(f"not separate interaction nodes.")
