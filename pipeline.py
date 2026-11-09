"""Cleaning + feature engineering for the DewDune survey."""
import numpy as np
import pandas as pd

CONCERNS = ["Acne", "Pigmentation", "Dryness", "Oiliness", "Sensitivity",
            "Dullness", "Fine lines", "Sun damage", "Hard-water hair fall"]
PRODUCTS = ["Cleanser", "Moisturiser", "Sunscreen", "Serum", "Toner",
            "Face mask", "Exfoliant", "Eye cream", "Lip balm", "Shower filter"]
BOX_ITEMS = ["SPF 50 sunscreen", "Hydrating serum", "Gentle cleanser",
             "Barrier moisturiser", "Shower filter", "Vitamin C serum",
             "Acne spot treatment", "Hair repair mask", "Lip SPF",
             "After-sun gel", "Sheet masks"]
LIKERT = ["q20_trust_personalised_reco", "q21_clean_halal_importance", "q22_price_sensitivity",
          "q23_influencer_influence", "q24_sustainability_importance", "q25_routine_satisfaction",
          "q26_derm_consult_interest", "q27_convenience_importance"]
AGE_ORD = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
INC_ORD = {"<5k": 0, "5-10k": 1, "10-20k": 2, "20-35k": 3, "35k+": 4}
INC_MID = {"<5k": 3500, "5-10k": 7500, "10-20k": 15000, "20-35k": 27500, "35k+": 45000}
FREQ_ORD = {"Rarely": 0, "Every 2-3 months": 1, "Monthly": 2, "Weekly": 3}

PRETTY = {
    "q1_age_group": "Age group", "q2_gender": "Gender", "q3_emirate": "Emirate",
    "q4_nationality_region": "Nationality region", "q5_occupation": "Occupation",
    "q6_monthly_income_aed": "Monthly income", "q7_years_in_uae": "Years in UAE",
    "q8_skin_type": "Skin type", "q10_outdoor_hours_day": "Outdoor hrs/day",
    "q11_ac_hours_day": "AC hrs/day", "q12_hard_water_impact": "Hard-water impact",
    "q13_monthly_skincare_spend_aed": "Monthly skincare spend", "q15_main_channel": "Main channel",
    "q16_purchase_frequency": "Purchase frequency", "q17_brands_tried_last_year": "Brands tried (12m)",
    "q18_routine_steps": "Routine steps", "q19_existing_subscriptions": "Existing subscriptions",
    "q20_trust_personalised_reco": "Trust in personalised reco", "q21_clean_halal_importance": "Clean/halal importance",
    "q22_price_sensitivity": "Price sensitivity", "q23_influencer_influence": "Influencer influence",
    "q24_sustainability_importance": "Sustainability importance", "q25_routine_satisfaction": "Routine satisfaction",
    "q26_derm_consult_interest": "Derm consult interest", "q27_convenience_importance": "Convenience importance",
    "q29_preferred_plan": "Preferred plan", "q30_subscribe_likelihood": "Subscribe likelihood",
    "q31_max_monthly_wtp_aed": "Max WTP (AED/month)", "n_concerns": "No. of concerns", "n_products": "No. of products used",
}


def pretty(c):
    if c in PRETTY:
        return PRETTY[c]
    for p in ("concern_", "uses_", "box_"):
        if c.startswith(p):
            return {"concern_": "Concern: ", "uses_": "Uses: ", "box_": "Box: "}[p] + c[len(p):].replace("_", " ")
    for k, v in PRETTY.items():
        if c.startswith(k + "_"):
            return f"{v} = {c[len(k) + 1:]}"
    return c


def clean(raw: pd.DataFrame):
    """Return (clean_df, log) where log is a list of dicts describing each step."""
    df = raw.copy()
    log = []

    def step(name, issue, action, n):
        log.append({"Step": name, "Issue found": issue, "Action taken": action, "Rows / cells affected": int(n)})

    # 1 duplicates
    n0 = len(df)
    df = df.drop_duplicates(subset="respondent_id", keep="first")
    step("Duplicates", "Same respondent_id submitted twice", "Kept first submission", n0 - len(df))

    # 2 messy labels
    emi_map = {"dubai": "Dubai", "dxb": "Dubai", "abu dhabi": "Abu Dhabi", "auh": "Abu Dhabi",
               "sharjah": "Sharjah", "shj": "Sharjah", "ajman": "Ajman", "ras al khaimah": "Ras Al Khaimah"}
    before = df["q3_emirate"].copy()
    df["q3_emirate"] = df["q3_emirate"].astype(str).str.strip().str.lower().map(emi_map)
    n_emi = (before != df["q3_emirate"]).sum()
    g_map = {"f": "Female", "female": "Female", "m": "Male", "male": "Male", "prefer not to say": "Prefer not to say"}
    before = df["q2_gender"].copy()
    df["q2_gender"] = df["q2_gender"].astype(str).str.strip().str.lower().map(g_map)
    n_g = (before != df["q2_gender"]).sum()
    step("Inconsistent labels", "e.g. 'DXB', ' dubai', 'AUH', 'F', 'male'", "Standardised to one spelling", n_emi + n_g)

    # 3 impossible values -> NaN
    n = 0
    m = (df["q11_ac_hours_day"] > 24) | (df["q11_ac_hours_day"] < 0); n += m.sum(); df.loc[m, "q11_ac_hours_day"] = np.nan
    m = (df["q10_outdoor_hours_day"] > 24) | (df["q10_outdoor_hours_day"] < 0); n += m.sum(); df.loc[m, "q10_outdoor_hours_day"] = np.nan
    m = df["q13_monthly_skincare_spend_aed"] < 0; n += m.sum(); df.loc[m, "q13_monthly_skincare_spend_aed"] = np.nan
    m = df["q7_years_in_uae"] > 60; n += m.sum(); df.loc[m, "q7_years_in_uae"] = np.nan
    step("Impossible values", ">24 hrs/day, negative hours or spend, >60 years in UAE", "Set to missing, then imputed", n)

    # 4 outliers: winsorise spend at 99th pct, WTP cap (> 3x 99th pct = fat-finger -> NaN)
    p99 = df["q13_monthly_skincare_spend_aed"].quantile(0.99)
    m = df["q13_monthly_skincare_spend_aed"] > p99
    df.loc[m, "q13_monthly_skincare_spend_aed"] = p99
    step("Outliers: spend", f"Spend above 99th percentile (AED {p99:,.0f})", "Capped at 99th percentile (winsorised)", m.sum())
    q1, q3 = df["q31_max_monthly_wtp_aed"].quantile([.25, .75])
    hi = q3 + 3 * (q3 - q1)
    m = df["q31_max_monthly_wtp_aed"] > hi
    df.loc[m, "q31_max_monthly_wtp_aed"] = np.nan
    step("Outliers: willingness to pay", f"WTP above Q3 + 3×IQR (AED {hi:,.0f}), e.g. 9,999", "Treated as typing error. Row dropped from regression only.", m.sum())

    # 5 straight-liners
    sl = (df[LIKERT].nunique(axis=1) == 1) & (df[LIKERT].iloc[:, 0] == 3)
    df = df[~sl]
    step("Straight-liners", "Answered 3 to every attitude question (low effort)", "Removed respondent", sl.sum())

    # 6 missing values
    miss = int(df.drop(columns=["q31_max_monthly_wtp_aed"]).isna().sum().sum())
    for c in ["q7_years_in_uae", "q10_outdoor_hours_day", "q11_ac_hours_day", "q13_monthly_skincare_spend_aed"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["q12_hard_water_impact", "q20_trust_personalised_reco", "q22_price_sensitivity", "q25_routine_satisfaction"]:
        df[c] = df[c].fillna(df[c].median()).round().astype(int)
    for c in ["q6_monthly_income_aed", "q8_skin_type", "q15_main_channel", "q3_emirate", "q2_gender"]:
        df[c] = df[c].fillna(df[c].mode()[0])
    step("Missing values", "Blank answers in 10 questions (about 2–5% each)", "Numbers: median. Categories: most common answer. WTP left blank.", miss)

    df = df.reset_index(drop=True)
    return df, pd.DataFrame(log)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-hot + ordinal columns used by the models."""
    out = df.copy()
    for col, items, pre in [("q9_skin_concerns", CONCERNS, "concern_"),
                            ("q14_products_used", PRODUCTS, "uses_"),
                            ("q28_box_items_wanted", BOX_ITEMS, "box_")]:
        s = out[col].fillna("")
        for it in items:
            out[pre + it.replace(" ", "_")] = s.str.split(";").apply(lambda l, it=it: int(it in l))
    out["n_concerns"] = out[[c for c in out if c.startswith("concern_")]].sum(axis=1)
    out["n_products"] = out[[c for c in out if c.startswith("uses_")]].sum(axis=1)
    out["age_ord"] = out["q1_age_group"].map(AGE_ORD)
    out["income_ord"] = out["q6_monthly_income_aed"].map(INC_ORD)
    out["income_mid"] = out["q6_monthly_income_aed"].map(INC_MID)
    out["freq_ord"] = out["q16_purchase_frequency"].map(FREQ_ORD)
    out["will_subscribe"] = (out["q30_subscribe_likelihood"] >= 4).astype(int)
    return out


NUMERIC = ["age_ord", "income_ord", "q7_years_in_uae", "q10_outdoor_hours_day", "q11_ac_hours_day",
           "q12_hard_water_impact", "q13_monthly_skincare_spend_aed", "freq_ord", "q17_brands_tried_last_year",
           "q18_routine_steps", "q19_existing_subscriptions"] + LIKERT + ["n_concerns", "n_products"]
CATEGORICAL = ["q2_gender", "q3_emirate", "q4_nationality_region", "q5_occupation", "q8_skin_type", "q15_main_channel"]
CLUSTER_COLS = ["age_ord", "income_ord", "q7_years_in_uae", "q10_outdoor_hours_day", "q11_ac_hours_day",
                "q12_hard_water_impact", "q13_monthly_skincare_spend_aed", "q17_brands_tried_last_year",
                "q18_routine_steps", "q20_trust_personalised_reco", "q22_price_sensitivity",
                "q23_influencer_influence", "q25_routine_satisfaction", "q27_convenience_importance"]


def model_matrix(df: pd.DataFrame, include_multi=True) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    X["log_spend"] = np.log1p(X.pop("q13_monthly_skincare_spend_aed"))
    X = pd.concat([X, pd.get_dummies(df[CATEGORICAL], drop_first=True, dtype=float)], axis=1)
    if include_multi:
        X = pd.concat([X, df[[c for c in df if c.startswith(("concern_", "uses_"))]].astype(float)], axis=1)
    return X
