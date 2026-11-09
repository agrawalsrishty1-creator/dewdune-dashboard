"""
Synthetic survey generator for DewDune — climate-smart personalised skincare
subscription for UAE residents.

Respondents are drawn from five latent consumer segments so that the
clustering, classification, regression and association-rule algorithms have
real structure to find. Noise is then deliberately injected (missing values,
outliers, impossible values, messy labels, duplicates, straight-liners) so the
cleaning step has something real to fix.

Run:  python generate_data.py          -> data/survey_raw.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 3600
rng = np.random.default_rng(SEED)

CONCERNS = ["Acne", "Pigmentation", "Dryness", "Oiliness", "Sensitivity",
            "Dullness", "Fine lines", "Sun damage", "Hard-water hair fall"]
PRODUCTS = ["Cleanser", "Moisturiser", "Sunscreen", "Serum", "Toner",
            "Face mask", "Exfoliant", "Eye cream", "Lip balm", "Shower filter"]
BOX_ITEMS = ["SPF 50 sunscreen", "Hydrating serum", "Gentle cleanser",
             "Barrier moisturiser", "Shower filter", "Vitamin C serum",
             "Acne spot treatment", "Hair repair mask", "Lip SPF",
             "After-sun gel", "Sheet masks"]

# ---------------------------------------------------------------- segments
SEGMENTS = {
    "Climate-struggling expat": 0.25,
    "Premium skin enthusiast": 0.18,
    "Busy professional minimalist": 0.22,
    "Budget student": 0.20,
    "Loyal pharmacy buyer": 0.15,
}
seg = rng.choice(list(SEGMENTS), size=N, p=list(SEGMENTS.values()))


def pick(options, probs_by_seg):
    out = np.empty(N, dtype=object)
    for s, p in probs_by_seg.items():
        m = seg == s
        out[m] = rng.choice(options, size=m.sum(), p=np.array(p) / np.sum(p))
    return out


def by_seg(fn):
    out = np.zeros(N)
    for s in SEGMENTS:
        m = seg == s
        out[m] = fn(s, m.sum())
    return out


def likert(mu_by_seg, sd=0.9):
    raw = by_seg(lambda s, n: rng.normal(mu_by_seg[s], sd, n))
    return np.clip(np.round(raw), 1, 5).astype(int)


S = list(SEGMENTS)  # order: expat, premium, busy, student, pharmacy

# ---------------------------------------------------------------- demographics
age = pick(["18-24", "25-34", "35-44", "45-54", "55+"], {
    S[0]: [10, 50, 28, 9, 3], S[1]: [12, 45, 30, 10, 3],
    S[2]: [5, 45, 35, 12, 3], S[3]: [85, 14, 1, 0.5, 0.5],
    S[4]: [5, 20, 30, 30, 15]})
gender = pick(["Female", "Male", "Prefer not to say"], {
    S[0]: [62, 35, 3], S[1]: [80, 17, 3], S[2]: [50, 47, 3],
    S[3]: [65, 32, 3], S[4]: [58, 39, 3]})
emirate = rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah"],
                     size=N, p=[0.45, 0.25, 0.18, 0.07, 0.05])
nationality = rng.choice(["South Asian", "Arab", "Western", "East/SE Asian", "African", "Other"],
                         size=N, p=[0.38, 0.24, 0.12, 0.15, 0.06, 0.05])
occupation = pick(["Student", "Salaried professional", "Business owner/Freelancer", "Homemaker", "Other"], {
    S[0]: [5, 70, 12, 10, 3], S[1]: [5, 60, 25, 8, 2], S[2]: [1, 75, 20, 2, 2],
    S[3]: [88, 8, 3, 0.5, 0.5], S[4]: [2, 50, 15, 28, 5]})
income = pick(["<5k", "5-10k", "10-20k", "20-35k", "35k+"], {
    S[0]: [8, 25, 40, 20, 7], S[1]: [2, 8, 25, 35, 30], S[2]: [2, 10, 35, 33, 20],
    S[3]: [65, 25, 8, 1.5, 0.5], S[4]: [8, 25, 35, 22, 10]})
income_num = pd.Series(income).map({"<5k": 0, "5-10k": 1, "10-20k": 2, "20-35k": 3, "35k+": 4}).values
years_uae = by_seg(lambda s, n: {
    S[0]: rng.gamma(1.5, 1.0, n), S[1]: rng.gamma(2.5, 2.5, n), S[2]: rng.gamma(2.5, 2.5, n),
    S[3]: rng.gamma(2.0, 2.5, n), S[4]: rng.gamma(4.0, 3.0, n)}[s]).round(1)

# ---------------------------------------------------------------- skin & environment
skin_type = pick(["Oily", "Dry", "Combination", "Normal", "Sensitive"], {
    S[0]: [12, 40, 20, 10, 18], S[1]: [20, 20, 30, 10, 20], S[2]: [25, 15, 30, 25, 5],
    S[3]: [45, 10, 30, 10, 5], S[4]: [15, 25, 25, 30, 5]})
outdoor_hours = by_seg(lambda s, n: rng.gamma({S[0]: 2.2, S[1]: 1.8, S[2]: 1.2, S[3]: 2.8, S[4]: 1.5}[s], 1.0, n)).round(1)
ac_hours = np.clip(by_seg(lambda s, n: rng.normal({S[0]: 15, S[1]: 14, S[2]: 16, S[3]: 12, S[4]: 13}[s], 3, n)), 2, 24).round(0)
hard_water = likert({S[0]: 4.4, S[1]: 3.5, S[2]: 3.0, S[3]: 3.0, S[4]: 2.3})

concern_p = {  # probability of each concern by segment
    S[0]: [.15, .30, .75, .10, .45, .50, .15, .45, .70],
    S[1]: [.20, .55, .40, .25, .35, .55, .55, .50, .35],
    S[2]: [.15, .25, .25, .30, .10, .40, .30, .30, .25],
    S[3]: [.70, .25, .15, .60, .20, .35, .03, .25, .20],
    S[4]: [.10, .30, .30, .15, .15, .25, .40, .20, .15]}
concern_mat = np.zeros((N, len(CONCERNS)), dtype=int)
for s, p in concern_p.items():
    m = seg == s
    concern_mat[m] = (rng.random((m.sum(), len(CONCERNS))) < np.array(p)).astype(int)
# ensure everyone has at least one concern
none = concern_mat.sum(1) == 0
concern_mat[none, rng.integers(0, len(CONCERNS), none.sum())] = 1

# ---------------------------------------------------------------- behaviour
spend = by_seg(lambda s, n: rng.lognormal(np.log({S[0]: 180, S[1]: 520, S[2]: 220, S[3]: 90, S[4]: 140}[s]), 0.45, n))
spend = (spend * (1 + 0.12 * (income_num - 2))).round(0)
routine_steps = np.clip(by_seg(lambda s, n: rng.poisson({S[0]: 4, S[1]: 7, S[2]: 2, S[3]: 3.5, S[4]: 3}[s], n)), 1, 12)
brands_tried = np.clip(by_seg(lambda s, n: rng.poisson({S[0]: 5, S[1]: 9, S[2]: 2, S[3]: 6, S[4]: 1.5}[s], n)), 0, 25)
channel = pick(["Pharmacy", "Sephora/Faces", "Amazon/Noon", "Brand website", "Instagram/TikTok shop", "Supermarket"], {
    S[0]: [25, 15, 30, 12, 10, 8], S[1]: [10, 45, 15, 22, 6, 2], S[2]: [20, 10, 45, 15, 3, 7],
    S[3]: [10, 12, 30, 5, 35, 8], S[4]: [55, 5, 10, 3, 2, 25]})
purchase_freq = pick(["Weekly", "Monthly", "Every 2-3 months", "Rarely"], {
    S[0]: [8, 50, 35, 7], S[1]: [20, 60, 17, 3], S[2]: [3, 35, 45, 17],
    S[3]: [12, 40, 35, 13], S[4]: [3, 25, 45, 27]})
existing_subs = np.clip(by_seg(lambda s, n: rng.poisson({S[0]: 2, S[1]: 3, S[2]: 3.5, S[3]: 1.5, S[4]: 1}[s], n)), 0, 10)

prod_p = {
    S[0]: [.95, .90, .60, .45, .25, .20, .15, .10, .40, .35],
    S[1]: [.98, .97, .85, .90, .70, .60, .55, .55, .60, .30],
    S[2]: [.95, .80, .55, .20, .10, .05, .05, .05, .25, .10],
    S[3]: [.92, .60, .35, .35, .30, .40, .30, .03, .50, .05],
    S[4]: [.95, .80, .40, .15, .15, .10, .05, .20, .45, .05]}
prod_mat = np.zeros((N, len(PRODUCTS)), dtype=int)
for s, p in prod_p.items():
    m = seg == s
    prod_mat[m] = (rng.random((m.sum(), len(PRODUCTS))) < np.array(p)).astype(int)

# ---------------------------------------------------------------- attitudes (1-5)
trust_ai = likert({S[0]: 3.8, S[1]: 4.1, S[2]: 3.4, S[3]: 3.6, S[4]: 2.3})
clean_halal = likert({S[0]: 3.4, S[1]: 3.9, S[2]: 3.0, S[3]: 3.1, S[4]: 3.3}, sd=1.1)
price_sens = likert({S[0]: 3.0, S[1]: 2.0, S[2]: 2.6, S[3]: 4.6, S[4]: 3.4})
influencer = likert({S[0]: 3.0, S[1]: 3.6, S[2]: 2.0, S[3]: 4.3, S[4]: 1.8})
sustainability = likert({S[0]: 3.3, S[1]: 3.8, S[2]: 3.0, S[3]: 3.6, S[4]: 2.7}, sd=1.0)
satisfaction = likert({S[0]: 2.2, S[1]: 3.0, S[2]: 3.3, S[3]: 2.8, S[4]: 4.3})
derm_interest = likert({S[0]: 4.0, S[1]: 4.2, S[2]: 3.2, S[3]: 3.3, S[4]: 2.4})
convenience = likert({S[0]: 3.7, S[1]: 3.4, S[2]: 4.6, S[3]: 3.0, S[4]: 3.2})

# ---------------------------------------------------------------- box preference (association rules)
box = np.zeros((N, len(BOX_ITEMS)), dtype=int)
bi = {b: i for i, b in enumerate(BOX_ITEMS)}
c = {k: concern_mat[:, i] for i, k in enumerate(CONCERNS)}
r = lambda: rng.random(N)
box[:, bi["SPF 50 sunscreen"]] = r() < 0.45 + 0.08 * np.minimum(outdoor_hours, 4) + 0.15 * c["Sun damage"]
box[:, bi["Lip SPF"]] = r() < np.where(box[:, bi["SPF 50 sunscreen"]] == 1, 0.42, 0.06) + 0.04 * np.minimum(outdoor_hours, 4)
box[:, bi["After-sun gel"]] = r() < np.where((box[:, bi["SPF 50 sunscreen"]] == 1) & (outdoor_hours > 2.5), 0.55, 0.08)
box[:, bi["Shower filter"]] = r() < 0.04 + 0.13 * (hard_water - 1) + 0.15 * c["Hard-water hair fall"]
box[:, bi["Hair repair mask"]] = r() < np.where(box[:, bi["Shower filter"]] == 1, 0.62, 0.08)
box[:, bi["Hydrating serum"]] = r() < 0.20 + 0.40 * c["Dryness"] + 0.02 * (ac_hours - 12)
box[:, bi["Barrier moisturiser"]] = r() < np.where(box[:, bi["Hydrating serum"]] == 1, 0.58, 0.14) + 0.12 * c["Sensitivity"]
box[:, bi["Vitamin C serum"]] = r() < 0.12 + 0.45 * c["Pigmentation"] + 0.15 * c["Dullness"]
box[:, bi["Acne spot treatment"]] = r() < 0.04 + 0.60 * c["Acne"]
box[:, bi["Gentle cleanser"]] = r() < np.where(box[:, bi["Acne spot treatment"]] == 1, 0.68, 0.22) + 0.1 * c["Oiliness"]
box[:, bi["Sheet masks"]] = r() < 0.10 + 0.08 * (influencer - 1)
empty = box.sum(1) == 0
box[empty, bi["SPF 50 sunscreen"]] = 1

# ---------------------------------------------------------------- targets
seg_effect = pd.Series(seg).map({S[0]: 0.6, S[1]: 0.5, S[2]: 0.1, S[3]: -0.2, S[4]: -1.0}).values
latent = (-1.05 + seg_effect
          + 0.55 * (trust_ai - 3) + 0.35 * (hard_water - 3) + 0.30 * (derm_interest - 3)
          + 0.28 * (convenience - 3) - 0.40 * (price_sens - 3) - 0.45 * (satisfaction - 3)
          + 0.12 * concern_mat.sum(1) + 0.25 * np.log1p(spend / 100)
          - 0.10 * np.maximum(existing_subs - 3, 0)
          + rng.logistic(0, 0.9, N))
likelihood = np.digitize(latent, [-1.4, -0.2, 1.0, 2.4]) + 1  # 1..5

wtp = (25 + 0.22 * spend + 18 * income_num + 14 * (trust_ai - 3) - 22 * (price_sens - 3)
       + 9 * concern_mat.sum(1) + 10 * (derm_interest - 3) + 12 * (likelihood - 3)
       + 6 * (clean_halal - 3) + rng.normal(0, 28, N))
wtp = np.clip(wtp, 0, None).round(0)

plan = np.where(likelihood >= 4,
                rng.choice(["Monthly box", "Quarterly box", "One-time kit"], N, p=[0.55, 0.35, 0.10]),
                rng.choice(["Monthly box", "Quarterly box", "One-time kit", "Not interested"], N, p=[0.12, 0.18, 0.35, 0.35]))

# ---------------------------------------------------------------- assemble
join = lambda mat, labels: [";".join(l for l, v in zip(labels, row) if v) for row in mat]
df = pd.DataFrame({
    "respondent_id": [f"R{i:05d}" for i in range(1, N + 1)],
    "q1_age_group": age, "q2_gender": gender, "q3_emirate": emirate, "q4_nationality_region": nationality,
    "q5_occupation": occupation, "q6_monthly_income_aed": income, "q7_years_in_uae": years_uae,
    "q8_skin_type": skin_type, "q9_skin_concerns": join(concern_mat, CONCERNS),
    "q10_outdoor_hours_day": outdoor_hours, "q11_ac_hours_day": ac_hours, "q12_hard_water_impact": hard_water,
    "q13_monthly_skincare_spend_aed": spend, "q14_products_used": join(prod_mat, PRODUCTS),
    "q15_main_channel": channel, "q16_purchase_frequency": purchase_freq, "q17_brands_tried_last_year": brands_tried,
    "q18_routine_steps": routine_steps, "q19_existing_subscriptions": existing_subs,
    "q20_trust_personalised_reco": trust_ai, "q21_clean_halal_importance": clean_halal,
    "q22_price_sensitivity": price_sens, "q23_influencer_influence": influencer,
    "q24_sustainability_importance": sustainability, "q25_routine_satisfaction": satisfaction,
    "q26_derm_consult_interest": derm_interest, "q27_convenience_importance": convenience,
    "q28_box_items_wanted": join(box, BOX_ITEMS), "q29_preferred_plan": plan,
    "q30_subscribe_likelihood": likelihood, "q31_max_monthly_wtp_aed": wtp,
})

# ---------------------------------------------------------------- noise injection
noise_log = {}
# 1. messy categorical labels
idx = rng.choice(N, 140, replace=False)
df.loc[idx, "q3_emirate"] = rng.choice(["dubai", " Dubai", "DXB", "abu dhabi", "AUH", "sharjah ", "SHJ"], 140)
idx = rng.choice(N, 60, replace=False)
df.loc[idx, "q2_gender"] = rng.choice(["F", "female", "M", "male"], 60)
noise_log["messy_labels"] = 200
# 2. impossible values
idx = rng.choice(N, 25, replace=False); df.loc[idx, "q11_ac_hours_day"] = rng.choice([26, 30, 48], 25)
idx = rng.choice(N, 15, replace=False); df.loc[idx, "q10_outdoor_hours_day"] = -1
idx = rng.choice(N, 12, replace=False); df.loc[idx, "q13_monthly_skincare_spend_aed"] = -rng.integers(10, 200, 12)
noise_log["impossible_values"] = 52
# 3. outliers (fat-finger / extreme spenders)
idx = rng.choice(N, 40, replace=False); df.loc[idx, "q13_monthly_skincare_spend_aed"] *= rng.choice([10, 15, 25], 40)
idx = rng.choice(N, 30, replace=False); df.loc[idx, "q31_max_monthly_wtp_aed"] = rng.choice([2500, 5000, 9999], 30)
idx = rng.choice(N, 20, replace=False); df.loc[idx, "q7_years_in_uae"] = rng.choice([99, 150], 20)
noise_log["outliers"] = 90
# 4. missing values (~3% across selected columns)
for col in ["q6_monthly_income_aed", "q8_skin_type", "q13_monthly_skincare_spend_aed", "q15_main_channel",
            "q20_trust_personalised_reco", "q22_price_sensitivity", "q25_routine_satisfaction",
            "q31_max_monthly_wtp_aed", "q7_years_in_uae", "q12_hard_water_impact"]:
    idx = rng.choice(N, int(N * rng.uniform(0.02, 0.05)), replace=False)
    df.loc[idx, col] = np.nan
# 5. straight-liners (all Likerts = 3) — low-quality responses
likert_cols = [c for c in df.columns if c.startswith(("q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27"))]
idx = rng.choice(N, 55, replace=False)
df.loc[idx, likert_cols] = 3
noise_log["straight_liners"] = 55
# 6. duplicate submissions
dups = df.sample(60, random_state=SEED)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
noise_log["duplicates"] = 60

if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "survey_raw.csv", index=False)
    # hidden ground-truth segment kept separately (NOT used by the models)
    print(f"Wrote {len(df)} rows -> data/survey_raw.csv")
    print("Noise injected:", noise_log)
    print("Subscribe (4-5) share:", (likelihood >= 4).mean().round(3))
    print("Median WTP:", np.median(wtp))
