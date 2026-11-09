"""Static content: startup plan and questionnaire (with algorithm mapping)."""

BRAND = "DewDune"
TAGLINE = "Climate-smart, personalised skincare — delivered monthly across the UAE"

PLAN = {
    "Problem": [
        "UAE residents live in an unusual skin environment: 40°C+ heat and UV index 11+ for half the year, "
        "15+ hours a day in air-conditioning that strips moisture, and desalinated hard water that leaves skin tight and hair brittle.",
        "Most products on UAE shelves were formulated for European or East Asian climates. Consumers, especially the ~88% expat population, "
        "go through trial-and-error buying, so they spend money on products that don't work.",
        "Advice is scattered across influencers, pharmacy staff and global brand websites, with nobody accounting for local conditions.",
    ],
    "Solution": [
        "A 3-minute online skin + lifestyle quiz (skin type, concerns, outdoor/AC exposure, hard-water impact, budget).",
        "A recommendation engine builds a 3–5 product routine from a curated catalogue of dermatologist-approved, halal-certified, climate-tested products "
        "(own-label hero SKUs + partner brands).",
        "Delivered as a Monthly or Quarterly box, with a free tele-derm check-in every quarter and automatic seasonal adjustments (summer UV vs. winter dryness).",
        "Add-ons that no competitor bundles: shower filters, after-sun care, hair repair for hard water.",
    ],
    "Target customer": [
        "Primary: urban residents aged 22–40 in Dubai, Abu Dhabi and Sharjah who earn AED 10k+ a month and already spend AED 150+ a month on skincare.",
        "Beachhead: recently arrived expats (<3 years in UAE) who are struggling with dryness, pigmentation and hard-water hair fall.",
        "The survey and clustering model confirm or reject this targeting. See the Clustering page.",
    ],
    "Revenue model": [
        "Monthly box: AED 149 (3 full-size or travel products, rotated). Quarterly box: AED 399 (saves ~11%).",
        "One-time Starter kit: AED 199. Add-ons at AED 39–129 (shower filter, after-sun gel, lip SPF).",
        "Affiliate/brand-placement fees from partner brands that want trial with targeted UAE consumers (year 2+).",
    ],
    "Go-to-market": [
        "Months 0–3: Instagram/TikTok quiz funnel and micro-influencers (10–50k followers) in Dubai, plus pop-ups at Dubai Hills Mall and university campuses.",
        "Months 4–12: Referral programme (give AED 50, get AED 50), Noon/Amazon storefront for one-time kits, and corporate wellness partnerships.",
        "Year 2: Expand to KSA (Riyadh/Jeddah), which has a similar climate with no local personalised player.",
    ],
    "Competition": [
        "Pharmacies (Aster, Boots, Life): trusted but generic. No personalisation and no subscription.",
        "Sephora/Faces: prestige range but expensive, and recommendations are driven by sales staff.",
        "Global D2C personalisation brands (e.g., Curology, Proven): not localised, no GCC shipping or halal positioning.",
        "DewDune's edge: climate data + hard-water focus + halal + subscription convenience.",
    ],
    "Key risks": [
        "Stated survey intent overstates real purchase. The financial model applies an intent-realisation discount.",
        "Subscription fatigue and churn. Quarterly plans and seasonal personalisation are designed to reduce it.",
        "Regulatory: UAE cosmetic product registration (Dubai Municipality / MoIAT) adds 2–4 months before launch.",
        "CAC inflation on Meta/TikTok in Q4 (festive season).",
    ],
}

# Each question: (id, text, type, options/scale, algorithm role)
ALGO_CLS = "Classification (feature)"
ALGO_REG = "Regression (feature)"
ALGO_CLU = "Clustering"
ALGO_ARM = "Association rules"
QUESTIONS = [
    ("Q1", "What is your age group?", "Single choice", "18-24 / 25-34 / 35-44 / 45-54 / 55+", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q2", "What is your gender?", "Single choice", "Female / Male / Prefer not to say", [ALGO_CLS, ALGO_REG]),
    ("Q3", "Which emirate do you live in?", "Single choice", "Dubai / Abu Dhabi / Sharjah / Ajman / Ras Al Khaimah", [ALGO_CLS, ALGO_REG, "Market sizing"]),
    ("Q4", "Which region best describes your nationality?", "Single choice", "South Asian / Arab / Western / East-SE Asian / African / Other", [ALGO_CLS, ALGO_REG]),
    ("Q5", "What is your occupation?", "Single choice", "Student / Salaried professional / Business owner-Freelancer / Homemaker / Other", [ALGO_CLS, ALGO_REG]),
    ("Q6", "What is your monthly personal income (AED)?", "Single choice", "<5k / 5-10k / 10-20k / 20-35k / 35k+", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q7", "How many years have you lived in the UAE?", "Numeric", "0–60", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q8", "What is your skin type?", "Single choice", "Oily / Dry / Combination / Normal / Sensitive", [ALGO_CLS, ALGO_REG]),
    ("Q9", "Which skin/hair concerns do you have? (select all)", "Multi-select", "Acne, Pigmentation, Dryness, Oiliness, Sensitivity, Dullness, Fine lines, Sun damage, Hard-water hair fall", [ALGO_CLS, ALGO_REG, ALGO_CLU, ALGO_ARM]),
    ("Q10", "On a typical day, how many hours are you outdoors?", "Numeric", "0–24", [ALGO_CLS, ALGO_CLU, ALGO_ARM]),
    ("Q11", "On a typical day, how many hours are you in air-conditioning?", "Numeric", "0–24", [ALGO_CLS, ALGO_CLU]),
    ("Q12", "How much does UAE tap water affect your skin/hair?", "Likert 1–5", "1 = Not at all … 5 = Severely", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q13", "How much do you spend on skincare per month (AED)?", "Numeric", "AED", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q14", "Which products do you currently use? (select all)", "Multi-select", "Cleanser, Moisturiser, Sunscreen, Serum, Toner, Face mask, Exfoliant, Eye cream, Lip balm, Shower filter", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q15", "Where do you mainly buy skincare?", "Single choice", "Pharmacy / Sephora-Faces / Amazon-Noon / Brand website / Instagram-TikTok shop / Supermarket", [ALGO_CLS, ALGO_REG]),
    ("Q16", "How often do you buy skincare?", "Single choice", "Weekly / Monthly / Every 2-3 months / Rarely", [ALGO_CLS, ALGO_REG]),
    ("Q17", "How many skincare brands did you try in the last 12 months?", "Numeric", "0–25", [ALGO_CLS, ALGO_CLU]),
    ("Q18", "How many steps are in your daily routine?", "Numeric", "1–12", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q19", "How many paid subscriptions do you currently have (any category)?", "Numeric", "0–10", [ALGO_CLS]),
    ("Q20", "I trust personalised / AI-based product recommendations.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q21", "Clean / halal-certified ingredients are important to me.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_REG]),
    ("Q22", "Price is the most important factor when I buy skincare.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_REG, ALGO_CLU]),
    ("Q23", "Influencers affect what skincare I buy.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_CLU]),
    ("Q24", "I prefer brands with sustainable packaging.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_REG]),
    ("Q25", "I am satisfied with my current skincare routine.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_CLU]),
    ("Q26", "I would value a free online dermatologist consultation.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_REG]),
    ("Q27", "Having products delivered automatically saves me time.", "Likert 1–5", "Strongly disagree … Strongly agree", [ALGO_CLS, ALGO_CLU]),
    ("Q28", "Which items would you want in your DewDune box? (select all)", "Multi-select", "SPF 50 sunscreen, Hydrating serum, Gentle cleanser, Barrier moisturiser, Shower filter, Vitamin C serum, Acne spot treatment, Hair repair mask, Lip SPF, After-sun gel, Sheet masks", [ALGO_ARM + " (basket)"]),
    ("Q29", "Which plan would you choose?", "Single choice", "Monthly box / Quarterly box / One-time kit / Not interested", ["Descriptive / pricing"]),
    ("Q30", "How likely are you to subscribe to DewDune?", "Likert 1–5", "1 = Very unlikely … 5 = Very likely", ["CLASSIFICATION TARGET (4–5 = Will subscribe)"]),
    ("Q31", "What is the most you would pay per month for a personalised box (AED)?", "Numeric", "AED", ["REGRESSION TARGET (willingness to pay)"]),
]

ALGORITHMS = [
    ("Classification", "Will this person subscribe? (Q30 → Yes if 4–5)",
     "Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN",
     "Lets us predict and target the people most likely to convert, which lowers CAC."),
    ("Regression", "How much will they pay per month? (Q31)",
     "Linear, Ridge, Lasso, Random Forest, Gradient Boosting regressors",
     "Sets the price point and tells us which traits raise willingness to pay."),
    ("Clustering", "Which customer segments exist? (Q1, Q6–Q13, Q17–Q18, Q20, Q22–Q23, Q25, Q27)",
     "K-Means (elbow + silhouette to choose k), PCA for visualisation",
     "Builds personas and picks the beachhead segment for go-to-market."),
    ("Association rules", "Which box items go together? (Q28)",
     "Apriori (support, confidence, lift)",
     "Designs bundles and cross-sell add-ons, which raises average order value."),
]
