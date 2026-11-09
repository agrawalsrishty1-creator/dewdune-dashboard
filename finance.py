"""36-month financial model and sustainability scorecard for DewDune.

Every number here is an ASSUMPTION the user can change in the dashboard.
Survey-derived inputs (conversion, price mix, WTP) come from the models.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

MONTHS = 36


@dataclass
class Assumptions:
    # market
    adults_3_emirates: int = 6_000_000      # residents aged 18–54 in Dubai, Abu Dhabi, Sharjah (assumption)
    reach_share: float = 0.10               # share of SAM DewDune can realistically reach in 3 yrs
    # demand
    intent_realisation: float = 0.25        # share of "likely to subscribe" who actually buy
    cost_per_quiz_lead: float = 40.0        # AED paid media cost per completed quiz
    use_model_targeting: bool = True        # spend only on the high-intent segments found by clustering
    # price
    monthly_price: float = 149.0
    quarterly_price: float = 399.0
    addon_attach: float = 0.20              # share of boxes with an add-on
    addon_value: float = 75.0
    # costs per box
    cogs_pct: float = 0.35
    fulfilment_per_box: float = 22.0
    payment_pct: float = 0.029
    derm_cost_per_sub_month: float = 6.0
    # retention
    monthly_churn: float = 0.10
    referral_uplift: float = 0.10           # extra subs from referrals as share of paid subs
    # spend
    marketing_m1: float = 60_000.0
    marketing_growth: float = 0.04          # per month
    fixed_y1: float = 85_000.0              # team + office + tech per month
    fixed_y2: float = 120_000.0
    fixed_y3: float = 160_000.0
    launch_cost: float = 250_000.0          # registration, first inventory, website, brand


def derive_inputs(df, clf, prof, a: Assumptions):
    """Turn survey + model outputs into model inputs."""
    stated_untargeted = df["will_subscribe"].mean()
    # target segments = clusters whose subscribe rate beats the overall average
    tgt = prof[prof["subscribe_rate"] > stated_untargeted]
    sam_share = tgt["share"].sum() if len(tgt) else 0.5
    stated_targeted = (tgt["subscribe_rate"] * tgt["size"]).sum() / tgt["size"].sum() if len(tgt) else stated_untargeted
    stated = stated_targeted if a.use_model_targeting else stated_untargeted
    conv = stated * a.intent_realisation
    cac = a.cost_per_quiz_lead / max(conv, 1e-6)
    likely = df[df["will_subscribe"] == 1]
    mix = likely["q29_preferred_plan"].value_counts(normalize=True)
    m_share = mix.get("Monthly box", 0)
    q_share = mix.get("Quarterly box", 0)
    tot = m_share + q_share
    m_share, q_share = (m_share / tot, q_share / tot) if tot else (0.6, 0.4)
    box_price = m_share * a.monthly_price + q_share * a.quarterly_price / 3
    wtp = df["q31_max_monthly_wtp_aed"].dropna()
    wtp_likely = likely["q31_max_monthly_wtp_aed"].dropna()
    sam = a.adults_3_emirates * sam_share
    som = sam * a.reach_share * stated_untargeted * a.intent_realisation
    return dict(target_segments=list(tgt["name"]), sam_share=sam_share, sam=sam, som_subs=som, stated=stated, stated_untargeted=stated_untargeted,
                stated_targeted=stated_targeted, conversion=conv, cac=cac, monthly_share=m_share,
                quarterly_share=q_share, box_price=box_price,
                wtp_ge_price_all=(wtp >= a.monthly_price).mean(),
                wtp_ge_price_likely=(wtp_likely >= a.monthly_price).mean(),
                wtp_median_likely=wtp_likely.median())


def unit_economics(a: Assumptions, d):
    rev = d["box_price"] + a.addon_attach * a.addon_value
    var = (a.cogs_pct * rev + a.fulfilment_per_box + a.payment_pct * rev + a.derm_cost_per_sub_month)
    contrib = rev - var
    lifetime = 1 / a.monthly_churn
    ltv = contrib * lifetime
    return dict(arpu=rev, var_cost=var, contribution=contrib, margin=contrib / rev, lifetime_months=lifetime,
                ltv=ltv, ltv_cac=ltv / d["cac"], payback=d["cac"] / contrib if contrib > 0 else np.inf)


def simulate(a: Assumptions, d, ue):
    rows, subs, cash = [], 0.0, -a.launch_cost
    cap = d["som_subs"]
    for m in range(1, MONTHS + 1):
        mkt = a.marketing_m1 * (1 + a.marketing_growth) ** (m - 1)
        paid = mkt / d["cac"]
        new = paid * (1 + a.referral_uplift)
        # saturation: harder to acquire as we approach the reachable market
        sat = max(0.0, 1 - subs / cap) if cap > 0 else 0
        new *= sat
        churned = subs * a.monthly_churn
        subs = subs + new - churned
        revenue = subs * ue["arpu"]
        var = subs * ue["var_cost"]
        fixed = a.fixed_y1 if m <= 12 else a.fixed_y2 if m <= 24 else a.fixed_y3
        ebitda = revenue - var - fixed - mkt
        cash += ebitda
        rows.append(dict(month=m, new_subs=new, churned=churned, subscribers=subs, revenue=revenue,
                         variable_costs=var, marketing=mkt, fixed_costs=fixed,
                         contribution=revenue - var, ebitda=ebitda, cumulative_cash=cash))
    sim = pd.DataFrame(rows)
    pos = sim.index[sim["ebitda"] > 0]
    be = int(sim.loc[pos[0], "month"]) if len(pos) else None
    # sustained break-even: first month after which EBITDA stays positive
    sust = None
    for i in range(len(sim)):
        if (sim["ebitda"].iloc[i:] > 0).all():
            sust = int(sim["month"].iloc[i]); break
    yearly = sim.assign(year=(sim["month"] - 1) // 12 + 1).groupby("year")[
        ["revenue", "variable_costs", "marketing", "fixed_costs", "ebitda"]].sum()
    yearly["ebitda_margin"] = yearly["ebitda"] / yearly["revenue"]
    return dict(sim=sim, breakeven=sust, first_positive=be, funding_need=-sim["cumulative_cash"].min() + 0,
                yearly=yearly, subs_end=sim["subscribers"].iloc[-1])


def scorecard(a, d, ue, fin, prof):
    tgt = prof.sort_values("value_index", ascending=False).iloc[0]
    be_subs = (a.fixed_y3) / ue["contribution"] if ue["contribution"] > 0 else np.inf
    checks = [
        ("Demand", "Stated subscribe intent (4–5 on Q30) among targeted prospects ≥ 40%",
         d["stated"], d["stated"] >= 0.40, f"{d['stated']:.0%}"),
        ("Price fit", f"Likely subscribers whose max WTP ≥ AED {a.monthly_price:.0f} is ≥ 50%",
         d["wtp_ge_price_likely"], d["wtp_ge_price_likely"] >= 0.50, f"{d['wtp_ge_price_likely']:.0%}"),
        ("Unit economics", "LTV : CAC ≥ 3", ue["ltv_cac"], ue["ltv_cac"] >= 3, f"{ue['ltv_cac']:.1f}×"),
        ("Payback", "CAC paid back in ≤ 12 months", ue["payback"], ue["payback"] <= 12, f"{ue['payback']:.1f} months"),
        ("Profitability", "EBITDA turns positive and stays positive within 36 months",
         fin["breakeven"] or 99, fin["breakeven"] is not None, f"Month {fin['breakeven']}" if fin["breakeven"] else "Not within 36 months"),
        ("Market headroom", "Subscribers needed to cover year-3 fixed costs ≤ 50% of reachable market",
         be_subs / d["som_subs"] if d["som_subs"] else np.inf, be_subs <= 0.5 * d["som_subs"],
         f"{be_subs:,.0f} needed vs {d['som_subs']:,.0f} reachable"),
        ("Clear beachhead", "One segment has above-average intent and median WTP ≥ price",
         tgt["subscribe_rate"], (tgt["subscribe_rate"] > d["stated_untargeted"]) and (tgt["wtp_median"] >= a.monthly_price),
         f"{tgt['name']}: {tgt['subscribe_rate']:.0%} intent, AED {tgt['wtp_median']:.0f} WTP"),
    ]
    sc = pd.DataFrame(checks, columns=["Test", "Rule", "value", "Pass", "Result"])
    n = int(sc["Pass"].sum())
    critical = sc.set_index("Test").loc[["Unit economics", "Profitability"], "Pass"].all()
    if n >= 6 and critical:
        verdict = ("Sustainable", "The idea passes the demand, pricing and unit-economics tests under these assumptions.")
    elif n >= 4 and critical:
        verdict = ("Sustainable with conditions", "The economics work, but some tests fail. Fix those before scaling.")
    elif n >= 5 and fin["breakeven"] is not None and ue["ltv_cac"] >= 2:
        verdict = ("Borderline: viable with thin margins",
                   "Demand and pricing are strong and the business breaks even, but LTV:CAC is below the 3× benchmark. "
                   "Cutting churn or acquisition cost is what makes it sustainable.")
    elif n >= 3:
        verdict = ("Not yet sustainable", "Demand exists but the economics don't close under these assumptions.")
    else:
        verdict = ("Not sustainable", "Too many core tests fail under these assumptions.")
    return sc, n, verdict, be_subs
