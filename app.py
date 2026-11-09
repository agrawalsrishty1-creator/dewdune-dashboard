"""DewDune — business viability dashboard (Streamlit)."""
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from content import BRAND, TAGLINE, PLAN, QUESTIONS, ALGORITHMS
from pipeline import clean, features, pretty, CONCERNS, BOX_ITEMS, LIKERT
from analysis import run_classification, run_regression, run_clustering, profile_clusters, run_arules
from finance import Assumptions, derive_inputs, unit_economics, simulate, scorecard

warnings.filterwarnings("ignore")
st.set_page_config(page_title=f"{BRAND} · Viability Dashboard", page_icon="💧", layout="wide")

# ------------------------------------------------------------------ look & feel
TEAL, SAND, INK, CORAL, SLATE, SKY = "#0F766E", "#C8963E", "#1F2937", "#D9534F", "#64748B", "#3B82F6"
PALETTE = [TEAL, SAND, SKY, CORAL, "#7C3AED", SLATE, "#059669", "#DB2777"]
pio.templates["dew"] = go.layout.Template(layout=dict(
    colorway=PALETTE, font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=10, r=10, t=50, b=10), title=dict(font=dict(size=15)),
    legend=dict(orientation="h", y=-0.18)))
pio.templates.default = "plotly_white+dew"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.verdict {border-radius: 14px; padding: 1.2rem 1.5rem; margin: .4rem 0 1rem; color: white;}
.verdict h2 {margin: 0 0 .3rem; color: white; font-size: 1.7rem;}
.verdict p {margin: 0; font-size: 1.02rem; opacity: .95;}
.card {border: 1px solid rgba(100,116,139,.25); border-radius: 12px; padding: 1rem 1.1rem; height: 100%;}
.card h4 {margin: 0 0 .4rem; font-size: 1.02rem;}
.small {font-size: .88rem; color: #64748B;}
.pill {display:inline-block; padding: .1rem .55rem; border-radius: 99px; background: rgba(15,118,110,.12);
       color: #0F766E; font-size: .78rem; margin: 0 .25rem .25rem 0;}
</style>""", unsafe_allow_html=True)

DATA = Path(__file__).parent / "data" / "survey_raw.csv"


# ------------------------------------------------------------------ cached compute
@st.cache_data(show_spinner="Cleaning survey data…")
def load():
    raw = pd.read_csv(DATA)
    df, log = clean(raw)
    return raw, features(df), log


@st.cache_resource(show_spinner="Training 5 classification models…")
def get_clf(_df):
    return run_classification(_df)


@st.cache_resource(show_spinner="Training 5 regression models…")
def get_reg(_df):
    return run_regression(_df)


@st.cache_resource(show_spinner="Clustering respondents…")
def get_clu(_df, k=None):
    res = run_clustering(_df, k)
    return res, profile_clusters(_df, res["labels"])


@st.cache_data(show_spinner="Mining association rules…")
def get_rules(_df, sup, conf):
    return run_arules(_df, sup, conf)


raw, df, clean_log = load()


def fmt_aed(x):
    if abs(x) >= 1e6:
        return f"AED {x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"AED {x / 1e3:,.0f}k"
    return f"AED {x:,.0f}"


def insight(text, kind="info"):
    getattr(st, kind)(text, icon={"info": "💡", "success": "✅", "warning": "⚠️", "error": "⛔"}[kind])


# ------------------------------------------------------------------ assumptions (sidebar, shared)
DEFAULTS = asdict(Assumptions())


def assumption_controls():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault("_" + k, v)

    def w(fn, label, key, **kw):
        val = fn(label, value=st.session_state["_" + key], key="w_" + key, **kw)
        st.session_state["_" + key] = val
        return val

    def pct(label, key, lo, hi, step=1, **kw):
        cur = int(round(st.session_state["_" + key] * 100))
        val = st.slider(label, min_value=lo, max_value=hi, value=cur, step=step, format="%d%%", key="w_" + key, **kw)
        st.session_state["_" + key] = val / 100
        return val / 100

    with st.sidebar:
        st.markdown("### Assumptions")
        st.caption("Change these to stress-test the verdict. Survey-based inputs come from the models.")
        with st.expander("Demand & acquisition", expanded=True):
            w(st.toggle, "Target only high-intent segments", "use_model_targeting",
              help="Spend ads only on the clusters with above-average subscribe intent")
            pct("Intent → actual purchase", "intent_realisation", 5, 60, 5,
              help="Share of people who say 'likely' (4–5) that actually subscribe. 20–35% is typical.")
            w(st.slider, "Ad cost per completed quiz (AED)", "cost_per_quiz_lead", min_value=10.0, max_value=100.0, step=5.0)
            pct("Monthly churn", "monthly_churn", 3, 25)
            pct("Referral uplift", "referral_uplift", 0, 40, 5)
        with st.expander("Pricing"):
            w(st.slider, "Monthly box price (AED)", "monthly_price", min_value=79.0, max_value=299.0, step=10.0)
            w(st.slider, "Quarterly box price (AED)", "quarterly_price", min_value=199.0, max_value=799.0, step=10.0)
            pct("Add-on attach rate", "addon_attach", 0, 60, 5)
        with st.expander("Costs"):
            pct("Product cost (% of revenue)", "cogs_pct", 20, 60)
            w(st.slider, "Fulfilment & delivery per box (AED)", "fulfilment_per_box", min_value=10.0, max_value=45.0, step=1.0)
            w(st.number_input, "Marketing budget, month 1 (AED)", "marketing_m1", min_value=10_000.0, step=5_000.0)
            pct("Marketing budget growth / month", "marketing_growth", 0, 10)
            w(st.number_input, "Fixed costs / month, year 1 (AED)", "fixed_y1", min_value=20_000.0, step=5_000.0)
            w(st.number_input, "Fixed costs / month, year 2 (AED)", "fixed_y2", min_value=20_000.0, step=5_000.0)
            w(st.number_input, "Fixed costs / month, year 3 (AED)", "fixed_y3", min_value=20_000.0, step=5_000.0)
            w(st.number_input, "Launch cost (AED)", "launch_cost", min_value=0.0, step=10_000.0)
        with st.expander("Market"):
            w(st.number_input, "Adults 18–54 in Dubai, Abu Dhabi, Sharjah", "adults_3_emirates", min_value=500_000, step=250_000)
            pct("Share of target market reachable in 3 yrs", "reach_share", 2, 40)
        if st.button("Reset to defaults", width="stretch"):
            for k, v in DEFAULTS.items():
                st.session_state["_" + k] = v
                st.session_state.pop("w_" + k, None)
            st.rerun()
    return Assumptions(**{k: st.session_state["_" + k] for k in DEFAULTS})


def run_finance(a):
    clf = get_clf(df)
    _, prof = get_clu(df)
    d = derive_inputs(df, clf, prof, a)
    ue = unit_economics(a, d)
    fin = simulate(a, d, ue)
    sc, n, verdict, be_subs = scorecard(a, d, ue, fin, prof)
    return clf, prof, d, ue, fin, sc, n, verdict, be_subs


# ================================================================== PAGES
def page_verdict():
    a = assumption_controls()
    clf, prof, d, ue, fin, sc, n, verdict, be_subs = run_finance(a)
    st.title(f"💧 {BRAND}")
    st.caption(TAGLINE + " · Is this business sustainable?")

    colour = {"Sustainable": "#047857", "Sustainable with conditions": TEAL,
              "Borderline: viable with thin margins": SAND, "Not yet sustainable": "#B45309"}.get(verdict[0], CORAL)
    st.markdown(f"""<div class="verdict" style="background:{colour}">
        <h2>Verdict: {verdict[0]}</h2><p>{verdict[1]} <b>{n} of {len(sc)} tests passed.</b></p></div>""",
                unsafe_allow_html=True)

    c = st.columns(3) + st.columns(3)
    c[0].metric("Stated intent (targeted)", f"{d['stated']:.0%}", border=True, help="Share answering 4–5 on Q30 in the segments we'd target")
    c[1].metric("Real conversion", f"{d['conversion']:.1%}", border=True, help="Stated intent × intent-to-purchase discount")
    c[2].metric("CAC", fmt_aed(d["cac"]), border=True, help="Ad cost per quiz ÷ real conversion")
    c[3].metric("LTV : CAC", f"{ue['ltv_cac']:.1f}×", border=True, help="Lifetime contribution per subscriber ÷ CAC. Benchmark ≥ 3×")
    c[4].metric("Break-even", f"Month {fin['breakeven']}" if fin["breakeven"] else "> 36 mo", border=True, help="First month after which EBITDA stays positive")
    c[5].metric("Funding needed", fmt_aed(fin["funding_need"]), border=True, help="Deepest point of cumulative cash incl. launch cost")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Sustainability scorecard")
        for _, r in sc.iterrows():
            ok = bool(r["Pass"])
            st.markdown(
                f'<div class="card" style="margin-bottom:.45rem;padding:.6rem .9rem;border-left:5px solid '
                f'{"#047857" if ok else CORAL}"><b>{"✅" if ok else "❌"} {r["Test"]}</b> · {r["Result"]}'
                f'<div class="small">{r["Rule"]}</div></div>', unsafe_allow_html=True)
    with right:
        st.subheader("What would change the verdict")
        max_cac = ue["ltv"] / 3
        max_churn = ue["contribution"] / (3 * d["cac"]) if d["cac"] else 0
        need_real = a.cost_per_quiz_lead * 3 / (ue["ltv"] * d["stated"]) if ue["ltv"] > 0 else np.nan
        st.markdown(f"""
- To hit **LTV:CAC = 3×**, CAC must be **≤ {fmt_aed(max_cac)}** (now {fmt_aed(d['cac'])}).
- Or keep CAC as it is and get monthly churn down to **≤ {max_churn:.1%}** (now {a.monthly_churn:.0%}).
- Or get **≥ {need_real:.0%}** of 'likely' respondents to actually buy (now {a.intent_realisation:.0%}).
- Each subscriber brings in **{fmt_aed(ue['arpu'])}/month** and contributes **{fmt_aed(ue['contribution'])}** after product, delivery and payment costs ({ue['margin']:.0%} margin).
- About **{be_subs:,.0f} active subscribers** cover year-3 fixed costs. That's {be_subs / d['som_subs']:.0%} of the {d['som_subs']:,.0f} subscribers we could realistically reach.
""")
        targeting = "on" if a.use_model_targeting else "off"
        st.caption(f"Targeting is {targeting}. Target segments: {', '.join(d['target_segments'])}.")

    st.subheader("36-month projection")
    sim = fin["sim"]
    t1, t2 = st.columns(2)
    with t1:
        fig = go.Figure()
        fig.add_bar(x=sim["month"], y=sim["revenue"], name="Revenue", marker_color=TEAL)
        fig.add_bar(x=sim["month"], y=-(sim["variable_costs"] + sim["marketing"] + sim["fixed_costs"]),
                    name="Total costs", marker_color="#CBD5E1")
        fig.add_scatter(x=sim["month"], y=sim["ebitda"], name="EBITDA", line=dict(color=INK, width=2.5))
        fig.update_layout(title="Monthly revenue, costs and EBITDA (AED)", barmode="relative", height=380,
                          xaxis_title="Month")
        st.plotly_chart(fig)
    with t2:
        fig = go.Figure()
        fig.add_scatter(x=sim["month"], y=sim["cumulative_cash"], name="Cumulative cash", fill="tozeroy",
                        line=dict(color=SAND, width=2.5))
        fig.add_scatter(x=sim["month"], y=sim["subscribers"], name="Active subscribers", yaxis="y2",
                        line=dict(color=TEAL, width=2.5, dash="dot"))
        fig.add_hline(y=0, line_color=SLATE, line_width=1)
        fig.update_layout(title="Cumulative cash (AED) and active subscribers", height=380, xaxis_title="Month",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Subscribers", rangemode="tozero"))
        st.plotly_chart(fig)

    yr = fin["yearly"].copy()
    yr.index = [f"Year {i}" for i in yr.index]
    yr = yr.rename(columns={"revenue": "Revenue", "variable_costs": "Variable costs", "marketing": "Marketing",
                            "fixed_costs": "Fixed costs", "ebitda": "EBITDA", "ebitda_margin": "EBITDA margin"})
    st.dataframe(yr.style.format({c: "AED {:,.0f}" for c in yr.columns if c != "EBITDA margin"} | {"EBITDA margin": "{:.1%}"}),
                 width="stretch")

    st.subheader("Sensitivity: LTV:CAC by churn and intent-to-purchase")
    churns = np.round(np.arange(0.05, 0.21, 0.025), 3)
    reals = np.round(np.arange(0.10, 0.51, 0.05), 2)
    grid = np.array([[ue["contribution"] / ch / (a.cost_per_quiz_lead / (d["stated"] * r)) for r in reals] for ch in churns])
    fig = px.imshow(grid, x=[f"{r:.0%}" for r in reals], y=[f"{c:.1%}" for c in churns], text_auto=".1f",
                    color_continuous_scale=[[0, CORAL], [0.3, "#FDE68A"], [0.45, "#A7F3D0"], [1, TEAL]],
                    zmin=0, zmax=8, aspect="auto",
                    labels=dict(x="Intent → purchase", y="Monthly churn", color="LTV:CAC"))
    fig.update_layout(height=360, title="Green cells (≥3×) are sustainable")
    st.plotly_chart(fig)

    st.subheader("What each algorithm told us")
    reg = get_reg(df)
    rules = get_rules(df, 0.05, 0.4)["rules"]
    top_seg = prof.sort_values("value_index", ascending=False).iloc[0]
    r0 = rules.iloc[0] if len(rules) else None
    k = st.columns(4)
    cards = [
        ("🎯 Classification", f"<b>{clf['best']}</b> predicts subscribers with ROC-AUC "
         f"<b>{clf['results'].iloc[0]['ROC-AUC']:.2f}</b>. Top-scored 10% subscribe at "
         f"<b>{clf['lift'].iloc[0, 1]:.0%}</b> vs a {clf['base_rate']:.0%} average, a "
         f"{clf['lift'].iloc[0, 1] / clf['base_rate']:.1f}× lift for ad targeting."),
        ("💰 Regression", f"<b>{reg['best']}</b> explains <b>{reg['results'].iloc[0]['R²']:.0%}</b> of willingness to pay. "
         f"Likely subscribers would pay a median <b>AED {d['wtp_median_likely']:.0f}</b>. "
         f"Current spend and price sensitivity are the main drivers."),
        ("🧩 Clustering", f"<b>{len(prof)} segments</b>. Best beachhead: <b>{top_seg['name']}</b> "
         f"({top_seg['share']:.0%} of market, {top_seg['subscribe_rate']:.0%} intent, AED {top_seg['wtp_median']:.0f} WTP)."),
        ("🛒 Association rules", (f"Strongest bundle: <b>{r0['antecedents']} → {r0['consequents']}</b> "
                                  f"(lift {r0['lift']:.1f}). Hard-water and sun-care items cluster together, which supports add-on bundles.")
         if r0 is not None else "No rules at current thresholds."),
    ]
    for col, (h, body) in zip(k, cards):
        col.markdown(f'<div class="card"><h4>{h}</h4><div class="small">{body}</div></div>', unsafe_allow_html=True)
    st.caption("Survey data is synthetic (simulated for this project). Financial inputs not taken from the survey are assumptions you can change in the sidebar.")


def page_plan():
    st.title("Startup plan")
    st.markdown(f"### {BRAND}\n*{TAGLINE}*")
    icons = {"Problem": "🌡️", "Solution": "🧪", "Target customer": "👥", "Revenue model": "💳",
             "Go-to-market": "🚀", "Competition": "⚔️", "Key risks": "⚠️"}
    keys = list(PLAN)
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for col, key in zip(cols, keys[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"#### {icons.get(key, '')} {key}")
                for b in PLAN[key]:
                    st.markdown(f"- {b}")
    st.subheader("Pricing")
    st.dataframe(pd.DataFrame({
        "Product": ["Monthly box", "Quarterly box", "Starter kit (one-time)", "Add-ons"],
        "Price (AED)": ["149", "399 (saves ~11%)", "199", "39–129"],
        "What's inside": ["3 products, rotated each month", "3 months of products, sent together",
                          "Full routine to try once", "Shower filter, after-sun gel, lip SPF, hair mask"]}),
        hide_index=True, width="stretch")


def page_questionnaire():
    st.title("Questionnaire ↔ algorithms")
    st.markdown("Every question exists to feed at least one algorithm. Q30 is the **classification target**, "
                "Q31 is the **regression target**, Q28 is the **basket** for association rules, and the behaviour and "
                "attitude questions drive **clustering**.")
    cols = st.columns(4)
    for col, (name, q, algos, why) in zip(cols, ALGORITHMS):
        col.markdown(f'<div class="card"><h4>{name}</h4><div class="small"><b>Question:</b> {q}<br><br>'
                     f'<b>Models:</b> {algos}<br><br><b>Business use:</b> {why}</div></div>', unsafe_allow_html=True)
    st.write("")
    qdf = pd.DataFrame([{"ID": q[0], "Question": q[1], "Type": q[2], "Options / scale": q[3], "Used for": ", ".join(q[4])}
                        for q in QUESTIONS])
    st.dataframe(qdf, hide_index=True, width="stretch", height=560)

    st.subheader("Coverage map")
    algos = ["Classification", "Regression", "Clustering", "Association rules"]
    mat = pd.DataFrame(0, index=[q[0] for q in QUESTIONS], columns=algos)
    for q in QUESTIONS:
        for role in q[4]:
            for al in algos:
                if role.upper().startswith(al.upper()):
                    mat.loc[q[0], al] = 2 if "TARGET" in role or "basket" in role else 1
    fig = px.imshow(mat.T, color_continuous_scale=["#F1F5F9", "#99F6E4", TEAL], aspect="auto")
    fig.update_layout(height=260, coloraxis_showscale=False, title="Light = input feature · Dark = target / basket")
    fig.update_traces(xgap=2, ygap=2)
    st.plotly_chart(fig)


def page_data():
    st.title("Data & cleaning")
    st.markdown("The survey responses were **simulated** from five hidden consumer types. Real-world problems were then "
                "added on purpose, and the cleaning step below finds and fixes them.")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Clean respondents", f"{len(df):,}")
    c[2].metric("Removed", f"{len(raw) - len(df):,}")
    c[3].metric("Questions", "31")
    st.subheader("Cleaning log")
    st.dataframe(clean_log, hide_index=True, width="stretch")

    st.subheader("Before vs after")
    a, b = st.columns(2)
    with a:
        miss = raw.isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        fig = px.bar(x=miss.values, y=[pretty(i) for i in miss.index], orientation="h",
                     labels={"x": "Share missing", "y": ""}, title="Missing answers in the raw data")
        fig.update_layout(height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig)
    with b:
        col = st.selectbox("Variable", ["q13_monthly_skincare_spend_aed", "q31_max_monthly_wtp_aed",
                                        "q11_ac_hours_day", "q7_years_in_uae"], format_func=pretty)
        both = pd.concat([pd.DataFrame({"value": raw[col], "data": "Raw"}),
                          pd.DataFrame({"value": df[col], "data": "Clean"})])
        fig = px.box(both, x="data", y="value", color="data", points="outliers",
                     title=f"{pretty(col)}: outliers removed", color_discrete_sequence=[CORAL, TEAL])
        fig.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig)
    e1, e2 = st.columns(2)
    with e1:
        vc = raw["q3_emirate"].value_counts()
        fig = px.bar(x=vc.index, y=vc.values, title="Emirate labels BEFORE cleaning", labels={"x": "", "y": "Count"},
                     color_discrete_sequence=[CORAL])
        st.plotly_chart(fig)
    with e2:
        vc = df["q3_emirate"].value_counts()
        fig = px.bar(x=vc.index, y=vc.values, title="Emirate labels AFTER cleaning", labels={"x": "", "y": "Count"},
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(fig)

    with st.expander("Preview raw data"):
        st.dataframe(raw.head(200), width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download raw CSV", raw.to_csv(index=False), "dewdune_survey_raw.csv", width="stretch")
    keep = [c for c in df.columns if c.startswith("q") or c == "respondent_id"]
    d2.download_button("⬇️ Download clean CSV", df[keep].to_csv(index=False), "dewdune_survey_clean.csv", width="stretch")


def page_eda():
    st.title("Exploratory analysis")
    with st.sidebar:
        st.markdown("### Filters")
        em = st.multiselect("Emirate", sorted(df["q3_emirate"].unique()), placeholder="All")
        ag = st.multiselect("Age group", ["18-24", "25-34", "35-44", "45-54", "55+"], placeholder="All")
        ge = st.multiselect("Gender", sorted(df["q2_gender"].unique()), placeholder="All")
    d = df.copy()
    if em: d = d[d["q3_emirate"].isin(em)]
    if ag: d = d[d["q1_age_group"].isin(ag)]
    if ge: d = d[d["q2_gender"].isin(ge)]
    if d.empty:
        st.warning("No respondents match these filters.")
        return
    c = st.columns(5)
    c[0].metric("Respondents", f"{len(d):,}")
    c[1].metric("Likely to subscribe (4–5)", f"{d['will_subscribe'].mean():.0%}")
    c[2].metric("Median skincare spend", f"AED {d['q13_monthly_skincare_spend_aed'].median():.0f}")
    c[3].metric("Median max WTP", f"AED {d['q31_max_monthly_wtp_aed'].median():.0f}")
    c[4].metric("Hard water hurts (4–5)", f"{(d['q12_hard_water_impact'] >= 4).mean():.0%}")

    r1 = st.columns(2)
    with r1[0]:
        cc = d[[f"concern_{x.replace(' ', '_')}" for x in CONCERNS]].mean().sort_values()
        fig = px.bar(x=cc.values, y=[i[8:].replace("_", " ") for i in cc.index], orientation="h",
                     title="Skin & hair concerns (Q9)", labels={"x": "Share of respondents", "y": ""})
        fig.update_layout(xaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig)
    with r1[1]:
        lk = d["q30_subscribe_likelihood"].value_counts().sort_index()
        fig = px.bar(x=lk.index.astype(str), y=lk.values, title="How likely to subscribe? (Q30)",
                     labels={"x": "1 = very unlikely … 5 = very likely", "y": "Respondents"},
                     color=lk.index >= 4, color_discrete_map={True: TEAL, False: "#CBD5E1"})
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig)
    r2 = st.columns(2)
    with r2[0]:
        fig = px.histogram(d, x="q31_max_monthly_wtp_aed", nbins=50, title="Max willingness to pay vs AED 149 price (Q31)",
                           labels={"q31_max_monthly_wtp_aed": "AED / month"}, color_discrete_sequence=[TEAL])
        fig.add_vline(x=149, line_dash="dash", line_color=CORAL, annotation_text="AED 149")
        fig.update_layout(height=380, yaxis_title="Respondents")
        st.plotly_chart(fig)
    with r2[1]:
        order = ["<5k", "5-10k", "10-20k", "20-35k", "35k+"]
        fig = px.box(d, x="q6_monthly_income_aed", y="q13_monthly_skincare_spend_aed", category_orders={"q6_monthly_income_aed": order},
                     title="Monthly skincare spend by income", labels={"q6_monthly_income_aed": "Income (AED)",
                                                                       "q13_monthly_skincare_spend_aed": "Spend (AED)"},
                     color_discrete_sequence=[SAND], points=False)
        fig.update_layout(height=380)
        st.plotly_chart(fig)
    r3 = st.columns(2)
    with r3[0]:
        g = d.assign(yrs=pd.cut(d["q7_years_in_uae"], [-0.1, 1, 3, 5, 10, 60], labels=["<1", "1–3", "3–5", "5–10", "10+"]))
        g = g.groupby("yrs", observed=True).agg(hard=("q12_hard_water_impact", "mean"), sub=("will_subscribe", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=g["yrs"].astype(str), y=g["hard"], name="Hard-water impact (1–5)", marker_color=SKY)
        fig.add_scatter(x=g["yrs"].astype(str), y=g["sub"], name="Likely to subscribe", yaxis="y2",
                        line=dict(color=TEAL, width=3))
        fig.update_layout(title="Newer expats feel hard water more and are keener to subscribe", height=380,
                          xaxis_title="Years in UAE", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
        st.plotly_chart(fig)
    with r3[1]:
        ch = d.groupby("q15_main_channel").agg(n=("respondent_id", "count"), sub=("will_subscribe", "mean")).reset_index()
        fig = px.scatter(ch, x="n", y="sub", size="n", text="q15_main_channel", title="Where they shop vs intent",
                         labels={"n": "Respondents", "sub": "Likely to subscribe"}, color_discrete_sequence=[SAND])
        fig.update_traces(textposition="top center")
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig)

    st.subheader("Attitudes: likely vs unlikely subscribers")
    att = d.groupby("will_subscribe")[LIKERT].mean().T
    att.index = [pretty(i) for i in att.index]
    att.columns = ["Unlikely (1–3)", "Likely (4–5)"]
    fig = go.Figure()
    for col, colr in zip(att.columns, ["#CBD5E1", TEAL]):
        fig.add_bar(y=att.index, x=att[col], name=col, orientation="h", marker_color=colr)
    fig.update_layout(barmode="group", height=440, xaxis_title="Mean score (1–5)")
    st.plotly_chart(fig)

    st.subheader("Correlations")
    num = ["q7_years_in_uae", "q10_outdoor_hours_day", "q11_ac_hours_day", "q12_hard_water_impact",
           "q13_monthly_skincare_spend_aed", "q17_brands_tried_last_year", "q18_routine_steps", "income_ord"] + LIKERT + \
          ["n_concerns", "q30_subscribe_likelihood", "q31_max_monthly_wtp_aed"]
    corr = d[num].corr()
    corr.index = corr.columns = [pretty(i).replace("income_ord", "Income") for i in num]
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", text_auto=".1f")
    fig.update_layout(height=640)
    fig.update_traces(textfont_size=9)
    st.plotly_chart(fig)


def page_classification():
    st.title("Classification: who will subscribe?")
    st.markdown("**Target:** Q30 = 4 or 5 → *will subscribe*. **Features:** demographics, skin profile, behaviour, "
                "attitudes. Data split 75/25 (stratified) plus 5-fold cross-validation.")
    clf = get_clf(df)
    res = clf["results"]
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"})
                 .highlight_max(subset=res.columns[1:], color="#CCFBF1"), hide_index=True, width="stretch")
    insight(f"**{clf['best']}** performs best (ROC-AUC {res.iloc[0]['ROC-AUC']:.2f}). "
            f"Base subscribe rate is {clf['base_rate']:.0%}.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, (fpr, tpr) in clf["rocs"].items():
            fig.add_scatter(x=fpr, y=tpr, name=name, mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#94A3B8"), name="Random", showlegend=False)
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
        st.plotly_chart(fig)
    with b:
        cm = clf["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Pred: won't", "Pred: will"], y=["Actual: won't", "Actual: will"],
                        color_continuous_scale=["#F1F5F9", TEAL], title=f"Confusion matrix · {clf['best']}")
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig)
    a, b = st.columns(2)
    with a:
        imp = clf["importance"].head(12)[::-1]
        fig = px.bar(x=imp.values, y=[pretty(i) if i != "log_spend" else "Skincare spend (log)" for i in imp.index],
                     orientation="h", title="Top drivers (Random Forest importance)", labels={"x": "Importance", "y": ""},
                     color_discrete_sequence=[TEAL])
        fig.update_layout(height=440)
        st.plotly_chart(fig)
    with b:
        coef = clf["coef"]
        top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)[::-1]
        fig = px.bar(x=top.values, y=[pretty(i) if i != "log_spend" else "Skincare spend (log)" for i in top.index],
                     orientation="h", color=top.values > 0, color_discrete_map={True: TEAL, False: CORAL},
                     title="Direction of effect (Logistic Regression, standardised)", labels={"x": "Coefficient", "y": ""})
        fig.update_layout(height=440, showlegend=False)
        st.plotly_chart(fig)
    st.subheader("Lift chart: how well the model ranks prospects")
    lift = clf["lift"]
    fig = px.bar(lift, x="Decile (1 = highest score)", y="Actual subscribe rate", color_discrete_sequence=[TEAL])
    fig.add_hline(y=clf["base_rate"], line_dash="dash", line_color=CORAL, annotation_text="Average")
    fig.update_layout(yaxis_tickformat=".0%", height=360, xaxis=dict(dtick=1))
    st.plotly_chart(fig)
    insight(f"Targeting the top 30% of scored prospects reaches people who subscribe at **{lift.iloc[:3, 1].mean():.0%}**, "
            f"versus {clf['base_rate']:.0%} if we advertise to everyone. That brings CAC down a lot. "
            "The strongest drivers are trust in personalised recommendations, current spend, hard-water pain and newness to the UAE. "
            "High price sensitivity and being happy with a current routine pull intent down.", "success")


def page_regression():
    st.title("Regression: how much will they pay?")
    st.markdown("**Target:** Q31 = the most a respondent would pay per month for a personalised box (AED). "
                "Typing-error outliers (e.g. AED 9,999) were removed before training.")
    reg = get_reg(df)
    res = reg["results"]
    st.dataframe(res.style.format({c: "{:.3f}" if "R²" in c else "{:.1f}" for c in res.columns if c != "Model"})
                 .highlight_max(subset=["R²", "CV R² (5-fold)"], color="#CCFBF1")
                 .highlight_min(subset=["MAE (AED)", "RMSE (AED)"], color="#CCFBF1"), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.scatter(x=reg["y_test"], y=reg["pred"], opacity=0.45, labels={"x": "Actual WTP (AED)", "y": "Predicted WTP (AED)"},
                         title=f"Actual vs predicted · {reg['best']}", color_discrete_sequence=[TEAL])
        mx = float(max(reg["y_test"].max(), reg["pred"].max()))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color=CORAL), showlegend=False)
        fig.update_layout(height=420)
        st.plotly_chart(fig)
    with b:
        sc = reg["coef"]
        keep = ["q22_price_sensitivity", "q20_trust_personalised_reco", "income_ord", "q26_derm_consult_interest",
                "n_concerns", "q21_clean_halal_importance", "q12_hard_water_impact", "n_products"]
        sc = sc.reindex([k for k in keep if k in sc.index]).sort_values()
        fig = px.bar(x=sc.values, y=[pretty(i).replace("income_ord", "Income band") for i in sc.index], orientation="h",
                     color=sc.values > 0, color_discrete_map={True: TEAL, False: CORAL},
                     title="AED change in WTP per +1 step (Linear Regression)", labels={"x": "AED / month", "y": ""})
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig)

    st.subheader("Price check: which monthly price earns the most?")
    likely = df[df["will_subscribe"] == 1]["q31_max_monthly_wtp_aed"].dropna()
    prices = np.arange(79, 300, 5)
    take = np.array([(likely >= p).mean() for p in prices])
    revenue = prices * take
    best_p = prices[revenue.argmax()]
    fig = go.Figure()
    fig.add_scatter(x=prices, y=take, name="Share of likely subscribers who'd accept", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=prices, y=revenue, name="Revenue index (price × take-up)", yaxis="y2", line=dict(color=SAND, width=3))
    fig.add_vline(x=149, line_dash="dash", line_color=SLATE, annotation_text="Plan: 149")
    fig.add_vline(x=best_p, line_dash="dot", line_color=CORAL, annotation_text=f"Revenue-max: {best_p}",
                  annotation_position="bottom right")
    fig.update_layout(height=400, xaxis_title="Monthly price (AED)", yaxis=dict(tickformat=".0%"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="AED per prospect"))
    st.plotly_chart(fig)
    insight(f"At **AED 149**, {(likely >= 149).mean():.0%} of likely subscribers say they'd pay. Survey revenue peaks around "
            f"**AED {best_p}**. Stated WTP usually runs high, so 149 is a safe launch price, with a premium tier around "
            f"AED {int(round(best_p / 10) * 10 + 20)} for the high-spend segment.", "success")


def page_clustering():
    st.title("Clustering: who are our customers?")
    st.markdown("K-Means on 14 standardised behaviour and attitude variables. The elbow and silhouette charts pick *k*.")
    base, _ = get_clu(df)
    a, b = st.columns(2)
    with a:
        fig = px.line(x=base["ks"], y=base["inertia"], markers=True, title="Elbow (inertia)",
                      labels={"x": "k", "y": "Inertia"}, color_discrete_sequence=[TEAL])
        fig.update_layout(height=300)
        st.plotly_chart(fig)
    with b:
        fig = px.line(x=base["ks"], y=base["sil"], markers=True, title="Silhouette score (higher = better)",
                      labels={"x": "k", "y": "Silhouette"}, color_discrete_sequence=[SAND])
        fig.update_layout(height=300)
        st.plotly_chart(fig)
    k = st.slider("Number of segments (k)", 2, 8, base["k"], help=f"Recommended: {base['k']}")
    res, prof = get_clu(df, k)
    labels = pd.Series(res["labels"]).map(prof["name"])
    fig = px.scatter(x=res["pca"][:, 0], y=res["pca"][:, 1], color=labels, opacity=0.55,
                     labels={"x": "PC1", "y": "PC2", "color": "Segment"}, title="Segments in 2-D (PCA)")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=460)
    st.plotly_chart(fig)

    st.subheader("Segment profiles")
    p = prof.sort_values("value_index", ascending=False)
    cols = st.columns(min(len(p), 4))
    for i, (_, r) in enumerate(p.iterrows()):
        with cols[i % len(cols)].container(border=True):
            tag = " 🎯 Beachhead" if i == 0 else ""
            st.markdown(f"**{r['name']}**{tag}")
            st.markdown(f"<span class='pill'>{r['share']:.0%} of market</span>"
                        f"<span class='pill'>{r['subscribe_rate']:.0%} likely</span>"
                        f"<span class='pill'>WTP AED {r['wtp_median']:.0f}</span>", unsafe_allow_html=True)
            st.caption(f"Spends AED {r['spend_aed']:.0f}/mo · {r['years_uae']:.1f} yrs in UAE · "
                       f"hard-water {r['hard_water']:.1f}/5 · trust in AI {r['trust_ai']:.1f}/5 · "
                       f"satisfaction {r['satisfaction']:.1f}/5 · price-sensitivity {r['price_sens']:.1f}/5")
            st.progress(min(r["value_index"] / 100, 1.0), text=f"Value index {r['value_index']:.0f}")

    heat_cols = {"spend_aed": "Spend", "income_aed": "Income", "years_uae": "Years in UAE", "hard_water": "Hard water",
                 "trust_ai": "Trust AI reco", "price_sens": "Price sensitivity", "influencer": "Influencer",
                 "satisfaction": "Satisfaction", "convenience": "Convenience", "routine_steps": "Routine steps",
                 "brands_tried": "Brands tried", "subscribe_rate": "Subscribe rate", "wtp_median": "WTP"}
    h = prof.set_index("name")[list(heat_cols)].rename(columns=heat_cols)
    hz = (h - h.mean()) / h.std(ddof=0)
    fig = px.imshow(hz, color_continuous_scale=[[0, CORAL], [0.5, "#F8FAFC"], [1, TEAL]], zmin=-2, zmax=2, aspect="auto",
                    title="How each segment differs from average (z-score)")
    fig.update_traces(text=h.apply(lambda c: c.map(lambda v: f"{v:,.0%}" if c.name == "Subscribe rate" else f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}")).values, texttemplate="%{text}")
    fig.update_layout(height=120 + 60 * len(h))
    st.plotly_chart(fig)
    top = p.iloc[0]
    insight(f"Lead with **{top['name']}**. They are {top['share']:.0%} of respondents, {top['subscribe_rate']:.0%} say they'd subscribe, "
            f"and their median WTP is AED {top['wtp_median']:.0f}. Segments with low intent and low WTP (often students or "
            "satisfied pharmacy loyalists) should get the one-time kit or Noon listing, not the subscription.", "success")


def page_rules():
    st.title("Association rules: what goes in the box?")
    st.markdown("Apriori on Q28 (items each respondent wants in their box). **Support** = how often the set appears, "
                "**confidence** = P(then | if), **lift** > 1 = items go together more than chance.")
    c1, c2, c3 = st.columns(3)
    sup = c1.slider("Min support", 0.02, 0.20, 0.05, 0.01)
    conf = c2.slider("Min confidence", 0.2, 0.9, 0.4, 0.05)
    minlift = c3.slider("Min lift", 1.0, 4.0, 1.2, 0.1)
    ar = get_rules(df, sup, conf)
    rules = ar["rules"][ar["rules"]["lift"] >= minlift]
    a, b = st.columns([1, 1.3])
    with a:
        s = ar["item_support"].sort_values()
        fig = px.bar(x=s.values, y=s.index, orientation="h", title="How often each item is wanted",
                     labels={"x": "Support", "y": ""}, color_discrete_sequence=[TEAL])
        fig.update_layout(height=420, xaxis_tickformat=".0%")
        st.plotly_chart(fig)
    with b:
        if len(rules):
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                             hover_data=["antecedents", "consequents"], color_continuous_scale=[SAND, TEAL],
                             title=f"{len(rules)} rules")
            fig.update_layout(height=420)
            st.plotly_chart(fig)
    if not len(rules):
        st.warning("No rules at these thresholds. Lower support or confidence.")
        return
    st.dataframe(rules.rename(columns={"antecedents": "If customer wants", "consequents": "They also want"})
                 .style.format({"support": "{:.1%}", "confidence": "{:.0%}", "lift": "{:.2f}"}),
                 hide_index=True, width="stretch", height=360)
    st.subheader("Suggested bundles")
    used, bundles = set(), []
    for _, r in rules.sort_values("lift", ascending=False).iterrows():
        items = frozenset(r["antecedents"].split(" + ") + r["consequents"].split(" + "))
        if any(len(items & u) >= 2 for u in used):
            continue
        used.add(items)
        bundles.append((items, r))
        if len(bundles) == 3:
            break
    names = ["Bundle A", "Bundle B", "Bundle C"]
    cols = st.columns(3)
    for col, nm, (items, r) in zip(cols, names, bundles):
        with col.container(border=True):
            st.markdown(f"**{nm}**")
            st.markdown("".join(f"<span class='pill'>{i}</span>" for i in sorted(items)), unsafe_allow_html=True)
            st.caption(f"Lift {r['lift']:.2f} · confidence {r['confidence']:.0%} · support {r['support']:.1%}")
    insight("Use these rules for themed boxes (a **Hard-Water Rescue** kit, a **Sun Shield** kit, a **Clear Skin** kit) "
            "and for 'customers also add' prompts at checkout. Raising add-on attach lifts ARPU, which moves the LTV:CAC needle.",
            "success")


pages = [
    st.Page(page_verdict, title="Verdict & financials", icon="🏁", default=True),
    st.Page(page_plan, title="Startup plan", icon="📋", url_path="plan"),
    st.Page(page_questionnaire, title="Questionnaire", icon="📝", url_path="questionnaire"),
    st.Page(page_data, title="Data & cleaning", icon="🧹", url_path="data"),
    st.Page(page_eda, title="Exploratory analysis", icon="🔎", url_path="eda"),
    st.Page(page_classification, title="Classification", icon="🎯", url_path="classification"),
    st.Page(page_regression, title="Regression", icon="💰", url_path="regression"),
    st.Page(page_clustering, title="Clustering", icon="🧩", url_path="clustering"),
    st.Page(page_rules, title="Association rules", icon="🛒", url_path="rules"),
]
nav = st.navigation({"Decision": pages[:1], "Business": pages[1:3], "Analysis": pages[3:]})
nav.run()
