"""Models: classification, regression, clustering, association rules."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
                             silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

from pipeline import model_matrix, CLUSTER_COLS, BOX_ITEMS

SEED = 42


# ------------------------------------------------------------------ classification
def run_classification(df):
    X = model_matrix(df)
    y = df["will_subscribe"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }
    rows, rocs, fitted = [], {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC-AUC": roc_auc_score(yte, p),
                     "CV AUC (5-fold)": cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()})
        fpr, tpr, _ = roc_curve(yte, p)
        rocs[name] = (fpr, tpr)
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = res.loc[0, "Model"]
    bm = fitted[best]
    p_best = bm.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, (p_best >= 0.5).astype(int))
    # feature importance: permutation-free — use RF importances (stable) and LR coefficients (direction)
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    lr = fitted["Logistic Regression"][-1]
    coef = pd.Series(lr.coef_[0], index=X.columns)
    # lift: score everyone, deciles
    p_all = bm.predict_proba(X)[:, 1]
    dec = pd.qcut(pd.Series(p_all).rank(method="first"), 10, labels=False)
    lift = (pd.DataFrame({"decile": 10 - dec, "y": y.values}).groupby("decile")["y"].mean().reset_index())
    lift.columns = ["Decile (1 = highest score)", "Actual subscribe rate"]
    return dict(results=res, rocs=rocs, best=best, cm=cm, importance=imp, coef=coef,
                lift=lift, base_rate=y.mean(), p_all=p_all, X_cols=list(X.columns))


# ------------------------------------------------------------------ regression
def run_regression(df):
    d = df.dropna(subset=["q31_max_monthly_wtp_aed"])
    X = model_matrix(d)
    y = d["q31_max_monthly_wtp_aed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.5, max_iter=5000)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    rows, preds, fitted = [], {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({"Model": name, "R²": r2_score(yte, p), "MAE (AED)": mean_absolute_error(yte, p),
                     "RMSE (AED)": mean_squared_error(yte, p) ** 0.5,
                     "CV R² (5-fold)": cross_val_score(m, X, y, cv=5, scoring="r2").mean()})
        preds[name] = p
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    best = res.loc[0, "Model"]
    lin = fitted["Linear Regression"]
    coef = pd.Series(lin[-1].coef_ / lin[0].scale_, index=X.columns)  # AED per unit
    std_coef = pd.Series(lin[-1].coef_, index=X.columns)  # AED per 1 SD
    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(results=res, best=best, y_test=yte.values, pred=preds[best], coef=coef, std_coef=std_coef,
                importance=imp, wtp=y)


# ------------------------------------------------------------------ clustering
def run_clustering(df, k=None):
    Z = df[CLUSTER_COLS].astype(float).copy()
    Z["q13_monthly_skincare_spend_aed"] = np.log1p(Z["q13_monthly_skincare_spend_aed"])
    Zs = StandardScaler().fit_transform(Z)
    ks = list(range(2, 9))
    inertia, sil = [], []
    for kk in ks:
        km = KMeans(kk, n_init=10, random_state=SEED).fit(Zs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Zs, km.labels_, sample_size=2000, random_state=SEED))
    if k is None:
        k = ks[int(np.argmax(sil))] if max(sil) > 0 else 4
        # prefer business-usable 4-6 when silhouette is close
        cand = [(s, kk) for s, kk in zip(sil, ks) if 4 <= kk <= 6]
        best_s, best_k = max(cand)
        if best_s >= max(sil) - 0.02:
            k = best_k
    km = KMeans(k, n_init=20, random_state=SEED).fit(Zs)
    labels = km.labels_
    pcs = PCA(2, random_state=SEED).fit_transform(Zs)
    return dict(ks=ks, inertia=inertia, sil=sil, k=k, labels=labels, pca=pcs)


def profile_clusters(df, labels):
    d = df.copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").agg(
        size=("respondent_id", "count"),
        age=("age_ord", "mean"),
        income_aed=("income_mid", "mean"),
        years_uae=("q7_years_in_uae", "median"),
        spend_aed=("q13_monthly_skincare_spend_aed", "median"),
        outdoor_hrs=("q10_outdoor_hours_day", "mean"),
        hard_water=("q12_hard_water_impact", "mean"),
        trust_ai=("q20_trust_personalised_reco", "mean"),
        price_sens=("q22_price_sensitivity", "mean"),
        influencer=("q23_influencer_influence", "mean"),
        satisfaction=("q25_routine_satisfaction", "mean"),
        convenience=("q27_convenience_importance", "mean"),
        routine_steps=("q18_routine_steps", "mean"),
        brands_tried=("q17_brands_tried_last_year", "mean"),
        subscribe_rate=("will_subscribe", "mean"),
        wtp_median=("q31_max_monthly_wtp_aed", "median"),
    )
    prof["share"] = prof["size"] / prof["size"].sum()
    prof["student_share"] = d.groupby("cluster")["q5_occupation"].apply(lambda s: (s == "Student").mean())
    prof["pharmacy_share"] = d.groupby("cluster")["q15_main_channel"].apply(lambda s: (s == "Pharmacy").mean())
    prof["wtp_ge_149"] = d.groupby("cluster")["q31_max_monthly_wtp_aed"].apply(lambda s: (s.dropna() >= 149).mean())
    names = [_name(r, prof) for _, r in prof.iterrows()]
    seen = {}
    for i, n in enumerate(names):
        seen[n] = seen.get(n, 0) + 1
        if names.count(n) > 1:
            names[i] = f"{n} {seen[n]}"
    prof["name"] = names
    # value score = share × subscribe rate × median WTP
    prof["value_index"] = prof["share"] * prof["subscribe_rate"] * prof["wtp_median"]
    prof["value_index"] = prof["value_index"] / prof["value_index"].max() * 100
    return prof


def _name(r, prof):
    m = prof.mean(numeric_only=True)
    if r["student_share"] > 0.5:
        return "Budget students"
    if r["satisfaction"] > m["satisfaction"] + 0.5 and r["trust_ai"] < m["trust_ai"]:
        return "Loyal pharmacy buyers"
    if r["spend_aed"] > m["spend_aed"] * 1.3 and r["routine_steps"] > m["routine_steps"]:
        return "Premium skin enthusiasts"
    if r["hard_water"] > m["hard_water"] + 0.3 and r["years_uae"] < m["years_uae"]:
        return "Climate-struggling new expats"
    if r["convenience"] > m["convenience"] + 0.3 and r["routine_steps"] < m["routine_steps"]:
        return "Busy minimalists"
    return "Mainstream"


# ------------------------------------------------------------------ association rules
def run_arules(df, min_support=0.05, min_conf=0.4):
    basket = df[[c for c in df if c.startswith("box_")]].astype(bool)
    basket.columns = [c[4:].replace("_", " ") for c in basket.columns]
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] > 1].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False)
    item_support = basket.mean().sort_values(ascending=False)
    return dict(rules=rules.reset_index(drop=True), freq=freq, item_support=item_support,
                basket_size=basket.sum(axis=1))
