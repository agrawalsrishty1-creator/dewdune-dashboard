"""Hyperparameter tuning for all four methods (report section 4).

Run:  python tuning.py   -> prints results and writes data/tuning_results.json
"""
import json, sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pipeline import clean, features, model_matrix, CLUSTER_COLS
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

SEED = 42
raw = pd.read_csv(HERE / 'data' / 'survey_raw.csv')
df, _ = clean(raw); df = features(df)
out = {}

# ---------------- classification
X = model_matrix(df); y = df['will_subscribe']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = {
  'Logistic Regression': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
      Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=3000, solver='liblinear'))]),
      {'m__C': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], 'm__penalty': ['l1', 'l2']}),
  'Decision Tree': (DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
      DecisionTreeClassifier(random_state=SEED),
      {'max_depth': [3, 4, 5, 6, 8, 10], 'min_samples_leaf': [5, 10, 20, 40, 80]}),
  'Random Forest': (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
      RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
      {'max_depth': [None, 6, 10], 'min_samples_leaf': [1, 5, 10, 20], 'max_features': ['sqrt', 0.3]}),
  'Gradient Boosting': (GradientBoostingClassifier(random_state=SEED),
      GradientBoostingClassifier(random_state=SEED),
      {'n_estimators': [100, 200, 300], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [2, 3, 4]}),
  'KNN': (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
      Pipeline([('s', StandardScaler()), ('m', KNeighborsClassifier())]),
      {'m__n_neighbors': [5, 15, 25, 35, 51, 75, 101], 'm__weights': ['uniform', 'distance']}),
}
rows = []
for name, (base, est, grid) in clf.items():
    t = time.time()
    base.fit(Xtr, ytr); pb = base.predict_proba(Xte)[:, 1]
    gs = GridSearchCV(est, grid, scoring='roc_auc', cv=cv, n_jobs=-1).fit(Xtr, ytr)
    pa = gs.predict_proba(Xte)[:, 1]
    n = int(np.prod([len(v) for v in grid.values()]))
    rows.append(dict(model=name, tried=n, auc_before=roc_auc_score(yte, pb), auc_after=roc_auc_score(yte, pa),
                     acc_before=accuracy_score(yte, pb >= .5), acc_after=accuracy_score(yte, pa >= .5),
                     cv_auc=gs.best_score_, best={k.replace('m__', ''): v for k, v in gs.best_params_.items()}))
    print(name, round(time.time() - t), 's', rows[-1], flush=True)
out['classification'] = rows

# ---------------- regression
d = df.dropna(subset=['q31_max_monthly_wtp_aed'])
X = model_matrix(d); y = d['q31_max_monthly_wtp_aed']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
kcv = KFold(5, shuffle=True, random_state=SEED)
reg = {
  'Ridge': (make_pipeline(StandardScaler(), Ridge(alpha=10)), Pipeline([('s', StandardScaler()), ('m', Ridge())]),
            {'m__alpha': [0.01, 0.1, 1, 3, 10, 30, 100, 300]}),
  'Lasso': (make_pipeline(StandardScaler(), Lasso(alpha=0.5, max_iter=5000)), Pipeline([('s', StandardScaler()), ('m', Lasso(max_iter=10000))]),
            {'m__alpha': [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 5]}),
  'Random Forest': (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
            {'max_depth': [None, 8, 12], 'min_samples_leaf': [1, 3, 5, 10], 'max_features': [1.0, 0.5]}),
  'Gradient Boosting': (GradientBoostingRegressor(random_state=SEED), GradientBoostingRegressor(random_state=SEED),
            {'n_estimators': [100, 200, 400], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [2, 3, 4]}),
}
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
rows = [dict(model='Linear Regression', tried=1, r2_before=r2_score(yte, lin.predict(Xte)), r2_after=r2_score(yte, lin.predict(Xte)),
             mae_after=mean_absolute_error(yte, lin.predict(Xte)), best='no hyperparameters')]
for name, (base, est, grid) in reg.items():
    base.fit(Xtr, ytr)
    gs = GridSearchCV(est, grid, scoring='r2', cv=kcv, n_jobs=-1).fit(Xtr, ytr)
    n = int(np.prod([len(v) for v in grid.values()]))
    rows.append(dict(model=name, tried=n, r2_before=r2_score(yte, base.predict(Xte)), r2_after=r2_score(yte, gs.predict(Xte)),
                     mae_after=mean_absolute_error(yte, gs.predict(Xte)), cv_r2=gs.best_score_,
                     best={k.replace('m__', ''): v for k, v in gs.best_params_.items()}))
    print(rows[-1], flush=True)
out['regression'] = rows

# ---------------- clustering: k x feature scaling x n_init
Z = df[CLUSTER_COLS].astype(float).copy()
Z['q13_monthly_skincare_spend_aed'] = np.log1p(Z['q13_monthly_skincare_spend_aed'])
Zs = StandardScaler().fit_transform(Z)
crow = []
for k in range(2, 9):
    for init in ['k-means++', 'random']:
        km = KMeans(k, n_init=20, init=init, random_state=SEED).fit(Zs)
        crow.append(dict(k=k, init=init, sil=silhouette_score(Zs, km.labels_, sample_size=2500, random_state=SEED)))
out['clustering'] = crow
print(pd.DataFrame(crow).round(3).to_string(), flush=True)

# ---------------- association rules: thresholds validated on a hold-out
basket = df[[c for c in df if c.startswith('box_')]].astype(bool)
basket.columns = [c[4:].replace('_', ' ') for c in basket.columns]
btr, bte = train_test_split(basket, test_size=0.3, random_state=SEED)


def lift_on(b, a, c):
    a, c = list(a), list(c)
    sa = b[a].all(axis=1).mean(); sc = b[c].all(axis=1).mean(); sac = b[a + c].all(axis=1).mean()
    return (sac / sa) / sc if sa and sc else np.nan


arow = []
for sup in [0.03, 0.05, 0.08, 0.10]:
    for conf in [0.3, 0.4, 0.5, 0.6]:
        fr = apriori(btr, min_support=sup, use_colnames=True)
        r = association_rules(fr, metric='confidence', min_threshold=conf)
        r = r[r['lift'] > 1.2]
        if len(r):
            test_lift = [lift_on(bte, a, c) for a, c in zip(r['antecedents'], r['consequents'])]
            valid = int(np.sum(np.array(test_lift) > 1.2))
        else:
            valid = 0
        arow.append(dict(support=sup, confidence=conf, rules=len(r), validated=valid, share=valid / len(r) if len(r) else 0))
out['arules'] = arow
print(pd.DataFrame(arow).to_string(), flush=True)

json.dump(out, open(HERE / 'data' / 'tuning_results.json', 'w'), default=str, indent=1)
