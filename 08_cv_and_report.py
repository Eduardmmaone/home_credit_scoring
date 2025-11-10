import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb
from settings import DATA_DIR, ART, REP

def gini(y,p): return 2*roc_auc_score(y,p)-1

if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR/"train_full.parquet").copy()
    y  = df["TARGET"]; X = df.drop(columns=["TARGET"]).fillna(0)

    # фич-лист (на всякий случай переупорядочим)
    feats = json.loads((ART/"features.json").read_text())
    X = X.reindex(columns=feats, fill_value=0)

    lgb_p = json.loads((ART/"tuning"/"best_params_lgb.json").read_text())
    xgb_p = json.loads((ART/"tuning"/"best_params_xgb.json").read_text())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    l_auc, x_auc = [], []
    l_gini, x_gini = [], []

    for tr_idx, va_idx in skf.split(X,y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # LGB
        lgbm = lgb.LGBMClassifier(**{**lgb_p, **dict(n_estimators=2000, random_state=42)})
        lgbm.fit(X_tr,y_tr, eval_set=[(X_va,y_va)], eval_metric="auc",
                 callbacks=[lgb.early_stopping(100)])
        p = lgbm.predict_proba(X_va)[:,1]
        l_auc.append(roc_auc_score(y_va,p)); l_gini.append(gini(y_va,p))

        # XGB
        dtr = xgb.DMatrix(X_tr,label=y_tr); dva = xgb.DMatrix(X_va,label=y_va)
        bst = xgb.train(xgb_p, dtr, num_boost_round=2000, evals=[(dva,"valid")],
                        early_stopping_rounds=100, verbose_eval=False)
        p2 = bst.predict(dva, iteration_range=(0, bst.best_iteration+1))
        x_auc.append(roc_auc_score(y_va,p2)); x_gini.append(gini(y_va,p2))

    print(f"[CV] LightGBM AUC:  {np.mean(l_auc):.5f} ± {np.std(l_auc):.5f} | Gini: {np.mean(l_gini):.5f} ± {np.std(l_gini):.5f}")
    print(f"[CV] XGBoost AUC:   {np.mean(x_auc):.5f} ± {np.std(x_auc):.5f} | Gini: {np.mean(x_gini):.5f} ± {np.std(x_gini):.5f}")

    REP.mkdir(parents=True, exist_ok=True)
    (REP/"final_report.md").write_text(
f"""# Финальная валидация

**LightGBM**  
- AUC:  {np.mean(l_auc):.5f} ± {np.std(l_auc):.5f}  
- Gini: {np.mean(l_gini):.5f} ± {np.std(l_gini):.5f}

**XGBoost**  
- AUC:  {np.mean(x_auc):.5f} ± {np.std(x_auc):.5f}  
- Gini: {np.mean(x_gini):.5f} ± {np.std(x_gini):.5f}
"""
    )
    print("✓ Финальный отчёт сохранён:", REP/"final_report.md")
