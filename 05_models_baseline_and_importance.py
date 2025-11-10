import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
from settings import DATA_DIR, ART

def load_xy():
    X_train = pd.read_parquet(DATA_DIR/"X_train.parquet")
    X_val   = pd.read_parquet(DATA_DIR/"X_val.parquet")
    y_train = pd.read_parquet(DATA_DIR/"y_train.parquet")["TARGET"]
    y_val   = pd.read_parquet(DATA_DIR/"y_val.parquet")["TARGET"]
    return X_train, X_val, y_train, y_val

def gini(p,y): auc=roc_auc_score(y,p); return 2*auc-1

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val = load_xy()

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr)
    p = lgb_model.predict_proba(X_val)[:,1]
    l_auc = roc_auc_score(y_val,p); l_gini = 2*l_auc-1
    print(f"LGB AUC={l_auc:.5f} | GINI={l_gini:.5f}")

    imp = (pd.DataFrame({"feature": X_tr.columns, "importance": lgb_model.feature_importances_})
            .sort_values("importance", ascending=False).head(20))
    plt.figure(figsize=(8,6))
    sns.barplot(data=imp, x="importance", y="feature")
    plt.title("LightGBM — топ-20 признаков"); plt.tight_layout()
    (ART/"lgb_importance.png").write_bytes(plt.gcf().canvas.buffer_rgba())

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    xgb_model.fit(X_tr, y_tr)
    p2 = xgb_model.predict_proba(X_val)[:,1]
    x_auc = roc_auc_score(y_val,p2); x_gini = 2*x_auc-1
    print(f"XGB AUC={x_auc:.5f} | GINI={x_gini:.5f}")

    # сохраняем краткий результат
    ART.mkdir(exist_ok=True, parents=True)
    (ART/"baseline").mkdir(exist_ok=True, parents=True)
    (ART/"baseline"/"lgb_auc_gini.json").write_text(json.dumps({"auc":float(l_auc),"gini":float(l_gini)}, indent=2))
    (ART/"baseline"/"xgb_auc_gini.json").write_text(json.dumps({"auc":float(x_auc),"gini":float(x_gini)}, indent=2))
