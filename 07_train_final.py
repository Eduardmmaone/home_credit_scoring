import json, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from settings import DATA_DIR, ART

def load_xy():
    X_train = pd.read_parquet(DATA_DIR/"X_train.parquet")
    X_val   = pd.read_parquet(DATA_DIR/"X_val.parquet")
    y_train = pd.read_parquet(DATA_DIR/"y_train.parquet")["TARGET"]
    y_val   = pd.read_parquet(DATA_DIR/"y_val.parquet")["TARGET"]
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val = load_xy()

    lgb_p = json.loads((ART/"tuning"/"best_params_lgb.json").read_text())
    xgb_p = json.loads((ART/"tuning"/"best_params_xgb.json").read_text())

    # LightGBM
    lgb_p.update(dict(n_estimators=3000, random_state=42, n_jobs=-1))
    lgbm = lgb.LGBMClassifier(**lgb_p)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], eval_metric="auc",
             callbacks=[lgb.early_stopping(200)])
    (ART/"lgbm_final").mkdir(parents=True, exist_ok=True)
    lgbm.booster_.save_model(str(ART/"lgbm_final"/"model.txt"))
    (ART/"lgbm_final"/"params.json").write_text(json.dumps(lgb_p, indent=2))

    # XGBoost
    dtr=xgb.DMatrix(X_tr,label=y_tr); dval=xgb.DMatrix(X_val,label=y_val)
    bst=xgb.train(xgb_p, dtr, num_boost_round=3000, evals=[(dval,"valid")],
                  early_stopping_rounds=200, verbose_eval=False)
    (ART/"xgb_final").mkdir(parents=True, exist_ok=True)
    bst.save_model(str(ART/"xgb_final"/"model.json"))
    (ART/"xgb_final"/"params.json").write_text(json.dumps(xgb_p, indent=2))

    # сохранить список фич
    (ART/"features.json").write_text(json.dumps(list(X_tr.columns), indent=2))
