import json, optuna, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb
from settings import DATA_DIR, ART

def load_xy():
    X_train = pd.read_parquet(DATA_DIR/"X_train.parquet")
    X_val   = pd.read_parquet(DATA_DIR/"X_val.parquet")
    y_train = pd.read_parquet(DATA_DIR/"y_train.parquet")["TARGET"]
    y_val   = pd.read_parquet(DATA_DIR/"y_val.parquet")["TARGET"]
    return X_train, X_val, y_train, y_val

def gini_from_proba(y_true, y_prob): 
    auc = roc_auc_score(y_true, y_prob); return auc, 2*auc-1

def make_patience(patience=6, min_delta=1e-5):
    state={"auc":-1e9,"gini":-1e9,"a":0,"g":0}
    def cb(study, trial):
        if trial.value is None: return
        auc = trial.value
        g   = trial.user_attrs.get("gini", -1e9)
        state["a"] = 0 if auc>state["auc"]+min_delta else state["a"]+1
        state["g"] = 0 if g  >state["gini"]+min_delta else state["g"]+1
        state["auc"]=max(state["auc"], auc); state["gini"]=max(state["gini"], g)
        if state["a"]>=patience and state["g"]>=patience: study.stop()
    return cb

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val = load_xy()
    pos = float(y_tr.sum()); neg = float(len(y_tr)-pos)
    spw = neg/max(pos,1.0)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=6)
    cb = make_patience(6)

    # LightGBM
    def obj_lgb(t):
        params=dict(
            objective="binary", metric="auc", boosting_type="gbdt",
            n_estimators=3000, learning_rate=t.suggest_float("lr",0.01,0.08,log=True),
            num_leaves=t.suggest_int("num_leaves",31,255),
            max_depth=t.suggest_int("max_depth",-1,12),
            min_child_samples=t.suggest_int("mcs",10,120),
            subsample=t.suggest_float("subsample",0.6,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
            reg_alpha=t.suggest_float("reg_alpha",0.0,0.8),
            reg_lambda=t.suggest_float("reg_lambda",0.0,0.8),
            random_state=42, n_jobs=-1, scale_pos_weight=spw, verbosity=-1
        )
        m=lgb.LGBMClassifier(**params)
        m.fit(X_tr,y_tr, eval_set=[(X_val,y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150)])
        proba=m.predict_proba(X_val)[:,1]
        auc,g=gini_from_proba(y_val,proba); t.set_user_attr("gini",g); return auc

    study_lgb=optuna.create_study(direction="maximize", pruner=pruner, study_name="lgbm_full")
    study_lgb.optimize(obj_lgb, n_trials=int(os.getenv("N_TRIALS","10")), callbacks=[cb], show_progress_bar=True)

    ART.mkdir(parents=True, exist_ok=True)
    (ART/"tuning").mkdir(parents=True, exist_ok=True)
    (ART/"tuning"/"best_params_lgb.json").write_text(json.dumps(study_lgb.best_params, indent=2))

    # XGBoost (через DMatrix + early stopping)
    def obj_xgb(t):
        params={
            "objective":"binary:logistic","eval_metric":"auc",
            "eta":t.suggest_float("eta",0.01,0.08,log=True),
            "max_depth":t.suggest_int("max_depth",3,8),
            "min_child_weight":t.suggest_int("min_child_weight",1,10),
            "subsample":t.suggest_float("subsample",0.6,1.0),
            "colsample_bytree":t.suggest_float("colsample_bytree",0.6,1.0),
            "gamma":t.suggest_float("gamma",0.0,2.0),
            "lambda":t.suggest_float("reg_lambda",0.0,2.0),
            "alpha":t.suggest_float("reg_alpha",0.0,2.0),
            "tree_method":"hist","scale_pos_weight":spw,"seed":42,"nthread":-1
        }
        dtr=xgb.DMatrix(X_tr,label=y_tr); dval=xgb.DMatrix(X_val,label=y_val)
        bst=xgb.train(params, dtr, num_boost_round=3000, evals=[(dval,"valid")],
                      early_stopping_rounds=150, verbose_eval=False)
        proba=bst.predict(dval, iteration_range=(0, bst.best_iteration+1))
        auc,g=gini_from_proba(y_val, proba); t.set_user_attr("gini",g); return auc

    import os
    study_xgb=optuna.create_study(direction="maximize", pruner=pruner, study_name="xgb_full")
    study_xgb.optimize(obj_xgb, n_trials=int(os.getenv("N_TRIALS","10")), callbacks=[cb], show_progress_bar=True)

    (ART/"tuning"/"best_params_xgb.json").write_text(json.dumps(study_xgb.best_params, indent=2))
