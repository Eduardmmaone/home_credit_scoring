import argparse, json, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from settings import ART

def load_models():
    feats = json.loads((ART/"features.json").read_text())
    lgbm = lgb.Booster(model_file=str(ART/"lgbm_final"/"model.txt"))
    xgbm = xgb.Booster(); xgbm.load_model(str(ART/"xgb_final"/"model.json"))
    return feats, lgbm, xgbm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Путь к файлу с новыми клиентами (CSV)")
    ap.add_argument("--model", choices=["lgbm","xgb","both"], default="both")
    args = ap.parse_args()

    feats, lgbm, xgbm = load_models()
    df = pd.read_csv(args.csv).reindex(columns=feats, fill_value=0)

    out = {}
    if args.model in ("lgbm","both"):
        out["lgbm_proba"] = lgbm.predict(df)
    if args.model in ("xgb","both"):
        out["xgb_proba"]  = xgbm.predict(xgb.DMatrix(df))

    res = pd.DataFrame(out)
    p = Path(args.csv).with_suffix(".pred.csv")
    res.to_csv(p, index=False)
    print("saved ->", p)

if __name__ == "__main__":
    main()
