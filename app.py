from flask import Flask, request, jsonify
import json, pandas as pd, lightgbm as lgb, xgboost as xgb
from settings import ART

app = Flask(__name__)

FEATURES = json.loads((ART/"features.json").read_text())
LGBM = lgb.Booster(model_file=str(ART/"lgbm_final"/"model.txt"))
XGBM = xgb.Booster(); XGBM.load_model(str(ART/"xgb_final"/"model.json"))

@app.get("/health")
def health():
    return {"status": "ok"}

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=FEATURES, fill_value=0)

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    df = pd.DataFrame([payload["features"]])
    X = prepare(df)
    proba = {
        "lgbm": float(LGBM.predict(X)[0]),
        "xgb":  float(XGBM.predict(xgb.DMatrix(X))[0]),
    }
    return jsonify({"proba": proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
