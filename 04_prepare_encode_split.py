import json, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from settings import DATA_DIR, ART

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    bad = r'["\'\\\[\]\{\}:,\n\r\t]'
    new_cols = (
        df.columns
          .str.replace(bad, "_", regex=True)
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^0-9A-Za-z_]", "_", regex=True)
          .str.replace(r"__+", "_", regex=True)
          .str.strip("_")
    )
    df = df.copy(); df.columns = new_cols
    if df.columns.duplicated().any():
        counts, safe = {}, []
        for c in df.columns:
            if c in counts: counts[c]+=1; safe.append(f"{c}__{counts[c]}")
            else: counts[c]=0; safe.append(c)
        df.columns = safe
    return df

def prepare():
    df = pd.read_parquet(DATA_DIR / "train_full.parquet").copy()
    y = df["TARGET"]; X = df.drop(columns=["TARGET"])

    # label encode категорий
    cat_cols = X.select_dtypes(include=["object","category"]).columns
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    X = sanitize_feature_names(X).fillna(0)

    # сохраняем список фич
    ART.mkdir(parents=True, exist_ok=True)
    (ART/"features.json").write_text(json.dumps(list(X.columns), ensure_ascii=False, indent=2))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
    X_train.to_parquet(DATA_DIR/"X_train.parquet"); X_val.to_parquet(DATA_DIR/"X_val.parquet")
    y_train.to_frame("TARGET").to_parquet(DATA_DIR/"y_train.parquet")
    y_val.to_frame("TARGET").to_parquet(DATA_DIR/"y_val.parquet")
    print(X_train.shape, X_val.shape)

if __name__ == "__main__":
    prepare()
