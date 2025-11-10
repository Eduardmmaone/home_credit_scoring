import pandas as pd
from settings import DATA_DIR
from pathlib import Path

def load_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    return pd.read_csv(p)

def build_train_full():
    app_train = load_csv("application_train.csv")
    app_test  = load_csv("application_test.csv")

    # Примеры простых агрегатов (как в ноутбуке)
    # --- bureau + bureau_balance
    bureau = load_csv("bureau.csv")
    bb     = load_csv("bureau_balance.csv")
    bb_agg = bb.groupby("SK_ID_BUREAU").MONTHS_BALANCE.mean().rename("bureau_bb_MONTHS_BALANCE_mean")
    bureau = bureau.merge(bb_agg, left_on="SK_ID_BUREAU", right_index=True, how="left")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT": ["min","max","mean","sum"],
        "AMT_CREDIT_SUM": ["min","max","mean","sum"],
        "AMT_CREDIT_SUM_DEBT": ["min","max","mean","sum"],
        "DAYS_CREDIT_ENDDATE": ["min","max","mean","sum"],
        "bureau_bb_MONTHS_BALANCE_mean": ["min","max","mean"]
    })
    bureau_agg.columns = ["bureau_" + "_".join(c) for c in bureau_agg.columns.to_flat_index()]

    # --- previous_application + cash/credit_card/pos/instalments (минимальный набор)
    prev = load_csv("previous_application.csv")
    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "AMT_ANNUITY": ["min","max","mean","sum"],
        "AMT_CREDIT": ["min","max","mean","sum"],
        "AMT_GOODS_PRICE": ["min","max","mean","sum"],
        "DAYS_DECISION": ["min","max","mean","sum"],
        "HOUR_APPR_PROCESS_START": ["min","max","mean","sum"],
        "NFLAG_INSURED_ON_APPROVAL": "mean"
    })
    prev_agg.columns = ["prev_" + "_".join(c) for c in prev_agg.columns.to_flat_index()]

    # join к train / test
    def attach(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(bureau_agg, left_on="SK_ID_CURR", right_index=True, how="left")
        out = out.merge(prev_agg, left_on="SK_ID_CURR", right_index=True, how="left")
        return out

    train_full = attach(app_train)
    test_full  = attach(app_test)

    # сохраняем как parquet (дальше всё из них)
    train_full.to_parquet(DATA_DIR / "train_full.parquet", index=False)
    test_full.to_parquet(DATA_DIR / "test_full.parquet", index=False)
    print(train_full.shape, test_full.shape)

if __name__ == "__main__":
    build_train_full()
