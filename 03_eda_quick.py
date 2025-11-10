import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from settings import DATA_DIR, REP

def quick_eda():
    df = pd.read_parquet(DATA_DIR / "train_full.parquet")
    df["AGE"] = -df["DAYS_BIRTH"]/365
    df["LOG_INCOME"] = np.log1p(df["AMT_INCOME_TOTAL"].clip(upper=df["AMT_INCOME_TOTAL"].quantile(0.99)))

    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axes = axes.flatten()
    sns.countplot(x="TARGET", data=df, ax=axes[0]); axes[0].set_title("TARGET")
    sns.histplot(df["AGE"], bins=50, kde=True, ax=axes[1]); axes[1].set_title("AGE")
    sns.histplot(df["LOG_INCOME"], bins=50, kde=True, ax=axes[2]); axes[2].set_title("LOG_INCOME")
    sns.kdeplot(data=df, x="AGE", hue="TARGET", common_norm=False, ax=axes[3]); axes[3].set_title("AGE vs TARGET")
    sns.boxplot(data=df, x="TARGET", y="LOG_INCOME", ax=axes[4]); axes[4].set_title("Income by TARGET")

    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].sample(min(6000, len(df)), random_state=42).corr()["TARGET"].sort_values(key=np.abs, ascending=False).head(11)
    corr.iloc[1:].plot(kind="barh", ax=axes[5]); axes[5].set_title("Top corr with TARGET")

    fig.tight_layout(); p = REP/"eda_panel.png"; fig.savefig(p, dpi=150)
    print("saved:", p)

if __name__ == "__main__":
    quick_eda()
