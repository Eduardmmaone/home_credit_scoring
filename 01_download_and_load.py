import os, zipfile, json
from pathlib import Path
from settings import DATA_DIR

def download_all():
    # требуются ~/.kaggle/kaggle.json
    cmd = f'kaggle competitions download -c home-credit-default-risk -p "{DATA_DIR}"'
    print(cmd)
    os.system(cmd)

def unzip_all():
    for z in DATA_DIR.glob("*.zip"):
        print("unzip:", z.name)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(DATA_DIR)

if __name__ == "__main__":
    download_all()
    unzip_all()
    # sanity
    print("CSV:", sorted([p.name for p in DATA_DIR.glob("*.csv")])[:8], "...")
