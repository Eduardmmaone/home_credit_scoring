from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "home_credit"
ART = ROOT / "artifacts"
REP = ROOT / "reports"

# создаём папки при импорте
for p in (DATA_DIR, ART, REP):
    p.mkdir(parents=True, exist_ok=True)
