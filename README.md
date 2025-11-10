```markdown
# 🏦 Home Credit Scoring Project  
**Полный ML-пайплайн для задачи кредитного скоринга по данным Home Credit (Kaggle).**  
Реализовано на Python с использованием LightGBM, XGBoost, Optuna, SHAP, кросс-валидации и Flask API.

---

## 📂 Структура проекта

```markdown

OTP_BANK/
├── artifacts/                 # сохранённые модели, энкодеры, scaler’ы
├── data/                      # исходные и промежуточные данные
├── reports/                   # отчёты и финальные результаты
├── 01_download_and_load.py    # загрузка и объединение данных
├── 02_preprocess_and_aggregate.py  # предобработка, агрегирование, feature engineering
├── 03_eda_quick.py            # разведочный анализ данных (EDA)
├── 04_prepare_encode_split.py # кодирование и разделение train/val/test
├── 05_models_baseline_and_importance.py  # базовые модели и важность признаков
├── 06_tune_optuna.py          # гиперпараметрический тюнинг (AUC, Gini)
├── 07_train_final.py          # финальное обучение моделей (LightGBM, XGBoost)
├── 08_cv_and_report.py        # кросс-валидация и финальный отчёт
├── app.py                     # Flask API для инференса
├── inference.py               # тестовый запуск модели
├── home_credit_scoring.ipynb  # ноутбук с пошаговым выполнением
├── requirements.txt           # список зависимостей
├── settings.py                # общие пути и константы
├── kaggle.json                # добавить свой ключ
└── README.md                  # описание проекта

````

---

## ⚙️ Установка и запуск

### 1️⃣ Создать виртуальное окружение
```bash
python3 -m venv home_credit
source home_credit/bin/activate   # macOS / Linux
home_credit\Scripts\activate      # Windows
````

### 2️⃣ Установить зависимости

```bash
pip install -r requirements.txt
```

### 3️⃣ Запуск пайплайна по шагам

```bash
python 01_download_and_load.py
python 02_preprocess_and_aggregate.py
python 03_eda_quick.py
python 04_prepare_encode_split.py
python 05_models_baseline_and_importance.py
python 06_tune_optuna.py
python 07_train_final.py
python 08_cv_and_report.py
```

---

## 📈 Финальные результаты

| Model    | Best AUC   | Best Gini  |
| -------- | ---------- | ---------- |
| LightGBM | **0.7867** | **0.5734** |
| XGBoost  | **0.7856** | **0.5715** |

---

## 🔍 Интерпретация модели

* **SHAP** анализ — важнейшие признаки:
  `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_ANNUITY`, `AMT_CREDIT`, `CODE_GENDER`
* **Feature importance** для обеих моделей визуализирован в `reports/`.

---

## 🧠 Инференс (предсказание)

Запуск Flask API:

```bash
python app.py
```

Локальный запрос:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"DAYS_BIRTH": -15000, "AMT_CREDIT": 500000, "EXT_SOURCE_2": 0.45, ...}'
```

---

## 📊 Отчёты и артефакты

* `reports/final_report.md` — финальные метрики, ROC-кривые, SHAP-графики.
* `artifacts/` — сохранённые модели (`.pkl` или `.txt`).
* `data/` — промежуточные данные после агрегации и кодирования.

---

## 🧾 Автор

**Eduard Gavrilov**
📧 [GitHub: Eduardmmaone](https://github.com/Eduardmmaone)
