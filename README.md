# Sales & Demand Forecast Analyzer
### Future Interns · Machine Learning Track · Task 1

-----------------------------------------------------------

## What this project does

This is a full-stack AI-powered forecasting web application built with Streamlit. Upload any sales, inventory, or business dataset — the app automatically cleans it, trains four ML models in parallel, generates a multi-period demand forecast, and delivers business planning insights for store owners, startup founders, and business managers.

No column mapping required. No configuration needed. Drop a file, get a complete report.

-----------------------------------------------------------

## Project Layout

```
FUTURE_ML_01/
│
├── app.py                        ← main Streamlit application (run this)
├── dataset/
│   └── amazon.csv                ← default dataset (loads automatically)
├── requirements.txt              ← all dependencies
└── README.md                     ← this file
```

-----------------------------------------------------------

## How to run it

```bash
git clone https://github.com/K-Naga774/FUTURE_ML_01.git
cd FUTURE_ML_01

pip install -r requirements.txt

streamlit run app.py
```

Opens at `http://localhost:8501` in your browser automatically.

-----------------------------------------------------------

## How to use it

**Default mode** — just run the app. It loads `dataset/amazon.csv` with 1,465 Amazon India products and shows a complete forecast report instantly.

**Upload your own data** — drag any CSV, Excel (.xlsx), or PDF file into the sidebar uploader. The app auto-detects all columns and rebuilds everything in seconds.

**Change the target** — use the Target Column selector below the hero banner to switch what the model predicts. Auto-selected as the highest-variance numeric column.

**Adjust settings** — use the sidebar sliders to change test set size (10–35%), forecast periods (3–24), and number of customer segments (2–8).

-----------------------------------------------------------

## What's inside — 6 tabs

|       Tab         |                                                            What it shows                                                            |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Data Overview     | Distribution histograms for all numeric columns, category breakdowns, missing value analysis, summary statistics                    |
| Model Results     | 4-model comparison table with R² visual bars, MAE/MAPE bar charts, actual vs predicted line + scatter, feature importance           |
| Forecast          | Historical + N-period forward forecast with ±12% confidence band, period-by-period table, category-level forecast lines             |
| Business Insights | Plain-language forecast summary, planning guides for store owners / founders / managers, key findings, data quality recommendations |
| Deep Analysis     | Pearson correlation heatmap, top correlations with target, K-Means cluster scatter + segment profiles                               |
| Raw Data          | Searchable full dataset with CSV download                                                                                           |

-----------------------------------------------------------

## Speed optimisations

**Parallel model training** — all 4 models train simultaneously using `joblib.Parallel` with thread-based parallelism, cutting training time by 3–4x vs sequential training.

**Cached processing** — `@st.cache_data` is applied to data cleaning, profiling, feature engineering, and model training. Changing the forecast slider does not retrain models. Only a new file upload triggers reprocessing.

**Adaptive complexity** — smaller datasets (<200 rows) automatically use lighter models (50 estimators, depth 3) while larger datasets use full models (150 estimators, depth 4).

**Auto-augmentation** — if the uploaded file has fewer than 50 usable rows, the app generates synthetic rows via ±5% statistical jitter on numeric columns to ensure reliable training.

-----------------------------------------------------------

## Models trained

|        Model      |                      Notes                               |
|-------------------|----------------------------------------------------------|
| Linear Regression | Baseline — fast, interpretable                           |
| Ridge Regression  | Regularised linear model, handles multicollinearity      |
| Random Forest     | 150 trees, parallel fitting, used for feature importance |
| Gradient Boosting | Sequential ensemble, best on skewed distributions        |

All models use 80/20 train-test split. Best model (highest R²) is selected automatically for forecasting and business insights.

-----------------------------------------------------------

## Auto-detection — what columns the app looks for

The app uses a 3-layer detection system:

1. **Exact keyword match** — column name matches exactly (e.g. `price` → price)
2. **Contains match** — column name contains keyword (e.g. `selling_price_usd` → price)
3. **Statistical heuristics** — uses data distribution when name-matching fails

|      Role      |                Detected from                          |
|----------------|-------------------------------------------------------|
| Demand / Sales | rating_count, sales, units, orders, revenue, quantity |
| Price          | discounted_price, price, selling_price, cost, amount  |
| Category       | category, department, type, segment, genre            |

-----------------------------------------------------------

## Supported file formats

| Format |                  Notes                       |
|--------|----------------------------------------------|
| CSV    | UTF-8, Latin-1, CP1252 encodings all handled |
| Excel  | .xlsx and .xls both supported                |
| PDF    | Extracts structured tables using pdfplumber  |

-----------------------------------------------------------

## Tech stack

|       Tool         |               Purpose                       |
|--------------------|---------------------------------------------|
| Python 3.x         | Core language                               |
| Streamlit          | Web application framework                   |
| Pandas             | Data loading, cleaning, feature engineering |
| NumPy              | Numerical operations                        |
| Scikit-learn       | All 4 ML models + metrics                   |
| Joblib             | Parallel model training                     |
| Matplotlib/Seaborn | All charts and visualisations               |
| pdfplumber         | PDF table extraction (optional)             |

-----------------------------------------------------------

## Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
pdfplumber
openpyxl
```

Install all at once:
```bash
pip install -r requirements.txt
```

-----------------------------------------------------------

## Key business findings (Amazon default dataset for training models)

1. Electronics dominates demand — 15.7M units, more than all other categories combined
2. The 40–60% discount bracket generates the highest revenue across every category
3. Products rated 4.0+ show higher demand regardless of price tier
4. Budget products (under ₹200) drive volume; premium products drive revenue
5. Office Products have the highest average rating (4.3+) with low competition — a growth opportunity

-----------------------------------------------------------

## Author

**[KOMMOJU KOTI NAGA HARSHAVARDHAN]**
B.Tech — Artificial Intelligence & Data Science
Machine Learning Intern — Future Interns
LinkedIn: [K K N HARSHAVARDHAN] | GitHub: [K-Naga774]

-----------------------------------------------------------

*The Future Interns ML Internship Program — Task 1: Sales & Demand Forecasting Analyzer*