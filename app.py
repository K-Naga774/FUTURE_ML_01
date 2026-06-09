import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, io, time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales & Demand Forecast Analyzer",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Nunito:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: #20304A;
}
.main {
    background:
        radial-gradient(circle at 15% 20%, rgba(66,133,244,0.10) 0%, transparent 22%),
        radial-gradient(circle at 85% 18%, rgba(234,67,53,0.10) 0%, transparent 22%),
        radial-gradient(circle at 70% 78%, rgba(52,168,83,0.10) 0%, transparent 20%),
        radial-gradient(circle at 30% 85%, rgba(251,188,5,0.12) 0%, transparent 22%),
        linear-gradient(180deg, #EAF4FF 0%, #F8FBFF 45%, #FFFFFF 100%);
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F7FAFF 0%, #EEF5FF 100%) !important;
    border-right: 1px solid rgba(66,133,244,0.12) !important;
}
[data-testid="stSidebar"] * { color: #425466 !important; }
[data-testid="stSidebar"] h2 {
    color: #4285F4 !important;
    font-family: 'Google Sans', 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
    color: #6B7A90 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(66,133,244,0.30) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.72) !important;
}
[data-testid="stSidebar"] .sidebar-logo {
    font-family: 'Google Sans', 'Nunito', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Animations ── */
@keyframes floatSoft {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes drift {
    0%   { transform: translateX(0px); }
    50%  { transform: translateX(8px); }
    100% { transform: translateX(0px); }
}
@keyframes shimmer {
    0% { background-position: 0% 0%; /* Start at the top */}
    100% { background-position: 0% 300%; /* Move straight down */}
}
@keyframes pulseDot {
    0%, 100% { transform: scale(1); opacity: 0.75; }
    50%      { transform: scale(1.3); opacity: 1; }
}
@keyframes cloudGlow {
    0%, 100% { opacity: 0.55; }
    50%      { opacity: 0.95; }
}
.gradient-text {
    background: linear-gradient(200deg, #E0E0FF 0%, #FBBC05 26%, #EA4335 60%, #E0E0FF 50%, #6699FF 80%, #E0E0FF 100%);
    background-size: 100% 300%; 
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.90) 0%, rgba(244,249,255,0.96) 100%);
    border: 1px solid rgba(66,133,244,0.12);
    border-radius: 30px;
    margin-top: 2rem;
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 50px rgba(66,133,244,0.08);
    animation: fadeSlideUp 0.6s ease-out both;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px;
    right: -60px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(66,133,244,0.18) 0%, rgba(66,133,244,0.06) 45%, transparent 72%);
    border-radius: 50%;
    animation: cloudGlow 5s ease-in-out infinite;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -90px;
    left: 6%;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(52,168,83,0.12) 0%, rgba(251,188,5,0.08) 40%, transparent 72%);
    border-radius: 50%;
    animation: drift 6s ease-in-out infinite;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #91b6f2;
    margin-bottom: 0.5rem;
    gap: 0.5rem;
    text-align: center;
}

.hero-title {
    font-family: 'Google Sans', 'Nunito', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    color: #1F3556;
    line-height: 1.08;
    margin: 0 0 1rem;
    text-align: center;
}
 
.hero-desc {
    color: #5F6F86;
    font-size: 0.96rem;
    line-height: 1.8;
    text-align: center;
    margin: 0 0 1.8rem;
}
.hero-pills {
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
    text-align: center;
}
.pill {
    background: rgba(66,133,244,0.08);
    border: 1px solid rgba(66,133,244,0.14);
    color: #356AC3;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 5px 13px;
    border-radius: 999px;
    transition: all 0.2s ease;
            
}
.pill:hover {
    transform: translateY(-1px);
    background: rgba(66,133,244,0.14);
}
.hero-stats {
    display: flex;
    gap: 3rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(66,133,244,0.10);
            justify-content: center;
          
}
.hero-stat-val {
    font-family: 'Google Sans', 'Nunito', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #1F3556;
    text-align: center;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: #7B8AA0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(66,133,244,0.10);
    border-radius: 20px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(66,133,244,0.06);
    animation: fadeSlideUp 0.5s ease-out both;
    transition: transform 0.2s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 40px rgba(66,133,244,0.10);
    border-color: rgba(66,133,244,0.22);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}
.kpi-purple::before { background: linear-gradient(90deg, #E0E0FF, #E0E0FF); }
.kpi-blue::before   { background: linear-gradient(90deg, #E0E0FF, #E0E0FF); }
.kpi-green::before  { background: linear-gradient(90deg, #E0E0FF, #E0E0FF); }
.kpi-orange::before { background: linear-gradient(90deg, #E0E0FF, #E0E0FF); }
.kpi-pink::before   { background: linear-gradient(90deg, #E0E0FF, #E0E0FF); }
.kpi-icon {
    font-size: 1.15rem;
    margin-bottom: 0.6rem;
    display: block;
    animation: floatSoft 3.5s ease-in-out infinite;
}
.kpi-lbl {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #7B8AA0;
}
.kpi-val {
    font-family: 'Google Sans', 'Nunito', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #1F3556;
    line-height: 1;
    margin: 6px 0;
}
.kpi-sub {
    font-size: 0.74rem;
    color: #7B8AA0;
}

/* ── Section Titles ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1rem;
    animation: fadeSlideUp 0.4s ease-out both;
}
.section-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #E0E0FF, #E0E0FF);
    animation: pulseDot 2s infinite;
    flex-shrink: 0;
}
.section-title-text {
    font-family: 'Google Sans', 'Nunito', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #24415F;
}
.section-subtitle {
    font-size: 0.84rem;
    color: #6E7E94;
    margin-bottom: 1.2rem;
    line-height: 1.7;
}

/* ── Chart Cards ── */
.chart-box {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(66,133,244,0.10);
    border-radius: 20px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(66,133,244,0.05);
    animation: fadeSlideUp 0.5s ease-out both;
    transition: transform 0.2s ease, border-color 0.25s ease;
}
.chart-box:hover {
    transform: translateY(-2px);
    border-color: rgba(66,133,244,0.18);
}
.chart-box-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: #35506F;
    margin-bottom: 2px;
}
.chart-box-sub {
    font-size: 0.74rem;
    color: #7B8AA0;
    margin-bottom: 1rem;
}

/* ── Insight Boxes ── */
.insight-box {
    background: rgba(66,133,244,0.08);
    border: 1px solid rgba(66,133,244,0.14);
    border-left: 4px solid #4285F4;
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: #3D5E87;
    line-height: 1.75;
}
.insight-box strong { color: #24415F; }

.warn-box {
    background: rgba(251,188,5,0.10);
    border: 1px solid rgba(251,188,5,0.18);
    border-left: 4px solid #FBBC05;
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: #8A6A00;
    line-height: 1.75;
}
.success-box {
    background: rgba(52,168,83,0.10);
    border: 1px solid rgba(52,168,83,0.16);
    border-left: 4px solid #34A853;
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: #2E7D4A;
    line-height: 1.75;
}

/* ── Loading Animation ── */           
.loader-wrap {
  background: #e3f2fd;
  border: 1px solid #90caf9;
  border-radius: 12px;
  padding: 1rem;
  margin: 1rem 0;
  animation: hideBox 7s forwards;
  animation-delay: 1.5s; /* wait 2 seconds before hiding */
}
@keyframes hideBox {
  to {
    opacity: 0;
    height: 0;
    margin: 0;
    padding: 0;
    visibility: hidden;
  }
}
            
.loader-dots { display: flex; gap: 6px; }
.loader-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4285F4;
    animation: pulseDot 1.3s ease-in-out infinite;
}
.loader-dots span:nth-child(2) { animation-delay: 0.18s; }
.loader-dots span:nth-child(3) { animation-delay: 0.36s; }
.loader-text {
    font-size: 0.83rem;
    color: #356AC3;
    font-weight: 700;
}

/* ── Tables ── */
.model-tbl, .fct-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
.model-tbl th, .fct-tbl th {
    text-align: left;
    padding: 8px 14px;
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7B8AA0;
    border-bottom: 1px solid rgba(66,133,244,0.10);
    background: rgba(66,133,244,0.03);
}
.model-tbl td, .fct-tbl td {
    padding: 11px 14px;
    border-bottom: 1px solid rgba(66,133,244,0.06);
    color: #35506F;
}
.model-tbl tr.win td {
    background: rgba(52,168,83,0.08);
    color: #245A39;
}
.model-tbl tr.win td:first-child::after { content: " ☁️"; }
.r2bg {
    background: rgba(66,133,244,0.10);
    border-radius: 999px;
    height: 6px;
}
.r2fill {
    background: linear-gradient(90deg, #4285F4, #34A853);
    border-radius: 999px;
    height: 6px;
}
.fct-tbl .up   { color: #34A853; font-weight: 700; }
.fct-tbl .dn   { color: #EA4335; font-weight: 700; }
.fct-tbl .flat { color: #FBBC05; font-weight: 700; }

/* ── Business Cards ── */
.biz-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(66,133,244,0.10);
    border-radius: 20px;
    padding: 1.4rem;
    height: 100%;
    box-shadow: 0 10px 28px rgba(66,133,244,0.05);
    transition: transform 0.2s ease, border-color 0.25s ease;
    animation: fadeSlideUp 0.5s ease-out both;
}
.biz-card:hover {
    border-color: rgba(66,133,244,0.18);
    transform: translateY(-3px);
}
.biz-tag {
    display: inline-block;
    font-size: 0.67rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
.tag-store   { background: rgba(52,168,83,0.12); color: #34A853; }
.tag-founder { background: rgba(66,133,244,0.12); color: #4285F4; }
.tag-manager { background: rgba(251,188,5,0.18); color: #B07A00; }
.biz-card h4 {
    font-size: 0.92rem;
    font-weight: 800;
    color: #24415F;
    margin-bottom: 0.6rem;
}
.biz-card p {
    font-size: 0.82rem;
    color: #6E7E94;
    line-height: 1.75;
    margin: 0;
}

/* ── Progress Bar ── */
.prog-bar {
    height: 6px;
    border-radius: 999px;
    background: rgba(66,133,244,0.10);
    margin: 0.4rem 0;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853);
    transition: width 0.8s ease-out;
}

/* ── Tabs / Inputs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.84) !important;
    border: 1px solid rgba(66,133,244,0.10) !important;
    border-radius: 16px !important;
    padding: 4px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #6E7E94 !important;
    padding: 7px 16px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #E0E0FF, #E0E0FF) !important;
    color: white !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(66,133,244,0.14) !important;
    border-radius: 12px !important;
    color: #35506F !important;
}
.stSlider [data-testid="stSliderThumbValue"] { color: #4285F4 !important; }
.stMetric {
    background: rgba(255,255,255,0.88);
    border-radius: 14px;
    border: 1px solid rgba(66,133,244,0.10);
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.73rem;
    color: #90A0B7;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid rgba(66,133,244,0.08);
    margin-top: 3rem;
    line-height: 2;
}
/* Style the outer alert wrapper */
.stAlert {
  margin-top: 3rem;
  background-color: #fff3cd !important; /* soft yellow for warning */
  color: #856404 !important;            /* dark text for contrast */
  border: 1px solid #ffeeba !important; /* subtle border */
  border-radius: 0.5rem;                /* rounded corners */
  padding: 1rem;                        /* spacing inside */
  margin-bottom: 1rem;                  /* spacing below */
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #EDF4FC; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #B7CCF8, #8EC5A4);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: #8FB3F4; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  COLOURS + CHART STYLE
# ─────────────────────────────────────────────────────────────
C = {
    "purple":"#8B5CF6","blue":"#60A5FA","green":"#34D399",
    "orange":"#F59E0B","red":"#F87171","pink":"#EC4899",
    "teal":"#2DD4BF","yellow":"#FCD34D","grey":"#4B5563",
}
PAL = ["#8B5CF6","#60A5FA","#34D399","#F59E0B","#F87171","#EC4899","#2DD4BF","#FCD34D","#A78BFA"]

plt.rcParams.update({
    "figure.facecolor":"#0F0F1A","axes.facecolor":"#0F0F1A",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#1F2937","axes.labelcolor":"#9CA3AF",
    "xtick.color":"#6B7280","ytick.color":"#6B7280",
    "grid.color":"#1F2937","grid.linewidth":0.6,
    "text.color":"#C4C9D4","font.family":"DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────
#  FAST HELPERS
# ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def fast_clean(df_bytes, fname):
    """Cached cleaning — runs once per file."""
    df = pd.read_csv(io.BytesIO(df_bytes)) if fname.endswith(".csv") else pd.read_excel(io.BytesIO(df_bytes))
    out = df.copy()
    for col in out.columns:
        sample = out[col].dropna().astype(str).head(30)
        ratio = (sample.str.replace(r"[₹$€£¥,% ]","",regex=True)
                       .str.replace(r"^-?\d+\.?\d*$","N",regex=True)
                       .str.contains("N").mean())
        if ratio > 0.5:
            cleaned = (out[col].astype(str)
                       .str.replace(r"[₹$€£¥, ]","",regex=True)
                       .str.replace("%","",regex=False).str.strip())
            conv = pd.to_numeric(cleaned, errors="coerce")
            if conv.notna().sum() >= out[col].notna().sum() * 0.7:
                out[col] = conv
        elif out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip().replace("nan", np.nan)
    return out

@st.cache_data(show_spinner=False)
def fast_profile(df_json, target_col):
    """Cached profiling."""
    import json
    df = pd.read_json(io.StringIO(df_json), orient="split")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    variances= df[num_cols].var().sort_values(ascending=False) if num_cols else pd.Series()
    cat_col  = None
    for c in cat_cols:
        if 2 <= df[c].nunique() <= max(20, len(df)*0.25):
            cat_col = c; break
    corr = df[num_cols].corr() if len(num_cols) >= 2 else pd.DataFrame()
    null_pct = (df.isnull().sum()/len(df)*100).round(2)
    return {
        "num_cols": num_cols, "cat_cols": cat_cols,
        "var_order": variances.index.tolist(),
        "category_col": cat_col,
        "null_pct": null_pct.to_dict(),
        "corr": corr.to_dict() if not corr.empty else {},
    }

@st.cache_data(show_spinner=False)
def fast_features(df_json, target_col):
    """Cached feature engineering."""
    df = pd.read_json(io.StringIO(df_json), orient="split")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    feat = pd.DataFrame(index=df.index)
    for c in num_cols:
        if c != target_col:
            feat[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    for c in cat_cols:
        if df[c].nunique() <= 50:
            le = LabelEncoder()
            feat[f"_enc_{c}"] = le.fit_transform(df[c].fillna("Unknown"))
    return feat.fillna(0)

def _fit_single(name_model, Xtr, ytr, Xte, yte):
    """Train one model — called in parallel."""
    name, model = name_model
    try:
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        return name, {
            "MAE":  mean_absolute_error(yte, p),
            "RMSE": np.sqrt(mean_squared_error(yte, p)),
            "R2":   r2_score(yte, p),
            "MAPE": np.mean(np.abs((yte.values-p)/(np.abs(yte.values)+1e-9)))*100,
        }, p, model
    except:
        return name, None, None, None

@st.cache_data(show_spinner=False)
def fast_train(Xtr_bytes, ytr_bytes, Xte_bytes, yte_bytes, n_small):
    """Parallel model training — cached per data."""
    Xtr = pd.read_json(io.StringIO(Xtr_bytes), orient="split")
    ytr = pd.read_json(io.StringIO(ytr_bytes), orient="split", typ="series")
    Xte = pd.read_json(io.StringIO(Xte_bytes), orient="split")
    yte = pd.read_json(io.StringIO(yte_bytes), orient="split", typ="series")

    n = 50 if n_small else 150
    d = 3  if n_small else 4
    model_list = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression",  Ridge(alpha=10)),
        ("Random Forest",     RandomForestRegressor(n_estimators=n, random_state=42,
                                                     n_jobs=-1, max_depth=d+2)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=n, learning_rate=0.1,
                                                         max_depth=d, random_state=42,
                                                         subsample=0.8)),
    ]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit_single)(nm, Xtr, ytr, Xte, yte)
        for nm in model_list
    )
    res, preds, trained = {}, {}, {}
    for name, r, p, m in results:
        if r is not None:
            res[name] = r; preds[name] = p; trained[name] = m
    return res, preds

def make_forecast(y, n=6):
    v = y.dropna().values
    if len(v) < 4: return None, None
    x = np.arange(len(v))
    z = np.polyfit(x, v, 1)
    fx= np.arange(len(v), len(v)+n)
    return v, np.maximum(np.poly1d(z)(fx), 0)

def make_cat_forecast(df, cat_col, target_col, n=6):
    if not cat_col or target_col not in df.columns: return None
    stats = (df.groupby(cat_col)[target_col].agg(["mean","sum"])
               .sort_values("sum", ascending=False).head(6))
    np.random.seed(7)
    fdf = pd.DataFrame({"Period": [f"P{i+1}" for i in range(n)]})
    for cat, row in stats.iterrows():
        base = row["mean"]
        vals = base*(1+0.08*np.sin(np.linspace(0,np.pi,n)))+np.random.normal(0,base*0.03,n)
        fdf[str(cat)[:18]] = np.maximum(vals,0).round(0)
    return fdf

def detect_col(df):
    KW = {
        "demand": ["rating_count","demand","sales","units_sold","units","quantity",
                   "orders","volume","total_sales","revenue","sold","purchases"],
        "price":  ["discounted_price","sale_price","selling_price","price","unit_price",
                   "cost","amount","final_price"],
    }
    result, used = {}, set()
    for role, kws in KW.items():
        for col in df.columns:
            cn = col.lower().replace(" ","_").replace("-","_")
            for kw in kws:
                if kw in cn or cn == kw:
                    if col not in used:
                        result[role] = col; used.add(col); break
            if role in result: break
    return result

def augment_if_needed(df, target_col, min_rows=50):
    if len(df) >= min_rows: return df, False
    needed = min_rows - len(df)
    num_c  = df.select_dtypes(include=[np.number]).columns.tolist()
    np.random.seed(42)
    rows = []
    for _ in range(needed):
        base = df.sample(1, random_state=np.random.randint(0,9999)).copy()
        for c in num_c:
            s = df[c].std()
            if pd.notna(s) and s > 0:
                base[c] = np.maximum(0, base[c].values[0]+np.random.normal(0,s*0.05))
        rows.append(base)
    return pd.concat([df]+rows, ignore_index=True), True

def load_file(f):
    if f.name.endswith(".csv"):
        for enc in ["utf-8","latin-1","cp1252"]:
            try: return pd.read_csv(f, encoding=enc), None
            except: f.seek(0)
        return None, "Cannot read CSV."
    if f.name.endswith((".xlsx",".xls")):
        try: return pd.read_excel(f), None
        except Exception as e: return None, str(e)
    if f.name.endswith(".pdf"):
        if not PDF_SUPPORT: return None, "pip install pdfplumber"
        try:
            dfs = []
            with pdfplumber.open(f) as pdf:
                for pg in pdf.pages:
                    for t in (pg.extract_tables() or []):
                        if t and len(t)>1:
                            dfs.append(pd.DataFrame([[str(c).strip() if c else "" for c in r] for r in t[1:]],
                                                     columns=[str(x).strip() for x in t[0]]))
            if dfs: return pd.concat(dfs, ignore_index=True), None
            return None, "No tables found in PDF."
        except Exception as e: return None, str(e)
    return None, "Unsupported file type."

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Veracity Of SDF ✦</div>', unsafe_allow_html=True)
    st.caption("Quicker like a cheetah, Fast machine learning analysis for your business data.")
    st.markdown("---")
    st.markdown("**Upload Dataset**",text_alignment="center")
    uploaded = st.file_uploader("", type=["csv","xlsx","xls","pdf"])
    st.markdown("---")
    st.markdown("**Settings**",text_alignment="center")
    test_pct   = st.slider("Test set %",     10, 35, 20, 5)
    n_forecast = st.slider("Forecast periods", 3, 24, 6, 1)
    n_clusters = st.slider("Segments",          2,  8, 3, 1)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.70rem;color:#374151;line-height:2;'>
    <span style='color:#A78BFA;font-weight:600;'>DELIVERABLE</span><br>
    ✦ Forecast model<br>
    ✦ Future predictions<br>
    ✦ Business insights<br>
    ✦ Store owner guide<br>
    ✦ Founder pitch data<br>
    ✦ Manager budget plan<br>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default():
    for p in ["dataset/amazon.csv","amazon.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df_raw, source_label = None, ""

if uploaded:
    with st.spinner(""):
        st.markdown("""<div class="loader-wrap"><div class="loader-dots"><span></span><span></span><span></span></div>
        <div class="loader-text">Loading your dataset…</div></div>""", unsafe_allow_html=True)
        df_raw, err = load_file(uploaded)
    if err: st.error(f"❌ {err}"); st.stop()
    source_label = uploaded.name
else:
    df_raw = load_default()
    source_label = "Amazon Dataset Sample"
    if df_raw is None:
        st.markdown("""<div class="warn-box" style="margin-top:2rem;">
        <strong> Upload a dataset to get started.</strong><br>
        Supports CSV, Excel, and PDF. Any sales, inventory, or business data.
        </div>""", unsafe_allow_html=True)
        st.stop()

# ─────────────────────────────────────────────────────────────
#  FAST CLEAN + PROFILE
# ─────────────────────────────────────────────────────────────
try:
    raw_bytes = df_raw.to_csv(index=False).encode()
    df_clean  = fast_clean(raw_bytes, source_label if source_label.endswith(".csv") else "f.csv")
except:
    df_clean = df_raw.copy()

num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_clean.select_dtypes(include=["object","category"]).columns.tolist()

if not num_cols:
    st.error("❌ No numeric columns found."); st.stop()

variances = df_clean[num_cols].var().sort_values(ascending=False)

# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">ML ANALYSIS INTELLIGENCE</div>
  <h1 class="hero-title">SALES & DEMAND FORECAST<br><span class="gradient-text">ANALYZER</span></h1>
  <p class="hero-desc">Machine learning analysis of <strong style="color:#6a6c73;">{len(df_raw):,} records</strong>
  from <em>{source_label}</em> — 4 models trained, future predictions generated,
  and business planning insights delivered.</p>
  <div class="hero-pills">
    <span class="pill">4 ML Models</span>
    <span class="pill">Demand Forecast</span>
    <span class="pill"> Business Insights</span>
    <span class="pill"> Correlation Analysis</span>
    <span class="pill"> Segmentation</span>
    <span class="pill"> Fast Processing</span>
  </div>
  <div class="hero-stats">
    <div><div class="hero-stat-val">{len(df_raw):,}</div><div class="hero-stat-lbl">Records</div></div>
    <div><div class="hero-stat-val">{df_raw.shape[1]}</div><div class="hero-stat-lbl">Columns</div></div>
    <div><div class="hero-stat-val">{len(num_cols)}</div><div class="hero-stat-lbl">Numeric</div></div>
    <div><div class="hero-stat-val">{n_forecast}</div><div class="hero-stat-lbl">Forecast Periods</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TARGET SELECTOR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="target-wrap">', unsafe_allow_html=True)
st.markdown('<div style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:0.5rem;"> Target Column — what the model predicts</div>', unsafe_allow_html=True)
tc1, tc2 = st.columns([2,3])
with tc1:
    target_col = st.selectbox("Target", variances.index.tolist(), index=0,
                               label_visibility="collapsed")
with tc2:
    ts = df_clean[target_col].describe()
    st.markdown(f"""
    
    <span style='background:rgba(139,92,246,0.15);color:#C4B5FD;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;'>mean {ts['mean']:,.1f}</span>
    <span style='background:rgba(52,211,153,0.12);color:#6EE7B7;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;'>max {ts['max']:,.0f}</span>
    <span style='background:rgba(96,165,250,0.12);color:#93C5FD;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;'>σ {ts['std']:,.1f}</span>
    <span style='background:rgba(255,255,255,0.06);color:#9CA3AF;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;'>{df_clean[target_col].notna().sum():,} rows</span>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  FAST AUGMENT + TRAIN
# ─────────────────────────────────────────────────────────────
df_work = df_clean.dropna(subset=[target_col]).reset_index(drop=True)
orig_n  = len(df_work)
df_work, augmented = augment_if_needed(df_work, target_col)

if augmented:
    st.markdown(f'<div class="insight-box">ℹ️ <strong>{orig_n} usable rows</strong> found — auto-generated {len(df_work)-orig_n} synthetic rows (±5% statistical jitter) for reliable training.</div>', unsafe_allow_html=True)

# Get profile (cached)
try:
    df_json = df_work.to_json(orient="split")
    prof    = fast_profile(df_json, target_col)
except:
    prof = {"num_cols":num_cols,"cat_cols":cat_cols,"var_order":variances.index.tolist(),
            "category_col":None,"null_pct":{},"corr":{}}

# Features (cached)
try:
    X_all = fast_features(df_json, target_col)
except:
    X_all = pd.DataFrame(index=df_work.index)
    for c in num_cols:
        if c != target_col:
            X_all[c] = df_work[c].fillna(0)

y_all  = df_work[target_col].astype(float)
n_rows = len(df_work)
split  = max(40, min(int(n_rows*(1-test_pct/100)), n_rows-10))
Xtr, Xte = X_all.iloc[:split], X_all.iloc[split:]
ytr, yte = y_all.iloc[:split], y_all.iloc[split:]

# Fast parallel training (cached)
with st.spinner(""):
    try:
        res_all, preds_all = fast_train(
            Xtr.to_json(orient="split"), ytr.to_json(orient="split"),
            Xte.to_json(orient="split"), yte.to_json(orient="split"),
            n_rows < 200
        )
        # Retrain to get model objects (can't cache sklearn objects directly)
        n = 50 if n_rows < 200 else 150
        d = 3  if n_rows < 200 else 4
        model_objs = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=10),
            "Random Forest":     RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1, max_depth=d+2),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n, learning_rate=0.1, max_depth=d, random_state=42),
        }
        trained_all = {}
        for nm, m in model_objs.items():
            if nm in res_all:
                try: m.fit(Xtr, ytr); trained_all[nm] = m
                except: pass
    except Exception as e:
        st.error(f"Training error: {e}"); st.stop()

if not res_all:
    st.error("All models failed. Try a different target column."); st.stop()

best_name  = max(res_all, key=lambda k: res_all[k]["R2"])
best_pred  = preds_all[best_name]
best_r     = res_all[best_name]
best_model = trained_all.get(best_name)

hist_v, fut_v = make_forecast(y_all, n_forecast)
cat_col       = prof.get("category_col")
fcast_df      = make_cat_forecast(df_work, cat_col, target_col, n_forecast)

# ─────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────
fut_avg  = fut_v.mean()  if fut_v is not None else 0
pct_chg  = ((fut_v[-1]-hist_v[-1])/(abs(hist_v[-1])+1e-9)*100) if fut_v is not None and hist_v is not None else 0

k1,k2,k3,k4,k5 = st.columns(5)
kpi_data = [
    (k1,"purple","","AVG HISTORICAL",    f"{y_all.mean():,.0f}",     f"σ = {y_all.std():,.0f}"),
    (k2,"blue",  "","AVG FORECAST",      f"{fut_avg:,.0f}",          f"next {n_forecast} periods"),
    (k3,"green", "","END PERIOD TREND",  f"{pct_chg:+.1f}%",        "vs last actual"),
    (k4,"orange","","BEST MODEL R²",     f"{max(0,best_r['R2']):.4f}",f"MAPE {best_r['MAPE']:.1f}%"),
    (k5,"pink",  "","FORECAST MAE",      f"{best_r['MAE']:,.0f}",    f"RMSE {best_r['RMSE']:,.0f}"),
]
for col_ui, color, icon, label, val, delta in kpi_data:
    with col_ui:
        st.markdown(f"""
        <div class="kpi-card kpi-{color}">
          <span class="kpi-icon">{icon}</span>
          <div class="kpi-lbl">{label}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-sub">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    " DATA OVERVIEW",
    " MODEL RESULTS",
    " FORECAST",
    " BUSINESS INSIGHTS",
    " DEEP ANALYSIS",
    " RAW DATA",
])

# ════════════════════════════════
#  TAB 1 — DATA OVERVIEW
# ════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Data Overview</div></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">Distribution and quality analysis of your {orig_n:,}-row dataset.</p>', unsafe_allow_html=True)

    show_num = num_cols[:8]
    npr = min(4, len(show_num))
    for ri in range((len(show_num)+npr-1)//npr):
        cols_ui = st.columns(npr)
        for ci, cn in enumerate(show_num[ri*npr:(ri+1)*npr]):
            with cols_ui[ci]:
                data  = df_work[cn].dropna()
                color = PAL[ci % len(PAL)]
                fig, ax = plt.subplots(figsize=(4, 2.8))
                ax.hist(data.clip(upper=data.quantile(0.97)), bins=25,
                        color=color, edgecolor="#0A0A0F", alpha=0.85, linewidth=0.3)
                ax.axvline(data.median(), color="#F87171", lw=1.3, ls="--", alpha=0.8)
                ax.set_title(cn[:22], fontsize=8.5, fontweight="bold", color="#E0E0FF", pad=5)
                ax.set_ylabel("Count", fontsize=7); ax.tick_params(labelsize=7)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    show_cat = [c for c in cat_cols if 2 <= df_work[c].nunique() <= 30][:4]
    if show_cat:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text" style="font-size:0.95rem;">Category Breakdown</div></div>', unsafe_allow_html=True)
        cols_ui = st.columns(len(show_cat))
        for i, cn in enumerate(show_cat):
            with cols_ui[i]:
                vc = df_work[cn].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.barh(vc.index[::-1].astype(str), vc.values[::-1],
                        color=PAL[i%len(PAL)], edgecolor="#0A0A0F", height=0.62, alpha=0.85)
                ax.set_title(cn[:22], fontsize=8.5, fontweight="bold", color="#E0E0FF")
                ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.25)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    null_s = pd.Series(prof["null_pct"]).sort_values(ascending=False)
    null_s = null_s[null_s > 0]
    if null_s.empty:
        st.markdown('<div class="success-box">✅ <strong>Zero missing values</strong> — dataset is clean.</div>', unsafe_allow_html=True)
    else:
        fig, ax = plt.subplots(figsize=(10, max(2, len(null_s)*0.4)))
        clrs = ["#F87171" if v>50 else "#F59E0B" if v>20 else "#60A5FA" for v in null_s.values]
        ax.barh(null_s.index, null_s.values, color=clrs, edgecolor="#0A0A0F", height=0.5)
        ax.set_xlabel("% Missing"); ax.set_title("Missing Data", fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.25); plt.tight_layout(); st.pyplot(fig); plt.close()

    summ = df_work[num_cols].describe().T
    st.dataframe(summ.style.format("{:.2f}").background_gradient(cmap="Purples", subset=["mean","std"]),
                 use_container_width=True)

# ════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Model Performance</div></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">4 algorithms trained in parallel to predict <strong style="color:#6a6c73;">{target_col}</strong>.</p>', unsafe_allow_html=True)

    m_df = (pd.DataFrame(res_all).T.reset_index().rename(columns={"index":"Model"})
              .sort_values("R2", ascending=False))
    max_r2 = max(0, m_df["R2"].max())

    tbl = """<div class="chart-box"><div class="chart-box-title">All 4 models — test set results</div>
    <div class="chart-box-sub">Higher R² = better  ·  Lower MAE / MAPE = better  ·  Green row = winner</div>
    <table class="model-tbl"><thead><tr>
    <th>Model</th><th>R²</th><th style="width:110px">R² Bar</th>
    <th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead><tbody>"""
    for _, row in m_df.iterrows():
        r2s = max(0, row["R2"])
        pct = int(r2s/max_r2*100) if max_r2>0 else 0
        win = "win" if row["Model"]==best_name else ""
        tbl += f"""<tr class="{win}"><td>{row['Model']}</td>
        <td><strong>{row['R2']:.4f}</strong></td>
        <td><div class="r2bg"><div class="r2fill" style="width:{pct}%"></div></div></td>
        <td>{row['MAE']:,.0f}</td><td>{row['RMSE']:,.0f}</td><td>{row['MAPE']:.1f}%</td></tr>"""
    tbl += "</tbody></table></div>"
    st.markdown(tbl, unsafe_allow_html=True)

    if best_r["R2"] >= 0.80:
        st.markdown(f'<div class="success-box">🎯 <strong>{best_name}</strong> — R² = {best_r["R2"]:.4f}. Explains {best_r["R2"]*100:.1f}% of variance. Strong enough for production forecasting.</div>', unsafe_allow_html=True)
    elif best_r["R2"] >= 0.5:
        st.markdown(f'<div class="insight-box">⚠️ <strong>{best_name}</strong> — R² = {best_r["R2"]:.4f}. Moderate fit — useful for directional planning.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">🔍 <strong>{best_name}</strong> — R² = {best_r["R2"]:.4f}. Add more features or data to improve.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    wi = m_df["Model"].tolist().index(best_name)
    clrs_m = [PAL[i%len(PAL)] for i in range(len(m_df))]
    clrs_m[wi] = C["green"]

    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        bars = ax.bar(m_df["Model"], m_df["R2"], color=clrs_m, edgecolor="#0A0A0F", width=0.5)
        bars[wi].set_edgecolor("#34D399"); bars[wi].set_linewidth(2)
        for bar, v in zip(bars, m_df["R2"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold", color="#E0E0FF")
        ax.set_title("R² Score", fontsize=9, fontweight="bold", color="#E0E0FF")
        ax.set_xticklabels(m_df["Model"], rotation=14, ha="right", fontsize=7.5)
        ax.set_ylim(-0.1, 1.2); ax.axhline(0, color=C["grey"], lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.25); plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        bars = ax.bar(m_df["Model"], m_df["MAE"], color=clrs_m, edgecolor="#0A0A0F", width=0.5)
        bars[wi].set_edgecolor("#34D399"); bars[wi].set_linewidth(2)
        for bar, v in zip(bars, m_df["MAE"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(m_df["MAE"])*0.01,
                    f"{v:,.0f}", ha="center", fontsize=8, fontweight="bold", color="#E0E0FF")
        ax.set_title("MAE (lower = better)", fontsize=9, fontweight="bold", color="#E0E0FF")
        ax.set_xticklabels(m_df["Model"], rotation=14, ha="right", fontsize=7.5)
        ax.grid(axis="y", alpha=0.25); plt.tight_layout(); st.pyplot(fig); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    idx = np.arange(len(yte))
    axes[0].plot(idx, yte.values, color=C["blue"], lw=1.8, label="Actual", alpha=0.9)
    axes[0].plot(idx, best_pred,  color=C["orange"], lw=2, ls="--", label=f"Predicted", alpha=0.9)
    axes[0].fill_between(idx, yte.values, best_pred, where=(best_pred>yte.values), alpha=0.08, color=C["orange"])
    axes[0].fill_between(idx, yte.values, best_pred, where=(best_pred<yte.values), alpha=0.08, color=C["blue"])
    axes[0].set_title("Actual vs Predicted", fontsize=9, fontweight="bold", color="#E0E0FF")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.2)

    axes[1].scatter(yte.values, best_pred, alpha=0.45, s=18, color=C["purple"], edgecolors="none")
    lim=max(yte.max(),best_pred.max()); mn=min(yte.min(),best_pred.min())
    axes[1].plot([mn,lim],[mn,lim], color=C["red"], lw=1.5, ls="--", label="Perfect")
    axes[1].set_title("Scatter — Actual vs Predicted", fontsize=9, fontweight="bold", color="#E0E0FF")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if "Random Forest" in trained_all:
        fi = pd.DataFrame({"Feature": X_all.columns,
                           "Importance": trained_all["Random Forest"].feature_importances_})\
               .sort_values("Importance", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(10, max(2.8, len(fi)*0.38)))
        cmap = plt.get_cmap("Purples")
        clrs_fi = [cmap(0.35+0.6*(i/len(fi))) for i in range(len(fi)-1,-1,-1)]
        ax.barh(fi["Feature"][::-1], fi["Importance"][::-1], color=clrs_fi, edgecolor="#0A0A0F", height=0.6)
        ax.set_title("Feature Importance", fontsize=9, fontweight="bold", color="#E0E0FF")
        ax.grid(axis="x", alpha=0.2); plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════
#  TAB 3 — FORECAST
# ════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Demand Forecast</div></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">Historical trend + {n_forecast}-period forward prediction with ±12% confidence band.</p>', unsafe_allow_html=True)

    if hist_v is not None and fut_v is not None:
        fig, ax = plt.subplots(figsize=(14, 4.8))
        hx = np.arange(len(hist_v)); fx = np.arange(len(hist_v), len(hist_v)+n_forecast)
        ax.plot(hx, hist_v, color=C["blue"], lw=1.8, label="Historical", alpha=0.9)
        ax.fill_between(hx, hist_v.min()*0.97, hist_v, alpha=0.04, color=C["blue"])
        ax.plot(fx, fut_v, color=C["orange"], lw=2.4, ls="--", marker="o", ms=6, label=f"Forecast", zorder=5)
        ax.fill_between(fx, fut_v*0.88, fut_v*1.12, alpha=0.12, color=C["orange"], label="±12% band")
        ax.axvline(len(hist_v)-1, color=C["grey"], lw=1.2, ls=":", alpha=0.6)
        for xi, yi in zip(fx, fut_v):
            ax.annotate(f"{yi:,.0f}", (xi,yi), textcoords="offset points",
                        xytext=(0,9), ha="center", fontsize=7.5, color="#FCD34D", fontweight="bold")
        ax.set_title(f"{target_col} — {n_forecast}-Period Forecast", fontsize=10, fontweight="bold", color="#E0E0FF")
        ax.set_xlabel("Periods"); ax.set_ylabel(target_col[:28])
        ax.legend(fontsize=8); ax.grid(alpha=0.2); plt.tight_layout(); st.pyplot(fig); plt.close()

        f1,f2,f3,f4 = st.columns(4)
        pct_c = (fut_v[-1]-hist_v[-1])/(abs(hist_v[-1])+1e-9)*100
        f1.metric("Avg Forecast",  f"{fut_v.mean():,.2f}")
        f2.metric("Peak Forecast", f"{fut_v.max():,.2f}")
        f3.metric("Min Forecast",  f"{fut_v.min():,.2f}")
        f4.metric("End Δ",         f"{pct_c:+.1f}%", "vs last actual")

        periods  = [f"Period {i+1}" for i in range(n_forecast)]
        pct_vs   = [(v-hist_v[-1])/(abs(hist_v[-1])+1e-9)*100 for v in fut_v]
        tbl = """<table class="fct-tbl"><thead><tr>
        <th>Period</th><th>Forecast</th><th>Lower −12%</th><th>Upper +12%</th><th>Δ vs Last</th>
        </tr></thead><tbody>"""
        for p, v, pv in zip(periods, fut_v, pct_vs):
            css = "up" if pv>=0 else "dn"
            tbl += f"<tr><td><strong>{p}</strong></td><td>{v:,.2f}</td><td>{v*0.88:,.2f}</td><td>{v*1.12:,.2f}</td><td class='{css}'>{pv:+.1f}%</td></tr>"
        tbl += "</tbody></table>"
        st.markdown(tbl, unsafe_allow_html=True)

        if pct_c > 5:
            msg = f"📈 <strong>+{pct_c:.1f}% growth</strong> projected. Scale inventory and marketing to capture rising demand."
        elif pct_c > -5:
            msg = f"➡️ <strong>Stable forecast</strong> ({pct_c:+.1f}%). Focus on margin optimisation."
        else:
            msg = f"📉 <strong>{pct_c:.1f}% decline</strong> projected. Review pricing and demand drivers."
        st.markdown(f'<div class="insight-box" style="margin-top:1rem;">{msg}</div>', unsafe_allow_html=True)

    if fcast_df is not None and len(fcast_df.columns) > 1:
        cat_cols_f = [c for c in fcast_df.columns if c != "Period"]
        fig, ax = plt.subplots(figsize=(13, 4.5))
        for i, cat in enumerate(cat_cols_f):
            ax.plot(fcast_df["Period"], fcast_df[cat], marker="o", ms=5, lw=2, color=PAL[i%len(PAL)], label=cat)
            ax.fill_between(fcast_df["Period"], fcast_df[cat]*0.93, fcast_df[cat]*1.07, alpha=0.07, color=PAL[i%len(PAL)])
        ax.set_title("Category-Level Forecast", fontsize=10, fontweight="bold", color="#E0E0FF")
        ax.legend(fontsize=8); ax.grid(alpha=0.2); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(fcast_df.set_index("Period").style.format("{:,.0f}").background_gradient(cmap="Purples"),
                     use_container_width=True)

# ════════════════════════════════
#  TAB 4 — BUSINESS INSIGHTS
# ════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Business Insights & Planning Guide</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">What this forecast means and how to act on it — for store owners, founders, and managers.</p>', unsafe_allow_html=True)

    if hist_v is not None and fut_v is not None:
        avg_h = hist_v.mean(); avg_f = fut_v.mean()
        pct_d = (avg_f-avg_h)/(abs(avg_h)+1e-9)*100
        trend = "upward" if pct_d>2 else ("downward" if pct_d<-2 else "flat")
        action= ("increase inventory and marketing" if trend=="upward"
                 else "optimise margins and reduce overstock" if trend=="downward"
                 else "maintain operations and focus on efficiency")
        st.markdown(f"""
        <div class="chart-box">
          <div class="chart-box-title">What the forecast means</div>
          <p style='font-size:0.87rem;color:#9CA3AF;line-height:1.8;margin:0.8rem 0 0;'>
          Analysed <strong style='color:#6a6c73;'>{orig_n:,} records</strong> →
          produced a <strong style='color:#6a6c73;'>{n_forecast}-period forecast</strong> for
          <strong style='color:#A78BFA;'>{target_col}</strong>.<br><br>
          • Historical avg: <strong style='color:#6a6c73;'>{avg_h:,.2f}</strong> &nbsp;→&nbsp;
            Forecasted avg: <strong style='color:#FCD34D;'>{avg_f:,.2f}</strong>
            ({pct_d:+.1f}%)<br>
          • Trend: <strong style='color:#{"34D399" if trend=="upward" else "F87171" if trend=="downward" else "FCD34D"};'>
            {trend.upper()}</strong> — recommended action: <em style='color:#C4C9D4;'>{action}</em><br>
          • Model confidence: R² = <strong style='color:#A78BFA;'>{max(0,best_r["R2"]):.4f}</strong>
            ({max(0,best_r["R2"])*100:.1f}% of variance explained)
          </p>
        </div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f"""<div class="biz-card">
          <div class="biz-tag tag-store"> Store Owner</div>
          <h4>Inventory & Restock Planning</h4>
          <p>Use the <strong>Point Forecast</strong> as your reorder baseline.
          Use <strong>Upper Bound (+12%)</strong> as safety stock.<br><br>
          Peak is Period {int(np.argmax(fut_v))+1 if fut_v is not None else "3"} —
          place orders 2–3 periods ahead. Set Lower Bound as the reorder trigger
          to avoid overstock costs.</p></div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="biz-card">
          <div class="biz-tag tag-founder"> Startup Founder</div>
          <h4>Investor & Growth Planning</h4>
          <p>R² = {max(0,best_r["R2"]):.3f} gives you a credible demand projection
          for pitch decks. {max(0,best_r["R2"])*100:.0f}% of variance explained.<br><br>
          Use category forecast to identify highest-growth segments for MVP.
          Present ±12% band as conservative/optimistic scenarios.</p></div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="biz-card">
          <div class="biz-tag tag-manager"> Business Manager</div>
          <h4>Budget & Resource Allocation</h4>
          <p>Planning baseline: <strong>{fut_v.mean():,.0f}</strong> avg forecast.<br>
          Add 12% contingency to variable spend.<br><br>
          Track MAE = {best_r["MAE"]:,.0f} monthly — if actuals exceed this
          consistently, re-run the model with fresh data.</p></div>""", unsafe_allow_html=True)

    corr_insight = ""
    if prof.get("corr") and target_col in prof["corr"]:
        corr_d = {k: v.get(target_col,0) for k,v in prof["corr"].items() if k != target_col}
        if corr_d:
            driver = max(corr_d, key=lambda k: abs(corr_d[k]))
            val    = corr_d[driver]
            if abs(val) > 0.3:
                d = "increases" if val > 0 else "decreases"
                corr_insight = f"🔗 <strong>{driver}</strong> has the strongest correlation with {target_col} (r = {val:.3f}) — when it goes up, {target_col} {d}."

    findings = [
        f"📊 <strong>{orig_n:,} records</strong> · {df_raw.shape[1]} columns · {'sufficient' if orig_n>=500 else 'moderate' if orig_n>=100 else 'limited'} for forecasting.",
        f"🤖 Best: <strong>{best_name}</strong> — R² {max(0,best_r['R2']):.4f}, MAE {best_r['MAE']:,.0f}.",
        f"📅 Forecast horizon <strong>{n_forecast} periods</strong> with ±12% confidence.",
    ]
    if corr_insight: findings.append(corr_insight)
    null_pct = pd.Series(prof["null_pct"])
    if (null_pct > 20).any():
        bad = null_pct[null_pct>20].index[:3].tolist()
        findings.append(f"🧹 Columns {', '.join(bad)} have >20% missing — filling these may improve accuracy.")
    else:
        findings.append("✅ Clean data — no major missing value issues.")
    if orig_n < 200:
        findings.append("⚠️ Small dataset — aim for 500+ rows for production-grade forecasting.")

    st.markdown('<div style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;margin:1.5rem 0 0.6rem;">KEY FINDINGS</div>', unsafe_allow_html=True)
    for f in findings:
        st.markdown(f'<div class="insight-box" style="margin-bottom:0.4rem;">{f}</div>', unsafe_allow_html=True)

    recs = []
    if not (null_pct < 20).all():
        for col, pct in null_pct[null_pct>20].items():
            recs.append(f"• <strong>{col}</strong>: {pct:.1f}% missing — {'drop or impute' if pct>50 else 'fill with median/mode'}.")
    if orig_n < 200:
        recs.append(f"• Only {orig_n} rows — collect more. 500+ rows = much better accuracy.")
    if best_r["R2"] < 0.5:
        recs.append("• Low R² — add time-based features (month, quarter) or external data.")

    st.markdown('<div style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;margin:1.2rem 0 0.6rem;">RECOMMENDATIONS</div>', unsafe_allow_html=True)
    if recs:
        st.markdown('<div class="warn-box">'+"<br>".join(recs)+"</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ <strong>No major data issues.</strong> Well-structured for forecasting.</div>', unsafe_allow_html=True)

# ════════════════════════════════
#  TAB 5 — DEEP ANALYSIS
# ════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Deep Analysis</div></div>', unsafe_allow_html=True)

    if len(num_cols) >= 2:
        cc   = num_cols[:10]
        corr = df_work[cc].corr()
        fig, ax = plt.subplots(figsize=(min(12,len(cc)*1.1+2), min(10,len(cc)+2)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdPu",
                    ax=ax, center=0, linewidths=0.5,
                    annot_kws={"size":8}, cbar_kws={"label":"Correlation"}, square=True)
        ax.set_title("Correlation Heatmap", fontsize=9, fontweight="bold", color="#E0E0FF")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if target_col in corr.columns:
            tc2 = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(9, max(2.8, len(tc2)*0.48)))
            clrs_tc = [C["green"] if v>0 else C["red"] for v in tc2.values]
            ax.barh(tc2.index, tc2.values, color=clrs_tc, edgecolor="#0A0A0F", height=0.52)
            ax.axvline(0, color=C["grey"], lw=1); ax.set_xlim(-1.1,1.1)
            ax.set_title(f"Top Correlations with '{target_col[:25]}'", fontsize=9, fontweight="bold", color="#E0E0FF")
            ax.grid(axis="x", alpha=0.2); plt.tight_layout(); st.pyplot(fig); plt.close()

    if len(num_cols) >= 2:
        cc6 = num_cols[:6]
        Xc  = df_work[cc6].fillna(0)
        km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_work["_cluster"] = km.fit_transform(StandardScaler().fit_transform(Xc)) if False else km.fit_predict(StandardScaler().fit_transform(Xc))
        vo = df_work[cc6].var().sort_values(ascending=False)
        cx, cy = vo.index[0], (vo.index[1] if len(vo)>1 else vo.index[0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for k in range(n_clusters):
            mk = df_work["_cluster"]==k
            axes[0].scatter(df_work.loc[mk,cx], df_work.loc[mk,cy],
                            s=25, alpha=0.55, edgecolors="none", color=PAL[k%len(PAL)],
                            label=f"Seg {k} (n={mk.sum()})")
        axes[0].set_title("Segment Scatter", fontsize=9, fontweight="bold", color="#E0E0FF")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.2)

        cs = df_work["_cluster"].value_counts().sort_index()
        axes[1].bar([f"Seg {k}" for k in cs.index], cs.values,
                    color=PAL[:n_clusters], edgecolor="#0A0A0F", width=0.5)
        for bar, v in zip(axes[1].patches, cs.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         f"{v}\n({v/len(df_work)*100:.0f}%)", ha="center", fontsize=8, fontweight="bold")
        axes[1].set_title("Segment Sizes", fontsize=9, fontweight="bold", color="#E0E0FF")
        axes[1].grid(axis="y", alpha=0.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        cm = df_work.groupby("_cluster")[cc6].mean().round(2)
        cm.index = [f"Segment {i}" for i in cm.index]
        st.dataframe(cm.style.background_gradient(cmap="Purples", axis=0).format("{:.2f}"),
                     use_container_width=True)

# ════════════════════════════════
#  TAB 6 — RAW DATA
# ════════════════════════════════
with tab6:
    st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Raw Data</div></div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([3,1])
    with sc1:
        search = st.text_input("Search", "", placeholder="Filter rows…", label_visibility="collapsed")
    with sc2:
        max_r = st.selectbox("Rows", [100,250,500,1000], index=0, label_visibility="collapsed")
    disp = df_raw
    if search:
        mask = df_raw.astype(str).apply(lambda c: c.str.contains(search,case=False)).any(axis=1)
        disp = df_raw[mask]
    st.dataframe(disp.head(max_r), use_container_width=True, height=420)
    st.caption(f"{min(max_r,len(disp)):,} of {len(disp):,} rows  |  {source_label}")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download Filtered CSV", disp.to_csv(index=False).encode(),
                           "filtered.csv","text/csv", use_container_width=True)
    with d2:
        st.download_button("Download Clean CSV", df_clean.to_csv(index=False).encode(),
                           "clean.csv","text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <strong style='color:#6B7280;'>V O S</strong> — ML Forecasting Platform &nbsp;·&nbsp;EDA &nbsp;·&nbsp;  Predictive Analytics &nbsp;·&nbsp;Data Visualization &nbsp;·&nbsp;Streamlit &nbsp;·&nbsp;Feature Engineering
</div>
""", unsafe_allow_html=True)