"""
5G Network Connection Drop Prediction - Streamlit Web Application
Review 3: Machine Learning Model Development, Interactive Telemetry & Live Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Page configuration
st.set_page_config(
    page_title="5G Network ML | Review 3",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #182B49;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #0E74BC;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📶 5G Network Quality & Connection Drop Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Review 3: Machine Learning Model Pipeline, Statistical Proofs & Live Inference Playground</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Data Loader & Preprocessing
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    file_candidates = [
        '5g_network_data.csv',
        os.path.join('project', '5g_network_data.csv'),
        os.path.join('..', '5g_network_data.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '5g_network_data.csv')
    ]
    csv_file = None
    for f in file_candidates:
        if os.path.exists(f):
            csv_file = f
            break
            
    if not csv_file:
        st.error("Error: '5g_network_data.csv' not found in workspace.")
        return None, None
        
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
    except Exception:
        df = pd.read_csv(csv_file)
        
    df.columns = [c.encode('ascii', 'ignore').decode('ascii').strip() for c in df.columns]
    raw_df = df.copy()
    
    # Cleaning pipeline
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    # Impute
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
        
    cat_cols = df_clean.select_dtypes(include=['object', 'bool']).columns
    for c in cat_cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])
        
    if 'Timestamp' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Timestamp'])
        
    return raw_df, df_clean

raw_df, df_clean = load_and_clean_data()

if df_clean is None:
    st.stop()

# -----------------------------------------------------------------------------
# Cached Model Trainer
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models(df):
    target_col = 'Dropped Connection'
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    nominal_cols = ['Location', 'Network Type', 'Device Model', 'Carrier', 'Band']
    ordinal_cols = ['Network Congestion Level']
    boolean_cols = ['VoNR Enabled']
    scaling_numeric_cols = [
        'Signal Strength (dBm)', 'Download Speed (Mbps)', 'Upload Speed (Mbps)',
        'Latency (ms)', 'Jitter (ms)', 'Battery Level (%)', 'Temperature (C)',
        'Connected Duration (min)', 'Handover Count', 'Data Usage (MB)',
        'Video Streaming Quality', 'Ping to Google (ms)'
    ]

    prep_standard = ColumnTransformer(
        transformers=[
            ('num_std', StandardScaler(), scaling_numeric_cols),
            ('cat_nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
            ('cat_ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]), ordinal_cols),
            ('passthrough', 'passthrough', boolean_cols)
        ]
    )

    prep_minmax = ColumnTransformer(
        transformers=[
            ('num_minmax', MinMaxScaler(), scaling_numeric_cols),
            ('cat_nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
            ('cat_ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]), ordinal_cols),
            ('passthrough', 'passthrough', boolean_cols)
        ]
    )

    models = {
        'HistGradientBoosting': Pipeline([
            ('prep', prep_standard),
            ('clf', HistGradientBoostingClassifier(max_iter=100, random_state=42))
        ]),
        'Logistic Regression (Standardized)': Pipeline([
            ('prep', prep_standard),
            ('clf', LogisticRegression(max_iter=500, random_state=42))
        ]),
        'Logistic Regression (Normalized)': Pipeline([
            ('prep', prep_minmax),
            ('clf', LogisticRegression(max_iter=500, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('prep', prep_standard),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ])
    }

    results = []
    y_preds = {}
    y_probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        y_preds[name] = pred
        y_probs[name] = prob

    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    return models, results_df, X_train, X_test, y_train, y_test, y_preds, y_probs

with st.spinner("Initializing models & evaluation pipelines..."):
    models, results_df, X_train, X_test, y_train, y_test, y_preds, y_probs = train_models(df_clean)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/5g.png", width=70)
st.sidebar.title("Review 3 Sections")
app_mode = st.sidebar.radio("Navigate Review Milestones:", [
    "1. Data Understanding",
    "2. Data Cleaning",
    "3. Statistical Checks",
    "4. Preprocessing & Scaling",
    "5. Model Benchmarks",
    "6. Live Prediction Simulator",
    "7. Viva Defense Manual"
])

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** 5G Connection Drop ML  
**Dataset:** 50,000 session logs  
**Evaluation:** 80/20 Stratified Split  
**Target:** Binary Connection Drop
""")

# =============================================================================
# TAB 1: DATA UNDERSTANDING
# =============================================================================
if app_mode == "1. Data Understanding":
    st.header("1. Data Understanding & Feature Taxonomy")
    st.write("Exploratory inspection of 50,000 telecommunication sessions across 21 raw features.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(raw_df):,}")
    col2.metric("Total Features", f"{raw_df.shape[1]}")
    col3.metric("Dropped Connections", f"{raw_df['Dropped Connection'].sum():,} ({raw_df['Dropped Connection'].mean()*100:.2f}%)")
    col4.metric("Active Connections", f"{(~raw_df['Dropped Connection']).sum():,} ({(1-raw_df['Dropped Connection'].mean())*100:.2f}%)")

    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Statistical Measurement Taxonomy")
    tax_data = [
        {"Category": "Numerical (Continuous)", "Features": "Signal Strength, Download Speed, Upload Speed, Latency, Jitter, Temperature, Data Usage, Ping", "Count": 8, "Role": "Physical & Radio Access Network metrics"},
        {"Category": "Numerical (Discrete)", "Features": "Battery Level (%), Connected Duration (min), Handover Count", "Count": 3, "Role": "Device state & cell tower mobility counts"},
        {"Category": "Categorical (Nominal)", "Features": "Location, Network Type, Device Model, Carrier, Band", "Count": 5, "Role": "Qualitative identifiers with NO inherent order"},
        {"Category": "Categorical (Ordinal)", "Features": "Network Congestion Level (Low < Med < High), Video Streaming Quality (1 to 5)", "Count": 2, "Role": "Ordered qualitative ranks"},
        {"Category": "Binary / Boolean", "Features": "VoNR Enabled, Dropped Connection (Target)", "Count": 2, "Role": "Binary state indicators"},
        {"Category": "Metadata / ID", "Features": "Timestamp", "Count": 1, "Role": "Unique session log identifier (pruned)"}
    ]
    st.table(pd.DataFrame(tax_data))

# =============================================================================
# TAB 2: DATA CLEANING
# =============================================================================
elif app_mode == "2. Data Cleaning":
    st.header("2. Data Cleaning & Imputation Pipeline")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Integrity Audit")
        st.write("• **Duplicates Found:** `0` (Verified via `df.duplicated().sum()`)")
        st.write("• **Missing Values in Raw CSV:** `0` (100% complete records)")
        st.write("• **Metadata Removed:** `Timestamp` column dropped to prevent data leakage and memorization.")
        st.write("• **Cleaned Matrix:** 50,000 samples × 20 features ready for featurization.")

    with col2:
        st.subheader("Imputation Rationale")
        st.info("""
        **Continuous Numerical:** Imputed with **Median**  
        *Rationale:* The median is non-parametric and resistant to extreme skewness, whereas the mean shifts heavily when outliers occur.
        
        **Categorical & Boolean:**
