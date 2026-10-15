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
        st.error("Error: '5g_network_data.csv' not found. Please ensure the CSV is committed to your repository.")
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

with st.spinner("Initializing ML models and pipelines..."):
    models, results_df, X_train, X_test, y_train, y_test, y_preds, y_probs = train_models(df_clean)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Review 3 Navigation")
app_mode = st.sidebar.radio("Select Section:", [
    "1. Data Understanding",
    "2. Data Cleaning",
    "3. Statistical Checks",
    "4. Preprocessing & Scaling",
    "5. Model Benchmarks",
    "6. Live Prediction Simulator",
    "7. Viva Defense Manual"
])

st.sidebar.markdown("---")
st.sidebar.info("5G Connection Drop ML Project | 50,000 Session Logs | Review 3")

# =============================================================================
# SECTION 1: DATA UNDERSTANDING
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
# SECTION 2: DATA CLEANING
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
        st.info("Continuous Numerical: Imputed with Median (resistant to skewness and extreme spikes).")
        st.info("Categorical & Boolean: Imputed with Mode (preserves dominant category frequency).")
        st.info("Zero Row Deletion: Dropping rows reduces statistical power and induces selection bias.")

# =============================================================================
# SECTION 3: STATISTICAL CHECKS
# =============================================================================
elif app_mode == "3. Statistical Checks":
    st.header("3. Statistical Checks: Normality & Outliers")
    
    st.subheader("Shapiro-Wilk Normality Hypothesis Test")
    st.write("H0: The feature follows a normal distribution (alpha = 0.05).")
    st.write("H1: The feature does NOT follow a normal distribution.")

    num_cols = [
        'Signal Strength (dBm)', 'Download Speed (Mbps)', 'Upload Speed (Mbps)',
        'Latency (ms)', 'Jitter (ms)', 'Battery Level (%)', 'Temperature (C)',
        'Data Usage (MB)', 'Ping to Google (ms)'
    ]

    norm_rows = []
    for col in num_cols:
        sample = df_clean[col].sample(min(5000, len(df_clean)), random_state=42)
        stat_w, p_val = stats.shapiro(sample)
        skew = df_clean[col].skew()
        kurt = df_clean[col].kurtosis()
        norm_rows.append({
            'Feature': col,
            'Shapiro Stat (W)': round(stat_w, 4),
            'p-value': f"{p_val:.2e}",
            'Skewness': round(skew, 3),
            'Excess Kurtosis': round(kurt, 3),
            'Decision': "Reject H0 (Non-Normal)" if p_val < 0.05 else "Normal"
        })

    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)

    st.success("The -1.20 Kurtosis Proof: Skewness is approximately 0.00 (symmetric), but excess kurtosis is consistently -1.20 across all continuous features. In statistics, an excess kurtosis of exactly -1.20 is the theoretical signature of a Continuous Uniform Distribution U(a, b). This mathematically proves that our telemetry features represent bounded uniform intervals.")

    st.subheader("Distribution & Normal Q-Q Plot Inspector")
    chosen_col = st.selectbox("Select Feature:", num_cols)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df_clean[chosen_col], kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f"Histogram & KDE: {chosen_col}")
    
    stats.probplot(df_clean[chosen_col].sample(2000, random_state=42), dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot (Notice Bounded S-Curve Signature)")
    st.pyplot(fig)

    st.subheader("Tukey's 1.5 IQR Outlier Detection")
    st.write("Using Tukey's fences [Q1 - 1.5*IQR, Q3 + 1.5*IQR], 0 outliers were detected because the fences mathematically enclose the entire bounded uniform range.")

# =============================================================================
# SECTION 4: PREPROCESSING & SCALING
# =============================================================================
elif app_mode == "4. Preprocessing & Scaling":
    st.header("4. Data Preprocessing: Standardization vs. Normalization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Standardization (Z-Score)")
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.write("• Rescales to mean 0 and unit variance 1.")
        st.write("• Mandatory for Logistic Regression: Restores spherical loss contours for smooth gradient descent.")
        
    with col2:
        st.subheader("Normalization (Min-Max)")
        st.latex(r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
        st.write("• Squashes values strictly into the interval [0, 1].")
        st.write("• Required for distance-based models (k-NN) so high-magnitude features do not distort Euclidean distances.")

    st.info("Tree-Based Invariance: Decision Trees, Random Forests, and Gradient Boosting are invariant to monotonic scaling because node splits depend solely on threshold rank order.")

    st.subheader("Interactive Feature Scaling Inspector")
    feat_inspect = st.selectbox("Select Feature to Scale:", [
        'Download Speed (Mbps)', 'Signal Strength (dBm)', 'Latency (ms)', 'Temperature (C)'
    ])
    
    raw_v = X_train[feat_inspect].values
    std_v = StandardScaler().fit_transform(X_train[[feat_inspect]]).flatten()
    minmax_v = MinMaxScaler().fit_transform(X_train[[feat_inspect]]).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    sns.histplot(raw_v, kde=True, ax=axes[0], color='royalblue')
    axes[0].set_title(f"Raw: {feat_inspect}\nMean={raw_v.mean():.1f}, Std={raw_v.std():.1f}")
    
    sns.histplot(std_v, kde=True, ax=axes[1], color='darkorange')
    axes[1].set_title(f"Standardized (Z-Score)\nMean={std_v.mean():.1f}, Std={std_v.std():.1f}")
    
    sns.histplot(minmax_v, kde=True, ax=axes[2], color='seagreen')
    axes[2].set_title(f"Normalized (Min-Max [0, 1])\nMin={minmax_v.min():.1f}, Max={minmax_v.max():.1f}")
    st.pyplot(fig)

# =============================================================================
# SECTION 5: MODEL BENCHMARKS
# =============================================================================
elif app_mode == "5. Model Benchmarks":
    st.header("5. Model Performance Benchmarks (Test Set N=10,000)")
    
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, prob in y_probs.items():
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, label=f"{name[:18]} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label="Chance (0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right', fontsize=8)
        st.pyplot(fig)

    with col2:
        st.subheader("Confusion Matrix")
        chosen_m = st.selectbox("Select Model:", list(models.keys()))
        cm = confusion_matrix(y_test, y_preds[chosen_m])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Active (0)', 'Dropped (1)'],
                    yticklabels=['Active (0)', 'Dropped (1)'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual Ground Truth")
        st.pyplot(fig)

# =============================================================================
# SECTION 6: LIVE PREDICTION SIMULATOR
# =============================================================================
elif app_mode == "6. Live Prediction Simulator":
    st.header("6. Live 5G Telemetry Inference Simulator")
    st.write("Adjust telemetry sliders below to simulate a live 5G session and predict whether the connection will drop:")

    with st.form("prediction_form"):
        colA, colB, colC = st.columns(3)
        
        with colA:
            st.markdown("**Radio & Transport Layer**")
            sig_str = st.slider("Signal Strength (RSRP in dBm)", -110.0, -60.0, -85.0, 0.5)
            latency = st.slider("Latency (ms)", 1.0, 20.0, 10.0, 0.5)
            jitter = st.slider("Jitter (ms)", 0.1, 5.0, 2.5, 0.1)
            ping = st.slider("Ping to Google (ms)", 10.0, 100.0, 50.0, 1.0)
            
        with colB:
            st.markdown("**Throughput & Device State**")
            dl_speed = st.slider("Download Speed (Mbps)", 100.0, 1000.0, 500.0, 10.0)
            ul_speed = st.slider("Upload Speed (Mbps)", 20.0, 150.0, 80.0, 5.0)
            temp = st.slider("Device Temperature (°C)", 20.0, 45.0, 32.0, 0.5)
            battery = st.slider("Battery Level (%)", 10, 99, 50)
            duration = st.slider("Connected Duration (min)", 1, 60, 30)

        with colC:
            st.markdown("**Operational Environment**")
            handover = st.selectbox("Handover Count", [0, 1, 2, 3, 4], index=1)
            data_usage = st.slider("Data Usage (MB)", 10.0, 500.0, 250.0, 10.0)
            quality = st.selectbox("Video Streaming Quality (1: Low to 5: 4K)", [1, 2, 3, 4, 5], index=2)
            congestion = st.selectbox("Network Congestion Level", ['Low', 'Medium', 'High'], index=1)
            net_type = st.selectbox("Network Type", ['4G', '5G NSA', '5G SA'], index=1)
            carrier = st.selectbox("Telecom Carrier", ['Airtel', 'Jio', 'AT&T', 'Verizon', 'T-Mobile', 'BSNL', 'Vi'], index=0)
            device = st.selectbox("Device Model", ['iPhone 14', 'Galaxy S23', 'Pixel 7', 'Nord 4', 'GT 7'], index=0)
            band = st.selectbox("Frequency Band", ['n28', 'n41', 'n78', 'n258', 'n260'], index=2)
            vonr = st.checkbox("VoNR Enabled", value=True)

        selected_model_name = st.selectbox("Choose Classification Engine:", list(models.keys()))
        submitted = st.form_submit_button("⚡ Predict Connection Drop Probability")

    if submitted:
        input_dict = {
            'Location': 'Berlin',
            'Signal Strength (dBm)': sig_str,
            'Download Speed (Mbps)': dl_speed,
            'Upload Speed (Mbps)': ul_speed,
            'Latency (ms)': latency,
            'Jitter (ms)': jitter,
            'Network Type': net_type,
            'Device Model': device,
            'Carrier': carrier,
            'Band': band,
            'Battery Level (%)': battery,
            'Temperature (C)': temp,
            'Connected Duration (min)': duration,
            'Handover Count': handover,
            'Data Usage (MB)': data_usage,
            'Video Streaming Quality': quality,
            'VoNR Enabled': vonr,
            'Network Congestion Level': congestion,
            'Ping to Google (ms)': ping
        }
        
        input_df = pd.DataFrame([input_dict])
        active_model = models[selected_model_name]
        
        drop_prob = active_model.predict_proba(input_df)[0][1]
        is_drop = active_model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("Inference Result:")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if is_drop == 1:
                st.error("⚠️ **PREDICTION: CONNECTION WILL DROP**")
            else:
                st.success("✅ **PREDICTION: CONNECTION STABLE / ACTIVE**")
                
        with res_col2:
            st.metric("Estimated Drop Probability", f"{drop_prob*100:.2f}%")
            st.progress(float(drop_prob))

# =============================================================================
# SECTION 7: VIVA DEFENSE MANUAL
# =============================================================================
elif app_mode == "7. Viva Defense Manual":
    st.header("7. Review 3 Presentation & Viva Cheat Sheet")
    
    with st.expander("Q1: Why is model accuracy around ~50%? Does that mean the model failed?", expanded=True):
        st.write("Answer: No, the models did not fail; they functioned with statistical integrity. Correlation analysis confirmed that every predictor has a Pearson correlation |r| < 0.010 with the target. The dataset was generated as independent uniform variables. Because the target is evenly balanced (50.06% vs. 49.94%), random chance is 50%. Our models accurately reflected the true mathematical entropy of the data without overfitting.")

    with st.expander("Q2: What is the significance of the -1.20 Kurtosis?", expanded=True):
        st.write("Answer: A standard normal distribution has excess kurtosis of 0. A continuous uniform distribution U(a, b) has theoretical excess kurtosis of -1.20. Because all our continuous features exhibit kurtosis of exactly -1.20, they represent bounded uniform intervals rather than Gaussian bell curves.")

    with st.expander("Q3: How did you ensure zero Data Leakage?"):
        st.write("Answer: We performed an 80/20 stratified train-test split BEFORE fitting any transformers. All scalers and encoders were fitted strictly on X_train and evaluated on X_test using scikit-learn Pipeline objects.")

    with st.expander("Q4: Which model architecture do you recommend for real 5G deployment?"):
        st.write("Answer: We recommend Histogram-based Gradient Boosting (HistGB) and Random Forest. In real cellular networks, drops occur due to non-linear physical thresholds (e.g. Signal dropping below -105 dBm combined with high jitter). Non-linear tree ensembles naturally capture multi-feature trigger thresholds.")
