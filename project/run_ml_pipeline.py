

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)


sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

print("=" * 80)
print("5G NETWORK MACHINE LEARNING REVIEW PIPELINE")
print("=" * 80)

# ==============================================================================
# 1. DATA UNDERSTANDING
# ==============================================================================
print("\n" + "=" * 50)
print("1. DATA UNDERSTANDING")
print("=" * 50)

# Load dataset (handling potential latin1 degree symbol)
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '5g_network_data.csv')
try:
    df = pd.read_csv(csv_path, encoding='latin1')
except Exception:
    df = pd.read_csv(csv_path)

# Normalize column names to ASCII to ensure complete cross-platform portability
df.columns = [c.encode('ascii', 'ignore').decode('ascii').strip() for c in df.columns]

print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Features (Columns): {df.shape[1]}")

# Categorization of features by statistical measurement type
feature_classification = {
    'Timestamp': ('Temporal / Metadata', 'Timestamp of the recording (discarded for modeling)'),
    'Location': ('Categorical (Nominal)', 'City where data was recorded (8 distinct cities)'),
    'Signal Strength (dBm)': ('Numerical (Continuous)', 'Radio Signal Received Power / RSRP [-110 to -60 dBm]'),
    'Download Speed (Mbps)': ('Numerical (Continuous)', 'Downlink network throughput [100 to 1000 Mbps]'),
    'Upload Speed (Mbps)': ('Numerical (Continuous)', 'Uplink network throughput [20 to 150 Mbps]'),
    'Latency (ms)': ('Numerical (Continuous)', 'One-way radio access transmission delay [1 to 20 ms]'),
    'Jitter (ms)': ('Numerical (Continuous)', 'Packet delay variation [0.1 to 5.0 ms]'),
    'Network Type': ('Categorical (Nominal)', 'Deployment mode: 4G, 5G NSA (Non-Standalone), 5G SA (Standalone)'),
    'Device Model': ('Categorical (Nominal)', 'User equipment model (iPhone 14, Galaxy S23, Pixel 7, Nord 4, GT 7)'),
    'Carrier': ('Categorical (Nominal)', 'Telecom service provider (AT&T, Airtel, BSNL, Jio, T-Mobile, Verizon, Vi)'),
    'Band': ('Categorical (Nominal)', 'Operating frequency band (n28, n41, n78, n258, n260)'),
    'Battery Level (%)': ('Numerical (Discrete)', 'Device battery percentage [10% to 99%]'),
    'Temperature (C)': ('Numerical (Continuous)', 'Operating temperature of device [20 to 45 C]'),
    'Connected Duration (min)': ('Numerical (Discrete)', 'Continuous duration of session [1 to 59 min]'),
    'Handover Count': ('Numerical (Discrete)', 'Number of base station cell handovers [0 to 4]'),
    'Data Usage (MB)': ('Numerical (Continuous)', 'Cumulative data consumed in session [10 to 500 MB]'),
    'Video Streaming Quality': ('Categorical (Ordinal)', 'Streaming resolution scale rating [1: Low to 5: 4K UHD]'),
    'VoNR Enabled': ('Binary / Boolean', 'Voice over New Radio feature status (True/False)'),
    'Network Congestion Level': ('Categorical (Ordinal)', 'Network traffic congestion state (Low, Medium, High)'),
    'Ping to Google (ms)': ('Numerical (Continuous)', 'Round-trip time to reference internet host [10 to 100 ms]'),
    'Dropped Connection': ('Binary / Target', 'Target variable: Network drop failure event (True/False)')
}

understanding_records = []
for col, (f_type, desc) in feature_classification.items():
    if col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample_vals = str(df[col].dropna().unique()[:3].tolist())
        understanding_records.append({
            'Feature': col,
            'Pandas Dtype': dtype,
            'Statistical Type': f_type,
            'Distinct Values': n_unique,
            'Sample Values': sample_vals,
            'Description': desc
        })

df_understanding = pd.DataFrame(understanding_records)
print(df_understanding[['Feature', 'Statistical Type', 'Distinct Values']].to_string(index=False))

# Plot 1: Feature Data Types Breakdown
fig, ax = plt.subplots(figsize=(10, 5))
type_counts = pd.Series([v[0].split(' ')[0] for v in feature_classification.values()]).value_counts()
sns.barplot(x=type_counts.values, y=type_counts.index, hue=type_counts.index, palette='viridis', legend=False, ax=ax)
ax.set_title('Feature Breakdown by Measurement Type (Total Features = 21)', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Features', fontsize=11)
for i, v in enumerate(type_counts.values):
    ax.text(v + 0.1, i, str(v), va='center', fontweight='bold', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_feature_data_types_summary.png'), dpi=300)
plt.close()
print("Saved: 01_feature_data_types_summary.png")


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
print("\n" + "=" * 50)
print("2. DATA CLEANING")
print("=" * 50)

df_clean = df.copy()

# Step 2.1: Duplicate verification
initial_duplicates = df_clean.duplicated().sum()
print(f"Duplicate records found: {initial_duplicates}")
if initial_duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"Dropped {initial_duplicates} duplicate rows.")

# Step 2.2: Missing values inspection & robust imputation strategy
missing_summary = df_clean.isnull().sum()
total_missing = missing_summary.sum()
print(f"Missing values found across all columns: {total_missing}")

# Imputation pipeline demonstration (vital for production readiness)
# - Continuous numeric features: Impute with Median (unaffected by extreme values)
# - Categorical & Boolean features: Impute with Mode (most frequent category)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()

for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"Imputed numeric column '{col}' with median: {median_val}")

for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode().iloc[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f"Imputed categorical column '{col}' with mode: {mode_val}")

# Step 2.3: Metadata removal
# Discard high-cardinality unique timestamp column to prevent memorization / spurious correlation
if 'Timestamp' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Timestamp'])
    print("Dropped 'Timestamp' identifier column to prevent leakage.")

print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Post-cleaning missing value count: {df_clean.isnull().sum().sum()}")


# ==============================================================================
# 3. STATISTICAL CHECKS: NORMALITY & OUTLIERS
# ==============================================================================
print("\n" + "=" * 50)
print("3. STATISTICAL CHECKS (NORMALITY & OUTLIERS)")
print("=" * 50)

numeric_features = [
    'Signal Strength (dBm)', 'Download Speed (Mbps)', 'Upload Speed (Mbps)',
    'Latency (ms)', 'Jitter (ms)', 'Battery Level (%)', 'Temperature (C)',
    'Connected Duration (min)', 'Handover Count', 'Data Usage (MB)',
    'Video Streaming Quality', 'Ping to Google (ms)'
]

normality_records = []
outlier_records = []

for col in numeric_features:
    series = df_clean[col].dropna()
    sample = series.sample(min(5000, len(series)), random_state=42)
    stat_shapiro, p_val = stats.shapiro(sample)
    skew = series.skew()
    kurt = series.kurtosis()
    is_normal = p_val > 0.05

    normality_records.append({
        'Feature': col,
        'Shapiro Stat (W)': round(stat_shapiro, 4),
        'p-value': f"{p_val:.2e}",
        'Skewness': round(skew, 3),
        'Kurtosis': round(kurt, 3),
        'Follows Normal Dist': "Yes (p > 0.05)" if is_normal else "No (p <= 0.05)"
    })

    # Outlier detection using Tukey's IQR Method
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(series)) * 100

    outlier_records.append({
        'Feature': col,
        'Q1 (25%)': round(q1, 2),
        'Q3 (75%)': round(q3, 2),
        'IQR': round(iqr, 2),
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outlier Count': outlier_count,
        'Outlier %': f"{outlier_pct:.2f}%"
    })

df_normality = pd.DataFrame(normality_records)
df_outliers = pd.DataFrame(outlier_records)

print("\n--- NORMALITY TEST RESULTS (Shapiro-Wilk & Moments) ---")
print(df_normality.to_string(index=False))

print("\n--- OUTLIER DETECTION RESULTS (Tukey's IQR Method) ---")
print(df_outliers[['Feature', 'Lower Bound', 'Upper Bound', 'Outlier Count', 'Outlier %']].to_string(index=False))

# Plot 2: Normality Visualization (Histograms + KDE and Q-Q Plots for key continuous features)
key_continuous = ['Signal Strength (dBm)', 'Download Speed (Mbps)', 'Latency (ms)', 'Temperature (C)']
fig, axes = plt.subplots(4, 2, figsize=(14, 14))

for i, col in enumerate(key_continuous):
    # Histogram + KDE
    sns.histplot(df_clean[col], kde=True, ax=axes[i, 0], color='steelblue', bins=30)
    axes[i, 0].set_title(f'Histogram & KDE: {col}', fontsize=11, fontweight='bold')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')

    # Q-Q plot vs Standard Normal
    sample_qq = df_clean[col].sample(min(2000, len(df_clean)), random_state=42)
    stats.probplot(sample_qq, dist="norm", plot=axes[i, 1])
    axes[i, 1].get_lines()[0].set_color('steelblue')
    axes[i, 1].get_lines()[1].set_color('crimson')
    axes[i, 1].set_title(f'Normal Q-Q Plot: {col}', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_normality_checks_distribution_qq.png'), dpi=300)
plt.close()
print("Saved: 02_normality_checks_distribution_qq.png")

# Plot 3: Box Plots for Outlier Inspection
fig, ax = plt.subplots(figsize=(14, 6))
# Normalize numeric columns to [0, 1] purely for unified boxplot visualization
norm_vis = (df_clean[numeric_features] - df_clean[numeric_features].min()) / (
    df_clean[numeric_features].max() - df_clean[numeric_features].min()
)
sns.boxplot(data=norm_vis, orient='h', palette='Set2', ax=ax)
ax.set_title('Outlier Inspection Across All Numeric Features (Scaled [0,1] for Visual Comparison)', fontsize=13, fontweight='bold')
ax.set_xlabel('Normalized Feature Scale (0 to 1)')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_outlier_detection_boxplots.png'), dpi=300)
plt.close()
print("Saved: 03_outlier_detection_boxplots.png")


# ==============================================================================
# 4. DATA PREPROCESSING (STANDARDIZATION VS NORMALIZATION)
# ==============================================================================
print("\n" + "=" * 50)
print("4. DATA PREPROCESSING (SCALING & ENCODING)")
print("=" * 50)

# Target separation
target_col = 'Dropped Connection'
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col].astype(int)

# Mandatory Featurization Ordering: Split train/test FIRST to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Testing set : {X_test.shape[0]:,} samples")
print(f"Class Balance: {y_train.value_counts(normalize=True).to_dict()}")

# Separate columns into scaling and encoding groups
nominal_cols = ['Location', 'Network Type', 'Device Model', 'Carrier', 'Band']
ordinal_cols = ['Network Congestion Level']
boolean_cols = ['VoNR Enabled']
scaling_numeric_cols = [
    'Signal Strength (dBm)', 'Download Speed (Mbps)', 'Upload Speed (Mbps)',
    'Latency (ms)', 'Jitter (ms)', 'Battery Level (%)', 'Temperature (C)',
    'Connected Duration (min)', 'Handover Count', 'Data Usage (MB)',
    'Video Streaming Quality', 'Ping to Google (ms)'
]

# Pipeline 1: Preprocessor with STANDARDIZATION (Z-Score: mean=0, std=1)
preprocessor_standard = ColumnTransformer(
    transformers=[
        ('num_std', StandardScaler(), scaling_numeric_cols),
        ('cat_nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
        ('cat_ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]), ordinal_cols),
        ('passthrough', 'passthrough', boolean_cols)
    ]
)

# Pipeline 2: Preprocessor with NORMALIZATION (MinMax: range [0, 1])
preprocessor_minmax = ColumnTransformer(
    transformers=[
        ('num_minmax', MinMaxScaler(), scaling_numeric_cols),
        ('cat_nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
        ('cat_ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]), ordinal_cols),
        ('passthrough', 'passthrough', boolean_cols)
    ]
)

# Fit and demonstrate the transformation on training data
sample_feat = 'Download Speed (Mbps)'
raw_vals = X_train[sample_feat].values
std_scaler = StandardScaler().fit(X_train[[sample_feat]])
std_vals = std_scaler.transform(X_train[[sample_feat]]).flatten()
minmax_scaler = MinMaxScaler().fit(X_train[[sample_feat]])
minmax_vals = minmax_scaler.transform(X_train[[sample_feat]]).flatten()

# Plot 4: Feature Scaling Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(raw_vals, ax=axes[0], color='royalblue', kde=True, bins=30)
axes[0].set_title(f'Raw Distribution: {sample_feat}\nMean={raw_vals.mean():.1f}, Std={raw_vals.std():.1f}', fontweight='bold')
axes[0].set_xlabel('Speed (Mbps)')

sns.histplot(std_vals, ax=axes[1], color='darkorange', kde=True, bins=30)
axes[1].set_title(f'Standardized (Z-Score Scaling)\nMean={std_vals.mean():.1f}, Std={std_vals.std():.1f}', fontweight='bold')
axes[1].set_xlabel('Standard Deviations (z)')

sns.histplot(minmax_vals, ax=axes[2], color='seagreen', kde=True, bins=30)
axes[2].set_title(f'Normalized (Min-Max Scaling [0, 1])\nMin={minmax_vals.min():.1f}, Max={minmax_vals.max():.1f}', fontweight='bold')
axes[2].set_xlabel('Normalized Scale [0, 1]')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04_feature_scaling_comparison.png'), dpi=300)
plt.close()
print("Saved: 04_feature_scaling_comparison.png")


# ==============================================================================
# 5. MODEL BUILDING & SELECTION
# ==============================================================================
print("\n" + "=" * 50)
print("5. MODEL BUILDING & EVALUATION")
print("=" * 50)

# Instantiate models with justifications:
# 1. Logistic Regression (Standardized): Interpretable linear baseline
# 2. Logistic Regression (Normalized): Direct empirical test of MinMax scaling
# 3. Random Forest Classifier: Non-linear ensemble of decision trees with bagging
# 4. HistGradientBoostingClassifier: Modern histogram-based gradient boosting

models = {
    'Logistic Regression (Standardized)': Pipeline([
        ('prep', preprocessor_standard),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Logistic Regression (Normalized)': Pipeline([
        ('prep', preprocessor_minmax),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest Classifier': Pipeline([
        ('prep', preprocessor_standard),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ]),
    'Gradient Boosting (HistGB)': Pipeline([
        ('prep', preprocessor_standard),
        ('clf', HistGradientBoostingClassifier(max_iter=150, random_state=42))
    ])
}

results = []
trained_models = {}
y_preds = {}
y_probs = {}

for name, model_pipeline in models.items():
    print(f"Training: {name}...")
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })

    trained_models[name] = model_pipeline
    y_preds[name] = y_pred
    y_probs[name] = y_prob

df_results = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL EVALUATION BENCHMARK")
print("=" * 70)
print(df_results.to_string(index=False))


# ==============================================================================
# 6. MODEL ANALYSIS & FINDINGS
# ==============================================================================
print("\n" + "=" * 50)
print("6. MODEL ANALYSIS, VISUALIZATIONS & FINDINGS")
print("=" * 50)

# Plot 5: Model Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
df_melt = pd.melt(df_results, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                  var_name='Metric', value_name='Score')
sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax)
ax.set_ylim(0.40, 0.60)
ax.set_title('Performance Metrics Comparison Across Models', fontsize=13, fontweight='bold')
ax.axhline(0.50, color='red', linestyle='--', linewidth=1, label='Random Chance Baseline (0.50)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_model_comparison_metrics.png'), dpi=300)
plt.close()
print("Saved: 05_model_comparison_metrics.png")

# Plot 6: Confusion Matrix Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(y_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Active (0)', 'Dropped (1)'],
                yticklabels=['Active (0)', 'Dropped (1)'], cbar=False)
    axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}', fontweight='bold', fontsize=11)
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('Actual Ground Truth')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_confusion_matrices.png'), dpi=300)
plt.close()
print("Saved: 06_confusion_matrices.png")

# Plot 7: ROC Curves for all models
fig, ax = plt.subplots(figsize=(8, 7))
for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Chance Line (AUC = 0.5000)', linewidth=1.5)
ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07_roc_curves.png'), dpi=300)
plt.close()
print("Saved: 07_roc_curves.png")

# Plot 8: Random Forest Feature Importance Analysis
rf_pipeline = trained_models['Random Forest Classifier']
rf_model = rf_pipeline.named_steps['clf']
feature_names = rf_pipeline.named_steps['prep'].get_feature_names_out()

# Clean feature names for clean presentation
cleaned_feat_names = [f.split('__')[-1] for f in feature_names]
rf_importances = pd.Series(rf_model.feature_importances_, index=cleaned_feat_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=rf_importances.head(15).values, y=rf_importances.head(15).index, hue=rf_importances.head(15).index, palette='mako', legend=False, ax=ax)
ax.set_title('Top 15 Feature Importances (Random Forest Classifier)', fontsize=13, fontweight='bold')
ax.set_xlabel('Relative Importance (Gini Impurity Reduction)', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '08_feature_importance.png'), dpi=300)
plt.close()
print("Saved: 08_feature_importance.png")

print("\n" + "=" * 80)
print("ALL 8 VISUALIZATIONS AND ARTIFACTS GENERATED SUCCESSFULLY!")
print("=" * 80)
