import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


df = pd.read_csv('5g_network_data.csv')
df_clean = df.copy()
df_clean = df_clean.drop(columns=['Timestamp'], errors='ignore')
df_clean = df_clean.drop_duplicates()
for col in df_clean.columns:
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif pd.api.types.is_bool_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
df = df_clean

X = df.drop(columns=['Dropped Connection'], errors='ignore')
y = df['Dropped Connection'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

standard_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
minmax_preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
standard_logreg = Pipeline([
    ('preprocess', standard_preprocessor),
    ('model', LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced'))
])
minmax_logreg = Pipeline([
    ('preprocess', minmax_preprocessor),
    ('model', LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced'))
])
rf_model = Pipeline([
    ('preprocess', standard_preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])
models = {
    'Logistic Regression (Standardized)': standard_logreg,
    'Logistic Regression (Normalized)': minmax_logreg,
    'Random Forest': rf_model
}

print('Class distribution (0 = No Drop, 1 = Dropped):')
print(y.value_counts().sort_index())

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False).reset_index(drop=True)
for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']:
    results_df[col] = results_df[col].round(4)
print('\nModel comparison table:')
print(results_df.to_string(index=False))

cm = confusion_matrix(y_test, rf_model.predict(X_test))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Drop', 'Dropped'], yticklabels=['No Drop', 'Dropped'], ax=ax)
ax.set_title('Confusion Matrix - Random Forest')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual Label')
fig.tight_layout()
fig.savefig('model_confusion_matrix.png')
print('\nplot_saved')
