import pandas as pd
from scipy import stats

# Load data
path = '5g_network_data.csv'
df = pd.read_csv(path)

print('SHAPE:', df.shape)
print('\nDTYPES:\n', df.dtypes)
print('\nNULLS:\n', df.isnull().sum())
print('\nDUPLICATES:', df.duplicated().sum())

print('\nSHAPIRO TEST (sampled):')
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    s = df[col].dropna()
    sample = s.sample(min(5000, len(s)), random_state=42)
    stat, p = stats.shapiro(sample)
    print(f'{col}: p={p:.6f} -> {"normal" if p > 0.05 else "not normal"}')

print('\nOUTLIER COUNTS (IQR):')
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = df[(df[col] < lower) | (df[col] > upper)]
    print(f'{col}: {len(out)} outliers')
