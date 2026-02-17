# Load feature dataframe
df = pd.read_csv('all_features.csv')

# Identify feature columns (excluding patient_id and response)
feature_cols = [col for col in df.columns if col.startswith('T0_') or col.startswith('T1_')]

# Create new columns for delta features
for col in feature_cols:
    if col.startswith('T0_'):
        base_name = col[3:]  # remove 'T0_'
        t1_col = 'T1_' + base_name
        if t1_col in df.columns:
            delta_name = 'delta_' + base_name
            # Relative change: (T1 - T0) / T0 (avoid division by zero)
            df[delta_name] = (df[t1_col] - df[col]) / (df[col] + 1e-6)

# Keep only delta features, patient_id, and response
delta_cols = [col for col in df.columns if col.startswith('delta_')]
df_delta = df[['patient_id', 'response'] + delta_cols].copy()
df_delta = df_delta.dropna()  # drop patients with missing features
