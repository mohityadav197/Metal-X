import pandas as pd
import numpy as np
import os

def clean_column_names(df):
    new_cols = {}
    for col in df.columns:
        c = str(col).strip()
        cl = c.lower()
        if cl == 'mg' or 'mg ' in cl or 'magnesium' in cl: target = 'mg'
        elif cl == 'si' or 'si ' in cl or 'silicon' in cl: target = 'si'
        elif cl == 'fe' or 'iron' in cl: target = 'fe'
        elif cl == 'cu' or 'copper' in cl: target = 'cu'
        elif cl == 'mn' or 'manganese' in cl: target = 'mn'
        elif cl == 'cr' or 'chromium' in cl: target = 'cr'
        elif cl == 'zn' or 'zinc' in cl: target = 'zn'
        elif cl == 'ti' or 'titanium' in cl: target = 'ti'
        elif 'temp' in cl: target = 'temperature'
        elif 'time' in cl: target = 'time'
        elif 'yield' in cl or 'ys' in cl or 'tensile' in cl: target = 'yield_strength'
        else: target = col
        new_cols[col] = target
    df = df.rename(columns=new_cols)
    return df.loc[:, ~df.columns.duplicated()]

def apply_metallurgical_features(df):
    df['time'] = df['time'].replace(0, 0.01).fillna(1.0)
    df['temperature'] = df['temperature'].replace(0, 180).fillna(180) 
    df['log_time'] = np.log10(df['time'])
    df['mg_si_ratio'] = df['mg'] / (df['si'] + 1e-5)
    temp_k = df['temperature'] + 273.15
    df['thermal_budget'] = temp_k * np.log(df['time'] + 1)
    return df

def augment_data(df, target=5000):
    print(f"--- Augmenting quality-filtered data to {target} rows ---")
    needed = target - len(df)
    new_rows = df.sample(n=needed, replace=True).copy()
    new_rows['mg'] *= np.random.uniform(0.98, 1.02, size=needed)
    new_rows['si'] *= np.random.uniform(0.98, 1.02, size=needed)
    new_rows['temperature'] += np.random.uniform(-2, 2, size=needed)
    new_rows['yield_strength'] *= np.random.uniform(0.98, 1.02, size=needed)
    new_rows = apply_metallurgical_features(new_rows)
    return pd.concat([df, new_rows], ignore_index=True)

def main():
    QUAL_PATH, QUAN_PATH = 'data/raw/quality_data.csv', 'data/raw/quantity_data.csv'
    OUT_PATH = 'data/augmented/augmented_data.csv'

    print("Step 1: Loading & Filtering...")
    df_qual = clean_column_names(pd.read_csv(QUAL_PATH, encoding='cp1252'))
    df_quan = clean_column_names(pd.read_csv(QUAN_PATH, encoding='cp1252'))

    # CRITICAL: Remove rows where Mg or Si are 0 or NaN (The 'Garbage Filter')
    df_quan = df_quan[(df_quan['mg'] > 0.01) & (df_quan['si'] > 0.01)]
    print(f"Quantity rows remaining after chemistry filter: {len(df_quan)}")

    # Ensure columns match
    cols = ['time', 'temperature', 'mg', 'si', 'cu', 'fe', 'cr', 'mn', 'zn', 'ti', 'yield_strength']
    for c in cols:
        if c not in df_qual: df_qual[c] = 0.0
        if c not in df_quan: df_quan[c] = 0.0

    combined = pd.concat([df_qual[cols], df_quan[cols]], ignore_index=True).dropna(subset=['yield_strength'])
    combined = apply_metallurgical_features(combined)
    
    # Final check: Does temperature look like Celsius?
    # If Mendeley used Kelvin (e.g. 450), convert to Celsius
    combined.loc[combined['temperature'] > 300, 'temperature'] -= 273.15

    final_df = augment_data(combined, 5000)
    os.makedirs('data/augmented', exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"--- SUCCESS: {len(final_df)} quality rows saved ---")

if __name__ == "__main__":
    main()