import pandas as pd

df = pd.read_csv('data/augmented/augmented_data.csv')
print("--- DATA AUDIT ---")
print(f"Total Rows: {len(df)}")
print(f"Average Mg: {df['mg'].mean():.4f}")
print(f"Average Si: {df['si'].mean():.4f}")
print(f"Average Temp: {df['temperature'].mean():.2f}")
print(f"Rows with 0 Mg: {len(df[df['mg'] == 0])}")
print(f"Rows with 0 Si: {len(df[df['si'] == 0])}")
print("------------------")