import pandas as pd

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Display first 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset information
print("\nDataset Info:\n")
print(df.info())

# Class distribution
print("\nClass Distribution:\n")
print(df['label'].value_counts())