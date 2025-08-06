import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_data import preprocess
import os

# Load raw data
raw_path = "../data/raw/train.csv"
df = pd.read_csv(raw_path)

# Preprocess first!
df = preprocess(df)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Output paths
os.makedirs("../data/processed", exist_ok=True)
train_df.to_csv("../data/processed/train_split.csv", index=False)
test_df.to_csv("../data/processed/test_split.csv", index=False)

print("Preprocessed data split completed and saved to 'data/processed/' directory.")
