import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Remove ID column if present
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def load_and_preprocess(file_path: str):
    df = pd.read_csv(file_path)
    df = preprocess(df)

    X = df.drop(columns=['Reached.on.Time_Y.N'])
    y = df['Reached.on.Time_Y.N']

    return X, y

# xu li data
if __name__ == "__main__":
    raw_path = "../data/raw/train.csv"

    # Load raw data
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Preprocess
    print("Preprocessing data...")
    df = preprocess(df)

    # Split train/test
    print("Splitting into train/test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save processed data
    os.makedirs("../data/processed", exist_ok=True)
    train_df.to_csv("../data/processed/train_split.csv", index=False)
    test_df.to_csv("../data/processed/test_split.csv", index=False)

    print("Preprocessed data split completed.")
    print("Files saved to '../data/processed/' directory.")