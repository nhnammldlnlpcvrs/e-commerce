import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame):
    df = df.copy()
    
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

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
