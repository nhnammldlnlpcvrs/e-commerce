import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess_data import preprocess

def evaluate(test_file: str, model_path: str = "models/model.pkl"):
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    df = pd.read_csv(test_file)
    df = preprocess(df)

    X_test = df.drop(columns=['Reached.on.Time_Y.N'])
    y_test = df['Reached.on.Time_Y.N']

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print(f"Number of test samples: {len(y_test)}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate("data/processed/test_split.csv")
