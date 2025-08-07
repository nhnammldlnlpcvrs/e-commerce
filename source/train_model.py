import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess_data import load_and_preprocess

def train(file_path: str, model_output: str = "models/model.pkl"):
    X, y = load_and_preprocess(file_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_output)

    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    train("data/processed/train_split.csv")
