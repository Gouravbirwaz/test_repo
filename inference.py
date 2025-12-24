
import joblib
import pandas as pd
import json
import sys

MODEL_PATH = "model.pkl"

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict(data_json):
    model = load_model()
    df = pd.DataFrame(data_json)
    # Ensure columns match training (simplified)
    # In production, we woud need the schema or pipeline
    predictions = model.predict(df)
    return predictions.tolist()

if __name__ == "__main__":
    # Example usage: python inference.py '[{"feature1": 1.0, ...}]'
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        print(json.dumps(predict(data)))
    else:
        print("Please provide data as JSON string argument")
