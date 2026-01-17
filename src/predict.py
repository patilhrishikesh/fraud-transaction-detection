import time
import joblib
import pandas as pd

# Load trained model from file (Random Forest model saved earlier)
model = joblib.load("rf_model.pkl")

def predict_transaction(transaction: dict):
    # Start a timer to measure how long prediction takes
    start_time = time.time()
    
    # Convert the input dictionary into a pandas DataFrame (model expects tabular data)
    df = pd.DataFrame([transaction])
    
    # Use the model to predict fraud (0 = legit, 1 = fraud). Take the first result.
    prediction = model.predict(df)[0]
    
    # Get the probability that this transaction is fraud (second column of predict_proba)
    probability = model.predict_proba(df)[0][1]
    
    # Calculate how long the prediction took (in milliseconds)
    latency = (time.time() - start_time) * 1000  # ms
    
    # Return results in a nice dictionary format
    return {
        "is_fraud": bool(prediction),              # True if fraud, False if legit
        "fraud_probability": round(probability, 4), # Probability of fraud (rounded to 4 decimals)
        "latency_ms": round(latency, 2)             # Time taken to predict (rounded to 2 decimals)
    }
    
    
if __name__ == "__main__":
    sample_transaction = {
        "Time": 100000,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }
    result = predict_transaction(sample_transaction)
    print(result)