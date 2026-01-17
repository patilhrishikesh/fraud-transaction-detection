import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(path):
    # Load the dataset
    df = pd.read_csv(path)
    
    # Featues and target
    X = df.drop("Class", axis = 1)
    y = df["Class"]
    
    # Scale only amount (Time is optional)
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
    
    # Train-test split (stratified because of imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test