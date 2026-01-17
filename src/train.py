from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report
import joblib

from preprocess import load_and_prepare_data

DATA_PATH = "data/transactions.csv"

def train_models():
    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)
    
    # Random forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    # fit the model
    rf.fit(X_train, y_train)
    
    # predict the X test1
    y_pred_rf = rf.predict(X_test)
    
    print("Random forest Results : ")
    print(classification_report(y_test, y_pred_rf))
    
    joblib.dump(rf, "rf_model.pkl")
    
    
    # isolation forest (unsupervised)
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.0017, # apporx fraud ratio
        random_state=42
    )
    
    iso.fit(X_train)
    y_pred_iso = iso.predict(X_test)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]
    
    print("Isolation forest Results:")
    print(classification_report(y_test, y_pred_iso))
    
    joblib.dump(iso, "iso_model.pkl")
    

if __name__ == "__main__":
    train_models()