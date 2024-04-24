#model.py
import joblib

# Load the trained model
def load_model(model_path="trained_model_rf.pkl"): # Update the model path to the one saved with Random Forest
    return joblib.load(model_path)

# Function to predict credit classification
def predict_credit_classification(model, input_data):
    try:
        # Perform prediction using the model
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        return str(e)
