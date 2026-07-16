import os
import joblib
from api.logger import get_logger

logger = get_logger("ModelLoader")
API_VERSION = "1.0.0"

MODEL_PATH = os.path.join("models", "bike_rental_model.pkl")

def load_prediction_pipeline():
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Model artifact not found at path: {MODEL_PATH}")
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model artifact from {MODEL_PATH}")
        return pipeline, API_VERSION
    except Exception as e:
        logger.critical(f"Failed to deserialize model pickle: {str(e)}")
        raise e