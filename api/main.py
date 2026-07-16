import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from api.logger import get_logger
from api.schema import BikePredictionInput, PredictionOutput
from api.model_loader import load_prediction_pipeline

logger = get_logger("API-Main")

model_pipeline = None
model_version = "unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, model_version
    try:
        model_pipeline, model_version = load_prediction_pipeline()
        logger.info("FastAPI lifecycle startup complete. Pipeline ready.")
    except Exception as e:
        logger.critical(f"Lifecycle startup failed: {str(e)}")
    yield
    
    logger.info("FastAPI lifecycle shutting down.")

app = FastAPI(
    title="Seoul Bike Rental Prediction API",
    description="Structured modular production API for bike demand forecasting.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: BikePredictionInput):
    request_timestamp = datetime.now().isoformat()
    logger.info("Processing inbound inference request.")

    try:
        datetime.strptime(payload.Date, "%d/%m/%Y")
    except ValueError:
        logger.warning(f"Validation Failure: Invalid date string passed -> {payload.Date}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format structure. Please use 'DD/MM/YYYY'."
        )

    try:

        input_df = pd.DataFrame([payload.model_dump(by_alias=True)])
        
        raw_prediction = model_pipeline.predict(input_df)[0]
        
        final_count = int(max(0, np.round(raw_prediction)))
        
        logger.info(f"Inference successfully evaluated: {final_count} bikes.")
        return PredictionOutput(
            predicted_bike_count=final_count,
            model_version=model_version,
            timestamp=request_timestamp
        )

    except Exception as e:
        logger.error(f"Execution Error within prediction cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference backend engine execution error."
        )