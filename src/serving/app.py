import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import mlflow.pyfunc

# Setup logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChurnInferenceService")

app = FastAPI(
    title="Customer Churn Prediction Service",
    description="Production API serving our dynamically updated MLflow model.",
    version="1.0.0"
)

model = None

class InferencePayload(BaseModel):
    """
    Strict Pydantic validation schema. 
    Ensures input data matches the exact feature shape your model expects.
    """
    features: list[float] = Field(..., example=[0.5, 2.1, -1.1, 0.9])

@app.on_event("startup")
def load_production_model():
    """Pulls the active production model from the MLflow Registry on startup."""
    global model
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        model_name = os.getenv("MODEL_NAME", "phoenix_core_model")
        model_uri = f"models://{model_name}@Production"
        
        logger.info(f"Connecting to MLflow. Loading: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Production model cached and ready to serve.")
    except Exception as e:
        logger.critical(f"Critical error loading model asset: {str(e)}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """AWS App Runner/ECS uses this endpoint to verify the container is alive."""
    if model is None:
        return {"status": "unhealthy", "error": "Model asset not initialized."}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: InferencePayload):
    """Accepts features, runs inference, and returns structural JSON."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model engine is offline."
        )
    try:
        data_input = np.array(payload.features).reshape(1, -1)
        predictions = model.predict(data_input)
        result = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        return {"predictions": result, "status": "success"}
    except Exception as e:
        logger.error(f"Inference crash: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
