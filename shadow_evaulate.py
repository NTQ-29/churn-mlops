import os
import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

# Configure logging for CI/CD output visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShadowEvaluation")

def run_shadow_evaluation():
    logger.info("Starting Shadow Model Evaluation gate...")
    
    # 1. Setup MLflow Connection
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.getenv("MODEL_NAME", "phoenix_core_model")
    
    # 2. Mocking validation data for pipeline validation
    # In a fully wired setup, you would load your test/validation sets here via DVC
    logger.info("Loading validation datasets...")
    X_val = np.random.rand(100, 4)
    y_true = np.random.choice([0, 1], size=100)
    
    # 3. Load Current Production Model (The Baseline)
    try:
        prod_model_uri = f"models://{model_name}@Production"
        logger.info(f"Loading current baseline model from: {prod_model_uri}")
        prod_model = mlflow.pyfunc.load_model(prod_model_uri)
        prod_preds = prod_model.predict(X_val)
        # Assuming classification for churn; change to MSE if doing regression
        prod_perf = accuracy_score(y_true, np.round(prod_preds))
        logger.info(f"Current Production Model Performance (Accuracy): {prod_perf:.4f}")
    except Exception as e:
        logger.warning(f"No existing production model found, or load failed: {str(e)}")
        logger.info("Setting baseline performance threshold to 0.0 (First-time deployment).")
        prod_perf = 0.0

    # 4. Load the Candidate Model (The Challenger)
    try:
        # Pulls the latest run or a staged candidate
        candidate_model_uri = f"models://{model_name}/Latest" 
        logger.info(f"Loading candidate challenger model from: {candidate_model_uri}")
        candidate_model = mlflow.pyfunc.load_model(candidate_model_uri)
        candidate_preds = candidate_model.predict(X_val)
        candidate_perf = accuracy_score(y_true, np.round(candidate_preds))
        logger.info(f"Candidate Model Performance (Accuracy): {candidate_perf:.4f}")
    except Exception as e:
        logger.critical(f"Failed to load candidate model for evaluation: {str(e)}")
        # Exit with a failure code to halt the GitHub Actions pipeline
        exit(1)

    # 5. The Gate Guard Logic
    # The candidate must perform better than or equal to the current production model
    performance_margin = 0.01  # Candidate must be within or greater than a 1% margin
    
    if candidate_perf >= (prod_perf - performance_margin):
        logger.info("🎉 SUCCESS: Candidate model passes the shadow evaluation gate.")
        logger.info("Promoting candidate model to @Production status.")
        
        # Code-driven MLflow promotion
        client = mlflow.tracking.MlflowClient()
        # Find latest version to transition
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model version {latest_version} successfully moved to Production.")
    else:
        logger.error("❌ FAILURE: Candidate model failed the shadow evaluation performance gate.")
        logger.error(f"Candidate Performance ({candidate_perf:.4f}) is lower than Production ({prod_perf:.4f}).")
        # Exit with code 1 to stop the GitHub deployment job from triggering
        exit(1)

if __name__ == "__main__":
    run_shadow_evaluation()
