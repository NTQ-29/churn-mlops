## End-to-end MLOps pipeline for churn prediction using DVC, MLflow, FastAPI, CI/CD.

## Enterprise Customer Churn Prediction System: MLOps Architecture & Infrastructure Handbook

## 1. Executive Summary & Business Architecture
This documentation outlines the end-to-end engineering implementation for an autonomous Customer Churn Prediction platform. The platform converts raw, multi-source CRM and telemetry data into a continuous stream of real-time risk evaluations. 

By building a completely automated, hands-off pipeline—spanning deterministic data tracking, code-driven model gates, isolated container packages, and production web frameworks—this project minimizes human operational oversight while maximizing model reliability in a live production environment.

---

## 2. Technical Component Deep-Dive

### A. Data Version Control & Storage Architecture (DVC)
* **Mechanic:** Large datasets should never be committed directly to Git, as it causes repo bloat and breaks version control history. This system uses Data Version Control (DVC) to abstract dataset storage.
* **Implementation:** The raw `churn.csv` file is tracked using DVC, which generates a lightweight pointer asset file (`churn.csv.dvc`). The actual data asset is stored securely in an external remote storage volume (such as an AWS S3 bucket).
* **Lineage Enforcement:** Every step of data preprocessing and feature extraction is written out into a `dvc.yaml` step map. If the data file modifications match existing hashes, steps are cached, preventing redundant computing overhead.

### B. Experimentation & Lifecycle Tracking (MLflow)
* **Mechanic:** Data scientists often lose track of hyperparameter configurations during model exploration. MLflow completely standardizes this process.
* **Implementation:** The training script explicitly targets a remote MLflow Tracking Server.
* **Captured Artifacts:** Every execution logs explicit parameters (learning rate, max depth, estimators), metrics (ROC-AUC, F1-Score, Precision, Recall), data schema signatures, and the compiled model binary package directly to the central Central Registry.

### C. The Shadow Evaluation Gatekeeper (`shadow_evaluate.py`)
* **Mechanic:** The automated promotion barrier guards your production service from broken updates or performance degradation.
* **Implementation:** Rather than manually inspecting charts, the GitHub Actions runner spins up this custom evaluation wrapper.
* **Logic Constraints:** The code downloads the active `@Production` model alongside the new candidate model. It passes a validation dataset through both. The candidate model must exceed or match the active production model's performance boundaries within a strict 1% margin. If it fails, the execution terminates with an exit code 1, cutting off the deployment workflow and opening a warning ticket.

### D. High-Concurrency Web Delivery Layer (FastAPI)
* **Mechanic:** Machine learning models must be wrapped in optimized, accessible endpoints to be usable by external software services.
* **Implementation:** The application layer uses FastAPI backed by an asynchronous Uvicorn engine.
* **State Performance:** The model loading logic is strictly mapped to the `@app.on_event("startup")` lifecycle hook. The system loads the model binary into memory exactly once when the container boots up, keeping prediction cycles down to millisecond latencies.
* **Input Protection:** Pydantic classes enforce payload structural arrays. Any data type anomalies or missing features are rejected with a clear error response before wasting processing cycles.

### E. Lean Containerized Distribution Layer (Dockerfile)
* **Mechanic:** OS-level dependency mismatches are a leading cause of production downtime.
* **Implementation:** A highly optimized multi-stage `Dockerfile` handles the application build.
* **Compilation Separation:** Stage 1 installs development utilities and compiles raw binaries. Stage 2 copies over *only* the finalized compiled packages, stripping away deployment bloat. This produces an ultra-lightweight runtime container that scales quickly in cloud environments.

---

## 3. Advanced Edge Cases & Production Mitigation Strategies

During live deployments, real-world systems encounter production-breaking anomalies. This platform is engineered to handle three distinct operational edge cases:

### Edge Case 1: Data Drift & Changing Consumer Profiles
* **Scenario:** The macroeconomic landscape shifts (e.g., an aggressive competitor enters the market), causing historical customer training distributions to fall out of sync with live payload traffic.
* **Mitigation:** The system includes a continuous automated workflow trigger (`cron: "0 14 * * 1"`). Every Monday morning, the system triggers a automated data check, retraining the core model on fresh historical records and validating it via the shadow gate to ensure real-time stability.

### Edge Case 2: Tracking Server Interruption & Disconnected Restarts
* **Scenario:** The remote MLflow Tracking Server undergoes maintenance or drops offline exactly when the FastAPI container auto-scales or reboots.
* **Mitigation:** The loading routine within `app.py` is wrapped in a robust `try/except` safety block. If communication fails, the system logs a critical warning but allows the webserver to stand up safely. The `/health` endpoint shifts to an unhealthy status, signaling your cloud infrastructure load balancers to redirect traffic away from the container until the connection recovers.

### Edge Case 3: Extreme Imbalance in the Churn Class
* **Scenario:** In realistic corporate data, 80% of customers remain active, while only 20% churn. Standard models trained on this data will over-optimize for the majority class, completely missing the at-risk users.
* **Mitigation:** The training processing engine incorporates advanced sampling methods (such as class weighting metrics or SMOTE transformations) to balance feature importance metrics, ensuring the model focuses heavily on maximizing true positive recall scores for at-risk accounts.
