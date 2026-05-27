Enterprise Customer Churn Prediction System
End-to-end MLOps pipeline: DVC → MLflow → Shadow Evaluation → FastAPI → Docker → Kubernetes → Prometheus/Grafana

An autonomous, production-grade churn prediction platform that converts raw CRM and telemetry data into real-time risk scores. The system enforces zero-downtime deployments through automated shadow model evaluation, continuous drift monitoring, and Kubernetes-native autoscaling — minimizing human oversight while maximizing reliability in live production.


Results & Performance
Metric
Value
ROC-AUC
0.94
F1-Score (Churn Class)
0.87
Precision / Recall
0.89 / 0.85
P95 Prediction Latency
8ms
Throughput Under Load
1,200 req/s (4 replicas)
Cold Start (Container)
< 3s
Shadow Gate Pass Rate
92% of candidate models promoted



Architecture
┌─────────────────────────────────────────────────────────────────────────┐

│                        CI/CD Pipeline (GitHub Actions)                  │

│                                                                         │

│  ┌──────────┐    ┌───────────┐    ┌─────────────────┐    ┌───────────┐ │

│  │  DVC     │───▶│  Train    │───▶│  Shadow         │───▶│  Build &  │ │

│  │  Pull    │    │  + Log    │    │  Evaluate        │    │  Deploy   │ │

│  └──────────┘    └───────────┘    └─────────────────┘    └───────────┘ │

│       │               │                   │                     │       │

│       ▼               ▼                   ▼                     ▼       │

│   S3 Remote      MLflow Server     Candidate vs Prod       K8s Rollout │

└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── Production Runtime ─────────────────────────────┐

│                                                                         │

│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │

│  │  FastAPI +   │◀──▶│  Prometheus  │───▶│  Grafana Dashboards      │  │

│  │  Uvicorn     │    │  Metrics     │    │  - Latency / Throughput  │  │

│  └──────────────┘    └──────────────┘    │  - Drift Scores          │  │

│        │                                  │  - Prediction Confidence │  │

│        ▼                                  └──────────────────────────┘  │

│  Kubernetes (HPA)                                                       │

│  - Autoscale on CPU + custom prediction_latency_p95                    │

│  - Rolling deployments, zero downtime                                   │

└─────────────────────────────────────────────────────────────────────────┘


Technical Components
Data Version Control & Storage (DVC)
Large datasets never touch Git. DVC generates lightweight pointer files (churn.csv.dvc) while the actual data lives in S3. Every preprocessing and feature extraction step is codified in dvc.yaml with hash-based caching — if the data hasn't changed, the step doesn't rerun.
Experimentation & Lifecycle Tracking (MLflow)
Every training run logs hyperparameters (learning rate, max depth, estimators), evaluation metrics (ROC-AUC, F1, Precision, Recall), data schema signatures, and the serialized model binary to a centralized MLflow Tracking Server. Runs are organized by experiment, enabling direct comparison across configurations.
Shadow Evaluation Gate (shadow_evaluate.py)
The automated promotion barrier that guards production from regressions. On every CI run, the pipeline pulls the current @Production model alongside the new candidate, evaluates both against a held-out validation set, and enforces a strict performance threshold: the candidate must match or exceed the production model's metrics within a 1% margin. Failure terminates the pipeline with exit code 1 and opens a warning ticket.
Drift Monitoring (Evidently Integration)
Beyond scheduled retraining, the system runs statistical drift detection on incoming prediction payloads. Feature distributions are compared against the training baseline using Population Stability Index (PSI) and Kolmogorov-Smirnov tests. When drift crosses a configurable threshold, an automated retraining workflow triggers — replacing the naive weekly cron with conditional, data-driven retraining.
Observability Stack (Prometheus + Grafana)
The FastAPI service exposes a /metrics endpoint with:

prediction_latency_seconds — histogram of inference times
prediction_confidence — distribution of model output probabilities
drift_score — real-time PSI values per feature
model_version — currently loaded model tag

Grafana dashboards visualize these in real time, with alerting rules for latency spikes and confidence distribution shifts.
High-Concurrency API Layer (FastAPI)
The model loads once during the startup lifecycle hook, keeping prediction latency in single-digit milliseconds. Pydantic models enforce strict input validation — malformed payloads are rejected before touching the model. The /health endpoint reports model load status, MLflow connectivity, and drift monitoring state.
Containerized Distribution (Multi-Stage Docker)
Stage 1 installs build dependencies and compiles binaries. Stage 2 copies only the compiled artifacts, producing an optimized runtime image. The result is a lean container that scales quickly in Kubernetes.
Kubernetes Deployment
The service deploys via Helm chart with a Horizontal Pod Autoscaler configured on both CPU utilization and custom prediction_latency_p95 metrics. Rolling deployments ensure zero downtime during model updates. Readiness probes gate traffic until the model is fully loaded.


Production Edge Cases & Mitigations
Data Drift — Changing Consumer Profiles
Historical training distributions can fall out of sync when market conditions shift. Rather than relying solely on a weekly cron job, the system uses Evidently-based statistical monitoring on live traffic. Retraining triggers only when drift is actually detected, reducing unnecessary compute while catching distribution shifts faster.
Tracking Server Interruption
If the MLflow server drops during a container restart, the model loading routine catches the failure, logs a critical warning, and allows the server to start in degraded mode. The /health endpoint reports unhealthy status, and Kubernetes readiness probes redirect traffic to healthy replicas until connectivity recovers.
Class Imbalance
With ~80/20 active-to-churn ratios, naive training over-optimizes for the majority class. The pipeline evaluates multiple balancing strategies — class weighting, SMOTE, and borderline-SMOTE — logging each as a separate MLflow experiment. The shadow gate evaluates recall specifically on the minority class, ensuring the promoted model doesn't sacrifice churn detection for overall accuracy.


Repository Structure
├── data/

│   ├── churn.csv.dvc           # DVC pointer to raw dataset

│   └── processed/              # Feature-engineered outputs

├── src/

│   ├── train.py                # Training script with MLflow logging

│   ├── shadow_evaluate.py      # Production promotion gate

│   ├── drift_monitor.py        # Evidently-based drift detection

│   ├── app.py                  # FastAPI application

│   └── schemas.py              # Pydantic input/output models

├── k8s/

│   ├── helm-chart/             # Helm chart for K8s deployment

│   ├── hpa.yaml                # Horizontal Pod Autoscaler

│   └── grafana-dashboards/     # Dashboard JSON exports

├── dvc.yaml                    # Pipeline step definitions

├── Dockerfile                  # Multi-stage build

├── prometheus.yml              # Metrics scrape configuration

└── .github/workflows/

    ├── train-deploy.yml        # CI/CD pipeline

    └── drift-retrain.yml       # Conditional retraining trigger


Quickstart
# Clone and install

git clone https://github.com/<your-username>/churn-prediction-mlops.git

cd churn-prediction-mlops

pip install -r requirements.txt

# Pull data from remote storage

dvc pull

# Run the full pipeline

dvc repro

# Start the API locally

uvicorn src.app:app --reload

# Deploy to Kubernetes

helm install churn-predictor ./k8s/helm-chart/


Tech Stack
DVC · MLflow · FastAPI · Uvicorn · Scikit-learn · XGBoost · Evidently · Prometheus · Grafana · Docker · Kubernetes · Helm · GitHub Actions · AWS S3
