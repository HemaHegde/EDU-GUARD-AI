import os
import sys
import time
import json
import warnings
import subprocess
import threading
import numpy as np
import pandas as pd
import requests
import concurrent.futures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "final_student_psychology_dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "ieee_results")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    "total_clicks", "avg_score", "active_days",
    "engagement_variability", "inactivity_days",
    "engagement_slope", "assessment_consistency",
]

print("=" * 60)
print("API Load Testing Benchmark (FastAPI + XGBoost)")
print("=" * 60)

# Load some sample data for payload
df = pd.read_csv(DATA_PATH)
X_test_samples = df[FEATURES].values[:1000].tolist()

def run_load_test(concurrent_users=10, total_requests=1000):
    latencies = []
    
    def send_request():
        # Pick a random sample
        sample = X_test_samples[np.random.randint(0, len(X_test_samples))]
        payload = {
            "features": [
                {
                    "total_clicks": sample[0],
                    "avg_score": sample[1],
                    "active_days": sample[2],
                    "engagement_variability": sample[3],
                    "inactivity_days": sample[4],
                    "engagement_slope": sample[5],
                    "assessment_consistency": sample[6]
                }
            ]
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post("http://127.0.0.1:8001/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        for _ in range(total_requests):
            executor.submit(send_request)
    
    total_time = time.perf_counter() - t_start
    throughput = len(latencies) / total_time if total_time > 0 else 0
    mean_lat = np.mean(latencies) if latencies else 0
    std_lat = np.std(latencies) if latencies else 0
    
    print(f"  Users: {concurrent_users:3d} | Reqs: {len(latencies):4d} | "
          f"Mean Latency: {mean_lat:6.2f} ms | Throughput: {throughput:7.1f} req/s")
    
    return {
        "concurrent_users": concurrent_users,
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "throughput_req_s": throughput,
        "total_requests": len(latencies)
    }

# Create a minimal FastAPI app script to run in a subprocess
app_script_path = os.path.join(SCRIPT_DIR, "temp_fastapi_app.py")
app_code = """
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "academic_risk_xgboost.pkl")
model = joblib.load(MODEL_PATH)

class FeatureRow(BaseModel):
    total_clicks: float
    avg_score: float
    active_days: float
    engagement_variability: float
    inactivity_days: float
    engagement_slope: float
    assessment_consistency: float

class PredictRequest(BaseModel):
    features: List[FeatureRow]

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([row.dict() for row in req.features])
    probs = model.predict_proba(df)[:, 1].tolist()
    return {"predictions": probs}
"""

with open(app_script_path, "w") as f:
    f.write(app_code)

# Start the FastAPI server
print("  Starting FastAPI server on port 8001...")
server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "temp_fastapi_app:app", "--host", "127.0.0.1", "--port", "8001", "--log-level", "critical"],
    cwd=SCRIPT_DIR
)
time.sleep(3) # Wait for server to start

results = []
user_levels = [1, 5, 10, 20, 50, 100]
TOTAL_REQUESTS = 1000

try:
    for users in user_levels:
        res = run_load_test(concurrent_users=users, total_requests=TOTAL_REQUESTS)
        results.append(res)
finally:
    # Kill the server
    server_process.terminate()
    server_process.wait()
    if os.path.exists(app_script_path):
        os.remove(app_script_path)

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "api_load_benchmark.csv"), index=False)
print("  [SAVED] api_load_benchmark.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

users = res_df["concurrent_users"]
lats = res_df["mean_latency_ms"]
thrus = res_df["throughput_req_s"]
errs = res_df["std_latency_ms"]

axes[0].errorbar(users, lats, yerr=errs, fmt="o-", color="#457B9D", lw=2, ms=8, capsize=5)
axes[0].set_xlabel("Concurrent Users", fontsize=11)
axes[0].set_ylabel("API Latency (ms)", fontsize=11)
axes[0].set_title("API Latency vs. Concurrent Users", fontweight="bold")
axes[0].grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(users, thrus, "o-", color="#E63946", lw=2, ms=8)
axes[1].set_xlabel("Concurrent Users", fontsize=11)
axes[1].set_ylabel("Throughput (req/sec)", fontsize=11)
axes[1].set_title("API Throughput vs. Concurrent Users", fontweight="bold")
axes[1].grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle("Figure M1: FastAPI Deployment Benchmarks (XGBoost Inference)\\n"
             "Local Network HTTP Requests",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_m1_api_latency.png"), dpi=300)
plt.close()
print("  [SAVED] fig_m1_api_latency.png")
print("\\n  API Load Testing complete.\\n")
