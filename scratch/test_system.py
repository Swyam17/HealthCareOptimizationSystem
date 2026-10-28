import sys
import os
import pandas as pd

# Add workspace to python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.ml_engine import HealthcareMLEngine
from backend.optimizer import HealthcareScheduleOptimizer
from backend.app import app

def test_full_system():
    print("==================================================")
    print("   HEALTHCARE SYSTEM END-TO-END VERIFICATION TEST")
    print("==================================================")

    data_path = os.path.join(BASE_DIR, "data", "Medical_NoShows.csv")
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    df = pd.read_csv(data_path)
    print(f"[1/6] Ingested Dataset: {len(df)} rows, columns: {list(df.columns)}")
    assert len(df) == 1500, f"Expected 1500 rows, found {len(df)}"

    # 2. Test ML Engine with real dataset
    ml = HealthcareMLEngine(data_path)
    metrics_rf = ml.train_model("RandomForest")
    print(f"[2/6] RandomForest Accuracy: {metrics_rf['accuracy']*100:.2f}%, ROC-AUC: {metrics_rf['roc_auc']:.4f}")
    assert metrics_rf["accuracy"] >= 0.75, "RandomForest accuracy below threshold"

    # Test GradientBoosting
    metrics_gb = ml.train_model("GradientBoosting")
    print(f"      GradientBoosting Accuracy: {metrics_gb['accuracy']*100:.2f}%, ROC-AUC: {metrics_gb['roc_auc']:.4f}")
    assert metrics_gb["accuracy"] >= 0.75, "GradientBoosting accuracy below threshold"

    # Retrain back to RandomForest
    ml.train_model("RandomForest")

    # 3. Test Model Loading (Instant Cold Start)
    ml_loader = HealthcareMLEngine(data_path)
    loaded = ml_loader.load_model()
    assert loaded == True, "Failed to load pre-trained model artifact"
    print(f"[3/6] Pre-trained Model Loaded: {ml_loader.model_name} (Acc: {ml_loader.metrics.get('accuracy')})")

    # 4. Single Risk Prediction
    sample = {
        "Age": 62, "Gender": "F", "LeadTimeDays": 21, "ScheduledHour": 14,
        "SMS_received": 0, "Hipertension": 1, "Diabetes": 0, "Alcoholism": 0, "Handcap": 0,
        "DayOfWeek": "Monday", "Priority": "Urgent", "Department": "Cardiology"
    }
    pred = ml.predict_appointment_risk(sample)
    print(f"[4/6] Single Patient Risk: {pred['no_show_percentage']}% ({pred['risk_level']})")
    print(f"      Drivers: {pred['risk_drivers']}")
    assert "no_show_probability" in pred and "risk_level" in pred

    # 5. Optimizer & Dynamic Rescheduling
    optimizer = HealthcareScheduleOptimizer()
    sample_apts = [
        {"AppointmentID": f"APT-TEST-{i}", "PatientID": f"P-{i}", "Age": 30 + i * 4,
         "LeadTimeDays": i * 3, "SMS_received": i % 2, "Hipertension": i % 2,
         "ExpectedDurationMin": 20, "Priority": "Urgent" if i == 0 else "Routine"}
        for i in range(8)
    ]
    fcfs = optimizer.run_fcfs_baseline(sample_apts)
    ml_opt = optimizer.run_ml_optimized_scheduler(sample_apts, ml)
    resched = optimizer.dynamic_reschedule_slot(
        ml_opt["schedule"], "APT-TEST-2",
        {"PatientName": "Standby Urgent Patient", "ExpectedDurationMin": 20}
    )
    print(f"[5/6] Optimizer Tested: FCFS Cost: {fcfs['performance']['total_cost']}, ML Cost: {ml_opt['performance']['total_cost']}")
    assert resched["reallocated"] == True, "Dynamic rescheduling failed"
    print("      Dynamic slot reallocation succeeded!")

    # 6. Test Flask API Routes & Vercel Entry Point
    with app.test_client() as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"
        print(f"[6/6] API /api/health -> Status: {r_health.json.get('status')}")

        r_summary = client.get("/api/data/summary")
        assert r_summary.status_code == 200, f"Summary failed: {r_summary.status_code}"
        print(f"      API /api/data/summary -> Records: {r_summary.json.get('total_records')}, NoShow%: {r_summary.json.get('no_show_rate_pct')}%")

        r_appts = client.get("/api/appointments")
        assert r_appts.status_code == 200
        print(f"      API /api/appointments -> Loaded {len(r_appts.json.get('appointments', []))} live appointments")

        r_docs = client.get("/api/doctors")
        assert r_docs.status_code == 200

        r_metrics = client.get("/api/ml/metrics")
        assert r_metrics.status_code == 200

        r_analytics = client.get("/api/analytics")
        assert r_analytics.status_code == 200

        r_index = client.get("/")
        assert r_index.status_code == 200
        print("      API / (UI HTML Template) -> 200 OK")

    # Import Vercel entry point
    import api.index as vercel_entry
    assert hasattr(vercel_entry, "app"), "Vercel entry point missing 'app'"
    print("      Vercel Entry Point api/index.py verified: 'app' exported successfully!")

    print("\n==================================================")
    print("   ALL TESTS PASSED! PROJECT IS 100% READY FOR VERCEL")
    print("==================================================")

if __name__ == "__main__":
    test_full_system()
