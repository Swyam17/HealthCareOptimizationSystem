import sys
import os

# Add workspace to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data_generator import generate_healthcare_dataset
from backend.ml_engine import HealthcareMLEngine
from backend.optimizer import HealthcareScheduleOptimizer

def test_full_system():
    print("--- Starting System Verification Test ---")

    # 1. Test Dataset Generation
    data_path = "data/Medical_NoShows.csv"
    df = generate_healthcare_dataset(data_path, num_records=200)
    assert len(df) == 200, "Dataset generation failed"
    print("[OK] Data Generator Test Passed!")

    # 2. Test ML Engine
    ml = HealthcareMLEngine(data_path)
    metrics = ml.train_model("DecisionTree")
    assert "accuracy" in metrics and metrics["accuracy"] > 0.5, "ML training failed"
    print(f"[OK] ML Engine Test Passed! Accuracy: {metrics['accuracy']}, ROC-AUC: {metrics['roc_auc']}")

    # Single Prediction Test
    sample = {
        "Age": 45, "LeadTimeDays": 12, "SMSReceived": 0, "PreviousNoShows": 2,
        "PreviousCancellations": 1, "AppointmentHour": 9, "ExpectedDurationMin": 20,
        "Priority": "Routine", "DayOfWeek": "Monday", "Department": "Cardiology"
    }
    pred = ml.predict_appointment_risk(sample)
    assert "no_show_probability" in pred, "Prediction output missing probability"
    print(f"[OK] ML Prediction Test Passed! Score: {pred['no_show_percentage']}% ({pred['risk_level']})")

    # 3. Test Optimizer
    samples = [
        {"AppointmentID": f"APT-{100+i}", "PatientID": f"P-{i}", "Age": 40+i, "LeadTimeDays": i*2,
         "SMSReceived": i%2, "PreviousNoShows": i%2, "PreviousCancellations": 0, "ExpectedDurationMin": 20,
         "Priority": "Urgent" if i==0 else "Routine", "DayOfWeek": "Monday", "Department": "Cardiology"}
        for i in range(10)
    ]

    opt = HealthcareScheduleOptimizer()
    fcfs_res = opt.run_fcfs_baseline(samples)
    ml_res = opt.run_ml_optimized_scheduler(samples, ml)

    assert "performance" in fcfs_res and "performance" in ml_res, "Optimizer evaluation failed"
    print(f"[OK] FCFS Baseline Wait Time: {fcfs_res['performance']['avg_wait_time_min']} min")
    print(f"[OK] ML-Optimized Wait Time: {ml_res['performance']['avg_wait_time_min']} min")

    # 4. Test Dynamic Rescheduling
    resched = opt.dynamic_reschedule_slot(ml_res["schedule"], "APT-102", {"PatientName": "Standby Test Patient", "ExpectedDurationMin": 20})
    assert resched["reallocated"] == True, "Dynamic rescheduling failed"
    print("[OK] Dynamic Rescheduling Test Passed!")

    print("\nALL BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_system()
