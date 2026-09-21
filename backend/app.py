import os
import sys
import random
import pandas as pd

# Ensure project root is in sys.path for Vercel serverless environment
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from backend.data_generator import generate_healthcare_dataset
from backend.ml_engine import HealthcareMLEngine
from backend.optimizer import HealthcareScheduleOptimizer

app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

DATA_PATH = os.path.join(BASE_DIR, "data", "Medical_NoShows.csv")

# Global Singletons
if not os.path.exists(DATA_PATH):
    try:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate_healthcare_dataset(DATA_PATH)
    except Exception:
        DATA_PATH = "/tmp/Medical_NoShows.csv"
        generate_healthcare_dataset(DATA_PATH)

ml_engine = HealthcareMLEngine(DATA_PATH)
ml_engine.train_model("RandomForest")

# In-memory store for Doctors & Live Appointments
DOCTORS = [
    {"id": "DOC-101", "name": "Dr. Aris Thorne", "specialty": "Cardiology", "room": "Cabinet 3A", "shift": "08:00 AM - 04:00 PM", "max_capacity": 16, "avatar": "👨‍⚕️"},
    {"id": "DOC-102", "name": "Dr. Bhupaesh Ghai", "specialty": "Orthopedics", "room": "Cabinet 1B", "shift": "08:00 AM - 04:00 PM", "max_capacity": 16, "avatar": "👨‍🏫"},
    {"id": "DOC-103", "name": "Dr. Elena Vance", "specialty": "Neurology", "room": "Cabinet 4C", "shift": "09:00 AM - 05:00 PM", "max_capacity": 14, "avatar": "👩‍⚕️"},
    {"id": "DOC-104", "name": "Dr. Marcus Chen", "specialty": "Pediatrics", "room": "Cabinet 2A", "shift": "08:00 AM - 04:00 PM", "max_capacity": 18, "avatar": "👨‍⚕️"},
    {"id": "DOC-105", "name": "Dr. Sarah Jenkins", "specialty": "Dermatology", "room": "Cabinet 5A", "shift": "09:00 AM - 05:00 PM", "max_capacity": 15, "avatar": "👩‍⚕️"}
]

LIVE_APPOINTMENTS = []

def init_live_appointments():
    global LIVE_APPOINTMENTS
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).head(25)
        LIVE_APPOINTMENTS = []
        for idx, row in df.iterrows():
            doc = random.choice(DOCTORS)
            apt_dict = {
                "AppointmentID": str(row["AppointmentID"]),
                "PatientID": str(row["PatientID"]),
                "PatientName": f"Patient #{row['PatientID'][-4:]}",
                "Age": int(row["Age"]),
                "Gender": str(row["Gender"]),
                "Department": str(row["Department"]),
                "DoctorID": doc["id"],
                "DoctorName": doc["name"],
                "Priority": str(row["Priority"]),
                "ScheduledDate": str(row["ScheduledDate"]),
                "AppointmentDate": str(row["AppointmentDate"]),
                "LeadTimeDays": int(row["LeadTimeDays"]),
                "DayOfWeek": str(row["DayOfWeek"]),
                "AppointmentHour": int(row["AppointmentHour"]),
                "SMSReceived": int(row["SMSReceived"]),
                "PreviousNoShows": int(row["PreviousNoShows"]),
                "PreviousCancellations": int(row["PreviousCancellations"]),
                "ExpectedDurationMin": int(row["ExpectedDurationMin"]),
                "NoShow": int(row["NoShow"]),
                "Status": "Scheduled"
            }
            risk = ml_engine.predict_appointment_risk(apt_dict)
            apt_dict.update(risk)
            LIVE_APPOINTMENTS.append(apt_dict)

init_live_appointments()

# --- WEB UI ROUTE ---
@app.route("/")
def index():
    return render_template("index.html")

# --- API ENDPOINTS ---

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "system": "Healthcare Appointment Optimization System",
        "version": "1.0.0",
        "dataset_exists": os.path.exists(DATA_PATH),
        "ml_model_loaded": ml_engine.model is not None,
        "active_appointments": len(LIVE_APPOINTMENTS)
    })

@app.route("/api/data/summary", methods=["GET"])
def data_summary():
    if not os.path.exists(DATA_PATH):
        generate_healthcare_dataset(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    no_show_pct = float(df["NoShow"].mean() * 100)
    avg_age = float(df["Age"].mean())
    avg_lead_time = float(df["LeadTimeDays"].mean())

    dept_counts = df["Department"].value_counts().to_dict()
    priority_counts = df["Priority"].value_counts().to_dict()

    return jsonify({
        "total_records": len(df),
        "no_show_rate_pct": round(no_show_pct, 1),
        "attendance_rate_pct": round(100.0 - no_show_pct, 1),
        "avg_age": round(avg_age, 1),
        "avg_lead_time_days": round(avg_lead_time, 1),
        "departments": dept_counts,
        "priorities": priority_counts
    })

@app.route("/api/ml/metrics", methods=["GET"])
def ml_metrics():
    return jsonify({
        "metrics": ml_engine.metrics,
        "feature_importances": ml_engine.feature_importances,
        "algorithm": ml_engine.model_name
    })

@app.route("/api/ml/train", methods=["POST"])
def ml_train():
    data = request.json or {}
    algo = data.get("algorithm", "RandomForest")
    metrics = ml_engine.train_model(algo)
    
    # Recalculate risks for live appointments
    for apt in LIVE_APPOINTMENTS:
        risk = ml_engine.predict_appointment_risk(apt)
        apt.update(risk)

    return jsonify({
        "message": f"Model successfully re-trained with {algo}",
        "metrics": metrics,
        "feature_importances": ml_engine.feature_importances
    })

@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    data = request.json or {}
    risk_info = ml_engine.predict_appointment_risk(data)
    return jsonify(risk_info)

@app.route("/api/doctors", methods=["GET"])
def get_doctors():
    return jsonify({"doctors": DOCTORS})

@app.route("/api/appointments", methods=["GET", "POST"])
def manage_appointments():
    global LIVE_APPOINTMENTS
    if request.method == "GET":
        dept = request.args.get("department")
        status = request.args.get("status")
        doc_id = request.args.get("doctor_id")

        filtered = LIVE_APPOINTMENTS
        if dept:
            filtered = [a for a in filtered if a["Department"] == dept]
        if status:
            filtered = [a for a in filtered if a["Status"] == status]
        if doc_id:
            filtered = [a for a in filtered if a["DoctorID"] == doc_id]

        return jsonify({"appointments": filtered, "total": len(filtered)})

    elif request.method == "POST":
        data = request.json or {}
        new_id = f"APT-{random.randint(20000, 99999)}"
        doc = next((d for d in DOCTORS if d["id"] == data.get("DoctorID")), random.choice(DOCTORS))

        new_apt = {
            "AppointmentID": new_id,
            "PatientID": data.get("PatientID", f"PAT-{random.randint(5000, 9999)}"),
            "PatientName": data.get("PatientName", "New Patient"),
            "Age": int(data.get("Age", 35)),
            "Gender": data.get("Gender", "M"),
            "Department": data.get("Department", doc["specialty"]),
            "DoctorID": doc["id"],
            "DoctorName": doc["name"],
            "Priority": data.get("Priority", "Routine"),
            "ScheduledDate": data.get("ScheduledDate", "2026-09-20"),
            "AppointmentDate": data.get("AppointmentDate", "2026-09-25"),
            "LeadTimeDays": int(data.get("LeadTimeDays", 5)),
            "DayOfWeek": data.get("DayOfWeek", "Monday"),
            "AppointmentHour": int(data.get("AppointmentHour", 10)),
            "SMSReceived": int(data.get("SMSReceived", 1)),
            "PreviousNoShows": int(data.get("PreviousNoShows", 0)),
            "PreviousCancellations": int(data.get("PreviousCancellations", 0)),
            "ExpectedDurationMin": int(data.get("ExpectedDurationMin", 20)),
            "NoShow": 0,
            "Status": "Scheduled"
        }
        
        # Calculate ML Risk
        risk = ml_engine.predict_appointment_risk(new_apt)
        new_apt.update(risk)

        LIVE_APPOINTMENTS.insert(0, new_apt)
        return jsonify({"message": "Appointment created successfully", "appointment": new_apt}), 201

@app.route("/api/appointments/<apt_id>/status", methods=["PUT"])
def update_appointment_status(apt_id):
    global LIVE_APPOINTMENTS
    data = request.json or {}
    new_status = data.get("status")

    for apt in LIVE_APPOINTMENTS:
        if apt["AppointmentID"] == apt_id:
            apt["Status"] = new_status
            if new_status == "No-Show":
                apt["NoShow"] = 1
                apt["is_noshow_occurred"] = True
            elif new_status == "Completed":
                apt["NoShow"] = 0
                apt["is_noshow_occurred"] = False
            return jsonify({"message": f"Appointment {apt_id} status updated to {new_status}", "appointment": apt})

    return jsonify({"error": "Appointment not found"}), 404

@app.route("/api/optimize", methods=["POST"])
def run_optimization():
    data = request.json or {}
    w_wait = float(data.get("w_wait", 1.0))
    w_idle = float(data.get("w_idle", 1.5))
    w_unused = float(data.get("w_unused", 2.0))
    w_overtime = float(data.get("w_overtime", 2.5))
    w_noshow = float(data.get("w_noshow", 1.2))

    optimizer = HealthcareScheduleOptimizer(
        weight_wait=w_wait,
        weight_idle=w_idle,
        weight_unused=w_unused,
        weight_overtime=w_overtime,
        weight_noshow=w_noshow
    )

    fcfs_result = optimizer.run_fcfs_baseline(LIVE_APPOINTMENTS)
    ml_result = optimizer.run_ml_optimized_scheduler(LIVE_APPOINTMENTS, ml_engine)

    # Compute comparative performance improvement percentage
    fcfs_perf = fcfs_result["performance"]
    ml_perf = ml_result["performance"]

    wait_improvement = round(((fcfs_perf["avg_wait_time_min"] - ml_perf["avg_wait_time_min"]) / max(0.1, fcfs_perf["avg_wait_time_min"])) * 100, 1)
    utilization_gain = round(ml_perf["doctor_utilization_pct"] - fcfs_perf["doctor_utilization_pct"], 1)
    cost_reduction = round(((fcfs_perf["total_cost"] - ml_perf["total_cost"]) / max(1.0, fcfs_perf["total_cost"])) * 100, 1)

    return jsonify({
        "fcfs": fcfs_result,
        "ml_optimized": ml_result,
        "comparison_summary": {
            "wait_time_reduction_pct": max(0.0, wait_improvement),
            "doctor_utilization_gain_pct": max(0.0, utilization_gain),
            "total_cost_reduction_pct": max(0.0, cost_reduction),
            "fcfs_cost": fcfs_perf["total_cost"],
            "ml_cost": ml_perf["total_cost"]
        }
    })

@app.route("/api/reschedule", methods=["POST"])
def dynamic_reschedule():
    data = request.json or {}
    cancelled_apt_id = data.get("cancelled_appointment_id")
    
    standby_patient = data.get("standby_patient") or {
        "PatientID": f"PAT-STANDBY-{random.randint(100, 999)}",
        "PatientName": "Standby Patient (Urgent Call-In)",
        "Age": 42,
        "Gender": "F",
        "Department": "General Medicine",
        "Priority": "Urgent",
        "ExpectedDurationMin": 20,
        "NoShow": 0
    }

    optimizer = HealthcareScheduleOptimizer()
    current_schedule = optimizer.run_ml_optimized_scheduler(LIVE_APPOINTMENTS, ml_engine)["schedule"]

    res = optimizer.dynamic_reschedule_slot(current_schedule, cancelled_apt_id, standby_patient)
    return jsonify(res)

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    optimizer = HealthcareScheduleOptimizer()
    fcfs_result = optimizer.run_fcfs_baseline(LIVE_APPOINTMENTS)
    ml_result = optimizer.run_ml_optimized_scheduler(LIVE_APPOINTMENTS, ml_engine)

    return jsonify({
        "fcfs_performance": fcfs_result["performance"],
        "ml_performance": ml_result["performance"],
        "ml_metrics": ml_engine.metrics,
        "total_appointments": len(LIVE_APPOINTMENTS),
        "high_risk_count": len([a for a in LIVE_APPOINTMENTS if a.get("risk_level") == "High Risk"]),
        "moderate_risk_count": len([a for a in LIVE_APPOINTMENTS if a.get("risk_level") == "Moderate Risk"]),
        "low_risk_count": len([a for a in LIVE_APPOINTMENTS if a.get("risk_level") == "Low Risk"])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
