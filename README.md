# Healthcare Appointment Optimization System

A data-driven **Healthcare Appointment Optimization System** integrating **Machine Learning No-Show Risk Prediction**, **Multi-Factor Schedule Optimization**, **Dynamic Slot Rescheduling**, and an **Executive Analytics Web Dashboard**.

Developed based on the Project Work Synopsis by **Swyam Arora, Yash Sharma, Ayush Sharma, Kanish Thakur, Saksham Beniwal** under the supervision of **Dr. Bhupaesh Ghai** (Department of Computer Science & Engineering - AI & ML, Chandigarh University).

---

## 🌟 Key Features

1. **Machine Learning No-Show Prediction Studio**:
   - Classifiers: Random Forest, Gradient Boosting, Logistic Regression.
   - Evaluates risk metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC score.
   - Live feature importance breakdown & single-appointment risk calculator.

2. **Multi-Factor Schedule Optimization Engine**:
   - Multi-objective cost formulation:
     $$\text{Cost} = w_1 \cdot \text{Wait Time} + w_2 \cdot \text{Doctor Idle Time} + w_3 \cdot \text{Unused Slot Cost} + w_4 \cdot \text{Overtime Cost} + w_5 \cdot \text{No-Show Impact}$$
   - Baseline First-Come-First-Served (FCFS) vs Intelligent ML-Optimized Scheduler.
   - Strategic overbooking buffers for high no-show probability appointments.

3. **Dynamic Slot Rescheduling**:
   - Real-time slot reallocation upon appointment cancellation or marked no-show.

4. **Executive Command & Control Dashboard**:
   - Modern dark glassmorphism interface with Chart.js visualizations.
   - Appointments console with real-time ML risk scoring.
   - Doctor schedule visualizer & capacity monitoring.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python run.py
```

### 3. Open Web Dashboard
Navigate to `http://localhost:5000` in your web browser.

---

## 🛠 Project Structure

```
HealthcareOptimizationSystem/
├── backend/
│   ├── app.py              # Flask API Web Application
│   ├── data_generator.py   # Synthetic Dataset Engine
│   ├── ml_engine.py        # ML No-Show Risk Classifier & Trainer
│   └── optimizer.py        # Schedule Optimizer & Dynamic Rescheduler
├── templates/
│   └── index.html          # Dashboard Web Application Template
├── static/
│   ├── css/style.css       # Glassmorphism Design System
│   └── js/app.js           # Frontend Controller & Chart.js Integration
├── data/
│   └── Medical_NoShows.csv # Healthcare Appointments Dataset
├── models/
│   └── no_show_model.pkl   # Serialized ML Model
├── run.py                  # Main Entry Point
└── requirements.txt        # Python Dependencies
```
