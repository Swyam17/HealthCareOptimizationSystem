import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "no_show_model.pkl")
TMP_MODEL_PATH = "/tmp/no_show_model.pkl"

FEATURE_COLS = [
    "Age",
    "LeadTimeDays",
    "IsSameDay",
    "ScheduledHour",
    "ScheduledDayOfWeek",
    "AppointmentDayOfWeek",
    "IsMondayOrFriday",
    "Gender_F",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "NumChronicConditions",
    "HasChronicCondition",
    "SMS_received",
    "SMS_x_LeadTime",
    "Age_x_LeadTime"
]

class HealthcareMLEngine:
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = os.path.join(BASE_DIR, "data", "Medical_NoShows.csv")
        else:
            self.data_path = data_path
        self.model = None
        self.model_name = "RandomForest"
        self.metrics = {}
        self.feature_importances = []
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
        except Exception:
            pass

    def _preprocess_dataframe(self, df):
        """
        Transforms raw dataset (either Kaggle Brazilian appointment dataset or UI dict)
        into a consistent numerical feature matrix matching FEATURE_COLS.
        """
        df_proc = df.copy()

        # 1. Target column (if present)
        y = None
        if "No-show" in df_proc.columns:
            y = (df_proc["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int)
        elif "NoShow" in df_proc.columns:
            y = pd.to_numeric(df_proc["NoShow"], errors="coerce").fillna(0).astype(int)

        # 2. Lead Time & Date Parsing
        if "ScheduledDay" in df_proc.columns and "AppointmentDay" in df_proc.columns:
            sched = pd.to_datetime(df_proc["ScheduledDay"], errors="coerce")
            appt = pd.to_datetime(df_proc["AppointmentDay"], errors="coerce")
            lead_time = (appt.dt.date - sched.dt.date).apply(lambda d: max(0, d.days) if pd.notnull(d) else 0)
            sched_hour = sched.dt.hour.fillna(9)
            sched_dow = sched.dt.dayofweek.fillna(0)
            appt_dow = appt.dt.dayofweek.fillna(0)
        elif "ScheduledDate" in df_proc.columns and "AppointmentDate" in df_proc.columns:
            sched = pd.to_datetime(df_proc["ScheduledDate"], errors="coerce")
            appt = pd.to_datetime(df_proc["AppointmentDate"], errors="coerce")
            lead_time = (appt.dt.date - sched.dt.date).apply(lambda d: max(0, d.days) if pd.notnull(d) else 0)
            sched_hour = pd.to_numeric(df_proc["AppointmentHour"], errors="coerce").fillna(9) if "AppointmentHour" in df_proc.columns else pd.Series(9, index=df_proc.index)
            sched_dow = sched.dt.dayofweek.fillna(0)
            appt_dow = appt.dt.dayofweek.fillna(0)
        else:
            lead_time = pd.to_numeric(df_proc["LeadTimeDays"], errors="coerce").fillna(0) if "LeadTimeDays" in df_proc.columns else pd.Series(0, index=df_proc.index)
            sched_hour = pd.to_numeric(df_proc["AppointmentHour"], errors="coerce").fillna(9) if "AppointmentHour" in df_proc.columns else pd.Series(9, index=df_proc.index)
            sched_dow = pd.Series(0, index=df_proc.index)
            appt_dow = pd.Series(0, index=df_proc.index)

        # Explicit LeadTimeDays takes priority if provided
        if "LeadTimeDays" in df_proc.columns and ("ScheduledDay" in df_proc.columns or "ScheduledDate" in df_proc.columns):
            explicit_lead = pd.to_numeric(df_proc["LeadTimeDays"], errors="coerce")
            lead_time = explicit_lead.combine_first(lead_time)

        # 3. Construct Feature Matrix
        X = pd.DataFrame(index=df_proc.index)
        age_col = pd.to_numeric(df_proc["Age"], errors="coerce").fillna(35) if "Age" in df_proc.columns else pd.Series(35, index=df_proc.index)
        X["Age"] = age_col.clip(0, 115)
        X["LeadTimeDays"] = lead_time.clip(0, 365)
        X["IsSameDay"] = (X["LeadTimeDays"] == 0).astype(int)
        X["ScheduledHour"] = sched_hour.clip(0, 23)
        X["ScheduledDayOfWeek"] = sched_dow
        X["AppointmentDayOfWeek"] = appt_dow

        if "DayOfWeek" in df_proc.columns:
            is_mf = df_proc["DayOfWeek"].astype(str).str.lower().isin(["monday", "friday"]).astype(int)
        else:
            is_mf = appt_dow.isin([0, 4]).astype(int)
        X["IsMondayOrFriday"] = is_mf

        # Gender
        if "Gender" in df_proc.columns:
            X["Gender_F"] = (df_proc["Gender"].astype(str).str.upper() == "F").astype(int)
        else:
            X["Gender_F"] = 0

        # Medical Conditions
        for col in ["Hipertension", "Diabetes", "Alcoholism"]:
            if col in df_proc.columns:
                X[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(0).astype(int)
            else:
                X[col] = 0

        if "Handcap" in df_proc.columns:
            handcap_val = pd.to_numeric(df_proc["Handcap"], errors="coerce").fillna(0).astype(int)
            X["Handcap"] = (handcap_val > 0).astype(int)
        else:
            X["Handcap"] = 0

        X["NumChronicConditions"] = X["Hipertension"] + X["Diabetes"] + X["Alcoholism"] + X["Handcap"]
        X["HasChronicCondition"] = (X["NumChronicConditions"] > 0).astype(int)

        # SMS Notification
        if "SMS_received" in df_proc.columns:
            sms = pd.to_numeric(df_proc["SMS_received"], errors="coerce").fillna(0).astype(int)
        elif "SMSReceived" in df_proc.columns:
            sms = pd.to_numeric(df_proc["SMSReceived"], errors="coerce").fillna(0).astype(int)
        else:
            sms = pd.Series(0, index=df_proc.index)
        X["SMS_received"] = sms

        # Interaction Features
        X["SMS_x_LeadTime"] = X["SMS_received"] * X["LeadTimeDays"]
        X["Age_x_LeadTime"] = (X["Age"] * X["LeadTimeDays"]) / 100.0

        # Ensure correct ordering and missing column safety
        for col in FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0.0

        return X[FEATURE_COLS], y

    def train_model(self, algorithm="RandomForest"):
        """
        Trains the specified ML classifier on the real dataset and computes evaluation metrics.
        """
        if not os.path.exists(self.data_path):
            from backend.data_generator import generate_healthcare_dataset
            generate_healthcare_dataset(self.data_path)

        df = pd.read_csv(self.data_path)
        X, y = self._preprocess_dataframe(df)

        if y is None or len(y) == 0:
            raise ValueError("Target variable 'No-show' / 'NoShow' not found in dataset.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        algo_norm = algorithm.lower().strip()
        if "forest" in algo_norm:
            algorithm = "RandomForest"
            clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42, n_jobs=1)
        elif "gradient" in algo_norm or "boost" in algo_norm:
            algorithm = "GradientBoosting"
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42)
        elif "logistic" in algo_norm:
            algorithm = "LogisticRegression"
            clf = LogisticRegression(max_iter=1000, random_state=42)
        elif "bayes" in algo_norm:
            algorithm = "NaiveBayes"
            clf = GaussianNB()
        else:
            algorithm = "DecisionTree"
            clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)

        clf.fit(X_train, y_train)
        self.model = clf
        self.model_name = algorithm

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        else:
            roc = 0.50

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute feature importances
        feature_imp = []
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            for feat, imp in zip(FEATURE_COLS, importances):
                feature_imp.append({"feature": feat, "importance": round(float(imp), 4)})
            feature_imp = sorted(feature_imp, key=lambda x: x["importance"], reverse=True)
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            total_c = np.sum(coefs) or 1.0
            for feat, imp in zip(FEATURE_COLS, coefs / total_c):
                feature_imp.append({"feature": feat, "importance": round(float(imp), 4)})
            feature_imp = sorted(feature_imp, key=lambda x: x["importance"], reverse=True)
        else:
            for feat in FEATURE_COLS:
                feature_imp.append({"feature": feat, "importance": round(1.0 / len(FEATURE_COLS), 4)})

        self.feature_importances = feature_imp
        self.metrics = {
            "algorithm": algorithm,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "confusion_matrix": cm,
            "dataset_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "no_show_rate": round(float(y.mean()), 4)
        }

        # Save model with fallback for serverless read-only filesystems
        saved = False
        for path in [MODEL_PATH, TMP_MODEL_PATH]:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                joblib.dump({
                    "model": clf,
                    "algorithm": algorithm,
                    "metrics": self.metrics,
                    "feature_importances": feature_imp
                }, path)
                saved = True
                break
            except Exception as e:
                continue

        if not saved:
            print("[ML Engine] Warning: Could not write model to disk, kept in memory.")

        print(f"[ML Engine] Trained {algorithm} - Accuracy: {acc*100:.2f}%, ROC-AUC: {roc:.4f}")
        return self.metrics

    def load_model(self):
        """
        Loads pre-trained model artifact if available from models/ or /tmp.
        """
        for path in [MODEL_PATH, TMP_MODEL_PATH]:
            if os.path.exists(path):
                try:
                    data = joblib.load(path)
                    self.model = data["model"]
                    self.model_name = data.get("algorithm", "RandomForest")
                    self.metrics = data.get("metrics", {})
                    self.feature_importances = data.get("feature_importances", [])
                    return True
                except Exception as e:
                    print(f"[ML Engine] Failed to load model from {path}: {e}")
        return False

    def predict_appointment_risk(self, appointment_dict):
        """
        Predicts no-show probability and generates clinical risk drivers for a single appointment dict.
        """
        if self.model is None:
            if not self.load_model():
                self.train_model("RandomForest")

        # Format dict into single row DF
        df_single = pd.DataFrame([appointment_dict])
        X_single, _ = self._preprocess_dataframe(df_single)

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_single)[0, 1])
        else:
            proba = 0.18

        if proba >= 0.40:
            risk_level = "High Risk"
            badge_class = "danger"
        elif proba >= 0.20:
            risk_level = "Moderate Risk"
            badge_class = "warning"
        else:
            risk_level = "Low Risk"
            badge_class = "success"

        # Clinical & Operational Risk Drivers
        drivers = []
        lead_time = float(X_single["LeadTimeDays"].iloc[0])
        is_same_day = int(X_single["IsSameDay"].iloc[0])
        sms = int(X_single["SMS_received"].iloc[0])
        num_cond = int(X_single["NumChronicConditions"].iloc[0])
        age = float(X_single["Age"].iloc[0])

        if lead_time > 14:
            drivers.append(f"Extended lead time ({int(lead_time)} days)")
        elif is_same_day == 1:
            drivers.append("Same-day appointment (High historical attendance)")

        if sms == 0 and lead_time > 2:
            drivers.append("No SMS confirmation sent")
        elif sms == 1:
            drivers.append("SMS confirmation received")

        if num_cond >= 2:
            drivers.append(f"{num_cond} chronic conditions (Elevated complexity)")
        elif num_cond == 1:
            drivers.append("Chronic condition present (Hypertension/Diabetes)")

        if age >= 65:
            drivers.append(f"Senior patient ({int(age)} yrs) mobility factor")
        elif age <= 12:
            drivers.append(f"Pediatric patient ({int(age)} yrs)")

        if not drivers:
            drivers = ["Standard scheduling parameters"]

        return {
            "no_show_probability": round(proba, 4),
            "no_show_percentage": round(proba * 100, 1),
            "risk_level": risk_level,
            "badge_class": badge_class,
            "risk_drivers": drivers
        }

if __name__ == "__main__":
    engine = HealthcareMLEngine()
    metrics = engine.train_model("RandomForest")
    print("Trained metrics:", metrics)
    sample = {
        "Age": 62, "LeadTimeDays": 15, "SMSReceived": 0, "Hipertension": 1,
        "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "Gender": "F"
    }
    pred = engine.predict_appointment_risk(sample)
    print("Sample Risk Prediction:", pred)
