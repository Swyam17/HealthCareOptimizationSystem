import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "no_show_model.pkl")

FEATURE_COLS = [
    "Age", "LeadTimeDays", "SMSReceived", "PreviousNoShows", "PreviousCancellations",
    "AppointmentHour", "ExpectedDurationMin", "IsUrgent", "IsMondayOrFriday", "Dept_Cardiology",
    "Dept_Orthopedics", "Dept_Neurology", "Dept_Pediatrics", "Dept_Dermatology"
]

class HealthcareMLEngine:
    def __init__(self, data_path="data/Medical_NoShows.csv"):
        self.data_path = data_path
        self.model = None
        self.model_name = "DecisionTree"
        self.metrics = {}
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
        except Exception:
            pass

    def _preprocess_dataframe(self, df):
        """Transforms raw dataframe into numerical feature matrix."""
        df_proc = df.copy()

        df_proc["IsUrgent"] = (df_proc["Priority"] == "Urgent").astype(int)
        df_proc["IsMondayOrFriday"] = df_proc["DayOfWeek"].isin(["Monday", "Friday"]).astype(int)

        # Department One-Hot Encodings
        for dept in ["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "Dermatology"]:
            df_proc[f"Dept_{dept}"] = (df_proc["Department"] == dept).astype(int)

        for col in FEATURE_COLS:
            if col not in df_proc.columns:
                df_proc[col] = 0

        X = df_proc[FEATURE_COLS]
        y = df_proc["NoShow"] if "NoShow" in df_proc.columns else None
        return X, y

    def train_model(self, algorithm="DecisionTree"):
        """Trains specified ML classifier and saves evaluation metrics."""
        if not os.path.exists(self.data_path):
            from backend.data_generator import generate_healthcare_dataset
            generate_healthcare_dataset(self.data_path)

        df = pd.read_csv(self.data_path)
        X, y = self._preprocess_dataframe(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        if algorithm == "LogisticRegression":
            clf = LogisticRegression(max_iter=500, random_state=42)
        elif algorithm == "NaiveBayes":
            clf = GaussianNB()
        else:
            algorithm = "DecisionTree"
            clf = DecisionTreeClassifier(max_depth=8, random_state=42)

        clf.fit(X_train, y_train)
        self.model = clf
        self.model_name = algorithm

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute feature importances if supported
        feature_imp = []
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            for feat, imp in zip(FEATURE_COLS, importances):
                feature_imp.append({"feature": feat, "importance": float(imp)})
            feature_imp = sorted(feature_imp, key=lambda x: x["importance"], reverse=True)
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            for feat, imp in zip(FEATURE_COLS, coefs):
                feature_imp.append({"feature": feat, "importance": float(imp)})
            feature_imp = sorted(feature_imp, key=lambda x: x["importance"], reverse=True)

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

        try:
            joblib.dump({"model": clf, "algorithm": algorithm, "metrics": self.metrics, "feature_importances": feature_imp}, MODEL_PATH)
        except Exception as e:
            print(f"[ML Engine] Model training completed in memory (file save skipped: {e})")
        print(f"[ML Engine] Trained {algorithm} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")
        return self.metrics

    def load_model(self):
        """Loads pre-trained model if available."""
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.model_name = data["algorithm"]
            self.metrics = data["metrics"]
            self.feature_importances = data["feature_importances"]
            return True
        return False

    def predict_appointment_risk(self, appointment_dict):
        """
        Predicts no-show probability for a single appointment dict.
        Returns probability (0-1), risk level, and risk drivers.
        """
        if self.model is None:
            if not self.load_model():
                self.train_model("RandomForest")

        # Format dict into single row DF
        df_single = pd.DataFrame([appointment_dict])
        X_single, _ = self._preprocess_dataframe(df_single)

        proba = float(self.model.predict_proba(X_single)[0, 1])

        if proba >= 0.50:
            risk_level = "High Risk"
            badge_class = "danger"
        elif proba >= 0.25:
            risk_level = "Moderate Risk"
            badge_class = "warning"
        else:
            risk_level = "Low Risk"
            badge_class = "success"

        # Key drivers explanation
        drivers = []
        lead_time = appointment_dict.get("LeadTimeDays", 0)
        prev_noshows = appointment_dict.get("PreviousNoShows", 0)
        sms = appointment_dict.get("SMSReceived", 1)

        if lead_time > 14:
            drivers.append(f"Long lead time ({lead_time} days)")
        if prev_noshows > 0:
            drivers.append(f"{prev_noshows} prior no-show history")
        if sms == 0 and lead_time > 2:
            drivers.append("No SMS confirmation sent")

        return {
            "no_show_probability": round(proba, 4),
            "no_show_percentage": round(proba * 100, 1),
            "risk_level": risk_level,
            "badge_class": badge_class,
            "risk_drivers": drivers if drivers else ["Normal scheduling factors"]
        }

if __name__ == "__main__":
    engine = HealthcareMLEngine()
    engine.train_model("RandomForest")
    sample = {
        "Age": 34, "LeadTimeDays": 18, "SMSReceived": 0, "PreviousNoShows": 2,
        "PreviousCancellations": 1, "AppointmentHour": 9, "ExpectedDurationMin": 30,
        "Priority": "Routine", "DayOfWeek": "Monday", "Department": "Orthopedics"
    }
    pred = engine.predict_appointment_risk(sample)
    print("Sample Risk Prediction:", pred)
