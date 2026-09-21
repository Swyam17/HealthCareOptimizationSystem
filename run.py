import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.data_generator import generate_healthcare_dataset
from backend.ml_engine import HealthcareMLEngine

def main():
    from backend.app import app
    print("=" * 70)
    print("  HEALTHCARE APPOINTMENT OPTIMIZATION SYSTEM")
    print("  Project Synopsis Implementation | Chandigarh University")
    print("  Authors: Swyam Arora, Yash Sharma, Ayush Sharma, Kanish Thakur, Saksham Beniwal")
    print("  Supervisor: Dr. Bhupaesh Ghai")
    print("=" * 70)

    data_path = "data/Medical_NoShows.csv"
    if not os.path.exists(data_path):
        print("\n[Initialization] Generating synthetic healthcare dataset...")
        generate_healthcare_dataset(data_path)

    print("\n[ML Engine] Verifying machine learning model...")
    ml_engine = HealthcareMLEngine(data_path)
    ml_engine.train_model("RandomForest")

    print("\n[Web Server] Starting Flask server on http://localhost:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()
