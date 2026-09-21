import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_healthcare_dataset(filepath="data/Medical_NoShows.csv", num_records=1200, seed=42):
    """
    Generates a realistic synthetic dataset for patient appointment attendance and no-show analysis.
    Features follow real-world distributions found in outpatient appointment datasets.
    """
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    departments = ["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "Dermatology", "General Medicine"]
    priorities = ["Routine", "Moderate", "Urgent"]
    genders = ["F", "M"]

    data = []
    base_date = datetime.now() - timedelta(days=90)

    for i in range(1, num_records + 1):
        patient_id = f"PAT-{1000 + (i % 350)}"  # Some repeating patients to reflect history
        gender = random.choice(genders)
        age = int(np.clip(np.random.normal(45, 18), 1, 90))
        department = random.choice(departments)
        priority = random.choice(priorities)

        # Scheduled date and Appointment date
        lead_time_days = int(np.clip(np.random.exponential(scale=7), 0, 45))
        scheduled_dt = base_date + timedelta(days=random.randint(0, 60))
        appointment_dt = scheduled_dt + timedelta(days=lead_time_days)
        
        # Day of week & hour
        day_of_week = appointment_dt.strftime("%A")
        appointment_hour = random.choice([8, 9, 10, 11, 13, 14, 15, 16])
        
        # Historical metrics for returning patients
        previous_no_shows = np.random.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.07, 0.03])
        previous_cancellations = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
        sms_received = 1 if (lead_time_days > 2 and random.random() > 0.3) else 0

        # Consultation expected duration (mins)
        expected_duration = random.choice([15, 20, 30, 45])

        # Calculate No-Show probability based on factors
        # High lead time + prior no shows + no SMS + Monday/Friday increase risk
        risk_score = 0.15
        risk_score += (lead_time_days / 45.0) * 0.35
        risk_score += (previous_no_shows * 0.20)
        risk_score += (0.15 if sms_received == 0 and lead_time_days > 3 else 0.0)
        risk_score += (0.08 if day_of_week in ["Monday", "Friday"] else 0.0)
        risk_score += (-0.10 if priority == "Urgent" else 0.0)

        risk_score = float(np.clip(risk_score, 0.05, 0.90))

        # Binary outcome: 1 = No-Show, 0 = Attended
        no_show = 1 if random.random() < risk_score else 0

        data.append({
            "AppointmentID": f"APT-{10000 + i}",
            "PatientID": patient_id,
            "Gender": gender,
            "Age": age,
            "Department": department,
            "Priority": priority,
            "ScheduledDate": scheduled_dt.strftime("%Y-%m-%d"),
            "AppointmentDate": appointment_dt.strftime("%Y-%m-%d"),
            "LeadTimeDays": lead_time_days,
            "DayOfWeek": day_of_week,
            "AppointmentHour": appointment_hour,
            "SMSReceived": sms_received,
            "PreviousNoShows": previous_no_shows,
            "PreviousCancellations": previous_cancellations,
            "ExpectedDurationMin": expected_duration,
            "NoShow": no_show
        })

    df = pd.DataFrame(data)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"[Dataset Generator] Successfully generated {num_records} records to '{filepath}'. No-Show rate: {df['NoShow'].mean()*100:.1f}%")
    except Exception as e:
        print(f"[Dataset Generator] Warning: Could not write dataset to file ({e}). Returning generated DataFrame in memory.")
    return df

if __name__ == "__main__":
    generate_healthcare_dataset()
