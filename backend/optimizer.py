import copy
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HealthcareScheduleOptimizer:
    def __init__(self, weight_wait=1.0, weight_idle=1.5, weight_unused=2.0, weight_overtime=2.5, weight_noshow=1.2):
        self.w_wait = weight_wait
        self.w_idle = weight_idle
        self.w_unused = weight_unused
        self.w_overtime = weight_overtime
        self.w_noshow = weight_noshow

    def evaluate_schedule_performance(self, schedule_slots, shift_length_min=480):
        """
        Calculates operational performance metrics & cost function for a given schedule configuration.
        """
        total_wait_time = 0.0
        total_doctor_idle = 0.0
        unused_slot_count = 0
        overtime_minutes = 0.0
        completed_count = 0
        no_show_count = 0
        cancelled_count = 0

        current_time = 0  # start of shift in minutes from 8:00 AM

        for slot in schedule_slots:
            status = slot.get("status", "Scheduled")
            duration = slot.get("duration", 20)
            risk = slot.get("no_show_probability", 0.15)
            is_noshow = slot.get("is_noshow_occurred", False)

            if status == "Cancelled":
                cancelled_count += 1
                unused_slot_count += 1
                total_doctor_idle += duration * 0.5
                continue

            if is_noshow:
                no_show_count += 1
                # Doctor experiences idle time when patient no-shows
                total_doctor_idle += duration
                unused_slot_count += 1
                continue

            completed_count += 1

            # Simulated actual service arrival & delay
            scheduled_start = slot.get("scheduled_start_min", 0)
            wait_time = max(0, current_time - scheduled_start)
            total_wait_time += wait_time

            # Update doctor current time after servicing patient
            current_time = max(current_time, scheduled_start) + duration

        # Check if shift ran over expected 8-hour capacity (480 mins)
        if current_time > shift_length_min:
            overtime_minutes = current_time - shift_length_min

        total_slots = len(schedule_slots)
        avg_wait_time = round(total_wait_time / max(1, completed_count), 2)
        doctor_utilization = round(max(0.0, min(100.0, ((shift_length_min - total_doctor_idle) / shift_length_min) * 100)), 1)
        completion_rate = round((completed_count / max(1, total_slots)) * 100, 1)

        total_cost = (
            (self.w_wait * total_wait_time) +
            (self.w_idle * total_doctor_idle) +
            (self.w_unused * unused_slot_count * 20) +
            (self.w_overtime * overtime_minutes) +
            (self.w_noshow * no_show_count * 25)
        )

        return {
            "total_cost": round(total_cost, 2),
            "avg_wait_time_min": avg_wait_time,
            "total_wait_time_min": round(total_wait_time, 1),
            "doctor_idle_min": round(total_doctor_idle, 1),
            "doctor_utilization_pct": doctor_utilization,
            "unused_slots": unused_slot_count,
            "overtime_min": round(overtime_minutes, 1),
            "completed_appointments": completed_count,
            "no_shows": no_show_count,
            "cancellations": cancelled_count,
            "completion_rate_pct": completion_rate
        }

    def run_fcfs_baseline(self, appointments):
        """
        Simulates traditional First-Come-First-Served fixed interval schedule.
        """
        slots = []
        current_min = 0

        for apt in appointments:
            slot = copy.deepcopy(apt)
            duration = slot.get("ExpectedDurationMin", 20)
            slot["scheduled_start_min"] = current_min
            slot["duration"] = duration
            slot["is_noshow_occurred"] = slot.get("NoShow", 0) == 1
            slots.append(slot)
            current_min += duration

        perf = self.evaluate_schedule_performance(slots)
        return {"schedule": slots, "performance": perf}

    def run_ml_optimized_scheduler(self, appointments, ml_engine):
        """
        Intelligent Schedule Optimizer incorporating No-Show Risk probabilities.
        - Sorts low-risk & urgent patients into prime slots.
        - Overbooks / pairs high-risk appointments in strategic buffer slots.
        - Minimizes idle time and cumulative wait delays.
        """
        scored_apts = []

        for apt in appointments:
            item = copy.deepcopy(apt)
            # Evaluate no-show risk using ML engine
            risk_info = ml_engine.predict_appointment_risk(item)
            item["no_show_probability"] = risk_info["no_show_probability"]
            item["risk_level"] = risk_info["risk_level"]
            item["is_noshow_occurred"] = item.get("NoShow", 0) == 1
            scored_apts.append(item)

        # Separate by risk category & priority
        urgent = [a for a in scored_apts if a.get("Priority") == "Urgent"]
        low_risk = [a for a in scored_apts if a.get("Priority") != "Urgent" and a["no_show_probability"] < 0.30]
        med_risk = [a for a in scored_apts if a.get("Priority") != "Urgent" and 0.30 <= a["no_show_probability"] < 0.55]
        high_risk = [a for a in scored_apts if a.get("Priority") != "Urgent" and a["no_show_probability"] >= 0.55]

        # Order strategy: Urgent first, then low risk, then staggered medium & high risk overbooking buffers
        ordered_schedule = []
        ordered_schedule.extend(sorted(urgent, key=lambda x: x["no_show_probability"]))
        ordered_schedule.extend(sorted(low_risk, key=lambda x: x["no_show_probability"]))

        # Interleave medium & high risk with buffer allowances
        for i, hr in enumerate(high_risk):
            if i < len(med_risk):
                # Pair high risk with medium risk buffer slot
                hr["is_buffered_slot"] = True
                ordered_schedule.append(hr)
                ordered_schedule.append(med_risk[i])
            else:
                ordered_schedule.append(hr)

        # Add remaining medium risk
        remaining_med = med_risk[len(high_risk):]
        ordered_schedule.extend(remaining_med)

        # Assign timelines
        current_min = 0
        final_slots = []

        for slot in ordered_schedule:
            duration = slot.get("ExpectedDurationMin", 20)
            if slot.get("is_buffered_slot", False):
                # Overbooked buffer slot: offset by only half duration since attendance probability is low
                slot["scheduled_start_min"] = current_min
                slot["duration"] = duration
                current_min += int(duration * 0.5)
            else:
                slot["scheduled_start_min"] = current_min
                slot["duration"] = duration
                current_min += duration
            
            final_slots.append(slot)

        perf = self.evaluate_schedule_performance(final_slots)
        return {"schedule": final_slots, "performance": perf}

    def dynamic_reschedule_slot(self, schedule_slots, cancelled_apt_id, standby_patient):
        """
        Reallocates a canceled or missed appointment slot dynamically to a standby patient.
        """
        updated_slots = copy.deepcopy(schedule_slots)
        reallocated = False

        for slot in updated_slots:
            if slot.get("AppointmentID") == cancelled_apt_id:
                slot["status"] = "Cancelled"
                slot["is_noshow_occurred"] = False
                
                if standby_patient:
                    # Create new reallocated slot
                    new_slot = copy.deepcopy(standby_patient)
                    new_slot["AppointmentID"] = f"REALLOC-{random.randint(1000, 9999)}"
                    new_slot["scheduled_start_min"] = slot["scheduled_start_min"]
                    new_slot["duration"] = slot["duration"]
                    new_slot["status"] = "Reallocated"
                    new_slot["is_noshow_occurred"] = False
                    new_slot["reallocated_from"] = cancelled_apt_id
                    
                    # Insert reallocated slot into schedule
                    idx = updated_slots.index(slot)
                    updated_slots.insert(idx + 1, new_slot)
                    reallocated = True
                break

        new_perf = self.evaluate_schedule_performance(updated_slots)
        return {
            "reallocated": reallocated,
            "schedule": updated_slots,
            "performance": new_perf
        }

if __name__ == "__main__":
    from backend.ml_engine import HealthcareMLEngine
    ml = HealthcareMLEngine()
    ml.train_model("RandomForest")

    # Sample list of 10 appointments
    samples = [
        {"AppointmentID": f"APT-{100+i}", "PatientID": f"P-{i}", "Age": 40+i, "LeadTimeDays": i*3,
         "SMSReceived": i%2, "PreviousNoShows": i%3, "PreviousCancellations": 0, "ExpectedDurationMin": 20,
         "Priority": "Urgent" if i==0 else "Routine", "DayOfWeek": "Monday", "Department": "Cardiology"}
        for i in range(10)
    ]

    opt = HealthcareScheduleOptimizer()
    fcfs_res = opt.run_fcfs_baseline(samples)
    ml_res = opt.run_ml_optimized_scheduler(samples, ml)

    print("FCFS Perf:", fcfs_res["performance"])
    print("ML-Opt Perf:", ml_res["performance"])
