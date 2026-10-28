import os
import sys
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

def test_all_endpoints():
    with app.test_client() as c:
        # 1. UI Root & Statics
        r = c.get('/')
        assert r.status_code == 200, f'Root failed: {r.status_code}'
        assert b'Healthcare' in r.data
        
        r_css = c.get('/static/css/style.css')
        assert r_css.status_code == 200, f'CSS failed: {r_css.status_code}'
        
        r_js = c.get('/static/js/app.js')
        assert r_js.status_code == 200, f'JS failed: {r_js.status_code}'

        # 2. Health
        r = c.get('/api/health')
        assert r.status_code == 200
        print('[PASSED] /api/health:', r.json['status'])

        # 3. Data Summary
        r = c.get('/api/data/summary')
        assert r.status_code == 200
        print('[PASSED] /api/data/summary: records =', r.json['total_records'], 'no_show_pct =', r.json['no_show_rate_pct'])

        # 4. Doctors
        r = c.get('/api/doctors')
        assert r.status_code == 200
        print('[PASSED] /api/doctors: count =', len(r.json['doctors']))

        # 5. Appointments GET & POST
        r = c.get('/api/appointments')
        assert r.status_code == 200
        initial_count = len(r.json['appointments'])
        print('[PASSED] /api/appointments GET: initial count =', initial_count)

        r_post = c.post('/api/appointments', json={
            'PatientName': 'Jane Doe', 'Age': 52, 'Gender': 'F',
            'Department': 'Cardiology', 'Priority': 'Urgent', 'LeadTimeDays': 7
        })
        assert r_post.status_code == 201
        new_apt = r_post.json['appointment']
        apt_id = new_apt['AppointmentID']
        print(f'[PASSED] /api/appointments POST: created = {apt_id}, Risk = {new_apt["risk_level"]}')

        # 6. Status Update
        r_put = c.put(f'/api/appointments/{apt_id}/status', json={'status': 'Completed'})
        assert r_put.status_code == 200
        print('[PASSED] /api/appointments PUT status:', r_put.json['message'])

        # 7. ML Metrics & Predict & Train
        r_met = c.get('/api/ml/metrics')
        assert r_met.status_code == 200
        print('[PASSED] /api/ml/metrics: accuracy =', r_met.json['metrics']['accuracy'])

        r_pred = c.post('/api/ml/predict', json={'Age': 60, 'LeadTimeDays': 20, 'Hipertension': 1})
        assert r_pred.status_code == 200
        print(f'[PASSED] /api/ml/predict: proba = {r_pred.json["no_show_percentage"]}%')

        r_train = c.post('/api/ml/train', json={'algorithm': 'RandomForest'})
        assert r_train.status_code == 200
        print('[PASSED] /api/ml/train:', r_train.json['message'])

        # 8. Optimize & Reschedule
        r_opt = c.post('/api/optimize', json={})
        assert r_opt.status_code == 200
        print(f'[PASSED] /api/optimize: wait_time_reduction = {r_opt.json["comparison_summary"]["wait_time_reduction_pct"]}%')

        r_resched = c.post('/api/reschedule', json={'cancelled_appointment_id': apt_id})
        assert r_resched.status_code == 200
        print('[PASSED] /api/reschedule: reallocated =', r_resched.json['reallocated'])

        # 9. Analytics
        r_ana = c.get('/api/analytics')
        assert r_ana.status_code == 200
        print('[PASSED] /api/analytics: total appointments =', r_ana.json['total_appointments'])

    print('\n>>> ALL 9 ROUTE GROUPS TESTED & PASSED WITH 100% SUCCESS! <<<')

if __name__ == '__main__':
    test_all_endpoints()
