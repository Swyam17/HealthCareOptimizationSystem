document.addEventListener("DOMContentLoaded", () => {
    // --- Global Application State ---
    let state = {
        appointments: [],
        doctors: [],
        analytics: null,
        mlMetrics: null,
        optResults: null,
        comparisonChart: null,
        riskPieChart: null,
        featImpChart: null
    };

    // --- DOM Elements ---
    const navItems = document.querySelectorAll(".nav-item");
    const tabPages = document.querySelectorAll(".tab-page");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");

    // Modal Elements
    const modalBook = document.getElementById("modal-book-apt");
    const btnOpenBook = document.getElementById("btn-open-book-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const formBookApt = document.getElementById("form-book-appointment");

    // Sliders
    const sliders = {
        w_wait: document.getElementById("slider-w-wait"),
        w_idle: document.getElementById("slider-w-idle"),
        w_unused: document.getElementById("slider-w-unused"),
        w_overtime: document.getElementById("slider-w-overtime")
    };

    // --- Tab Navigation Setup ---
    const tabTitles = {
        "tab-dashboard": { title: "Executive Analytics Dashboard", sub: "Real-time attendance predictions & multi-objective schedule optimization" },
        "tab-appointments": { title: "Appointments Console & Management", sub: "Filter, track, and update patient appointment records" },
        "tab-doctors": { title: "Doctor Schedules & Resource Utilization", sub: "Monitor doctor shift workloads and slot capacities" },
        "tab-ml-studio": { title: "Machine Learning No-Show Studio", sub: "Train classification algorithms and evaluate predictive risk factors" },
        "tab-optimizer": { title: "Schedule Optimizer & Simulator", sub: "Simulate baseline FCFS vs. ML-Optimized multi-factor scheduling" }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        navItems.forEach(n => n.classList.remove("active"));
        tabPages.forEach(p => p.classList.remove("active"));

        const activeNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
        const activePage = document.getElementById(tabId);

        if (activeNav && activePage) {
            activeNav.classList.add("active");
            activePage.classList.add("active");

            if (tabTitles[tabId]) {
                pageTitle.textContent = tabTitles[tabId].title;
                pageSubtitle.textContent = tabTitles[tabId].sub;
            }
        }
    }

    // --- Toast Notifications ---
    function showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${message}`;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    // --- API Data Fetchers ---
    async function fetchDashboardData() {
        try {
            const [analyticsRes, apptsRes, docsRes, mlRes] = await Promise.all([
                fetch("/api/analytics"),
                fetch("/api/appointments"),
                fetch("/api/doctors"),
                fetch("/api/ml/metrics")
            ]);

            state.analytics = await analyticsRes.json();
            const apptsData = await apptsRes.json();
            state.appointments = apptsData.appointments;
            const docsData = await docsRes.json();
            state.doctors = docsData.doctors;
            state.mlMetrics = await mlRes.json();

            updateDashboardUI();
            renderAppointmentsTable();
            renderDoctorsGrid();
            updateMLStudioUI();
            populateRescheduleDropdown();
        } catch (err) {
            console.error("Error loading system data:", err);
            showToast("Failed to connect to backend server.", "danger");
        }
    }

    // --- Update Executive Dashboard ---
    function updateDashboardUI() {
        if (!state.analytics) return;

        const mlPerf = state.analytics.ml_performance;
        const fcfsPerf = state.analytics.fcfs_performance;
        const mlMet = state.analytics.ml_metrics;

        document.getElementById("dash-wait-time").textContent = `${mlPerf.avg_wait_time_min} min`;
        document.getElementById("dash-doctor-util").textContent = `${mlPerf.doctor_utilization_pct}%`;
        document.getElementById("dash-ml-acc").textContent = `${(mlMet.accuracy * 100).toFixed(1)}%`;
        document.getElementById("dash-ml-roc").textContent = `ROC-AUC: ${mlMet.roc_auc}`;
        document.getElementById("dash-high-risk-count").textContent = state.analytics.high_risk_count;

        // Render Comparison Chart
        renderComparisonChart(fcfsPerf, mlPerf);
        renderRiskPieChart();
        renderDashboardRecentTable();
    }

    function renderComparisonChart(fcfs, ml) {
        const ctx = document.getElementById("chart-comparison").getContext("2d");

        if (state.comparisonChart) state.comparisonChart.destroy();

        state.comparisonChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Avg Wait Time (min)", "Doctor Utilization (%)", "Unused Slots", "Overtime (min)"],
                datasets: [
                    {
                        label: "Baseline FCFS",
                        data: [fcfs.avg_wait_time_min, fcfs.doctor_utilization_pct, fcfs.unused_slots, fcfs.overtime_min],
                        backgroundColor: "rgba(255, 23, 68, 0.6)",
                        borderColor: "rgba(255, 23, 68, 1)",
                        borderWidth: 1
                    },
                    {
                        label: "ML-Optimized Schedule",
                        data: [ml.avg_wait_time_min, ml.doctor_utilization_pct, ml.unused_slots, ml.overtime_min],
                        backgroundColor: "rgba(0, 242, 254, 0.6)",
                        borderColor: "rgba(0, 242, 254, 1)",
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#f0f4f8" } }
                },
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
                }
            }
        });
    }

    function renderRiskPieChart() {
        const ctx = document.getElementById("chart-risk-pie").getContext("2d");
        if (state.riskPieChart) state.riskPieChart.destroy();

        state.riskPieChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Low Risk (<25%)", "Moderate Risk (25-50%)", "High Risk (>50%)"],
                datasets: [{
                    data: [state.analytics.low_risk_count, state.analytics.moderate_risk_count, state.analytics.high_risk_count],
                    backgroundColor: ["#00e676", "#ff9100", "#ff1744"],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { color: "#f0f4f8" } }
                }
            }
        });
    }

    function renderDashboardRecentTable() {
        const tbody = document.querySelector("#dashboard-recent-table tbody");
        tbody.innerHTML = "";

        state.appointments.slice(0, 6).forEach(apt => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${apt.AppointmentID}</strong></td>
                <td>${apt.PatientName} <span class="muted">(${apt.PatientID})</span></td>
                <td>${apt.DoctorName} <br><small class="muted">${apt.Department}</small></td>
                <td><span class="badge ${apt.Priority === 'Urgent' ? 'badge-danger' : 'badge-info'}">${apt.Priority}</span></td>
                <td>${apt.LeadTimeDays} days</td>
                <td><span class="badge badge-${apt.badge_class}">${apt.no_show_percentage}% (${apt.risk_level})</span></td>
                <td><span class="badge ${apt.Status === 'Completed' ? 'badge-success' : 'badge-warning'}">${apt.Status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // --- Appointments Console ---
    function renderAppointmentsTable() {
        const tbody = document.getElementById("appointments-table-body");
        const searchVal = document.getElementById("apt-search").value.toLowerCase();
        const deptVal = document.getElementById("filter-dept").value;
        const statusVal = document.getElementById("filter-status").value;

        tbody.innerHTML = "";

        const filtered = state.appointments.filter(apt => {
            const matchesSearch = apt.PatientName.toLowerCase().includes(searchVal) ||
                                  apt.PatientID.toLowerCase().includes(searchVal) ||
                                  apt.DoctorName.toLowerCase().includes(searchVal);
            const matchesDept = !deptVal || apt.Department === deptVal;
            const matchesStatus = !statusVal || apt.Status === statusVal;
            return matchesSearch && matchesDept && matchesStatus;
        });

        filtered.forEach(apt => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${apt.AppointmentID}</strong></td>
                <td>${apt.PatientName}<br><small class="muted">${apt.Age} yrs, ${apt.Gender}</small></td>
                <td>${apt.DoctorName}</td>
                <td>${apt.Department}</td>
                <td><span class="badge ${apt.Priority === 'Urgent' ? 'badge-danger' : 'badge-info'}">${apt.Priority}</span></td>
                <td>${apt.LeadTimeDays} d</td>
                <td><span class="badge badge-${apt.badge_class}">${apt.no_show_percentage}%</span></td>
                <td><span class="badge ${apt.Status === 'Completed' ? 'badge-success' : (apt.Status === 'Cancelled' ? 'badge-danger' : 'badge-warning')}">${apt.Status}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline btn-status-update" data-id="${apt.AppointmentID}" data-status="Completed"><i class="fa-solid fa-check"></i></button>
                    <button class="btn btn-sm btn-outline btn-status-update" data-id="${apt.AppointmentID}" data-status="No-Show"><i class="fa-solid fa-user-slash"></i></button>
                    <button class="btn btn-sm btn-outline btn-status-update" data-id="${apt.AppointmentID}" data-status="Cancelled"><i class="fa-solid fa-xmark"></i></button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        document.querySelectorAll(".btn-status-update").forEach(btn => {
            btn.addEventListener("click", async () => {
                const id = btn.getAttribute("data-id");
                const newStatus = btn.getAttribute("data-status");
                await updateAppointmentStatus(id, newStatus);
            });
        });
    }

    async function updateAppointmentStatus(aptId, newStatus) {
        try {
            const res = await fetch(`/api/appointments/${aptId}/status`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status: newStatus })
            });

            if (res.ok) {
                showToast(`Appointment ${aptId} status updated to ${newStatus}`, "success");
                await fetchDashboardData();
            }
        } catch (err) {
            showToast("Failed to update status", "danger");
        }
    }

    document.getElementById("apt-search").addEventListener("input", renderAppointmentsTable);
    document.getElementById("filter-dept").addEventListener("change", renderAppointmentsTable);
    document.getElementById("filter-status").addEventListener("change", renderAppointmentsTable);

    // --- Render Doctor Cards ---
    function renderDoctorsGrid() {
        const container = document.getElementById("doctors-container");
        container.innerHTML = "";

        state.doctors.forEach(doc => {
            const docAppts = state.appointments.filter(a => a.DoctorID === doc.id);
            const highRisks = docAppts.filter(a => a.risk_level === "High Risk").length;

            const card = document.createElement("div");
            card.className = "doctor-card glass-card";
            card.innerHTML = `
                <div class="doctor-header">
                    <div class="doc-avatar">${doc.avatar}</div>
                    <div class="doc-info">
                        <h4>${doc.name}</h4>
                        <p>${doc.specialty} • ${doc.room}</p>
                    </div>
                </div>
                <div class="doc-stats">
                    <div><span>Shift:</span> <strong>${doc.shift}</strong></div>
                    <div><span>Scheduled:</span> <strong>${docAppts.length}/${doc.max_capacity}</strong></div>
                    <div><span>High Risk:</span> <strong class="badge badge-danger">${highRisks}</strong></div>
                </div>
            `;
            container.appendChild(card);
        });

        // Also populate modal doctor dropdown
        const bookDocSelect = document.getElementById("book-doctor");
        bookDocSelect.innerHTML = "";
        state.doctors.forEach(doc => {
            const opt = document.createElement("option");
            opt.value = doc.id;
            opt.textContent = `${doc.name} (${doc.specialty})`;
            bookDocSelect.appendChild(opt);
        });
    }

    // --- ML Studio Setup ---
    function updateMLStudioUI() {
        if (!state.mlMetrics) return;

        const met = state.mlMetrics.metrics;
        document.getElementById("ml-stat-acc").textContent = `${(met.accuracy * 100).toFixed(1)}%`;
        document.getElementById("ml-stat-prec").textContent = `${(met.precision * 100).toFixed(1)}%`;
        document.getElementById("ml-stat-rec").textContent = `${(met.recall * 100).toFixed(1)}%`;
        document.getElementById("ml-stat-f1").textContent = `${(met.f1_score * 100).toFixed(1)}%`;
        document.getElementById("ml-stat-roc").textContent = met.roc_auc;

        renderFeatureImportanceChart(state.mlMetrics.feature_importances);
    }

    function renderFeatureImportanceChart(features) {
        const ctx = document.getElementById("chart-feature-importance").getContext("2d");
        if (state.featImpChart) state.featImpChart.destroy();

        const top8 = features.slice(0, 8);
        state.featImpChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: top8.map(f => f.feature),
                datasets: [{
                    label: "Importance Score",
                    data: top8.map(f => f.importance),
                    backgroundColor: "rgba(127, 0, 255, 0.6)",
                    borderColor: "rgba(127, 0, 255, 1)",
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
                }
            }
        });
    }

    document.getElementById("btn-retrain-ml").addEventListener("click", async () => {
        const algo = document.getElementById("ml-algo-select").value;
        try {
            const res = await fetch("/api/ml/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ algorithm: algo })
            });
            const data = await res.json();
            showToast(data.message, "success");
            await fetchDashboardData();
        } catch (err) {
            showToast("Failed to re-train ML model", "danger");
        }
    });

    document.getElementById("btn-calc-risk").addEventListener("click", async () => {
        const sample = {
            Age: parseInt(document.getElementById("pred-age").value),
            LeadTimeDays: parseInt(document.getElementById("pred-lead").value),
            PreviousNoShows: parseInt(document.getElementById("pred-noshows").value),
            PreviousCancellations: parseInt(document.getElementById("pred-cancels").value),
            SMSReceived: parseInt(document.getElementById("pred-sms").value),
            Department: document.getElementById("pred-dept").value,
            Priority: document.getElementById("pred-priority").value,
            DayOfWeek: document.getElementById("pred-day").value,
            AppointmentHour: 10,
            ExpectedDurationMin: 20
        };

        const res = await fetch("/api/ml/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sample)
        });

        const data = await res.json();

        const box = document.getElementById("predictor-result-box");
        box.classList.remove("hidden");
        document.getElementById("result-score-circle").textContent = `${data.no_show_percentage}%`;
        document.getElementById("result-risk-level").textContent = `${data.risk_level}`;
        document.getElementById("result-drivers").textContent = `Key Drivers: ${data.risk_drivers.join(", ")}`;
    });

    // --- Schedule Optimizer Simulator ---
    Object.keys(sliders).forEach(key => {
        sliders[key].addEventListener("input", () => {
            const valSpan = document.getElementById(`val-${key.replace('_', '-')}`);
            if (valSpan) valSpan.textContent = sliders[key].value;
        });
    });

    document.getElementById("btn-run-simulation").addEventListener("click", runSimulation);

    async function runSimulation() {
        const payload = {
            w_wait: parseFloat(sliders.w_wait.value),
            w_idle: parseFloat(sliders.w_idle.value),
            w_unused: parseFloat(sliders.w_unused.value),
            w_overtime: parseFloat(sliders.w_overtime.value)
        };

        const res = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        state.optResults = data;

        showToast(`Simulation completed! Wait time reduced by ${data.comparison_summary.wait_time_reduction_pct}%`, "success");
        renderSimulationTimeline(data.ml_optimized.schedule);
    }

    function renderSimulationTimeline(schedule) {
        const tbody = document.querySelector("#simulation-timeline-table tbody");
        tbody.innerHTML = "";

        schedule.forEach(slot => {
            const startHour = 8 + Math.floor(slot.scheduled_start_min / 60);
            const startMin = slot.scheduled_start_min % 60;
            const timeStr = `${startHour.toString().padStart(2, '0')}:${startMin.toString().padStart(2, '0')} AM`;

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${timeStr}</strong></td>
                <td>${slot.PatientName || slot.PatientID}</td>
                <td><span class="badge ${slot.Priority === 'Urgent' ? 'badge-danger' : 'badge-info'}">${slot.Priority}</span></td>
                <td><span class="badge badge-${slot.badge_class || 'info'}">${(slot.no_show_probability * 100).toFixed(1)}%</span></td>
                <td>${slot.is_buffered_slot ? '<span class="badge badge-warning">Overbooking Buffer Slot</span>' : 'Standard Slot'}</td>
                <td><span class="badge badge-success">Scheduled</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    function populateRescheduleDropdown() {
        const select = document.getElementById("select-reschedule-apt");
        select.innerHTML = "";
        state.appointments.forEach(apt => {
            const opt = document.createElement("option");
            opt.value = apt.AppointmentID;
            opt.textContent = `${apt.AppointmentID} - ${apt.PatientName} (${apt.risk_level})`;
            select.appendChild(opt);
        });
    }

    document.getElementById("btn-trigger-reschedule").addEventListener("click", async () => {
        const aptId = document.getElementById("select-reschedule-apt").value;

        const res = await fetch("/api/reschedule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cancelled_appointment_id: aptId })
        });

        const data = await res.json();
        const logBox = document.getElementById("reschedule-log-box");
        logBox.classList.remove("hidden");
        logBox.innerHTML = `
            <div class="alert alert-success">
                <strong>Dynamic Slot Reallocation Triggered!</strong><br>
                Slot cancelled for <code>${aptId}</code> was instantly reallocated to Standby Urgent Patient.
                Saved clinic idle time: <strong>20 minutes</strong>.
            </div>
        `;
        showToast("Dynamic slot reallocated successfully!", "success");
    });

    // --- Modal Handling ---
    btnOpenBook.addEventListener("click", () => modalBook.classList.remove("hidden"));
    btnCloseModal.addEventListener("click", () => modalBook.classList.add("hidden"));

    formBookApt.addEventListener("submit", async (e) => {
        e.preventDefault();

        const docId = document.getElementById("book-doctor").value;
        const selectedDoc = state.doctors.find(d => d.id === docId);

        const newApt = {
            PatientName: document.getElementById("book-name").value,
            Age: parseInt(document.getElementById("book-age").value),
            Gender: document.getElementById("book-gender").value,
            DoctorID: docId,
            Department: selectedDoc ? selectedDoc.specialty : "General Medicine",
            Priority: document.getElementById("book-priority").value,
            LeadTimeDays: parseInt(document.getElementById("book-lead").value),
            SMSReceived: parseInt(document.getElementById("book-sms").value),
            PreviousNoShows: parseInt(document.getElementById("book-prior-noshows").value),
            ExpectedDurationMin: 20
        };

        const res = await fetch("/api/appointments", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(newApt)
        });

        if (res.ok) {
            const data = await res.json();
            showToast(`Booked appointment ${data.appointment.AppointmentID} (${data.appointment.risk_level})`, "success");
            modalBook.classList.add("hidden");
            formBookApt.reset();
            await fetchDashboardData();
        }
    });

    document.getElementById("btn-refresh").addEventListener("click", () => {
        fetchDashboardData();
        showToast("Dashboard data refreshed", "info");
    });

    // Initial Load
    fetchDashboardData();
});
