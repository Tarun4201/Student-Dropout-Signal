/**
 * The Dropout Signal — Frontend Application
 * Fetches data from Flask APIs and renders the premium dashboard.
 */

// ========================================================================
// GLOBALS
// ========================================================================
let currentPage = 1;
let currentTier = 'all';
let currentSort = 'risk_score';
let currentOrder = 'desc';
let searchTimeout = null;

// Chart.js global defaults
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.legend.labels.usePointStyle = true;

// ========================================================================
// INIT
// ========================================================================
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    initNavbar();
    initNavLinks();
    loadStats();
    loadRiskDistribution();
    loadFeatures();
    loadStudents();
    loadFairness();
    loadPipeline();
    initTableControls();
    initModal();
    // Phase 2
    loadRedZone();
    initSimulator();
    loadMitigation();
});

// ========================================================================
// NAVBAR
// ========================================================================
function initNavbar() {
    const nav = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
    });
}

function initNavLinks() {
    const links = document.querySelectorAll('.nav-link[data-section]');
    const sections = {};
    links.forEach(link => {
        const id = link.dataset.section;
        sections[id] = document.getElementById(id);
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                links.forEach(l => l.classList.remove('active'));
                const activeLink = document.querySelector(`.nav-link[data-section="${entry.target.id}"]`);
                if (activeLink) activeLink.classList.add('active');
            }
        });
    }, { threshold: 0.3, rootMargin: '-80px 0px 0px 0px' });

    Object.values(sections).forEach(s => { if (s) observer.observe(s); });
}

// ========================================================================
// ANIMATED COUNTER
// ========================================================================
function animateCounter(el, target, duration = 1200) {
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = Math.round(start + (target - start) * eased);
        el.textContent = current.toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ========================================================================
// STATS / KPI
// ========================================================================
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        // Hero stats
        animateCounter(document.getElementById('hero-total'), data.total_students);
        animateCounter(document.getElementById('hero-atrisk'), data.at_risk_predicted);

        // KPI cards
        const kpiTotal = document.querySelector('[data-counter="total"]');
        const kpiDropouts = document.querySelector('[data-counter="dropouts"]');
        const kpiAtrisk = document.querySelector('[data-counter="atrisk"]');
        const kpiHigh = document.querySelector('[data-counter="high"]');

        animateCounter(kpiTotal, data.total_students);
        animateCounter(kpiDropouts, data.actual_dropouts);
        animateCounter(kpiAtrisk, data.at_risk_predicted);
        animateCounter(kpiHigh, data.intervention_tiers.high);

        document.getElementById('kpi-rate').textContent = `${data.dropout_rate}% rate`;

        // Pill counts
        const total = data.total_students;
        document.getElementById('pill-count-all').textContent = total;
        document.getElementById('pill-count-high').textContent = data.intervention_tiers.high;
        document.getElementById('pill-count-medium').textContent = data.intervention_tiers.medium;
        document.getElementById('pill-count-low').textContent = data.intervention_tiers.low;

        // Target distribution chart
        renderTargetChart(data.target_distribution);
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

// ========================================================================
// CHARTS
// ========================================================================
async function loadRiskDistribution() {
    try {
        const res = await fetch('/api/risk-distribution');
        const data = await res.json();
        renderRiskDistChart(data.histogram);
        renderTierChart(data.tier_distribution);
    } catch (err) {
        console.error('Failed to load risk distribution:', err);
    }
}

function renderRiskDistChart(histogram) {
    const ctx = document.getElementById('riskDistChart').getContext('2d');

    const colors = histogram.counts.map((_, i) => {
        const ratio = i / histogram.counts.length;
        if (ratio < 0.4) return 'rgba(16, 185, 129, 0.7)';
        if (ratio < 0.7) return 'rgba(245, 158, 11, 0.7)';
        return 'rgba(244, 63, 94, 0.7)';
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histogram.labels.map((l, i) => {
                // Show every 4th label
                return i % 4 === 0 ? l.split('-')[0] : '';
            }),
            datasets: [{
                label: 'Students',
                data: histogram.counts,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 23, 0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { weight: '600' },
                    callbacks: {
                        title: (items) => `Risk: ${histogram.labels[items[0].dataIndex]}`,
                        label: (item) => `${item.raw} students`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });
}

function renderTierChart(tiers) {
    const ctx = document.getElementById('tierChart').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [tiers.high || 0, tiers.medium || 0, tiers.low || 0],
                backgroundColor: [
                    'rgba(244, 63, 94, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                ],
                borderColor: [
                    'rgba(244, 63, 94, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(16, 185, 129, 1)',
                ],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 11, weight: '500' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 23, 0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                }
            }
        }
    });
}

function renderTargetChart(dist) {
    const ctx = document.getElementById('targetChart').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Dropout', 'Graduate', 'Enrolled'],
            datasets: [{
                data: [dist.Dropout, dist.Graduate, dist.Enrolled],
                backgroundColor: [
                    'rgba(244, 63, 94, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                ],
                borderColor: [
                    'rgba(244, 63, 94, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(59, 130, 246, 1)',
                ],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 11, weight: '500' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 23, 0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                }
            }
        }
    });
}

// ========================================================================
// FEATURES CHART
// ========================================================================
async function loadFeatures() {
    try {
        const res = await fetch('/api/features');
        const data = await res.json();
        renderFeatureChart(data.features);
    } catch (err) {
        console.error('Failed to load features:', err);
    }
}

function renderFeatureChart(features) {
    const ctx = document.getElementById('featureChart').getContext('2d');
    const top10 = features.slice(0, 10);

    const gradient = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
    gradient.addColorStop(1, 'rgba(167, 139, 250, 0.8)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top10.map(f => f.display_name),
            datasets: [{
                label: 'Importance',
                data: top10.map(f => f.importance),
                backgroundColor: gradient,
                borderColor: 'rgba(139, 92, 246, 0.6)',
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 23, 0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (item) => `Correlation: ${item.raw.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { font: { size: 10 } },
                    title: {
                        display: true,
                        text: 'Absolute Correlation with Dropout',
                        font: { size: 11, weight: '500' },
                        color: '#5c5f72'
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11, weight: '500' } }
                }
            }
        }
    });
}

// ========================================================================
// STUDENTS TABLE
// ========================================================================
function initTableControls() {
    // Filter pills
    document.querySelectorAll('.pill[data-tier]').forEach(pill => {
        pill.addEventListener('click', () => {
            document.querySelectorAll('.pill[data-tier]').forEach(p => p.classList.remove('pill-active'));
            pill.classList.add('pill-active');
            currentTier = pill.dataset.tier;
            currentPage = 1;
            loadStudents();
        });
    });

    // Search
    const searchInput = document.getElementById('student-search');
    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            currentPage = 1;
            loadStudents();
        }, 400);
    });

    // Sort columns
    document.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const sort = th.dataset.sort;
            if (currentSort === sort) {
                currentOrder = currentOrder === 'desc' ? 'asc' : 'desc';
            } else {
                currentSort = sort;
                currentOrder = 'desc';
            }
            loadStudents();
        });
    });

    // Pagination
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) { currentPage--; loadStudents(); }
    });
    document.getElementById('next-page').addEventListener('click', () => {
        currentPage++;
        loadStudents();
    });
}

async function loadStudents() {
    const search = document.getElementById('student-search').value.trim();
    const params = new URLSearchParams({
        page: currentPage,
        per_page: 25,
        sort: currentSort,
        order: currentOrder,
    });
    if (currentTier !== 'all') params.set('tier', currentTier);
    if (search) params.set('search', search);

    try {
        const res = await fetch(`/api/students?${params}`);
        const data = await res.json();
        renderStudentsTable(data.students);
        updatePagination(data);
    } catch (err) {
        console.error('Failed to load students:', err);
    }
}

function renderStudentsTable(students) {
    const tbody = document.getElementById('students-tbody');

    if (!students || students.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="table-loading">No students found.</td></tr>';
        return;
    }

    tbody.innerHTML = students.map(s => {
        const tierClass = `risk-badge-${s.intervention_tier.toLowerCase()}`;
        const tierLabel = s.intervention_tier;

        let statusClass = 'status-enrolled';
        if (s.target === 'Dropout') statusClass = 'status-dropout';
        else if (s.target === 'Graduate') statusClass = 'status-graduate';

        const gd = s.grade_delta;
        const gdClass = gd > 0 ? 'grade-positive' : gd < 0 ? 'grade-negative' : 'grade-neutral';
        const gdSign = gd > 0 ? '+' : '';

        return `
            <tr>
                <td><span style="font-family:var(--font-mono);font-weight:600;">#${s.student_id}</span></td>
                <td>
                    <span class="risk-badge ${tierClass}">${s.risk_score.toFixed(3)}</span>
                </td>
                <td><span class="risk-badge ${tierClass}">${tierLabel}</span></td>
                <td><span class="status-badge ${statusClass}">${s.target}</span></td>
                <td><span class="${gdClass}" style="font-family:var(--font-mono);">${gdSign}${gd != null ? gd.toFixed(1) : '—'}</span></td>
                <td><span style="font-family:var(--font-mono);">${s.financial_stress_index != null ? s.financial_stress_index.toFixed(0) : '—'}/5</span></td>
                <td>${s.gender_label || '—'}</td>
                <td><span class="reason-cell" title="${escapeHtml(s.reason_text || '')}">${s.reason_text || '—'}</span></td>
                <td><button class="btn-detail" onclick="openStudentModal(${s.student_id})">View</button></td>
            </tr>
        `;
    }).join('');

    lucide.createIcons();
}

function updatePagination(data) {
    document.getElementById('page-info').textContent =
        `Page ${data.page} of ${data.total_pages} (${data.total.toLocaleString()} records)`;
    document.getElementById('prev-page').disabled = data.page <= 1;
    document.getElementById('next-page').disabled = data.page >= data.total_pages;
}

function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

// ========================================================================
// STUDENT MODAL
// ========================================================================
function initModal() {
    const overlay = document.getElementById('student-modal');
    document.getElementById('modal-close').addEventListener('click', () => {
        overlay.classList.remove('active');
    });
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.classList.remove('active');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') overlay.classList.remove('active');
    });
}

async function openStudentModal(studentId) {
    const overlay = document.getElementById('student-modal');
    const body = document.getElementById('modal-body');

    body.innerHTML = '<div class="loading-shimmer" style="height:300px;"></div>';
    overlay.classList.add('active');

    try {
        const res = await fetch(`/api/students/${studentId}`);
        const s = await res.json();

        const tierClass = s.intervention_tier || 'low';
        const riskColor = tierClass === 'high' ? 'var(--rose-light)' :
                          tierClass === 'medium' ? 'var(--amber-light)' : 'var(--emerald-light)';

        body.innerHTML = `
            <div class="modal-header-section">
                <div class="modal-risk-ring modal-risk-ring-${tierClass}">
                    <span class="modal-risk-score" style="color:${riskColor}">
                        ${(s.risk_score || 0).toFixed(2)}
                    </span>
                </div>
                <div class="modal-student-info">
                    <h2>Student #${s.student_id}</h2>
                    <span class="meta">
                        <span class="risk-badge risk-badge-${tierClass}" style="margin-right:0.5rem;">
                            ${(tierClass).toUpperCase()} RISK
                        </span>
                        <span class="status-badge status-${(s.target||'').toLowerCase()}">${s.target}</span>
                        &nbsp;·&nbsp; ${s.gender_label || '—'}
                        &nbsp;·&nbsp; Age ${s.age_at_enrollment || '—'} at enrollment
                    </span>
                </div>
            </div>

            <div class="modal-reason">
                <div class="modal-reason-label">★ Advisor Reason Text</div>
                <div class="modal-reason-text">"${s.reason_text || 'No reason text available.'}"</div>
            </div>

            <div class="modal-grid">
                <div class="modal-stat">
                    <div class="modal-stat-label">Grade Delta (Sem2 − Sem1)</div>
                    <div class="modal-stat-value" style="color:${(s.grade_delta||0) < 0 ? 'var(--rose-light)' : 'var(--emerald-light)'}">
                        ${(s.grade_delta||0) > 0 ? '+' : ''}${(s.grade_delta||0).toFixed(2)}
                    </div>
                </div>
                <div class="modal-stat">
                    <div class="modal-stat-label">Financial Stress Index</div>
                    <div class="modal-stat-value">${(s.financial_stress_index||0).toFixed(0)}/5</div>
                </div>
                <div class="modal-stat">
                    <div class="modal-stat-label">Absenteeism Trend</div>
                    <div class="modal-stat-value">${((s.absenteeism_trend||0)*100).toFixed(1)}%</div>
                </div>
                <div class="modal-stat">
                    <div class="modal-stat-label">Engagement Score</div>
                    <div class="modal-stat-value">${(s.engagement_score||0).toFixed(2)}</div>
                </div>
                <div class="modal-stat">
                    <div class="modal-stat-label">Admission Grade</div>
                    <div class="modal-stat-value">${(s.admission_grade||0).toFixed(1)}</div>
                </div>
                <div class="modal-stat">
                    <div class="modal-stat-label">Socioeconomic Group</div>
                    <div class="modal-stat-value" style="font-family:var(--font-sans);font-size:0.9rem;">
                        ${(s.socioeconomic_group||'—').replace('_',' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                </div>
            </div>

            <div class="modal-shap-title">
                <i data-lucide="brain"></i>
                Top-3 Contributing Risk Factors (SHAP)
            </div>
            <div class="modal-shap-list">
                ${[1,2,3].map(i => {
                    const factor = s[`shap_factor_${i}`] || '—';
                    const value = s[`shap_value_${i}`] || 0;
                    const valClass = value > 0 ? 'shap-positive' : 'shap-negative';
                    return `
                        <div class="modal-shap-item">
                            <span class="shap-item-rank">#${i}</span>
                            <span class="shap-item-name">${factor.replace(/_/g,' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                            <span class="shap-item-value ${valClass}">${value > 0 ? '+' : ''}${value.toFixed(4)}</span>
                        </div>
                    `;
                }).join('')}
            </div>

            <!-- Phase 2: Action Plan -->
            <div class="modal-action-plan">
                <button class="action-plan-toggle" onclick="toggleActionPlan(${s.student_id}, this)">
                    <i data-lucide="lightbulb"></i>
                    Show Recommended Action Plan
                </button>
                <div class="action-plan-content" id="action-plan-${s.student_id}" style="display:none;"></div>
            </div>

            <!-- Phase 2: Nudge Message -->
            <div class="modal-nudge">
                <button class="nudge-toggle" onclick="toggleNudge(${s.student_id}, this)">
                    <i data-lucide="mail"></i>
                    Generate Outreach Message
                </button>
                <div class="nudge-content" id="nudge-${s.student_id}" style="display:none;"></div>
            </div>
        `;

        lucide.createIcons();
    } catch (err) {
        body.innerHTML = '<p style="color:var(--rose-light);padding:2rem;">Failed to load student details.</p>';
        console.error(err);
    }
}

// ========================================================================
// FAIRNESS
// ========================================================================
async function loadFairness() {
    try {
        const res = await fetch('/api/fairness');
        const data = await res.json();
        renderFairnessMetrics(data.metrics);
        renderFairnessCharts(data.metrics);
    } catch (err) {
        console.error('Failed to load fairness:', err);
    }
}

function renderFairnessMetrics(metrics) {
    const marginal = metrics.filter(m => m.audit_type === 'marginal');
    const intersectional = metrics.filter(m => m.audit_type === 'intersectional');

    // Marginal
    const marginalEl = document.getElementById('marginal-metrics');
    marginalEl.innerHTML = marginal.map(m => {
        const dpClass = m.demographic_parity_diff > 0.10 ? 'fm-num-bad' :
                        m.demographic_parity_diff > 0.05 ? 'fm-num-warn' : 'fm-num-ok';
        const eoClass = m.equal_opportunity_diff > 0.10 ? 'fm-num-bad' :
                        m.equal_opportunity_diff > 0.05 ? 'fm-num-warn' : 'fm-num-ok';
        return `
            <div class="fairness-metric-row">
                <div class="fm-label">
                    <span class="fm-groups">${formatGroup(m.group_a)} vs ${formatGroup(m.group_b)}</span>
                    <span class="fm-type">${m.group_type} · n=${m.group_a_size}+${m.group_b_size}</span>
                </div>
                <div class="fm-values">
                    <div class="fm-value">
                        <span class="fm-num ${dpClass}">${m.demographic_parity_diff.toFixed(3)}</span>
                        <span class="fm-label-small">DP Diff</span>
                    </div>
                    <div class="fm-value">
                        <span class="fm-num ${eoClass}">${m.equal_opportunity_diff.toFixed(3)}</span>
                        <span class="fm-label-small">EO Diff</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Intersectional
    const interEl = document.getElementById('intersectional-metrics');
    interEl.innerHTML = intersectional.map(m => {
        const dpClass = m.demographic_parity_diff > 0.10 ? 'fm-num-bad' :
                        m.demographic_parity_diff > 0.05 ? 'fm-num-warn' : 'fm-num-ok';
        const eoClass = m.equal_opportunity_diff > 0.10 ? 'fm-num-bad' :
                        m.equal_opportunity_diff > 0.05 ? 'fm-num-warn' : 'fm-num-ok';
        return `
            <div class="fairness-metric-row">
                <div class="fm-label">
                    <span class="fm-groups">${formatGroup(m.group_a)} vs ${formatGroup(m.group_b)}</span>
                    <span class="fm-type">intersectional · n=${m.group_a_size}+${m.group_b_size}</span>
                </div>
                <div class="fm-values">
                    <div class="fm-value">
                        <span class="fm-num ${dpClass}">${m.demographic_parity_diff.toFixed(3)}</span>
                        <span class="fm-label-small">DP Diff</span>
                    </div>
                    <div class="fm-value">
                        <span class="fm-num ${eoClass}">${m.equal_opportunity_diff.toFixed(3)}</span>
                        <span class="fm-label-small">EO Diff</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderFairnessCharts(metrics) {
    // DP chart
    const dpCtx = document.getElementById('dpChart').getContext('2d');
    const dpLabels = metrics.map(m => `${shortGroup(m.group_a)} vs ${shortGroup(m.group_b)}`);
    const dpValues = metrics.map(m => m.demographic_parity_diff);
    const dpColors = dpValues.map(v => v > 0.10 ? 'rgba(244,63,94,0.7)' :
                                        v > 0.05 ? 'rgba(245,158,11,0.7)' : 'rgba(16,185,129,0.7)');

    new Chart(dpCtx, {
        type: 'bar',
        data: {
            labels: dpLabels,
            datasets: [{
                label: 'DP Difference',
                data: dpValues,
                backgroundColor: dpColors,
                borderColor: dpColors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,17,23,0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                },
                annotation: {
                    annotations: {
                        threshold: {
                            type: 'line',
                            xMin: 0.10,
                            xMax: 0.10,
                            borderColor: 'rgba(244,63,94,0.5)',
                            borderWidth: 2,
                            borderDash: [6, 4],
                        }
                    }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { font: { size: 10 } } },
                y: { grid: { display: false }, ticks: { font: { size: 10 } } }
            }
        }
    });

    // EO chart
    const eoCtx = document.getElementById('eoChart').getContext('2d');
    const eoValues = metrics.map(m => m.equal_opportunity_diff);
    const eoColors = eoValues.map(v => v > 0.10 ? 'rgba(244,63,94,0.7)' :
                                        v > 0.05 ? 'rgba(245,158,11,0.7)' : 'rgba(16,185,129,0.7)');

    new Chart(eoCtx, {
        type: 'bar',
        data: {
            labels: dpLabels,
            datasets: [{
                label: 'EO Difference',
                data: eoValues,
                backgroundColor: eoColors,
                borderColor: eoColors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,17,23,0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { font: { size: 10 } } },
                y: { grid: { display: false }, ticks: { font: { size: 10 } } }
            }
        }
    });
}

function formatGroup(g) {
    return g.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function shortGroup(g) {
    const parts = g.split('_');
    if (parts.length > 2) {
        return parts[0].charAt(0).toUpperCase() + '/' + parts.slice(1).map(p => p.charAt(0).toUpperCase()).join('');
    }
    return formatGroup(g);
}

// ========================================================================
// PIPELINE
// ========================================================================
async function loadPipeline() {
    try {
        const res = await fetch('/api/pipeline');
        const data = await res.json();
        renderPipeline(data.layers);
    } catch (err) {
        console.error('Failed to load pipeline:', err);
    }
}

function renderPipeline(layers) {
    const container = document.getElementById('pipeline-flow');
    const icons = ['database', 'sparkles', 'brain', 'shield-check', 'eye', 'trophy'];
    const nodeClasses = ['bronze', 'silver', 'model', 'audit', 'shap', 'gold'];

    let html = '';
    layers.forEach((layer, i) => {
        const statusClass = layer.status === 'complete' ? 'ps-complete' :
                            layer.status === 'in_progress' ? 'ps-progress' : 'ps-pending';
        const statusLabel = layer.status === 'complete' ? '✓ Complete' :
                            layer.status === 'in_progress' ? '⟳ In Progress' : '○ Pending';

        html += `
            <div class="pipeline-step">
                <div class="pipeline-node pipeline-node-${nodeClasses[i]}">
                    <i data-lucide="${icons[i]}"></i>
                </div>
                <span class="pipeline-step-name">${layer.name}</span>
                <span class="pipeline-step-table">${layer.table}</span>
                <span class="pipeline-step-status ${statusClass}">${statusLabel}</span>
            </div>
        `;
        if (i < layers.length - 1) {
            html += `
                <div class="pipeline-arrow">
                    <i data-lucide="chevron-right"></i>
                </div>
            `;
        }
    });

    container.innerHTML = html;
    lucide.createIcons();
}

// ========================================================================
// PHASE 2: RED ZONE
// ========================================================================
async function loadRedZone() {
    try {
        const res = await fetch('/api/red-zone?per_page=15');
        const data = await res.json();

        // Stats
        animateCounter(document.getElementById('rz-total'), data.total_red_zone);
        document.getElementById('rz-rate').textContent = `${data.red_zone_rate}%`;
        document.getElementById('rz-avg-risk').textContent = data.avg_risk_score.toFixed(3);
        document.getElementById('rz-avg-sentiment').textContent = data.avg_sentiment.toFixed(3);

        // Table
        const tbody = document.getElementById('rz-tbody');
        if (!data.students || data.students.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="table-loading">No students in the Red Zone.</td></tr>';
            return;
        }

        tbody.innerHTML = data.students.map(s => {
            const tierClass = `risk-badge-${s.intervention_tier.toLowerCase()}`;
            const tierLabel = s.intervention_tier;
            const sentClass = s.sentiment_score < 0.3 ? 'sentiment-low' :
                              s.sentiment_score < 0.5 ? 'sentiment-medium' : 'sentiment-high';
            return `
                <tr>
                    <td><span style="font-family:var(--font-mono);font-weight:600;">#${s.student_id}</span></td>
                    <td><span class="risk-badge ${tierClass}">${s.risk_score.toFixed(3)}</span></td>
                    <td><span class="risk-badge ${tierClass}">${tierLabel}</span></td>
                    <td>
                        <span class="sentiment-indicator ${sentClass}">
                            <span class="sentiment-dot"></span>
                            ${s.sentiment_score.toFixed(3)}
                        </span>
                    </td>
                    <td><span style="font-family:var(--font-mono);">${s.financial_stress_index.toFixed(0)}/5</span></td>
                    <td>${s.gender_label || '—'}</td>
                    <td><span class="reason-cell" title="${escapeHtml(s.reason_text || '')}">${s.reason_text || '—'}</span></td>
                    <td><button class="btn-detail" onclick="openStudentModal(${s.student_id})">View</button></td>
                </tr>
            `;
        }).join('');

        lucide.createIcons();
    } catch (err) {
        console.error('Failed to load Red Zone:', err);
    }
}

// ========================================================================
// PHASE 2: ACTION PLAN
// ========================================================================
async function toggleActionPlan(studentId, btn) {
    const container = document.getElementById(`action-plan-${studentId}`);
    if (!container) return;

    if (container.style.display !== 'none') {
        container.style.display = 'none';
        btn.innerHTML = '<i data-lucide="lightbulb"></i> Show Recommended Action Plan';
        lucide.createIcons();
        return;
    }

    btn.innerHTML = '<i data-lucide="loader"></i> Loading...';
    lucide.createIcons();

    try {
        const res = await fetch(`/api/students/${studentId}/action-plan`);
        const data = await res.json();

        container.innerHTML = data.actions.map((a, i) => `
            <div class="action-card">
                <div class="action-rank action-rank-${i + 1}">#${i + 1}</div>
                <div class="action-body">
                    <div class="action-name">${a.name}</div>
                    <span class="action-type">${a.type}</span>
                    <div class="action-desc">${a.description}</div>
                    <div class="action-rationale">${a.rationale}</div>
                </div>
                <div class="action-impact">
                    <span class="action-impact-value">${(a.impact_score * 100).toFixed(0)}%</span>
                    <span class="action-impact-label">Impact</span>
                </div>
            </div>
        `).join('');

        container.style.display = 'flex';
        btn.innerHTML = '<i data-lucide="lightbulb"></i> Hide Action Plan';
        lucide.createIcons();
    } catch (err) {
        container.innerHTML = '<p style="color:var(--rose-light);font-size:0.8rem;">Failed to load action plan.</p>';
        container.style.display = 'flex';
        console.error(err);
    }
}

// ========================================================================
// PHASE 2: NUDGE MESSAGE
// ========================================================================
async function toggleNudge(studentId, btn) {
    const container = document.getElementById(`nudge-${studentId}`);
    if (!container) return;

    if (container.style.display !== 'none') {
        container.style.display = 'none';
        btn.innerHTML = '<i data-lucide="mail"></i> Generate Outreach Message';
        lucide.createIcons();
        return;
    }

    btn.innerHTML = '<i data-lucide="loader"></i> Generating...';
    lucide.createIcons();

    try {
        const res = await fetch(`/api/students/${studentId}/nudge`);
        const data = await res.json();

        const currentStatus = data.status || 'pending';

        container.innerHTML = `
            <div class="nudge-message-box">${escapeHtml(data.message)}</div>
            <div class="nudge-actions">
                <button class="nudge-btn nudge-btn-copy" onclick="copyNudge(this)">
                    <i data-lucide="copy"></i> Copy to Clipboard
                </button>
                <div class="nudge-status-group">
                    <button class="nudge-status-btn ${currentStatus==='pending'?'active-pending':''}" onclick="updateNudgeStatus(${studentId},'pending',this)">Pending</button>
                    <button class="nudge-status-btn ${currentStatus==='sent'?'active-sent':''}" onclick="updateNudgeStatus(${studentId},'sent',this)">Sent</button>
                    <button class="nudge-status-btn ${currentStatus==='resolved'?'active-resolved':''}" onclick="updateNudgeStatus(${studentId},'resolved',this)">Resolved</button>
                </div>
            </div>
        `;

        container.style.display = 'block';
        btn.innerHTML = '<i data-lucide="mail"></i> Hide Outreach Message';
        lucide.createIcons();
    } catch (err) {
        container.innerHTML = '<p style="color:var(--rose-light);font-size:0.8rem;">Failed to generate message.</p>';
        container.style.display = 'block';
        console.error(err);
    }
}

function copyNudge(btn) {
    const messageBox = btn.closest('.nudge-content').querySelector('.nudge-message-box');
    if (!messageBox) return;
    navigator.clipboard.writeText(messageBox.textContent).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = '<i data-lucide="check"></i> Copied!';
        lucide.createIcons();
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = '<i data-lucide="copy"></i> Copy to Clipboard';
            lucide.createIcons();
        }, 2000);
    });
}

async function updateNudgeStatus(studentId, status, btn) {
    try {
        await fetch(`/api/students/${studentId}/status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status }),
        });

        // Update button states
        const group = btn.closest('.nudge-status-group');
        group.querySelectorAll('.nudge-status-btn').forEach(b => {
            b.className = 'nudge-status-btn';
        });
        btn.classList.add(`active-${status}`);
    } catch (err) {
        console.error('Failed to update status:', err);
    }
}

// ========================================================================
// PHASE 2: POLICY SIMULATOR
// ========================================================================
function initSimulator() {
    // FSI slider label
    const fsiSlider = document.getElementById('sim-fsi');
    const fsiLabel = document.getElementById('fsi-value');
    if (fsiSlider && fsiLabel) {
        fsiSlider.addEventListener('input', () => {
            fsiLabel.textContent = fsiSlider.value;
        });
    }

    // Run button
    const runBtn = document.getElementById('sim-run-btn');
    if (runBtn) {
        runBtn.addEventListener('click', runSimulation);
    }
}

async function runSimulation() {
    const btn = document.getElementById('sim-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader"></i> Simulating...';
    lucide.createIcons();

    const params = {
        grant_scholarship: document.getElementById('sim-scholarship').checked,
        waive_fees: document.getElementById('sim-fees').checked,
        fsi_reduction: parseFloat(document.getElementById('sim-fsi').value) || 0,
    };

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        const data = await res.json();

        // Update badge
        const badge = document.getElementById('sim-status-badge');
        badge.textContent = 'Simulation complete';
        badge.classList.add('active');

        // Render results
        const body = document.getElementById('sim-results-body');
        const riskDelta = data.impact.risk_reduction;
        const riskColor = riskDelta > 0 ? 'var(--emerald-light)' : 'var(--text-muted)';

        body.innerHTML = `
            <div class="sim-impact-grid">
                <div class="sim-impact-card">
                    <span class="sim-impact-value" style="color:${riskColor}">
                        ${riskDelta > 0 ? '-' : ''}${(Math.abs(riskDelta) * 100).toFixed(2)}%
                    </span>
                    <span class="sim-impact-label">Avg Risk Reduction</span>
                </div>
                <div class="sim-impact-card">
                    <span class="sim-impact-value" style="color:var(--emerald-light)">
                        ${data.impact.moved_from_high}
                    </span>
                    <span class="sim-impact-label">Moved from High Risk</span>
                </div>
                <div class="sim-impact-card">
                    <span class="sim-impact-value" style="color:var(--cyan)">
                        ${data.impact.dropouts_potentially_saved}
                    </span>
                    <span class="sim-impact-label">Potential Dropouts Saved</span>
                </div>
            </div>

            <div class="sim-tier-comparison">
                <div class="sim-tier-col">
                    <div class="sim-tier-col-header">Before</div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--rose-light)">High</span>
                        <span class="sim-tier-count">${data.before.tiers.High || 0}</span>
                    </div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--amber-light)">Medium</span>
                        <span class="sim-tier-count">${data.before.tiers.Medium || 0}</span>
                    </div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--emerald-light)">Low</span>
                        <span class="sim-tier-count">${data.before.tiers.Low || 0}</span>
                    </div>
                </div>
                <div class="sim-arrow-col">
                    <i data-lucide="arrow-right"></i>
                    <i data-lucide="arrow-right"></i>
                    <i data-lucide="arrow-right"></i>
                </div>
                <div class="sim-tier-col">
                    <div class="sim-tier-col-header">After</div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--rose-light)">High</span>
                        <span class="sim-tier-count">${data.after.tiers.High || 0}</span>
                    </div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--amber-light)">Medium</span>
                        <span class="sim-tier-count">${data.after.tiers.Medium || 0}</span>
                    </div>
                    <div class="sim-tier-row">
                        <span class="sim-tier-name" style="color:var(--emerald-light)">Low</span>
                        <span class="sim-tier-count">${data.after.tiers.Low || 0}</span>
                    </div>
                </div>
            </div>
        `;

        lucide.createIcons();
    } catch (err) {
        console.error('Simulation failed:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="play"></i> Run Simulation';
        lucide.createIcons();
    }
}

// ========================================================================
// PHASE 2: BIAS MITIGATION
// ========================================================================
async function loadMitigation() {
    try {
        const res = await fetch('/api/fairness/mitigation');
        const data = await res.json();
        renderMitigationCards(data);
        renderMitigationChart(data);
    } catch (err) {
        console.error('Failed to load mitigation:', err);
    }
}

function renderMitigationCards(data) {
    const grid = document.getElementById('mitigation-grid');

    let html = data.before.map((b, i) => {
        const a = data.after[i];
        const dpBefore = (b.dp_diff > 0.10) ? 'fm-num-bad' : (b.dp_diff > 0.05) ? 'fm-num-warn' : 'fm-num-ok';
        const eoBefore = (b.eo_diff > 0.10) ? 'fm-num-bad' : (b.eo_diff > 0.05) ? 'fm-num-warn' : 'fm-num-ok';
        const dpAfter = (a.dp_diff > 0.10) ? 'fm-num-bad' : (a.dp_diff > 0.05) ? 'fm-num-warn' : 'fm-num-ok';
        const eoAfter = (a.eo_diff > 0.10) ? 'fm-num-bad' : (a.eo_diff > 0.05) ? 'fm-num-warn' : 'fm-num-ok';

        return `
            <div class="mitigation-card">
                <div class="mitigation-card-header">${formatGroup(b.group)}</div>
                <div class="mitigation-metric-pair">
                    <div class="mitigation-metric">
                        <span class="mitigation-value ${dpBefore}">${b.dp_diff.toFixed(3)}</span>
                        <span class="mitigation-label">DP Diff</span>
                    </div>
                    <div class="mitigation-metric">
                        <span class="mitigation-value ${eoBefore}">${b.eo_diff.toFixed(3)}</span>
                        <span class="mitigation-label">EO Diff</span>
                    </div>
                </div>
                <div class="mitigation-arrow">
                    <i data-lucide="arrow-down"></i>
                    After Constraint
                </div>
                <div class="mitigation-after">
                    <div class="mitigation-metric-pair">
                        <div class="mitigation-metric">
                            <span class="mitigation-value ${dpAfter}">${a.dp_diff.toFixed(3)}</span>
                            <span class="mitigation-label">DP Diff</span>
                        </div>
                        <div class="mitigation-metric">
                            <span class="mitigation-value ${eoAfter}">${a.eo_diff.toFixed(3)}</span>
                            <span class="mitigation-label">EO Diff</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    html += `
        <div class="mitigation-auc-note" style="grid-column: 1 / -1;">
            AUC Impact: <strong>${data.accuracy_impact.before_auc}</strong> → <strong>${data.accuracy_impact.after_auc}</strong>
            (−${(data.accuracy_impact.auc_cost * 100).toFixed(1)}%) · ${data.accuracy_impact.note}
        </div>
    `;

    grid.innerHTML = html;
    lucide.createIcons();
}

function renderMitigationChart(data) {
    const ctx = document.getElementById('mitigationChart').getContext('2d');
    const labels = data.before.map(b => formatGroup(b.group));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'EO Diff (Before)',
                    data: data.before.map(b => b.eo_diff),
                    backgroundColor: 'rgba(244, 63, 94, 0.6)',
                    borderColor: 'rgba(244, 63, 94, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'EO Diff (After)',
                    data: data.after.map(a => a.eo_diff),
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'DP Diff (Before)',
                    data: data.before.map(b => b.dp_diff),
                    backgroundColor: 'rgba(245, 158, 11, 0.5)',
                    borderColor: 'rgba(245, 158, 11, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'DP Diff (After)',
                    data: data.after.map(a => a.dp_diff),
                    backgroundColor: 'rgba(6, 182, 212, 0.5)',
                    borderColor: 'rgba(6, 182, 212, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        padding: 16,
                        font: { size: 11, weight: '500' },
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 23, 0.95)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (item) => `${item.dataset.label}: ${item.raw.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11, weight: '500' } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { font: { size: 10 } },
                    title: {
                        display: true,
                        text: 'Disparity Score',
                        font: { size: 11, weight: '500' },
                        color: '#5c5f72'
                    }
                }
            }
        }
    });
}
