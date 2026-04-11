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
        const tierClass = `risk-badge-${s.intervention_tier}`;
        const tierLabel = s.intervention_tier.charAt(0).toUpperCase() + s.intervention_tier.slice(1);

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
