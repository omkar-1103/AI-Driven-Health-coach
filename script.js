/* ──────────────────────────────────────────────────
   VitalSync — Dashboard Script
   Polls /api/dashboard every 5 s and updates UI
   ────────────────────────────────────────────────── */

const token    = sessionStorage.getItem('token');
const username = sessionStorage.getItem('username');

// Redirect to login if not authenticated
if (!token) window.location.href = '/';

// ── Auth header helper ────────────────────────────
const authHeaders = () => ({ 'Authorization': `Bearer ${token}` });

// ── Previous values (for trend arrows) ───────────
let prev = {};

// ── Chart instances ───────────────────────────────
let miniCharts = {};
let bigCharts  = {};

// ── Chart colour palette ──────────────────────────
const COLORS = {
  hr:   { line: '#f472b6', fill: 'rgba(244,114,182,0.12)' },
  temp: { line: '#fbbf24', fill: 'rgba(251,191,36,0.12)'  },
  spo2: { line: '#22d3ee', fill: 'rgba(34,211,238,0.12)'  },
  bpS:  { line: '#3b82f6', fill: 'rgba(59,130,246,0.10)'  },
  bpD:  { line: '#a855f7', fill: 'rgba(168,85,247,0.08)'  },
};

const CHART_DEFAULTS = {
  type: 'line',
  options: {
    responsive: true,
    animation: { duration: 400 },
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: {
      x: { display: false },
      y: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#5a5a7a', font: { size: 10 }, maxTicksLimit: 4 },
      },
    },
    elements: { point: { radius: 0, hoverRadius: 4 } },
  },
};

function makeDataset(color, data = []) {
  return {
    data,
    borderColor: color.line,
    backgroundColor: color.fill,
    borderWidth: 2,
    fill: true,
    tension: 0.4,
  };
}

function initCharts() {
  // Mini (inside metric cards)
  const miniConf = (color) => ({
    ...CHART_DEFAULTS,
    data: { labels: [], datasets: [makeDataset(color)] },
    options: {
      ...CHART_DEFAULTS.options,
      scales: { x: { display: false }, y: { display: false } },
    },
  });

  miniCharts.hr   = new Chart(document.getElementById('chart-hr'),   miniConf(COLORS.hr));
  miniCharts.temp = new Chart(document.getElementById('chart-temp'), miniConf(COLORS.temp));
  miniCharts.spo2 = new Chart(document.getElementById('chart-spo2'), miniConf(COLORS.spo2));
  miniCharts.bp   = new Chart(document.getElementById('chart-bp'),   {
    ...CHART_DEFAULTS,
    data: {
      labels: [],
      datasets: [makeDataset(COLORS.bpS), makeDataset(COLORS.bpD)],
    },
    options: {
      ...CHART_DEFAULTS.options,
      scales: { x: { display: false }, y: { display: false } },
    },
  });

  // Big charts (trend overview section)
  const bigConf = (color, yMin, yMax) => ({
    ...CHART_DEFAULTS,
    data: { labels: [], datasets: [makeDataset(color)] },
    options: {
      ...CHART_DEFAULTS.options,
      scales: {
        x: { display: false },
        y: {
          display: true,
          min: yMin, max: yMax,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#5a5a7a', font: { size: 10 }, maxTicksLimit: 4 },
        },
      },
    },
  });

  bigCharts.hr   = new Chart(document.getElementById('chart-hr-big'),   bigConf(COLORS.hr,   55, 185));
  bigCharts.temp = new Chart(document.getElementById('chart-temp-big'), bigConf(COLORS.temp, 35.8, 38.5));
  bigCharts.spo2 = new Chart(document.getElementById('chart-spo2-big'), bigConf(COLORS.spo2, 94, 101));
  bigCharts.bp   = new Chart(document.getElementById('chart-bp-big'), {
    ...CHART_DEFAULTS,
    data: {
      labels: [],
      datasets: [makeDataset(COLORS.bpS), makeDataset(COLORS.bpD)],
    },
    options: {
      ...CHART_DEFAULTS.options,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#9999bb', font: { size: 10 }, boxWidth: 10 },
        },
      },
      scales: {
        x: { display: false },
        y: {
          min: 65, max: 145,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#5a5a7a', font: { size: 10 }, maxTicksLimit: 5 },
        },
      },
    },
  });

  // Label BP datasets
  bigCharts.bp.data.datasets[0].label = 'Systolic';
  bigCharts.bp.data.datasets[1].label = 'Diastolic';
}

// ── Push data into a chart (keep last N points) ───
function pushChart(chart, labels, ...seriesArrays) {
  chart.data.labels = labels;
  seriesArrays.forEach((arr, i) => {
    chart.data.datasets[i].data = arr;
  });
  chart.update('none');
}

// ── Format timestamp for labels ───────────────────
function fmtTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── Trend arrow ───────────────────────────────────
function trendHtml(curr, prevVal, unit = '') {
  if (prevVal === undefined) return '';
  const diff = curr - prevVal;
  if (Math.abs(diff) < 0.05) return `<span class="trend-same">→ ${unit}</span>`;
  if (diff > 0) return `<span class="trend-up">▲ +${Math.abs(diff).toFixed(1)}${unit}</span>`;
  return `<span class="trend-down">▼ ${diff.toFixed(1)}${unit}</span>`;
}

// ── Status dot helper ─────────────────────────────
function setStatus(id, level) {
  const el = document.getElementById(id);
  el.className = `card-status-dot status-${level}`;
}

function hrStatus(v) {
  if (v < 60 || v > 100) return 'danger';
  if (v > 90) return 'warning';
  return 'normal';
}
function tempStatus(v) {
  if (v > 37.5) return 'danger';
  if (v > 37.2) return 'warning';
  return 'normal';
}
function spo2Status(v) {
  if (v < 95) return 'danger';
  if (v < 97) return 'warning';
  return 'normal';
}
function bpStatus(sys, dia) {
  if (sys > 135 || dia > 88) return 'danger';
  if (sys > 125 || dia > 83) return 'warning';
  return 'normal';
}

// ── Range bar fill % ──────────────────────────────
function pct(v, lo, hi) { return Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100)).toFixed(1) + '%'; }

// ── Flash animation on value change ──────────────
function flashEl(id) {
  const el = document.getElementById(id);
  el.classList.remove('value-updated');
  void el.offsetWidth; // reflow
  el.classList.add('value-updated');
}

// ── Steps ring ────────────────────────────────────
const RING_CIRCUMFERENCE = 326.7;
const STEP_GOAL = 10000;

function updateSteps(steps) {
  const pctVal = Math.min(steps / STEP_GOAL, 1);
  const offset = RING_CIRCUMFERENCE * (1 - pctVal);
  document.getElementById('ring-progress').style.strokeDashoffset = offset;
  document.getElementById('val-steps').textContent = steps.toLocaleString();
  const rem = Math.max(0, STEP_GOAL - steps);
  document.getElementById('steps-remaining').textContent = rem.toLocaleString();
  document.getElementById('steps-pct').textContent = (pctVal * 100).toFixed(1) + '%';
}

// ── Header setup ──────────────────────────────────
function setupHeader() {
  document.getElementById('user-name-header').textContent = username;
  document.getElementById('user-avatar').textContent = (username || '?')[0].toUpperCase();

  function updateDate() {
    document.getElementById('header-date').textContent =
      new Date().toLocaleDateString([], { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
  }
  updateDate();
  setInterval(updateDate, 60000);
}

// ── Main fetch & render ───────────────────────────
async function fetchDashboard() {
  try {
    const res = await fetch('/api/dashboard', { headers: authHeaders() });
    if (res.status === 401) { sessionStorage.clear(); window.location.href = '/'; return; }
    const data = await res.json();
    if (!data.latest) { document.getElementById('last-updated').textContent = 'Waiting for first reading…'; return; }

    renderLatest(data.latest);
    renderHistory(data.history);
    document.getElementById('last-updated').textContent =
      'Updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch (err) {
    console.error('Dashboard fetch error:', err);
  }
}

function renderLatest(d) {
  // Heart Rate
  const hrEl = document.getElementById('val-hr');
  hrEl.textContent = d.heart_rate.toFixed(1);
  document.getElementById('trend-hr').innerHTML = trendHtml(d.heart_rate, prev.heart_rate, ' bpm');
  setStatus('hr-status', hrStatus(d.heart_rate));
  if (prev.heart_rate !== undefined) flashEl('val-hr');

  // Temperature
  document.getElementById('val-temp').textContent = d.temperature.toFixed(2) + ' °C';
  document.getElementById('trend-temp').innerHTML = trendHtml(d.temperature, prev.temperature, '°C');
  setStatus('temp-status', tempStatus(d.temperature));
  if (prev.temperature !== undefined) flashEl('val-temp');

  // Blood Oxygen
  document.getElementById('val-spo2').textContent = d.blood_oxygen.toFixed(1) + '%';
  document.getElementById('trend-spo2').innerHTML = trendHtml(d.blood_oxygen, prev.blood_oxygen, '%');
  setStatus('spo2-status', spo2Status(d.blood_oxygen));
  if (prev.blood_oxygen !== undefined) flashEl('val-spo2');

  // Blood Pressure
  document.getElementById('val-bp').textContent = `${d.bp_systolic} / ${d.bp_diastolic}`;
  document.getElementById('trend-bp').innerHTML = trendHtml(d.bp_systolic, prev.bp_systolic, ' sys');
  setStatus('bp-status', bpStatus(d.bp_systolic, d.bp_diastolic));
  if (prev.bp_systolic !== undefined) flashEl('val-bp');

  // Steps
  updateSteps(d.step_count);

  // Save previous
  prev = { ...d };
}

function renderHistory(history) {
  if (!history || history.length === 0) return;

  const labels = history.map(r => fmtTime(r.timestamp));
  const hrArr   = history.map(r => r.heart_rate);
  const tmpArr  = history.map(r => r.temperature);
  const spo2Arr = history.map(r => r.blood_oxygen);
  const bpSArr  = history.map(r => r.bp_systolic);
  const bpDArr  = history.map(r => r.bp_diastolic);

  pushChart(miniCharts.hr,   labels, hrArr);
  pushChart(miniCharts.temp, labels, tmpArr);
  pushChart(miniCharts.spo2, labels, spo2Arr);
  pushChart(miniCharts.bp,   labels, bpSArr, bpDArr);

  pushChart(bigCharts.hr,   labels, hrArr);
  pushChart(bigCharts.temp, labels, tmpArr);
  pushChart(bigCharts.spo2, labels, spo2Arr);
  pushChart(bigCharts.bp,   labels, bpSArr, bpDArr);
}

// ── Logout ────────────────────────────────────────
async function handleLogout() {
  try {
    await fetch('/api/logout', { method: 'POST', headers: authHeaders() });
  } finally {
    sessionStorage.clear();
    window.location.href = '/';
  }
}

// ── SVG gradient for ring (injected inline) ───────
function injectRingGradient() {
  const svg = document.querySelector('.steps-ring');
  if (!svg) return;
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#a855f7"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>`;
  svg.prepend(defs);
}

// ── Boot ──────────────────────────────────────────
setupHeader();
injectRingGradient();
initCharts();
fetchDashboard();
setInterval(fetchDashboard, 5000);
