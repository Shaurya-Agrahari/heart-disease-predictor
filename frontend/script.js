const API_URL = "https://heart-disease-predictor-d4oh.onrender.com";

const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");

const resultIdle = document.getElementById("result-idle");
const resultLoading = document.getElementById("result-loading");
const resultActive = document.getElementById("result-active");

const riskLevelEl = document.getElementById("risk-level");
const probabilityEl = document.getElementById("probability-value");
const predictionTextEl = document.getElementById("prediction-text");
const bmiValueEl = document.getElementById("bmi-value");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeNeedle = document.getElementById("gauge-needle");

const GAUGE_CIRCUMFERENCE = 283;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  showLoading();
  submitBtn.disabled = true;

  const payload = {
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    height: document.getElementById("height").value,
    weight: document.getElementById("weight").value,
    ap_hi: document.getElementById("ap_hi").value,
    ap_lo: document.getElementById("ap_lo").value,
    cholesterol: document.getElementById("cholesterol").value,
    gluc: document.getElementById("gluc").value,
    smoke: document.getElementById("smoke").value,
    alco: document.getElementById("alco").value,
    active: document.getElementById("active").value,
  };

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    renderResult(data);
  } catch (err) {
    showError(
      err.message === "Failed to fetch"
        ? "Can't reach the backend. Is 'python app.py' running on port 5000?"
        : err.message
    );
    showIdle();
  } finally {
    submitBtn.disabled = false;
  }
});

function renderResult(data) {
  const { prediction, probability, risk_level, bmi, recommendations } = data;

  riskLevelEl.textContent = risk_level;
  riskLevelEl.className = "reading-value " + risk_level.toLowerCase();

  predictionTextEl.textContent =
    prediction === 1 ? "Disease indicated" : "No disease indicated";
  bmiValueEl.textContent = bmi;

  animateCount(probabilityEl, probability);

  const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * probability) / 100;
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke =
      risk_level === "High" ? "var(--coral)" :
      risk_level === "Moderate" ? "var(--amber)" : "var(--teal)";
  });

  const angle = -90 + (probability / 100) * 180;
  requestAnimationFrame(() => {
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  });

  renderRecommendations(recommendations, risk_level);

  resultLoading.hidden = true;
  resultIdle.hidden = true;
  resultActive.hidden = false;
}

function renderRecommendations(recommendations, riskLevel) {
  const recSection = document.getElementById("rec-section");
  const recList = document.getElementById("rec-list");

  recList.innerHTML = "";

  if (!recommendations || recommendations.length === 0) {
    recSection.hidden = true;
    return;
  }

  const flagClass = riskLevel === "High" ? "flag-high" : riskLevel === "Moderate" ? "flag-moderate" : "";

  recommendations.forEach((rec) => {
    const li = document.createElement("li");
    li.className = "rec-item " + flagClass;
    li.innerHTML = `
      <div class="rec-title">${rec.title}</div>
      <div class="rec-detail">${rec.detail}</div>
    `;
    recList.appendChild(li);
  });

  recSection.hidden = false;
}

function animateCount(el, target) {
  const duration = 700;
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    el.textContent = (target * progress).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function showLoading() {
  resultIdle.hidden = true;
  resultActive.hidden = true;
  resultLoading.hidden = false;
}
function showIdle() {
  resultLoading.hidden = true;
  resultActive.hidden = true;
  resultIdle.hidden = false;
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}
