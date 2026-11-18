// PathSnap Evaluator Interface Client Logic

let tripsData = [];

function getBaseUrl() {
    let url = document.getElementById("apiBaseUrl").value.trim();
    if (url.endsWith("/")) {
        url = url.slice(0, -1);
    }
    return url;
}

function logMessage(msg) {
    const logElem = document.getElementById("logStream");
    const timestamp = new Date().toLocaleTimeString();
    logElem.innerText += `\n[${timestamp}] ${msg}`;
    logElem.scrollTop = logElem.scrollHeight;
}

async function checkHealth() {
    const base = getBaseUrl();
    const statusElem = document.getElementById("serverStatus");
    try {
        logMessage(`Pinging API at ${base}/api/health...`);
        const res = await fetch(`${base}/api/health`);
        if (res.ok) {
            const data = await res.json();
            statusElem.className = "inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500 text-white";
            statusElem.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-white mr-1.5 animate-pulse"></span>Connected (${data.model_loaded})`;
            logMessage(`Backend Status: HEALTHY | Model: ${data.model_loaded} | Evaluation Trips: ${data.evaluation_trips_count}`);
            await loadEvaluationTrips();
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (err) {
        statusElem.className = "inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500 text-white";
        statusElem.innerHTML = `Disconnected`;
        logMessage(`ERROR connecting to API: ${err.message}`);
    }
}

async function loadEvaluationTrips() {
    const base = getBaseUrl();
    const selectElem = document.getElementById("tripSelect");
    try {
        logMessage(`Fetching evaluation trips from ${base}/api/trips...`);
        const res = await fetch(`${base}/api/trips`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        tripsData = data.evaluation_trips || [];

        selectElem.innerHTML = "";
        tripsData.forEach((t, idx) => {
            const opt = document.createElement("option");
            opt.value = t.trip_id;
            opt.innerText = `Evaluation Trip ${idx + 1}: ID ${t.trip_id} (${t.num_track_points} GPS points, ${t.num_arcs} arcs, ${t.duration_seconds.toFixed(0)}s)`;
            selectElem.appendChild(opt);
        });

        if (tripsData.length > 0) {
            updateTripMetadata(tripsData[0].trip_id);
            logMessage(`Loaded ${tripsData.length} evaluation trips successfully.`);
        }
    } catch (err) {
        logMessage(`Failed to load evaluation trips: ${err.message}`);
        selectElem.innerHTML = `<option value="">Error loading trips</option>`;
    }
}

function updateTripMetadata(tripId) {
    const trip = tripsData.find(t => t.trip_id === tripId);
    if (!trip) return;
    document.getElementById("metaPoints").innerText = trip.num_track_points;
    document.getElementById("metaNodes").innerText = trip.num_nodes;
    document.getElementById("metaArcs").innerText = trip.num_arcs;
    document.getElementById("metaDuration").innerText = `${trip.duration_seconds.toFixed(0)}s`;
}

document.getElementById("tripSelect").addEventListener("change", (e) => {
    updateTripMetadata(e.target.value);
});

async function runEvaluation() {
    const tripId = document.getElementById("tripSelect").value;
    if (!tripId) {
        alert("Please select an evaluation trip.");
        return;
    }

    const base = getBaseUrl();
    const evalBtn = document.getElementById("evalBtn");
    evalBtn.disabled = true;
    evalBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Processing Trip ${tripId}...</span>`;

    logMessage(`--- Initiating Full Evaluation for Trip ${tripId} ---`);
    logMessage(`Sending POST request to ${base}/api/predict/${tripId}...`);

    try {
        const res = await fetch(`${base}/api/predict/${tripId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        logMessage(`Response received in ${data.latency.total_latency_ms} ms!`);
        logMessage(`Payload: ${JSON.stringify(data.metrics, null, 2)}`);

        // Update KPIs
        const m = data.metrics;
        document.getElementById("valHmmAcc").innerText = m.hmm_viterbi_accuracy != null ? `${m.hmm_viterbi_accuracy}%` : "N/A";
        document.getElementById("valClfAcc").innerText = m.classifier_top1_accuracy != null ? `${m.classifier_top1_accuracy}%` : "N/A";
        document.getElementById("lblAccGain").innerHTML = `<i class="fa-solid fa-arrow-trend-up mr-1 text-emerald-500"></i> Baseline: ${m.classifier_top1_accuracy}% <span class="ml-1 text-emerald-600 font-bold">(+${m.accuracy_gain_pct}%)</span>`;

        document.getElementById("valHmmCont").innerText = m.hmm_viterbi_continuity != null ? `${m.hmm_viterbi_continuity}%` : "N/A";
        document.getElementById("valClfCont").innerText = m.classifier_continuity != null ? `${m.classifier_continuity}%` : "N/A";
        document.getElementById("lblContGain").innerHTML = `<i class="fa-solid fa-arrow-trend-up mr-1 text-blue-500"></i> Classifier only: ${m.classifier_continuity}% <span class="ml-1 text-blue-600 font-bold">(+${m.continuity_gain_pct}%)</span>`;

        document.getElementById("valMissRate").innerText = m.candidate_miss_rate != null ? `${m.candidate_miss_rate}%` : "0.0%";
        document.getElementById("valLatency").innerText = `${data.latency.total_latency_ms} ms`;
        document.getElementById("valPerObsLatency").innerText = `${data.latency.ms_per_observation} ms`;

        // Update Map Frame
        const mapUrl = `${base}${data.map_url}?t=${new Date().getTime()}`;
        document.getElementById("mapFrame").src = mapUrl;
        document.getElementById("openFullMapBtn").href = mapUrl;
        document.getElementById("mapPlaceholder").style.display = "none";

        logMessage(`Trip ${tripId} evaluated successfully! Interactive Folium map updated.`);
    } catch (err) {
        logMessage(`Evaluation Error: ${err.message}`);
        alert(`Evaluation Failed: ${err.message}`);
    } finally {
        evalBtn.disabled = false;
        evalBtn.innerHTML = `<i class="fa-solid fa-bolt"></i><span>Run Full Map-Matching Pipeline</span>`;
    }
}

// Initial Boot
window.addEventListener("DOMContentLoaded", () => {
    if (window.location.protocol.startsWith("http")) {
        const input = document.getElementById("apiBaseUrl");
        if (input && (!input.value || input.value.includes("localhost"))) {
            input.value = window.location.origin;
        }
    }
    checkHealth();
});
