"""Mobile/operator UI helpers for the Theseus Hive node."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def mobile_operator_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#101214">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Theseus Hive">
  <link rel="manifest" href="/operator.webmanifest">
  <link rel="apple-touch-icon" href="/operator-icon-180.png">
  <title>Theseus Hive</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:#101214;
      color:#e8edf2;
      --bg:#101214;
      --panel:#171b20;
      --panel2:#1f252b;
      --border:#303942;
      --muted:#98a4af;
      --soft:#b7c0ca;
      --good:#39d98a;
      --warn:#f6c453;
      --bad:#ff6b6b;
      --accent:#4e9bd8;
    }
    html { -webkit-text-size-adjust: 100%; font-size:16px; }
    * { -webkit-tap-highlight-color: transparent; }
    body { margin:0; min-height:100vh; background:var(--bg); }
    main { max-width:760px; margin:0 auto; padding:12px 16px calc(22px + env(safe-area-inset-bottom)); display:grid; gap:12px; }
    header { position:sticky; top:0; z-index:2; padding:calc(10px + env(safe-area-inset-top)) 16px 12px; background:#101214f2; border-bottom:1px solid var(--border); backdrop-filter: blur(16px); }
    .topbar { max-width:760px; margin:0 auto; display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center; }
    h1 { margin:0; font-size:20px; letter-spacing:0; }
    h2 { margin:0 0 10px; font-size:14px; color:var(--soft); font-weight:650; letter-spacing:0; }
    .grid { display:grid; gap:10px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card { border:1px solid var(--border); border-radius:8px; background:var(--panel); padding:12px; }
    .metric { display:grid; gap:2px; }
    .metric.wide { grid-column:1 / -1; }
    .metric span { color:var(--muted); font-size:12px; }
    .metric strong { font-size:20px; overflow-wrap:anywhere; }
    input, textarea, select, button { width:100%; box-sizing:border-box; border:1px solid var(--border); border-radius:8px; background:var(--panel2); color:#e8edf2; padding:11px; font:inherit; min-height:42px; }
    textarea { resize:vertical; }
    button { background:#26313a; cursor:pointer; font-weight:650; }
    button.primary { background:#315b7b; border-color:#3e79a8; }
    button.warn { background:#604323; border-color:#9b6930; }
    button.quick { background:#233126; border-color:#315f40; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .triple { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }
    .quick-grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px; }
    .list { display:grid; gap:8px; }
    .item { border-top:1px solid #273039; padding-top:8px; font-size:13px; color:var(--soft); overflow-wrap:anywhere; }
    .item b { color:#e8edf2; }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:300px; overflow:auto; margin:0; font-size:12px; }
    .muted { color:var(--muted); font-size:13px; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; justify-content:center; min-height:32px; padding:0 10px; border:1px solid var(--border); border-radius:999px; background:var(--panel2); color:var(--soft); font-size:12px; font-weight:650; white-space:nowrap; }
    .dot { width:9px; height:9px; border-radius:999px; background:var(--bad); box-shadow:0 0 0 3px #ff6b6b22; }
    .dot.running { background:var(--warn); box-shadow:0 0 0 3px #f6c45322; }
    .dot.connected { background:var(--good); box-shadow:0 0 0 3px #39d98a22; }
    .section-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:10px; }
    .section-head h2 { margin:0; }
    .compact-button { width:auto; min-height:34px; padding:7px 10px; font-size:13px; }
    .hidden { display:none !important; }
    @media (max-width: 560px) {
      .row, .triple, .quick-grid { grid-template-columns:1fr; }
      main { padding-left:10px; padding-right:10px; }
      .topbar { grid-template-columns:1fr; }
      .status-pill { justify-content:flex-start; }
    }
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <h1>Theseus Hive</h1>
      <div class="muted" id="statusLine">Connecting...</div>
    </div>
    <div class="status-pill"><span id="statusDot" class="dot"></span><span id="statusPill">Offline</span></div>
  </div>
</header>
<main>
  <section id="accessCard" class="card">
    <div class="section-head">
      <h2>Access</h2>
      <button id="refresh" class="compact-button primary">Refresh</button>
    </div>
    <input id="token" placeholder="Hive token" type="password" autocomplete="current-password">
    <div class="row">
      <button id="save">Save Token</button>
      <button id="clearToken" class="warn">Clear Token</button>
    </div>
    <div class="muted" id="accessLine">No user loaded.</div>
  </section>

  <section class="grid">
    <div class="card metric wide"><span>Node</span><strong id="nodeMetric">--</strong></div>
    <div class="card metric"><span>Peers</span><strong id="peerMetric">--</strong></div>
    <div class="card metric"><span>Apple MLX</span><strong id="mlxMetric">--</strong></div>
    <div class="card metric wide"><span>Hive Version</span><strong id="versionMetric">--</strong></div>
    <div class="card metric wide"><span>Transfer</span><strong id="transferMetric">--</strong></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Notifications</h2>
      <button id="ackNotifications" class="compact-button">Mark Read</button>
    </div>
    <div class="grid">
      <div class="metric"><span>Unread</span><strong id="notifyUnreadMetric">--</strong></div>
      <div class="metric"><span>Total</span><strong id="notifyTotalMetric">--</strong></div>
    </div>
    <div id="notificationList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Network</h2>
      <button id="runNetworkDoctor" class="compact-button quick">Run Doctor</button>
    </div>
    <div class="grid">
      <div class="metric"><span>State</span><strong id="networkStateMetric">--</strong></div>
      <div class="metric"><span>Coordinator</span><strong id="networkCoordinatorMetric">--</strong></div>
      <div class="metric"><span>Remote Peers</span><strong id="networkPeersMetric">--</strong></div>
      <div class="metric"><span>Stale</span><strong id="networkStaleMetric">--</strong></div>
    </div>
    <div id="networkList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Roaming</h2>
      <button id="buildRoamingProfile" class="compact-button quick">Profile</button>
    </div>
    <div class="grid">
      <div class="metric"><span>Active</span><strong id="roamingActiveMetric">--</strong></div>
      <div class="metric"><span>Endpoints</span><strong id="roamingEndpointMetric">--</strong></div>
      <div class="metric"><span>Cellular</span><strong id="roamingCellularMetric">--</strong></div>
      <div class="metric"><span>Inbound Only</span><strong id="roamingInboundMetric">--</strong></div>
    </div>
    <div class="row">
      <button id="copyRoamingLink">Copy Import Link</button>
      <button id="copyRoamingJson">Copy JSON</button>
    </div>
    <textarea id="roamingProfile" rows="4" readonly placeholder="Profile output"></textarea>
    <div id="roamingList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Always Active</h2>
      <button id="utilSweep" class="compact-button quick">Sweep</button>
    </div>
    <div class="grid">
      <div class="metric"><span>State</span><strong id="utilStateMetric">--</strong></div>
      <div class="metric"><span>Coverage</span><strong id="utilCoverageMetric">--</strong></div>
      <div class="metric"><span>Queued</span><strong id="utilQueuedMetric">--</strong></div>
      <div class="metric"><span>Blocked</span><strong id="utilBlockedMetric">--</strong></div>
    </div>
    <div class="quick-grid">
      <button id="utilPause" class="warn">Pause</button>
      <button id="utilResume" class="quick">Resume</button>
      <button id="utilStop" class="warn">Stop</button>
      <button id="utilClearStop">Clear Stop</button>
      <button id="utilRunRound" class="quick">Train</button>
    </div>
    <div id="utilNodeList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Solo Learning</h2>
      <button id="soloRefresh" class="compact-button quick">Refresh</button>
    </div>
    <div class="grid">
      <div class="metric"><span>State</span><strong id="soloStateMetric">--</strong></div>
      <div class="metric"><span>MLX</span><strong id="soloMlxMetric">--</strong></div>
      <div class="metric"><span>Promoted</span><strong id="soloPromotedMetric">--</strong></div>
      <div class="metric"><span>Failed</span><strong id="soloFailedMetric">--</strong></div>
    </div>
    <div id="soloList" class="list"></div>
  </section>

  <section class="card">
    <h2>Overnight</h2>
    <div id="overnightList" class="list"></div>
  </section>

  <section class="card">
    <h2>Assistant</h2>
    <div class="grid">
      <div class="metric"><span>State</span><strong id="assistantStateMetric">--</strong></div>
      <div class="metric"><span>VCM</span><strong id="assistantVcmMetric">--</strong></div>
      <div class="metric"><span>Tools</span><strong id="assistantToolMetric">--</strong></div>
      <div class="metric"><span>Code</span><strong id="assistantCodeMetric">--</strong></div>
    </div>
    <div id="assistantList" class="list"></div>
  </section>

  <section class="card">
    <h2>Chat</h2>
    <select id="chatTarget"></select>
    <div class="row">
      <select id="chatIntent">
        <option value="auto">Auto</option>
        <option value="chat">Chat</option>
        <option value="code">Code</option>
        <option value="tool">Tools</option>
        <option value="planning">Planning</option>
      </select>
      <select id="chatFeedback">
        <option value="completed">Completed</option>
        <option value="accepted">Accepted</option>
        <option value="missed">Missed</option>
        <option value="ignored">Ignored</option>
        <option value="corrected">Corrected</option>
      </select>
    </div>
    <textarea id="prompt" rows="4" placeholder="Ask Theseus..."></textarea>
    <button id="sendChat" class="primary">Send</button>
    <div class="quick-grid">
      <button id="markAccepted" class="quick">Accepted</button>
      <button id="markMissed" class="warn">Missed</button>
      <button id="markIgnored">Ignored</button>
      <button id="markCorrected" class="warn">Corrected</button>
      <button id="markCompleted">Completed</button>
    </div>
  </section>

  <section class="card">
    <h2>Task</h2>
    <div class="row">
      <select id="taskKind"></select>
      <select id="taskTarget"></select>
    </div>
    <textarea id="taskPayload" rows="4">{}</textarea>
    <button id="sendTask">Queue Task</button>
  </section>

  <section class="card">
    <h2>Remote Control</h2>
    <div class="row">
      <select id="controlTarget"></select>
      <select id="controlProvider"></select>
    </div>
    <button id="requestControl" class="primary">Open Control Session</button>
    <div id="controlList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Voice Following</h2>
      <button id="refreshVoice" class="compact-button primary">Refresh Route</button>
    </div>
    <div class="grid">
      <div class="metric"><span>Room</span><strong id="voiceRoomMetric">--</strong></div>
      <div class="metric"><span>Route</span><strong id="voiceRouteMetric">--</strong></div>
    </div>
    <button id="voicePresenceTest" class="quick">Mark Heard Here</button>
    <div id="voiceList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Spatial</h2>
      <button id="refreshSpatial" class="compact-button primary">Refresh Scene</button>
    </div>
    <div class="grid">
      <div class="metric"><span>Rooms</span><strong id="spatialRoomMetric">--</strong></div>
      <div class="metric"><span>Nodes</span><strong id="spatialNodeMetric">--</strong></div>
      <div class="metric"><span>Route</span><strong id="spatialRouteMetric">--</strong></div>
      <div class="metric"><span>Storage</span><strong id="spatialStorageMetric">--</strong></div>
    </div>
    <div id="spatialList" class="list"></div>
  </section>

  <section class="card">
    <h2>Quick Actions</h2>
    <div class="quick-grid">
      <button id="quickCheckpoint" class="quick">Checkpoint</button>
      <button id="quickReadiness" class="quick">Readiness</button>
      <button id="quickProbe" class="quick">Probe</button>
      <button id="quickMlxEval" class="quick">MLX Eval</button>
      <button id="quickMlxTrain" class="quick">MLX Train</button>
      <button id="quickMlxRollout" class="quick">MLX Control</button>
    </div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Distributed Training</h2>
      <button id="quickTrainRound" class="compact-button quick">Run Round</button>
    </div>
    <div id="trainingList" class="list"></div>
  </section>

  <section class="card">
    <h2>Teacher Governance</h2>
    <div class="grid">
      <div class="metric"><span>Teacher Share</span><strong id="teacherShareMetric">--</strong></div>
      <div class="metric"><span>Within Cap</span><strong id="teacherCapMetric">--</strong></div>
      <div class="metric"><span>Self Rows</span><strong id="teacherSelfMetric">--</strong></div>
      <div class="metric"><span>Rights</span><strong id="teacherRightsMetric">--</strong></div>
    </div>
    <div id="teacherGovernanceList" class="list"></div>
  </section>

  <section class="card">
    <div class="section-head">
      <h2>Governance Audit</h2>
      <button id="requestGovernanceAudit" class="compact-button primary">Export Audit</button>
    </div>
    <div class="grid">
      <div class="metric"><span>State</span><strong id="governanceAuditStateMetric">--</strong></div>
      <div class="metric"><span>Artifacts</span><strong id="governanceAuditArtifactMetric">--</strong></div>
    </div>
    <div id="governanceAuditList" class="list"></div>
  </section>

  <section class="card">
    <h2>Fleet</h2>
    <div id="fleetList" class="list"></div>
  </section>

  <section class="card">
    <h2>Storage</h2>
    <div class="row">
      <select id="storageShare"></select>
      <input id="storagePath" placeholder="folder path">
    </div>
    <button id="browseStorage">Browse Files</button>
    <div id="storageList" class="list"></div>
    <div id="storagePreview" class="list"></div>
  </section>

  <section class="card">
    <h2>Benchmarks And Games</h2>
    <div id="benchList" class="list"></div>
  </section>

  <section class="card">
    <h2>Output</h2>
    <pre id="out">Ready.</pre>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let state = {};
const params = new URLSearchParams(location.search);
const hashParams = new URLSearchParams(location.hash.replace(/^#/, ""));
const initialToken = params.get("token") || params.get("t") || params.get("operator_token") || params.get("access_token") || hashParams.get("token") || hashParams.get("t") || hashParams.get("operator_token") || hashParams.get("access_token") || localStorage.theseus_hive_token || "";
const nativeMode = params.get("native") === "1" || params.get("app") === "1" || hashParams.get("native") === "1" || hashParams.get("app") === "1";
const operatorSessionId = localStorage.theseus_hive_session_id || (`mobile_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`);
localStorage.theseus_hive_session_id = operatorSessionId;
let lastChatReport = localStorage.theseus_hive_last_chat_report || "";
$("token").value = initialToken;
if (initialToken || location.hash) {
  const clean = new URL(location.href);
  ["token", "t", "operator_token", "access_token", "native", "app"].forEach((key) => clean.searchParams.delete(key));
  clean.hash = "";
  history.replaceState(null, "", clean.pathname + clean.search);
}
if (nativeMode && $("token").value.trim()) {
  $("accessCard").classList.add("hidden");
}
function token(){ return $("token").value.trim(); }
function save(){ localStorage.theseus_hive_token = token(); }
function headers(){ return {"Content-Type":"application/json", "X-Theseus-Hive-Secret": token()}; }
async function api(path, options={}){
  save();
  options.headers = Object.assign(headers(), options.headers || {});
  const res = await fetch(path, options);
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = {ok:false, body:text}; }
  if (!res.ok) data.ok = false;
  return data;
}
function fmt(x){
  if (x === null || x === undefined || x === "") return "--";
  if (typeof x === "number") return x.toFixed(3).replace(/0+$/,"").replace(/\.$/,"");
  return String(x);
}
function esc(x){
  const map = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
  return String(x || "").replace(/[&<>"']/g, (c) => map[c]);
}
function opt(value, label){ return `<option value="${esc(value)}">${esc(label)}</option>`; }
function valueOr(){
  for (let i = 0; i < arguments.length; i += 1) {
    if (arguments[i] !== null && arguments[i] !== undefined && arguments[i] !== "") return arguments[i];
  }
  return 0;
}
async function copyText(value){
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(value);
    return true;
  }
  $("roamingProfile").focus();
  $("roamingProfile").select();
  return document.execCommand("copy");
}
function storageChoices(){
  const nodes = ((state.storage || {}).nodes || []);
  const choices = [];
  nodes.forEach((node) => {
    (node.shares || []).forEach((share) => {
      if (!share || !share.share_id) return;
      const nodeId = node.node_id || "local";
      const nodeName = node.node_name || (node.is_local ? "This node" : nodeId);
      choices.push({
        key: `${nodeId}::${share.share_id}`,
        node_id: nodeId,
        share_id: share.share_id,
        label: `${nodeName} / ${share.name || share.share_id}`,
        accessible: share.accessible !== false,
      });
    });
  });
  return choices;
}
function selectedStorage(){
  const value = $("storageShare").value || "";
  const split = value.indexOf("::");
  if (split < 0) return {node_id:"local", share_id:value};
  return {node_id:value.slice(0, split) || "local", share_id:value.slice(split + 2)};
}
function render(){
  const hive = state.hive || {};
  const local = hive.local_node || {};
  const learning = state.learning || {};
  const hv = ((state.updates || {}).hive_version || {});
  const mlx = (((state.accelerators || {}).apple_mlx) || {});
  const peerCount = hive.peer_count || 0;
  const access = state.access || {};
  const isRunning = !!state.ok;
  const isConnected = isRunning && peerCount > 0;
  $("statusLine").textContent = state.ok ? `hive ${hive.hive_id || "--"} / ${hive.tier || "--"}` : (state.error || "offline");
  $("accessLine").textContent = access.authenticated ? `${access.display_name || access.user_id || "Hive user"} / ${access.role || "member"}` : "Paste a Hive or user token.";
  $("statusPill").textContent = isConnected ? "Connected" : (isRunning ? "Running" : "Offline");
  $("statusDot").className = `dot ${isConnected ? "connected" : (isRunning ? "running" : "")}`;
  $("nodeMetric").textContent = local.node_name || "--";
  $("peerMetric").textContent = `${peerCount}`;
  $("mlxMetric").textContent = mlx.available ? `${mlx.node_count || 1} ready` : "missing";
  $("transferMetric").textContent = fmt(learning.broad_pass_rate || learning.public_pass_rate);
  $("versionMetric").textContent = hv.local_version_id || "--";
  const notifications = state.notifications || {};
  $("notifyUnreadMetric").textContent = fmt(notifications.unread_count);
  $("notifyTotalMetric").textContent = fmt(notifications.notification_count);
  $("ackNotifications").disabled = !notifications.unread_count;
  $("notificationList").innerHTML = (notifications.latest || []).map((note) => `<div class="item"><b>${esc((note.severity || "info").toUpperCase())} ${esc(note.title || "")}</b><br>${esc(note.body || "")}<br>${esc(note.created_utc || "")}</div>`).join("") || "<div class='item'>No active Hive notifications.</div>";
  const network = state.network || {};
  $("networkStateMetric").textContent = network.state || "--";
  $("networkCoordinatorMetric").textContent = network.coordinator_reachable === true ? "reachable" : (network.coordinator_reachable === false ? "blocked" : "--");
  $("networkPeersMetric").textContent = fmt(network.remote_peer_reachable_count);
  $("networkStaleMetric").textContent = fmt(network.stale_peer_count);
  $("networkList").innerHTML = (network.top_findings || []).map((finding) => `<div class="item"><b>${esc(finding.severity || "INFO")} ${esc(finding.code || "")}</b><br>${esc(finding.title || "")}</div>`).join("") || "<div class='item'>Run doctor for fresh LAN, peer, and roaming checks.</div>";
  const roaming = state.roaming || {};
  const roamingEndpoints = roaming.endpoints || [];
  const cellularReady = roamingEndpoints.some((endpoint) => (endpoint.url || "").startsWith("https://"));
  $("roamingActiveMetric").textContent = (monitorActiveEndpoint() || location.host || "--");
  $("roamingEndpointMetric").textContent = fmt(roamingEndpoints.length || (roaming.node_urls || []).length + (roaming.relay_urls || []).length);
  $("roamingCellularMetric").textContent = cellularReady ? "ready" : "tunnel";
  $("roamingInboundMetric").textContent = fmt(network.remote_peer_inbound_only_count || 0);
  $("roamingList").innerHTML = roamingEndpoints.map((endpoint) => {
    const label = endpoint.kind || ((endpoint.url || "").includes(":8793") ? "relay" : "node");
    const transport = endpoint.transport || endpoint.scope || "";
    return `<div class="item"><b>${esc(label)}</b><br>${esc(endpoint.url || "")}<br>${esc(transport)}</div>`;
  }).join("") || "<div class='item'>No roaming endpoints loaded.</div>";
  const util = state.utilization || {};
  const utilSummary = util.summary || {};
  const utilState = util.trigger_state || "--";
  const activeNodes = valueOr(utilSummary.active_or_planned_nodes, util.active_or_planned_nodes, 0);
  const blockedNodes = valueOr(utilSummary.blocked_nodes, util.blocked_nodes, 0);
  const plannedActions = valueOr(utilSummary.planned_actions, util.planned_actions, 0);
  const executedActions = valueOr(utilSummary.executed_actions, util.executed_actions, 0);
  $("utilStateMetric").textContent = utilState;
  $("utilCoverageMetric").textContent = `${activeNodes}/${(util.nodes || []).length || hive.target_count || 1}`;
  $("utilQueuedMetric").textContent = `${plannedActions}/${executedActions}`;
  $("utilBlockedMetric").textContent = `${blockedNodes}`;
  $("utilSweep").disabled = !(hive.allowed_task_kinds || []).includes("utilization_sweep");
  $("utilRunRound").disabled = !(hive.allowed_task_kinds || []).includes("training_orchestrate");
  $("utilNodeList").innerHTML = (util.nodes || []).map((node) => {
    const stateText = node.intended_state || (node.reachable === false ? "unreachable" : "unknown");
    const blockers = (node.resource_blockers || []).concat(node.training_blockers || []).filter(Boolean).join(", ");
    return `<div class="item"><b>${esc(node.node_name || node.node_id)}</b><br>${esc(stateText)} / idle ${fmt(node.idle_slots)} / busy ${fmt(node.busy_slots)}<br>${esc(blockers || node.api_url || "")}</div>`;
  }).join("") || "<div class='item'>No utilization sweep loaded.</div>";
  const solo = state.solo_learning || {};
  $("soloStateMetric").textContent = solo.stopped ? "stopped" : (solo.paused ? "paused" : (solo.state || "--"));
  $("soloMlxMetric").textContent = solo.mlx_available ? "ready" : (solo.mlx_status || "missing");
  $("soloPromotedMetric").textContent = fmt(solo.promoted_arm_count || solo.promotion_count || 0);
  $("soloFailedMetric").textContent = fmt(solo.failed_count || 0);
  const soloRows = [];
  (solo.best_by_arm || []).forEach((arm) => soloRows.push(`<div class="item"><b>${esc(arm.arm_id || "arm")}</b><br>score ${fmt(arm.score)} / ${esc(arm.backend || "")}<br>${esc(arm.activated_artifact || arm.active_manifest || "")}</div>`));
  (solo.recent || []).slice(-4).reverse().forEach((event) => soloRows.push(`<div class="item"><b>${esc(event.task_kind || "worker")}</b><br>${esc(event.promotion_decision || "")} / score ${fmt(event.score)}<br>${esc(event.job_id || event.source_report || "")}</div>`));
  if (!soloRows.length && (solo.next_actions || []).length) {
    soloRows.push(`<div class="item">${esc((solo.next_actions || [])[0])}</div>`);
  }
  $("soloList").innerHTML = soloRows.join("") || "<div class='item'>Run solo status to load local learning state.</div>";
  const recentJobs = (((state.training || {}).recent_jobs) || []).slice(-6).reverse();
  const overnight = ((state.training || {}).overnight) || {};
  const recentSweeps = (util.recent_sweeps || []).slice(-3).reverse();
  const recentTasks = (((state.tasks || {}).recent_results) || []).slice(-6).reverse();
  const overnightRows = [];
  if (overnight.created_utc) {
    overnightRows.push(`<div class="item"><b>Training Window</b><br>workers ${fmt(overnight.worker_report_count)} / promoted ${fmt(overnight.promotion_count)} / failed ${fmt(overnight.failed_count)}<br>${esc((overnight.next_actions || [])[0] || overnight.created_utc)}</div>`);
  }
  recentJobs.forEach((job) => overnightRows.push(`<div class="item"><b>${esc(job.kind || "training")}</b><br>${esc(job.status || "")} / ${esc(job.arm_id || "")}<br>${esc(job.job_id || "")}</div>`));
  recentSweeps.forEach((sweep) => overnightRows.push(`<div class="item"><b>Sweep ${esc(sweep.trigger_state || "")}</b><br>planned ${fmt(sweep.planned_actions)} / executed ${fmt(sweep.executed_actions)} / blocked ${fmt(sweep.blocked_nodes)}<br>${esc(sweep.created_utc || "")}</div>`));
  if (!overnightRows.length) {
    recentTasks.forEach((task) => overnightRows.push(`<div class="item"><b>${esc(task.kind || "task")}</b><br>${esc(task.status || "")} / rc ${fmt(task.returncode)}<br>${esc(((task.job || {}).job_id) || "")}</div>`));
  }
  $("overnightList").innerHTML = overnightRows.join("") || "<div class='item'>No completed work loaded.</div>";
  const assistant = state.assistant || {};
  const latestAssistant = assistant.latest || {};
  const assistantBenchmark = assistant.benchmark || {};
  const statusSource = (hive.local_status_source || {});
  $("assistantStateMetric").textContent = assistant.state || assistant.runtime_state || "--";
  $("assistantVcmMetric").textContent = assistant.vcm_ready ? "ready" : "--";
  $("assistantToolMetric").textContent = assistant.tool_evidence_state ? `${assistant.tool_evidence_state} / ${fmt(assistant.tool_evidence_result_count)}` : "--";
  $("assistantCodeMetric").textContent = assistant.private_code_probe_state ? `${assistant.private_code_probe_state} / ${fmt(assistant.private_code_probe_selected_pass_rate)}` : "--";
  $("assistantList").innerHTML = [
    `<div class="item"><b>Runtime</b><br>${esc(assistant.canonical_runtime || "")}<br>${esc(assistant.local_chat_mode || "")}</div>`,
    `<div class="item"><b>Benchmark</b><br>${esc(assistantBenchmark.latest_public_run || "--")}<br>${fmt(assistantBenchmark.latest_public_score)} / tasks ${fmt(assistantBenchmark.latest_public_task_count)} / ${esc(assistantBenchmark.latest_public_measurement_kind || "")}<br>${esc((assistantBenchmark.latest_public_cards || []).join(", "))}</div>`,
    `<div class="item"><b>Status Freshness</b><br>${esc(statusSource.kind || "unknown")} / age ${fmt(statusSource.age_seconds)}s<br>max ${fmt(statusSource.max_age_seconds)}s</div>`,
    `<div class="item"><b>Latest</b><br>${esc(latestAssistant.trigger_state || "--")} / ${esc(latestAssistant.intent || "--")} / ${esc(latestAssistant.feedback || "--")}<br>history ${fmt(latestAssistant.checkpoint_history_turns_loaded)} / dogfood ${latestAssistant.dogfood_event_written ? "written" : "pending"}</div>`,
  ].join("");
  const targets = [{node_id:"auto", node_name:"Auto"}].concat(hive.targets || []);
  const targetHtml = targets.map((t) => opt(t.node_id, `${t.node_name || t.node_id}${t.is_local ? " local" : ""}`)).join("");
  $("chatTarget").innerHTML = targetHtml;
  $("taskTarget").innerHTML = targetHtml;
  const control = state.remote_control || {};
  const controlNodes = control.nodes || [];
  const controlTargets = controlNodes.length ? controlNodes : (hive.targets || []);
  const previousControlTarget = $("controlTarget").value;
  $("controlTarget").innerHTML = controlTargets.map((t) => opt(t.node_id, `${t.node_name || t.node_id}${t.is_local ? " local" : ""}`)).join("");
  if (previousControlTarget && controlTargets.some((t) => t.node_id === previousControlTarget)) $("controlTarget").value = previousControlTarget;
  const selectedControl = controlTargets.find((t) => t.node_id === $("controlTarget").value) || controlTargets[0] || {};
  const controlProviders = (selectedControl.providers || []).map((p) => p.id).filter(Boolean);
  const providerOptions = ["auto"].concat(Array.from(new Set(controlProviders)));
  $("controlProvider").innerHTML = providerOptions.map((provider) => opt(provider, provider)).join("");
  $("requestControl").disabled = !controlTargets.length;
  $("controlList").innerHTML = controlTargets.map((node) => {
    const providers = (node.providers || []).map((p) => `${p.id}${p.ready ? " ready" : ""}`).join(", ") || "no providers";
    return `<div class="item"><b>${esc(node.node_name || node.node_id)}</b><br>${esc(providers)}<br>${esc(node.api_url || "")}</div>`;
  }).join("") || "<div class='item'>No remote-control provider status yet.</div>";
  const voice = state.voice_following || {};
  const voiceRoute = voice.route || {};
  const listen = voiceRoute.listen_node || {};
  const respond = voiceRoute.respond_node || {};
  $("voiceRoomMetric").textContent = voiceRoute.active_room_name || "idle";
  $("voiceRouteMetric").textContent = respond.node_name ? `${respond.room_name || respond.node_name}` : (voiceRoute.state || "--");
  $("voicePresenceTest").disabled = !voice.enabled;
  const voiceNodes = voice.nodes || [];
  $("voiceList").innerHTML = voiceNodes.map((node) => {
    const bits = [
      node.ready_to_listen ? "listening" : "mic idle",
      node.ready_to_respond ? "speaker" : "no speaker",
      `score ${fmt((node.presence || {}).score)}`,
    ].join(" / ");
    const marker = node.node_id && node.node_id === listen.node_id ? "heard" : (node.node_id && node.node_id === respond.node_id ? "respond" : "room");
    return `<div class="item"><b>${esc(node.room_name || node.node_name || node.node_id)}</b><br>${esc(marker)} / ${esc(bits)}<br>${esc(node.node_name || "")}</div>`;
  }).join("") || "<div class='item'>No voice-following nodes visible.</div>";
  const spatial = state.spatial || {};
  const spatialRoute = spatial.active_voice_route || {};
  const spatialNodes = spatial.nodes || [];
  $("spatialRoomMetric").textContent = fmt(spatial.room_count);
  $("spatialNodeMetric").textContent = fmt(spatial.node_count);
  $("spatialRouteMetric").textContent = spatialRoute.active_room_name || spatialRoute.state || "--";
  $("spatialStorageMetric").textContent = fmt((spatial.nearby_storage || []).length);
  $("spatialList").innerHTML = spatialNodes.map((node) => {
    const room = node.room || {};
    const caps = node.capabilities || {};
    const badges = [
      caps.microphone ? "mic" : "",
      caps.speaker ? "speaker" : "",
      caps.display ? "display" : "",
      caps.storage ? "storage" : "",
      caps.remote_control ? "control" : "",
      (caps.accelerators || []).slice(0,2).join(", "),
    ].filter(Boolean).join(" / ");
    const work = (node.work || {}).summary || "";
    return `<div class="item"><b>${esc(room.name || node.node_name || node.node_id)}</b><br>${esc(node.node_name || "")} / ${esc(work || node.discovery_state || "")}<br>${esc(badges || "operator anchor")}</div>`;
  }).join("") || "<div class='item'>No spatial scene loaded.</div>";
  const allowed = hive.allowed_task_kinds || [];
  $("taskKind").innerHTML = allowed.map((kind) => opt(kind, kind)).join("");
  $("quickCheckpoint").disabled = !allowed.includes("checkpoint_chat");
  $("quickReadiness").disabled = !allowed.includes("readiness_check");
  $("quickProbe").disabled = !allowed.includes("resource_probe");
  $("quickMlxEval").disabled = !(mlx.available && allowed.includes("mlx_eval_chunk"));
  $("quickMlxTrain").disabled = !(mlx.available && allowed.includes("mlx_training_chunk"));
  $("quickMlxRollout").disabled = !(mlx.available && allowed.includes("mlx_rollout_chunk"));
  $("quickTrainRound").disabled = !allowed.includes("training_orchestrate");
  const training = state.training || {};
  const macLocal = training.mac_local || {};
  const trainingRows = [];
  if (macLocal.available) {
    trainingRows.push(`<div class="item"><b>Mac Local ${esc(macLocal.state || "")}</b><br>${esc(macLocal.backend || macLocal.worker_canary || "local canary")} / smoke ${macLocal.bounded_smoke_allowed ? "ready" : "blocked"} / long ${macLocal.long_training_allowed ? "ready" : "blocked"}<br>eval ${fmt(macLocal.eval_accuracy)} / teacher ${String(macLocal.teacher_used === true ? "used" : "free")} / external ${fmt(macLocal.external_inference_calls || 0)}</div>`);
  }
  (training.arms || []).forEach((arm) => trainingRows.push(`<div class="item"><b>${esc(arm.display_name || arm.arm_id)}</b><br>${esc(arm.owner_node_name || "unassigned")} / ${esc(arm.task_kind || "")}<br>${esc(arm.slot_type || "")} / score ${fmt(arm.best_score)}</div>`));
  $("trainingList").innerHTML = trainingRows.join("") || `<div class="item">${esc(training.last_round_id || "No training round planned yet.")}</div>`;
  const teacher = state.teacher_governance || {};
  $("teacherShareMetric").textContent = fmt(teacher.teacher_share_of_accepted_training_rows);
  $("teacherCapMetric").textContent = teacher.teacher_share_within_cap ? "OK" : "review";
  $("teacherSelfMetric").textContent = fmt(teacher.verified_self_generated_rows);
  $("teacherRightsMetric").textContent = `${fmt(teacher.passed_governance_right_fixture_count)}/${fmt(teacher.governance_right_fixture_count)}`;
  const teacherRows = [
    `<div class="item"><b>${esc(teacher.trigger_state || "unknown")} ${esc(teacher.teacher_share_ledger_state || "")}</b><br>teacher ${fmt(teacher.teacher_accepted_rows)} / accepted ${fmt(teacher.accepted_training_rows)} / self ${fmt(teacher.verified_self_generated_rows)}<br>${esc(teacher.trend_state || "")} / ${esc(teacher.teacher_share_target_trend || "")}</div>`,
    `<div class="item"><b>Governance Rights ${esc(teacher.governance_rights_state || "")}</b><br>fixtures ${fmt(teacher.passed_governance_right_fixture_count)}/${fmt(teacher.governance_right_fixture_count)} / constitutional ${fmt(teacher.passed_constitutional_fixture_count)}/${fmt(teacher.constitutional_fixture_count)}<br>runtime external ${fmt(teacher.runtime_external_inference_calls)} / public rows ${fmt(teacher.public_training_rows_written)} / fallback ${fmt(teacher.fallback_return_count)}</div>`,
  ];
  $("teacherGovernanceList").innerHTML = teacher.operator_visible ? teacherRows.join("") : "<div class='item'>Teacher governance report not loaded.</div>";
  const governanceAudit = state.governance_audit || {};
  $("governanceAuditStateMetric").textContent = governanceAudit.trigger_state || "--";
  $("governanceAuditArtifactMetric").textContent = `${fmt(governanceAudit.payload_citation_ok_count)}/${fmt(governanceAudit.payload_citation_applicable_count || governanceAudit.artifact_count)}`;
  $("governanceAuditList").innerHTML = governanceAudit.latest_report ? [
    `<div class="item"><b>${esc(governanceAudit.request_kind || "audit")}</b><br>${esc(governanceAudit.latest_report)}<br>claims ${fmt(governanceAudit.claim_record_count)} / rights ${fmt(governanceAudit.governance_right_record_count)} / clean ${governanceAudit.no_cheat_clean ? "yes" : "review"}</div>`,
    `<div class="item"><b>Boundary</b><br>runtime external ${fmt(governanceAudit.runtime_external_inference_calls)} / public rows ${fmt(governanceAudit.public_training_rows_written)} / fallback ${fmt(governanceAudit.fallback_return_count)}<br>${esc(governanceAudit.non_claim || "")}</div>`,
  ].join("") : "<div class='item'>No live governance audit export has been generated yet.</div>";
  $("fleetList").innerHTML = (hive.targets || []).map((t) => {
    const reach = t.is_local ? "local" : (t.discovery_state || (t.reachable ? "reachable" : "unknown"));
    return `<div class="item"><b>${esc(t.node_name || t.node_id)}</b><br>${esc(reach)} / ${esc(t.api_url || "")}<br>${esc((t.accelerator_ids || []).slice(0,4).join(", ") || "cpu")} / storage ${t.storage_share_count || 0}<br>${esc((t.capability_ids || []).slice(0,6).join(", "))}</div>`;
  }).join("") || "<div class='item'>No peers yet.</div>";
  const shares = storageChoices();
  const currentShare = $("storageShare").value;
  $("storageShare").innerHTML = shares.map((share) => opt(share.key, `${share.label}${share.accessible ? "" : " unavailable"}`)).join("");
  if (currentShare && shares.some((share) => share.key === currentShare)) $("storageShare").value = currentShare;
  if (!shares.length) $("storageList").innerHTML = "<div class='item'>No storage shares configured in reachable Hive nodes.</div>";
  const broad = (((state.benchmarks || {}).broad_cards) || []).slice(0,5).map((c) => `<div class="item"><b>${c.source || c.card_id || c.benchmark || "card"}</b><br>pass ${fmt(c.pass_rate || c.real_public_task_pass_rate || c.score)} / sts ${fmt(c.sts_delta)}</div>`);
  const games = (((state.games || {}).rl_cards) || []).slice(0,4).map((c) => `<div class="item"><b>${c.name || c.card_id || "rl"}</b><br>${c.status || c.state || ""}</div>`);
  $("benchList").innerHTML = broad.concat(games).join("") || "<div class='item'>No benchmark/game cards loaded.</div>";
}
function monitorActiveEndpoint(){
  return (((state.access || {}).active_endpoint) || ((state.roaming || {}).active_endpoint) || "");
}
async function refresh(){
  const data = await api("/api/hive/operator/status");
  state = data;
  render();
  $("out").textContent = JSON.stringify(data, null, 2);
}
$("save").onclick = () => { save(); $("out").textContent = "Token saved on this device."; };
$("clearToken").onclick = () => { $("token").value = ""; localStorage.removeItem("theseus_hive_token"); $("out").textContent = "Token cleared on this device."; };
$("refresh").onclick = refresh;
$("ackNotifications").onclick = async () => {
  const data = await api("/api/hive/operator/notifications/ack", {method:"POST", body:JSON.stringify({all_current:true})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
$("requestGovernanceAudit").onclick = async () => {
  const data = await api("/api/hive/operator/governance-audit", {method:"POST", body:JSON.stringify({request_kind:"audit_export"})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
$("sendChat").onclick = async () => {
  const data = await api("/api/hive/operator/chat", {method:"POST", body:JSON.stringify({
    target_node_id:$("chatTarget").value,
    prompt:$("prompt").value,
    checkpoint_id:"live",
    session_id:operatorSessionId,
    intent:$("chatIntent").value,
    feedback:$("chatFeedback").value
  })});
  if (((data.assistant_runtime || {}).report)) {
    lastChatReport = data.assistant_runtime.report;
    localStorage.theseus_hive_last_chat_report = lastChatReport;
  }
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
async function markAssistantFeedback(outcome){
  const latest = (((state.assistant || {}).latest) || {});
  const artifactRef = lastChatReport || latest.report || "";
  const data = await api("/api/hive/operator/assistant-feedback", {method:"POST", body:JSON.stringify({
    outcome,
    session_id:operatorSessionId,
    artifact_ref:artifactRef
  })});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
}
$("markAccepted").onclick = () => markAssistantFeedback("accepted");
$("markMissed").onclick = () => markAssistantFeedback("missed");
$("markIgnored").onclick = () => markAssistantFeedback("ignored");
$("markCorrected").onclick = () => markAssistantFeedback("corrected");
$("markCompleted").onclick = () => markAssistantFeedback("completed");
$("sendTask").onclick = async () => {
  let payload = {};
  try { payload = JSON.parse($("taskPayload").value || "{}"); } catch { $("out").textContent = "Task payload must be JSON."; return; }
  const data = await api("/api/hive/operator/task", {method:"POST", body:JSON.stringify({target_node_id:$("taskTarget").value, kind:$("taskKind").value, task_payload:payload})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
async function utilizationControl(action){
  const data = await api("/api/hive/operator/utilization", {method:"POST", body:JSON.stringify({action})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
}
$("utilSweep").onclick = () => utilizationControl("sweep");
$("utilPause").onclick = () => utilizationControl("pause");
$("utilResume").onclick = () => utilizationControl("resume");
$("utilStop").onclick = () => utilizationControl("stop");
$("utilClearStop").onclick = () => utilizationControl("clear_stop");
$("utilRunRound").onclick = () => queueQuick("training_orchestrate", {profile:"smoke", source:"mobile_operator", sync_artifacts:true}, "local");
$("soloRefresh").onclick = refresh;
$("runNetworkDoctor").onclick = async () => {
  const data = await api("/api/hive/network-doctor?timeout=1.5");
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
let latestRoamingProfile = {};
$("buildRoamingProfile").onclick = async () => {
  const data = await api("/api/hive/operator/roaming-profile?include_token=1");
  latestRoamingProfile = data;
  $("roamingProfile").value = data.ios_app_url || data.qr_join_url || JSON.stringify(data, null, 2);
  $("out").textContent = JSON.stringify({
    ok: data.ok,
    endpoint_count: ((data.roaming || {}).endpoints || []).length,
    token_included: !!data.operator_token,
    revocation: data.revocation || {},
  }, null, 2);
};
$("copyRoamingLink").onclick = async () => {
  const value = $("roamingProfile").value || latestRoamingProfile.ios_app_url || latestRoamingProfile.qr_join_url || "";
  if (!value) { $("out").textContent = "Build a roaming profile first."; return; }
  await copyText(value);
  $("out").textContent = "Import link copied.";
};
$("copyRoamingJson").onclick = async () => {
  if (!latestRoamingProfile.ok) { $("out").textContent = "Build a roaming profile first."; return; }
  await copyText(JSON.stringify(latestRoamingProfile, null, 2));
  $("out").textContent = "Roaming profile JSON copied.";
};
$("controlTarget").onchange = render;
$("requestControl").onclick = async () => {
  const data = await api("/api/hive/remote-control/session", {method:"POST", body:JSON.stringify({target_node_id:$("controlTarget").value || "local", provider:$("controlProvider").value || "auto", mode:"control"})});
  $("out").textContent = JSON.stringify(data, null, 2);
  const connect = (((data.session || {}).connect || {}).connect_url) || "";
  if (connect) window.location.href = connect;
};
$("refreshVoice").onclick = async () => {
  const data = await api("/api/hive/voice/route");
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
$("refreshSpatial").onclick = async () => {
  const data = await api("/api/hive/spatial/status");
  $("out").textContent = JSON.stringify(data, null, 2);
  if (data.ok) state.spatial = data;
  render();
};
$("voicePresenceTest").onclick = async () => {
  const data = await api("/api/hive/voice/presence", {method:"POST", body:JSON.stringify({score:0.76, source:"mobile_operator_test"})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
};
async function browseStorage(path){
  const selected = selectedStorage();
  if (!selected.share_id) { $("storageList").innerHTML = "<div class='item'>No storage share selected.</div>"; return; }
  if (path !== undefined) $("storagePath").value = path;
  const query = new URLSearchParams({node_id: selected.node_id, share_id: selected.share_id, path: $("storagePath").value || "", limit: "100"});
  const data = await api("/api/hive/storage/peer/browse?" + query.toString());
  if (!data.ok) { $("storageList").innerHTML = `<div class="item">${esc(data.error || "browse failed")}</div>`; $("out").textContent = JSON.stringify(data, null, 2); return; }
  $("storageList").innerHTML = (data.entries || []).map((entry) => {
    const label = entry.is_dir ? "Open" : "View";
    return `<div class="item"><b>${esc(entry.is_dir ? "Folder" : entry.kind || "File")}: ${esc(entry.name)}</b><br>${esc(entry.path)}<br>${entry.size_bytes || ""}<button class="storage-entry" data-path="${esc(entry.path)}" data-dir="${entry.is_dir ? "1" : "0"}">${label}</button></div>`;
  }).join("") || "<div class='item'>This folder is empty.</div>";
  document.querySelectorAll(".storage-entry").forEach((button) => {
    button.onclick = () => button.dataset.dir === "1" ? browseStorage(button.dataset.path || "") : previewStorageFile(button.dataset.path || "");
  });
  $("out").textContent = JSON.stringify({ok:true, node_id: selected.node_id, share_id: selected.share_id, path: data.path, entry_count: data.entry_count}, null, 2);
}
async function previewStorageFile(path){
  const selected = selectedStorage();
  const query = new URLSearchParams({node_id: selected.node_id, share_id: selected.share_id, path, raw: "1"});
  const res = await fetch("/api/hive/storage/peer/file?" + query.toString(), {headers: headers()});
  if (!res.ok) { $("storagePreview").innerHTML = `<div class="item">File could not be opened.</div>`; return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const safePath = esc(path);
  if (blob.type.startsWith("image/")) {
    $("storagePreview").innerHTML = `<div class="item"><b>${safePath}</b><br><img alt="${safePath}" src="${url}" style="max-width:100%;border-radius:8px;margin-top:8px"><br><a href="${url}" download>Download</a></div>`;
  } else {
    $("storagePreview").innerHTML = `<div class="item"><b>${safePath}</b><br>${Math.round(blob.size / 1024)} KB<br><a href="${url}" download>Download</a></div>`;
  }
}
async function queueQuick(kind, payload={}, target="local"){
  const data = await api("/api/hive/operator/task", {method:"POST", body:JSON.stringify({target_node_id:target, kind, task_payload:payload})});
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
}
$("quickCheckpoint").onclick = () => queueQuick("checkpoint_chat", {checkpoint_id:"mobile", prompt:"Mobile operator checkpoint"});
$("quickReadiness").onclick = () => queueQuick("readiness_check", {source:"mobile_operator"});
$("quickProbe").onclick = () => queueQuick("resource_probe", {source:"mobile_operator"});
$("quickMlxEval").onclick = () => {
  const mlx = (((state.accelerators || {}).apple_mlx) || {});
  queueQuick("mlx_eval_chunk", {profile:"smoke", chunk_id:"mobile_mlx_eval", steps:1, eval_limit:128, train_limit:128}, mlx.queue_target || "auto");
};
$("quickMlxTrain").onclick = () => {
  const mlx = (((state.accelerators || {}).apple_mlx) || {});
  queueQuick("mlx_training_chunk", {profile:"smoke", chunk_id:"mobile_mlx_train", steps:4, eval_limit:128, train_limit:256}, mlx.queue_target || "auto");
};
$("quickMlxRollout").onclick = () => {
  const mlx = (((state.accelerators || {}).apple_mlx) || {});
  queueQuick("mlx_rollout_chunk", {profile:"smoke", chunk_id:"mobile_mlx_rollout", epochs:3, cases_per_task:48, eval_cases:48, hv_dim:512, obs_dim:24, seq_len:24}, mlx.queue_target || "auto");
};
$("quickTrainRound").onclick = () => queueQuick("training_orchestrate", {profile:"smoke", source:"mobile_operator", sync_artifacts:true}, "local");
$("browseStorage").onclick = () => browseStorage();
refresh();
</script>
</body>
</html>"""


def operator_webmanifest() -> str:
    return json.dumps(
        {
            "name": "Project Theseus Hive",
            "short_name": "Theseus Hive",
            "description": "Private Hive operator surface.",
            "start_url": "/mobile",
            "scope": "/",
            "display": "standalone",
            "background_color": "#101214",
            "theme_color": "#101214",
            "icons": [
                {"src": "/operator-icon-180.png", "sizes": "180x180", "type": "image/png", "purpose": "any"},
                {"src": "/operator-icon-1024.png", "sizes": "1024x1024", "type": "image/png", "purpose": "any maskable"},
            ],
        },
        indent=2,
    )


def operator_icon_png(route_path: str) -> bytes:
    icon_name = "AppIcon-1024.png" if route_path.endswith("1024.png") else "AppIcon-180.png"
    icon_path = ROOT / "ios" / "TheseusHive" / "TheseusHive" / "Assets.xcassets" / "AppIcon.appiconset" / icon_name
    try:
        return icon_path.read_bytes()
    except OSError:
        return b""
