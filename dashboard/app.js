let state = {};

const $ = (id) => document.getElementById(id);

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) {
      throw new Error(`status ${res.status}`);
    }
    state = await res.json();
    delete state.dashboard_error;
    render();
  } catch (error) {
    state.dashboard_error = `Dashboard reconnecting: ${error.message || error}`;
    renderHeader();
  }
}

function connectEvents() {
  const events = new EventSource("/api/events");
  events.onmessage = (event) => {
    state = JSON.parse(event.data);
    render();
  };
  events.onerror = () => {
    setTimeout(fetchStatus, 2000);
  };
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const text = await res.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (error) {
    payload = { ok: false, error: `Non-JSON response from ${url}`, body: text.slice(0, 500) };
  }
  await fetchStatus();
  if (!res.ok || payload.ok === false) {
    const message = payload.message || payload.error || "Request failed";
    alert(message);
  }
  return payload;
}

function render() {
  renderHeader();
  renderMetrics();
  renderBenchmarks();
  renderQueue();
  renderDaemon();
  renderVIEA();
  renderLearningTruth();
  renderDataInventory();
  renderSyntheticData();
  renderRLRegistry();
  renderOnlineSourceCatalog();
  renderCapabilityMatrix();
  renderLicense();
  renderComputeMarket();
  renderOpenAICompat();
  renderUpdates();
  renderResourceGovernor();
  renderHive();
  renderHiveOperatorOS();
  renderLaunchReadiness();
  renderArmLifecycle();
  renderATTD();
  renderSelfEvolution();
  renderArchitectureExperiments();
  renderArmTransferPlan();
  renderAutoresearchAudit();
  renderBenchmarkAdapterFactory();
  renderLoopClosure();
  renderAutonomousGoal();
  renderCheckpoints();
  renderCheckpointChat();
  renderTeacher();
  renderJobs();
  renderDiscovery();
  renderKnowledgeSources();
  renderContextPackets();
  renderReports();
  drawBenchmarkChart();
  drawHistoryChart();
}

function renderHeader() {
  const s = state.sparkstream || {};
  if (state.dashboard_error) {
    $("status-line").textContent = state.dashboard_error;
    return;
  }
  $("status-line").textContent = `${s.phase || "idle"} - ${s.message || "No active daemon"} - ${state.updated_utc || ""}`;
}

function renderMetrics() {
  const pre = state.preflight || {};
  const cand = state.candidate || {};
  const benches = Array.isArray(state.benchmarks) ? state.benchmarks : [];
  const frontier = benches.find((b) => b.lifecycle === "frontier") || benches[0] || {};
  $("metric-training").textContent = pre.heavy_training_allowed ? "allowed" : "blocked";
  $("metric-training").className = pre.heavy_training_allowed ? "good" : "bad";
  $("metric-candidate").textContent = cand.promote ? "promote" : `${cand.passed || 0}/${cand.total || 0}`;
  $("metric-candidate").className = cand.promote ? "good" : "warn";
  if ($("metric-learning")) {
    const learning = state.learning_scoreboard || {};
    const transfer = learning.public_transfer || {};
    $("metric-learning").textContent = learning.trigger_state
      ? `${learning.trigger_state} ${fmt(transfer.real_public_task_pass_rate)}`
      : "--";
    $("metric-learning").className = learning.trigger_state === "RED" ? "bad" : (learning.trigger_state === "YELLOW" ? "warn" : "good");
  }
  if ($("metric-viea")) {
    const viea = state.viea_autonomy_spine || {};
    const vieaSummary = viea.summary || {};
    $("metric-viea").textContent = viea.trigger_state ? `${viea.trigger_state} ${vieaSummary.feedback_action_count || 0}` : "--";
    $("metric-viea").className = viea.trigger_state === "RED" ? "bad" : (viea.trigger_state === "YELLOW" ? "warn" : "good");
  }
  $("metric-frontier").textContent = frontier.benchmark_name ? `${frontier.benchmark_name}: ${fmt(frontier.score)}` : "--";
  $("metric-teacher").textContent = ((state.teacher || {}).calls || []).length;
  $("metric-data").textContent = ((((state.data_inventory || {}).summary || {}).files) || 0);
  const synthetic = state.synthetic_data || {};
  if ($("metric-synthetic")) {
    $("metric-synthetic").textContent = synthetic.training_ready ? `${synthetic.accepted_count || 0}` : "check";
    $("metric-synthetic").className = synthetic.training_ready ? "good" : "warn";
  }
  $("metric-rl").textContent = ((((state.rl_registry || {}).summary || {}).local_envs) || 0);
  if ($("metric-online-sources")) {
    $("metric-online-sources").textContent = ((((state.online_source_catalog || {}).summary || {}).sources) || 0);
  }
  $("metric-efficiency").textContent = fmt((((state.resource_governor || {}).efficiency || {}).score));
  if ($("metric-hive")) {
    const hive = state.hive || {};
    const peers = ((hive.peers || {}).peer_count) || 0;
    const nodes = (((hive.scheduler || {}).summary || {}).nodes) || (peers ? peers + 1 : 1);
    $("metric-hive").textContent = `${nodes}/${peers}`;
    $("metric-hive").className = peers ? "good" : "warn";
  }
  if ($("metric-operator-os")) {
    const operator = state.hive_operator_os || {};
    const summary = operator.summary || {};
    $("metric-operator-os").textContent = operator.trigger_state
      ? `${operator.trigger_state} ${summary.board_tasks || 0}`
      : "--";
    $("metric-operator-os").className = operator.trigger_state === "RED" ? "bad" : (operator.trigger_state === "YELLOW" ? "warn" : "good");
  }
  if ($("metric-market")) {
    const market = state.compute_market || {};
    const balance = ((market.balances || {}).available_micro_twc) || 0;
    $("metric-market").textContent = `${fmtMicro(balance)} ${(((market.currency || {}).symbol) || "TWC")}`;
    $("metric-market").className = market.exchange_enabled ? "warn" : "good";
  }
  if ($("metric-license")) {
    const license = state.license || {};
    const entitlement = license.entitlement || {};
    $("metric-license").textContent = license.registration_complete ? (entitlement.tier || "registered") : "register";
    $("metric-license").className = license.registration_complete ? "good" : "bad";
  }
  if ($("metric-openai")) {
    const compat = state.openai_compat || {};
    $("metric-openai").textContent = compat.live ? "live" : (compat.enabled ? "enabled" : "off");
    $("metric-openai").className = compat.live ? "good" : (compat.enabled ? "warn" : "bad");
  }
  if ($("metric-updates")) {
    const updates = state.updates || {};
    $("metric-updates").textContent = updates.update_available ? "available" : (updates.installed || {}).active_update_id ? "current" : "check";
    $("metric-updates").className = updates.update_available ? "warn" : "good";
  }
  const launch = state.launch_readiness || {};
  $("metric-readiness").textContent = launch.ready_for_autonomous_training ? "ready" : "check";
  $("metric-readiness").className = launch.ready_for_autonomous_training ? "good" : "warn";
  const arms = ((state.arm_lifecycle_governance || {}).summary || {});
  $("metric-arms").textContent = arms.arms == null ? "--" : `${arms.arms}/${arms.proposal_count || 0}`;
  const context = ((state.context_packets || {}).summary || {});
  if ($("metric-context")) $("metric-context").textContent = context.active_packet_count == null ? "--" : `${context.active_packet_count}/${context.packet_count || 0}`;
  const capability = ((state.capability_matrix || {}).summary || {});
  if ($("metric-capabilities")) {
    $("metric-capabilities").textContent = capability.capabilities == null ? "--" : `${capability.ready_or_active || 0}/${capability.capabilities || 0}`;
    $("metric-capabilities").className = (capability.partial_or_blocked || 0) ? "warn" : "good";
  }
  const evolution = state.self_evolution_governance || {};
  if ($("metric-evolution")) {
    const allowed = (((evolution.teacher_apply || {}).allowed_now));
    const lanes = evolution.lanes || [];
    $("metric-evolution").textContent = allowed ? "apply" : `${lanes.length || 0} lanes`;
    $("metric-evolution").className = allowed ? "warn" : "good";
  }
  const attd = state.attd || {};
  if ($("metric-attd")) {
    $("metric-attd").textContent = attd.trigger_state ? `${attd.trigger_state} ${fmt(attd.attd_score)}` : "--";
    $("metric-attd").className = attd.trigger_state === "RED" ? "bad" : (attd.trigger_state === "YELLOW" ? "warn" : "good");
  }
}

function renderVIEA() {
  if (!$("viea-summary") || !$("viea-list")) return;
  const spine = state.viea_autonomy_spine || {};
  const spineSummary = spine.summary || {};
  const kernel = state.viea_artifact_kernel || {};
  const kernelSummary = kernel.summary || {};
  const executor = state.viea_command_executor || {};
  $("viea-summary").textContent = spine.policy
    ? `${spine.trigger_state} / objects ${kernelSummary.object_count || 0} / command ${executor.trigger_state || "--"} / actions ${spineSummary.feedback_action_count || 0}`
    : "VIEA spine report not found.";
  const spineRows = [
    {
      title: "artifact kernel",
      tag: kernel.trigger_state || "--",
      meta: `${kernelSummary.object_count || 0} objects / ${kernelSummary.relationship_count || 0} relationships`
    },
    {
      title: "latest command execution",
      tag: executor.trigger_state || "--",
      meta: `route ${(executor.route_plan || []).length || 0} steps / packets ${((executor.runtime_packets || []).length) || 0}`
    },
    {
      title: "digital runtime",
      tag: (state.digital_runtime_adapter || {}).trigger_state || "--",
      meta: (((state.digital_runtime_adapter || {}).dashboard_actions) || []).slice(0, 4).join(" / ")
    },
    {
      title: "report map",
      tag: (state.viea_report_map || {}).trigger_state || "--",
      meta: `${(((state.viea_report_map || {}).summary || {}).subsystem_count) || 0} subsystems`
    }
  ];
  $("viea-list").innerHTML = spineRows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("");

  const feedback = state.feedback_action_queue || {};
  const feedbackSummary = feedback.summary || {};
  const actionExecutor = state.viea_action_executor || {};
  const actionExecutorSummary = actionExecutor.summary || {};
  $("viea-actions-summary").textContent = feedback.policy
    ? `${feedback.trigger_state} / ${feedbackSummary.action_count || 0} actions / critical ${feedbackSummary.critical_count || 0} / executor ready ${actionExecutorSummary.ready_action_count || 0} / paused ${String(actionExecutorSummary.paused || false)}`
    : "No feedback action queue yet.";
  $("viea-actions-list").innerHTML = ((feedback.actions || []).slice(0, 12)).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || item.kind || "action")}</span>
        <span class="pill">${escapeHtml(item.priority || "")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(item.action_id || "")}<br>
        ${escapeHtml(item.suggested_action || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Feedback actions appear after the VIEA spine refreshes.</div></div>`;

  const broad = state.broad_transfer_closure || {};
  const broadSummary = broad.summary || {};
  $("viea-transfer-summary").textContent = broad.policy
    ? `${broad.trigger_state} / pass ${fmt(broadSummary.aggregate_pass_rate)} / gap ${fmt(broadSummary.aggregate_floor_gap)} / next ${broadSummary.selected_next_card || "--"}`
    : "No broad transfer closure report yet.";
  $("viea-transfer-list").innerHTML = ((broad.rows || []).slice(0, 8)).map((row) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(row.card_id || "card")}</span>
        <span class="pill">${fmt(row.pass_rate)} / ${escapeHtml((row.blockers || []).length ? "blocked" : "regression")}</span>
      </div>
      <div class="item-meta">
        tasks ${escapeHtml(row.public_task_count)} / STS ${fmt(row.sts_delta)}<br>
        ${escapeHtml(row.next_action || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No broad transfer rows yet.</div></div>`;

  const repo = state.repo_repair_main_curriculum || {};
  const repoSummary = repo.summary || {};
  const repoLearner = state.viea_repo_repair_learner || {};
  const repoLearnerSummary = repoLearner.summary || {};
  $("viea-repo-summary").textContent = repo.policy
    ? `${repo.trigger_state} / tasks ${repoSummary.task_count || 0} / STS ${repoSummary.sts_row_count || 0} / traces ${repoLearnerSummary.validated_private_trace_count || 0} / learner rows ${repoLearnerSummary.code_lm_row_count || 0}`
    : "No repo-repair curriculum report yet.";
  $("viea-repo-list").innerHTML = ((repo.loop || []).map((step) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(step)}</span>
        <span class="pill">loop</span>
      </div>
      <div class="item-meta">private hidden-test repo-repair curriculum stage</div>
    </div>
  `)).join("") || `<div class="item"><div class="item-meta">Repo-repair loop stages will appear after refresh.</div></div>`;

  const sym = state.symliquid_state_engine_queue || {};
  const symEngine = state.symliquid_state_engine || {};
  const symSummary = symEngine.summary || {};
  $("viea-sym-summary").textContent = sym.policy
    ? `${sym.trigger_state} / slots ${((sym.queue || []).length) || 0} / active ${symSummary.active_slot_count || 0} / top ${symSummary.strongest_action_kind || "--"}`
    : "No SymLiquid state-engine queue yet.";
  $("viea-sym-list").innerHTML = ((sym.queue || []).slice(0, 12)).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.capability || "capability")}</span>
        <span class="pill">${escapeHtml(item.status || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.role || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">SymLiquid state slots will appear after refresh.</div></div>`;

  const teacher = state.teacher_architect_closure || {};
  const teacherSummary = teacher.summary || {};
  const teacherRunner = state.teacher_architect_experiment_runner || {};
  const teacherRunnerSummary = teacherRunner.summary || {};
  $("viea-teacher-summary").textContent = teacher.policy
    ? `${teacher.trigger_state} / experiments ${teacherSummary.experiment_count || 0} / runner stages ${teacherRunnerSummary.executed_stage_count || 0} / teacher queue ${String(teacherSummary.teacher_queue_allowed)}`
    : "No teacher architecture closure report yet.";
  $("viea-teacher-list").innerHTML = ((teacher.closures || []).slice(0, 10)).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.id || "experiment")}</span>
        <span class="pill">${escapeHtml(item.teacher_request_status || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.hypothesis || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Teacher architecture closures will appear after refresh.</div></div>`;
}

function renderLearningTruth() {
  if (!$("learning-summary") || !$("learning-list")) return;
  const learning = state.learning_scoreboard || {};
  const publicTransfer = learning.public_transfer || {};
  const privateTraining = learning.private_training || {};
  const promotion = learning.promotion || {};
  const stale = learning.stale_or_superseded_lanes || [];
  $("learning-summary").textContent = learning.trigger_state
    ? `${learning.trigger_state} / public ${fmt(publicTransfer.real_public_task_pass_rate)} / promote ${String(promotion.promotion_allowed)} / stale retired ${stale.length}`
    : "Learning scoreboard not found.";
  const rows = [
    {
      title: "public transfer",
      tag: publicTransfer.promotion_evidence ? "eligible" : "calibration",
      meta: `source ${publicTransfer.candidate_source || "--"} / full-body ${publicTransfer.full_body_public_pass_count || 0} / templates ${publicTransfer.template_like_candidate_count || 0}`
    },
    {
      title: "private training",
      tag: "not promotion",
      meta: `code delta ${fmt(privateTraining.private_pass_rate_delta)} / conversation rows ${privateTraining.open_conversation_train_rows || 0} / conversation STS ${privateTraining.open_conversation_sts_rows || 0}`
    },
    ...stale.map((lane) => ({
      title: lane.lane || "stale lane",
      tag: lane.status || lane.trigger_state || "stale",
      meta: `${lane.why_it_matters || ""} Raw stale lane status is historical context, not active learning truth.`
    }))
  ];
  $("learning-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("");
}

function renderBenchmarks() {
  const list = $("benchmark-list");
  const benches = Array.isArray(state.benchmarks) ? state.benchmarks : [];
  list.innerHTML = benches.map((b) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(b.benchmark_name || "benchmark")}</span>
        <span class="pill">${escapeHtml(b.lifecycle || "")}</span>
      </div>
      <div class="item-meta">
        score ${fmt(b.score)} / residual ${fmt(b.residual)} / wall ${escapeHtml(b.wall_type || "none")}<br>
        threshold ${fmt((((b.graduation_policy || {}).current_threshold)))} / floor ${fmt((((b.graduation_policy || {}).floor_threshold)))}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No benchmark ledger found.</div></div>`;
}

function renderQueue() {
  const list = $("queue-list");
  const items = (((state.autonomy || {}).queue || {}).items || []);
  list.innerHTML = items.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || item.kind || "item")}</span>
        <span class="pill">${escapeHtml(item.priority || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.suggested_action || item.reason || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Queue is empty.</div></div>`;
}

function renderDaemon() {
  const autonomy = state.autonomy || {};
  const status = state.sparkstream || {};
  const ledger = autonomy.daemon_ledger_tail || [];
  $("daemon-summary").textContent = `${status.phase || "idle"} / cycle ${status.cycle || 0} / paused ${autonomy.pause_flag ? "yes" : "no"} / stop ${autonomy.stop_flag ? "yes" : "no"}`;
  $("daemon-list").innerHTML = ledger.slice(-10).reverse().map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.event || "event")}</span>
        <span class="pill">${escapeHtml(item.profile || "")}</span>
      </div>
      <div class="item-meta">
        cycle ${escapeHtml(item.cycle)} / ${escapeHtml(item.created_utc || "")}<br>
        ${item.returncode == null ? "" : `return ${escapeHtml(item.returncode)} / runtime ${escapeHtml(item.runtime_ms)}ms`}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No daemon ledger entries yet.</div></div>`;
}

function renderDataInventory() {
  const inventory = state.data_inventory || {};
  const summary = inventory.summary || {};
  const byRole = summary.by_role || {};
  const conv = state.open_conversation_training_pantry || {};
  const convSummary = conv.summary || {};
  $("data-summary").textContent = `${summary.files || 0} files / ${fmtBytes(summary.bytes || 0)} / ${Object.keys(byRole).length} roles / conversation ${convSummary.private_train_rows || 0} train, ${convSummary.sts_rows || 0} STS`;
  const files = inventory.files || [];
  $("data-list").innerHTML = files.slice(0, 18).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.path || "data")}</span>
        <span class="pill">${escapeHtml(item.role || "")}</span>
      </div>
      <div class="item-meta">
        ${fmtBytes(item.bytes || 0)} / ${item.line_count == null ? "?" : item.line_count} lines<br>
        ${escapeHtml(item.sha256 ? item.sha256.slice(0, 16) : "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Run inventory refresh to populate data files.</div></div>`;
}

function renderSyntheticData() {
  if (!$("synthetic-summary") || !$("synthetic-list")) return;
  const report = state.synthetic_data || {};
  const verification = report.verification || {};
  $("synthetic-summary").textContent = `ready ${report.training_ready ? "yes" : "no"} / accepted ${report.accepted_count || 0} / blend ${fmt(report.blend_synthetic_ratio)} / quality ${fmt(verification.mean_quality_score)}`;
  const checks = verification.checks || [];
  const ruleRows = Object.entries(report.by_rule || {}).slice(0, 8).map(([rule, count]) => ({
    title: rule,
    tag: "rule",
    meta: `${count} accepted examples`
  }));
  const checkRows = checks.map((item) => ({
    title: item.gate || "gate",
    tag: item.passed ? "ok" : "check",
    meta: item.evidence || ""
  }));
  const artifactRows = Object.entries(report.artifacts || {}).map(([name, path]) => ({
    title: name,
    tag: "artifact",
    meta: path
  }));
  $("synthetic-list").innerHTML = [...checkRows, ...ruleRows, ...artifactRows].slice(0, 20).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "synthetic")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No synthetic data curator report yet.</div></div>`;
}

function renderRLRegistry() {
  const registry = state.rl_registry || {};
  const summary = registry.summary || {};
  $("rl-summary").textContent = `${summary.local_envs || 0} local envs / ${summary.puffer_ocean_envs || 0} Puffer-Ocean / ${summary.discovered_candidates || 0} discovered`;
  const recs = registry.recommended_frontier || [];
  const local = (((registry.local_rl_inventory || {}).local_envs) || []).slice(0, 16);
  const discovered = (registry.discovered_candidates || []).slice(-8).reverse();
  $("rl-list").innerHTML = [
    ...recs.map((item) => ({
      title: item.name,
      tag: item.priority || "frontier",
      meta: item.next_step || item.status || ""
    })),
    ...local.map((item) => ({
      title: item.name,
      tag: item.kind || "local",
      meta: `${item.status || ""} ${item.path || ""}`
    })),
    ...discovered.map((item) => ({
      title: item.name,
      tag: item.audit_status || "candidate",
      meta: `${item.license_spdx || ""} ${item.url || ""}`
    }))
  ].map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "rl item")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No RL registry yet.</div></div>`;
}

function renderOnlineSourceCatalog() {
  if (!$("source-catalog-summary") || !$("source-catalog-list")) return;
  const catalog = state.online_source_catalog || {};
  const summary = catalog.summary || {};
  $("source-catalog-summary").textContent = `${summary.sources || 0} sources / ${summary.benchmark_candidates || 0} benchmarks / ${summary.training_data_candidates || 0} data candidates / ${summary.imported_or_present || 0} staged`;
  const rows = [
    ...((catalog.benchmark_candidates || []).map((item) => ({...item, tag: "benchmark"}))),
    ...((catalog.training_data_candidates || []).map((item) => ({...item, tag: "training data"})))
  ];
  $("source-catalog-list").innerHTML = rows.slice(0, 24).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.name || item.id || "source")}</span>
        <span class="pill">${escapeHtml(item.tag || item.category || "")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(item.decision || "")} / ${escapeHtml(item.import_policy || "")} / ${escapeHtml(item.license_spdx || "")}<br>
        ${escapeHtml(item.url || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No online source catalog report yet.</div></div>`;
}

function renderCapabilityMatrix() {
  if (!$("capability-summary") || !$("capability-list")) return;
  const report = state.capability_matrix || {};
  const summary = report.summary || {};
  $("capability-summary").textContent = `${summary.capabilities || 0} capabilities / maturity ${fmt(summary.average_maturity)} / ready+active ${summary.ready_or_active || 0} / differentiated ${summary.differentiated_count || 0} / behind market ${summary.behind_market_count || 0}`;
  const matrix = report.matrix || [];
  const rows = matrix
    .slice()
    .sort((a, b) => Number(a.maturity || 0) - Number(b.maturity || 0))
    .slice(0, 24);
  $("capability-list").innerHTML = rows.map((item) => {
    const gap = (item.gaps || [])[0] || "";
    const action = (item.next_actions || [])[0] || "";
    return `
      <div class="item">
        <div class="item-title">
          <span>${escapeHtml(item.name || item.capability_id || "capability")}</span>
          <span class="pill">${escapeHtml(item.status || "")} ${fmt(item.maturity)}</span>
        </div>
        <div class="item-meta">
          ${escapeHtml(item.market_position || "")}<br>
          Gap: ${escapeHtml(gap)}<br>
          Next: ${escapeHtml(action)}
        </div>
      </div>
    `;
  }).join("") || `<div class="item"><div class="item-meta">No capability matrix yet. Press Refresh or run an autonomy cycle.</div></div>`;
}

function renderResourceGovernor() {
  const resource = state.resource_governor || {};
  const decision = resource.decision || {};
  const gpu = (((resource.current_resources || {}).gpu) || {});
  const disk = (((resource.current_resources || {}).disk) || {});
  $("resource-summary").textContent = `can run ${decision.can_run_requested_profile === false ? "no" : "yes"} / recommended ${decision.recommended_profile || "--"} / owner ${decision.execution_owner || "--"}`;
  const rows = [
    {
      title: "GPU",
      tag: gpu.available === false ? "missing" : "available",
      meta: `${gpu.name || "--"} / free ${fmt(gpu.memory_free_mib)} MiB / util ${fmt(gpu.utilization_gpu_percent)}%`
    },
    {
      title: "Disk",
      tag: "local",
      meta: `${fmt(disk.free_gib)} GiB free at ${disk.root || ""}`
    },
    {
      title: "Throttle",
      tag: decision.can_run_requested_profile === false ? "active" : "clear",
      meta: (decision.throttle_reasons || []).join(", ") || "none"
    },
    {
      title: "Efficiency",
      tag: fmt((((resource.efficiency || {}).score))),
      meta: ((resource.efficiency || {}).objectives || []).slice(0, 4).join(" / ")
    }
  ];
  $("resource-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("");
}

function renderLicense() {
  if (!$("license-summary") || !$("license-list")) return;
  const license = state.license || {};
  const entitlement = license.entitlement || {};
  const feature = license.feature_summary || {};
  const verification = license.license_verification || {};
  $("license-summary").textContent =
    `registration ${license.registration_complete ? "complete" : "missing"} / tier ${entitlement.tier || "--"} / source ${entitlement.source || "--"} / paid ${entitlement.paid ? "yes" : "no"} / nodes ${feature.nodes_used || 0}/${feature.node_limit || entitlement.node_limit || "--"}`;
  const rows = [
    {
      title: "Local Registration",
      tag: license.registration_complete ? "active" : "required",
      meta: `${((license.registration || {}).usage) || "unregistered"} / ${((license.registration || {}).commercial_use) ? "commercial" : "non-commercial"} / ${license.next_action || ""}`
    },
    {
      title: "Signed License",
      tag: license.license_file_present ? (verification.valid ? "valid" : "invalid") : "not installed",
      meta: `${verification.reason || "no_license_file"} / issuer keys ${license.issuer_public_keys_configured ? "configured" : "not configured"}`
    },
    {
      title: "Private Hive",
      tag: feature.can_create_private_hive ? "allowed" : "blocked",
      meta: `friends/family ${feature.can_create_friends_family_hive ? "allowed" : "blocked"}`
    },
    {
      title: "Worker Chunks",
      tag: feature.can_run_worker_chunks ? "allowed" : "blocked",
      meta: "CUDA/MLX distributed training/eval task chunks"
    },
    {
      title: "Company/Public",
      tag: feature.can_create_company_hive || feature.can_operate_public_gateway ? "licensed" : "paid only",
      meta: `company ${feature.can_create_company_hive ? "allowed" : "blocked"} / public gateway ${feature.can_operate_public_gateway ? "allowed" : "blocked"}`
    },
    ...((license.gates || []).map((gate) => ({
      title: gate.name || "gate",
      tag: gate.ok ? "ok" : "blocked",
      meta: gate.detail || ""
    })))
  ];
  $("license-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("");
}

function renderComputeMarket() {
  if (!$("market-summary") || !$("market-list")) return;
  const market = state.compute_market || {};
  const currency = market.currency || {};
  const balances = market.balances || {};
  const summary = market.summary || {};
  const lastQuote = market.last_quote || {};
  $("market-summary").textContent =
    `${market.mode || "accounting"} / ${fmtMicro(balances.available_micro_twc || 0)} ${currency.symbol || "TWC"} available / ${fmtMicro(balances.earned_micro_twc || 0)} earned / exchange ${market.exchange_enabled ? "on" : "off"}`;
  const rows = [
    {
      title: "Wallet",
      tag: (market.wallet || {}).account_id ? "local" : "missing",
      meta: `${((market.wallet || {}).account_id) || "--"} / custodial ${String(((market.wallet || {}).custodial) || false)}`
    },
    {
      title: "Last Quote",
      tag: lastQuote.task_kind || "none",
      meta: lastQuote.quote_id ? `${fmtMicro(lastQuote.gas_estimate_micro_twc || 0)} ${lastQuote.currency_symbol || currency.symbol || "TWC"} / payout ${fmtMicro(lastQuote.provider_payout_micro_twc || 0)}` : "No quote yet."
    },
    {
      title: "Ledger Tail",
      tag: `${summary.ledger_tail_events || 0} events`,
      meta: `gas ${fmtMicro(summary.gas_micro_twc_tail || 0)} / payout ${fmtMicro(summary.provider_payout_micro_twc_tail || 0)} / fees ${fmtMicro(summary.protocol_fee_micro_twc_tail || 0)}`
    },
    {
      title: "Public Accounting",
      tag: (((market.license || {}).can_account_public_work)) ? "allowed" : "gated",
      meta: ((market.license || {}).next_action) || market.next_action || ""
    },
    ...((market.recent_ledger || []).slice(-6).reverse().map((event) => ({
      title: event.task_kind || event.event || "market event",
      tag: event.event || "settled",
      meta: `${fmtMicro(event.gas_micro_twc || 0)} ${currency.symbol || "TWC"} / ${event.created_utc || ""}`
    })))
  ];
  $("market-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("");
}

function renderOpenAICompat() {
  if (!$("openai-summary") || !$("openai-list")) return;
  const compat = state.openai_compat || {};
  $("openai-summary").textContent =
    `${compat.live ? "live" : "stopped"} / ${compat.enabled ? "enabled" : "disabled"} / base ${compat.base_url || "--"} / model ${compat.model || "--"}`;
  if ($("openai-enabled")) $("openai-enabled").checked = Boolean(compat.enabled);
  if ($("openai-host") && compat.host) $("openai-host").value = compat.host;
  if ($("openai-port") && compat.port) $("openai-port").value = compat.port;
  if ($("openai-model") && compat.model) $("openai-model").value = compat.model;
  if ($("openai-checkpoint") && compat.checkpoint_id) $("openai-checkpoint").value = compat.checkpoint_id;
  const rows = [
    {
      title: "Base URL",
      tag: compat.live ? "ready" : "copy when live",
      meta: compat.base_url || "http://127.0.0.1:8789/v1"
    },
    {
      title: "Chat Completions",
      tag: "OpenAI-compatible",
      meta: compat.chat_completions_url || "http://127.0.0.1:8789/v1/chat/completions"
    },
    {
      title: "Models",
      tag: "GET",
      meta: compat.models_url || "http://127.0.0.1:8789/v1/models"
    },
    {
      title: "API Key",
      tag: compat.require_token ? "required" : "not required",
      meta: compat.require_token ? "Use the configured local token." : "Use any placeholder key if the client requires one."
    },
    {
      title: "Teacher",
      tag: compat.allow_teacher ? "blocked by policy" : "off",
      meta: "Endpoint responses stay local and report-grounded."
    },
    {
      title: "License",
      tag: (((compat.license || {}).allowed)) ? "allowed" : "blocked",
      meta: ((compat.license || {}).next_action) || ""
    }
  ];
  $("openai-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("");
}

function renderUpdates() {
  if (!$("updates-summary") || !$("updates-list")) return;
  const updates = state.updates || {};
  const offer = updates.current_offer || {};
  const installed = updates.installed || {};
  const protectedInfo = updates.protected || {};
  const client = updates.client || {};
  const checkin = updates.last_checkin || {};
  const notice = offer.what_users_should_notice || [];
  if ($("updates-mode")) $("updates-mode").value = client.mode || "notify";
  if ($("updates-channel")) $("updates-channel").value = client.channel || "community";
  if ($("updates-track")) $("updates-track").value = client.track || "stable";
  if ($("updates-catalog-url")) $("updates-catalog-url").value = client.catalog_url || "";
  if ($("updates-check-start")) $("updates-check-start").checked = !!client.check_on_start;
  if ($("updates-auto-soft")) $("updates-auto-soft").checked = !!client.auto_install_soft;
  if ($("updates-auto-hard")) $("updates-auto-hard").checked = !!client.auto_install_hard;
  if ($("updates-prerelease")) $("updates-prerelease").checked = !!client.allow_prerelease;
  if ($("updates-offer")) {
    const selected = offer.update_id || "";
    const offers = updates.catalog_offers || [];
    $("updates-offer").innerHTML = `<option value="">latest compatible</option>` + offers.map((item) => {
      const label = `${item.update_id || "update"} ${item.checkpoint_id || ""} ${item.published_utc || ""}`;
      return `<option value="${escapeHtml(item.update_id || "")}">${escapeHtml(label)}</option>`;
    }).join("");
    if (selected) $("updates-offer").value = selected;
  }
  $("updates-summary").textContent = offer.update_id
    ? `${updates.update_available ? "update available" : "installed/current"} / ${client.mode || "notify"} / ${checkin.catalog_source || "local"} / checkpoint ${offer.checkpoint_id || "--"}`
    : "No accepted-candidate update offer yet.";
  const rows = [
    {
      title: "Catalog",
      tag: checkin.catalog_ok ? "ok" : "check",
      meta: `${checkin.created_utc || "never"} / ${client.catalog_url || "local Hive fallback"}`
    },
    {
      title: offer.headline || "Candidate update channel",
      tag: updates.update_available ? "available" : "idle",
      meta: `soft ${String(updates.soft_update_available)} / hard ${String(updates.hard_update_available)} / restart ${String(updates.restart_required)}`
    },
    {
      title: "Installed",
      tag: installed.active_update_id ? "active" : "none",
      meta: `${installed.active_update_id || "--"} / checkpoint ${installed.active_checkpoint_id || "--"}`
    },
    {
      title: "Protection",
      tag: `${(protectedInfo.protected_arms || []).length || 0} arms`,
      meta: `${protectedInfo.protected_path_patterns || 0} protected path patterns; company/local arms are preserved`
    },
    ...notice.map((line) => ({ title: "Better at", tag: "improvement", meta: line })),
    ...((updates.last_events || []).slice(-4).reverse().map((event) => ({
      title: event.kind || "update event",
      tag: (event.created_utc || "").slice(11, 19),
      meta: `${event.update_id || ""} ${event.checkpoint_id || ""}`
    })))
  ];
  $("updates-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("");
}

function renderHive() {
  if (!$("hive-summary") || !$("hive-list")) return;
  const hive = state.hive || {};
  const status = hive.status || {};
  const peers = hive.peers || {};
  const scheduler = hive.scheduler || {};
  const relay = hive.relay || {};
  const publicContribution = hive.public_contribution || {};
  const summary = scheduler.summary || {};
  $("hive-summary").textContent =
    `node ${status.node_name || "--"} / peers ${peers.peer_count || 0} / relay ${relay.port || "--"} / public ${publicContribution.enabled ? "contributing" : "off"} / worker chunks ${summary.real_worker_chunks || 0} / CUDA ${summary.best_cuda_node || "--"} / MLX ${summary.best_mlx_node || "--"}`;
  const localCaps = (status.capabilities || []).map((cap) => ({
    title: cap.id || "capability",
    tag: "local",
    meta: `${cap.detail || ""} / score ${fmt(cap.score)}`
  }));
  const peerRows = (peers.peers || []).map((peer) => ({
    title: peer.node_name || peer.node_id || "peer",
    tag: "peer",
    meta: `${peer.api_url || ""} / ${(peer.capabilities || []).map((cap) => cap.id).join(", ")}`
  }));
  const placementRows = (scheduler.placements || []).slice(0, 10).map((placement) => ({
    title: placement.task_kind || "task",
    tag: placement.target || "placement",
    meta: `${placement.node_name || placement.node_id || "--"} / ${placement.reason || ""}`
  }));
  const chunkRows = (hive.worker_chunks || []).slice(-4).reverse().map((chunk) => ({
    title: chunk.kind || "worker_chunk",
    tag: chunk.ok ? "chunk ok" : "chunk fail",
    meta: `${chunk.backend || "--"} / ${chunk.profile || "--"} / ${chunk.runtime_ms || 0}ms`
  }));
  const relayRows = relay.policy ? [{
    title: "Relay",
    tag: "rendezvous",
    meta: `${relay.host || "0.0.0.0"}:${relay.port || ""} / mobile ${relay.mobile_operator_ui ? "on" : "off"}`
  }] : [];
  const publicRows = publicContribution.policy ? [{
    title: "Public Contribution",
    tag: publicContribution.can_connect_now ? "ready" : "gated",
    meta: `${publicContribution.mode || "off"} / ${publicContribution.next_action || ""}`
  }] : [];
  $("hive-list").innerHTML = [...relayRows, ...publicRows, ...chunkRows, ...localCaps, ...peerRows, ...placementRows].slice(0, 28).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No Hive report yet. Press Probe or Start Hive.</div></div>`;
}

function renderHiveOperatorOS() {
  if (!$("hive-operator-summary") || !$("hive-operator-list")) return;
  const operator = state.hive_operator_os || {};
  const board = state.hive_work_board || operator.work_board || {};
  const boardExecutor = state.hive_work_board_executor || {};
  const channels = state.hive_channel_contract || operator.channels || {};
  const skills = state.hive_skill_registry || operator.skills || {};
  const hooks = state.hive_tool_hooks || operator.tool_hooks || {};
  const background = state.hive_background_tasks || operator.background_tasks || {};
  const goals = state.hive_persistent_goals || operator.persistent_goals || {};
  const safety = state.hive_execution_safety || operator.execution_safety || {};
  const transferScheduler = state.high_transfer_curriculum_scheduler || {};
  const morning = state.hive_morning_report || {};
  const nodeRegistry = state.hive_node_registry || {};
  const overnightProof = state.hive_overnight_proof || {};
  const summary = operator.summary || {};
  $("hive-operator-summary").textContent = operator.policy
    ? `${operator.trigger_state} / channels ${summary.implemented_channels || 0}/${summary.channels || 0} / board ${summary.board_tasks || 0} / active skills ${summary.active_skills || 0}/${summary.skills || 0} / hooks ${summary.hook_targets || 0}`
    : "No Hive operator OS report yet. Press Refresh.";
  const boardSummary = board.summary || {};
  const channelSummary = channels.summary || {};
  const skillSummary = skills.summary || {};
  const hookSummary = hooks.summary || {};
  const bgSummary = background.summary || {};
  const goalSummary = goals.summary || {};
  const safetySummary = safety.summary || {};
  const rows = [
    {
      title: "Shared command channels",
      tag: `${channelSummary.implemented_count || 0}/${channelSummary.channel_count || 0}`,
      meta: `commands ${channelSummary.command_count || 0}; future chat channels stay adapters over the same envelope`
    },
    {
      title: "Durable work board",
      tag: board.trigger_state || "--",
      meta: `${boardSummary.total_tasks || 0} tasks / ready ${boardSummary.ready_or_active || 0} / blocked ${boardSummary.blocked || 0}`
    },
    {
      title: "Work board executor",
      tag: boardExecutor.trigger_state || "--",
      meta: `${((boardExecutor.summary || {}).executed_tasks) || 0} executed / ready ${((boardExecutor.summary || {}).ready_tasks) || 0} / paused ${String(((boardExecutor.summary || {}).paused) || false)}`
    },
    {
      title: "High-transfer curriculum",
      tag: transferScheduler.trigger_state || "--",
      meta: `${((transferScheduler.summary || {}).ready_task_count) || 0} ready concept tasks / checks ${((transferScheduler.summary || {}).donor_receiver_checks) || 0}`
    },
    {
      title: "Node registry",
      tag: nodeRegistry.trigger_state || (nodeRegistry.ok ? "GREEN" : "--"),
      meta: `${((nodeRegistry.summary || {}).trusted_node_count) || 0}/${((nodeRegistry.summary || {}).node_count) || 0} trusted / training ${((nodeRegistry.summary || {}).training_eligible_node_count) || 0} / inference ${((nodeRegistry.summary || {}).best_inference_node) || "--"}`
    },
    {
      title: "Morning report",
      tag: morning.trigger_state || "--",
      meta: `${((morning.summary || {}).improvement_events) || 0} improvements / ${((morning.summary || {}).no_progress_or_failure_events) || 0} residuals`
    },
    {
      title: "Overnight proof",
      tag: overnightProof.trigger_state || "--",
      meta: `unfed ${((overnightProof.summary || {}).unfed_node_count) || 0} / artifacts ${String(((overnightProof.summary || {}).artifact_sync_ok) || false)} / no-cheat ${((overnightProof.summary || {}).public_no_cheat_violations) || 0}`
    },
    {
      title: "Background tasks",
      tag: background.trigger_state || "--",
      meta: `${bgSummary.task_count || 0} watchable jobs / schedules ${bgSummary.schedule_template_count || 0}`
    },
    {
      title: "Persistent goals",
      tag: goals.trigger_state || "--",
      meta: `${goalSummary.goal_count || 0} active or historical goals`
    },
    {
      title: "Hive skills",
      tag: skills.trigger_state || "--",
      meta: `${skillSummary.active_skill_count || 0} active / conflicts ${skillSummary.conflict_count || 0} / stale ${skillSummary.stale_count || 0}`
    },
    {
      title: "Tool hooks",
      tag: hooks.trigger_state || "--",
      meta: `${hookSummary.hook_target_count || 0} targets / live ${hookSummary.live_target_count || 0} / ledger ${hookSummary.ledger_event_count || 0}`
    },
    {
      title: "Execution safety",
      tag: safety.trigger_state || "--",
      meta: `${safetySummary.implemented_count || 0}/${safetySummary.contract_count || 0} implemented / git dirty ${String(safetySummary.git_dirty || false)}`
    },
    ...((skills.active_skill_set || []).slice(0, 8).map((skillId) => ({
      title: skillId,
      tag: "active skill",
      meta: "Loaded by current task terms or core operator role"
    }))),
    ...((board.tasks || []).slice(0, 8).map((task) => ({
      title: task.title || task.task_id,
      tag: task.status || task.priority || "task",
      meta: `${task.kind || ""} / ${task.assignee || ""} / ${task.task_id || ""}`
    })))
  ];
  $("hive-operator-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No operator OS rows yet.</div></div>`;
}

function renderLaunchReadiness() {
  const launch = state.launch_readiness || {};
  const blockers = launch.blocker_failures || [];
  const warnings = launch.warning_failures || [];
  $("readiness-summary").textContent = `training ${launch.ready_for_autonomous_training ? "ready" : "blocked"} / teacher ${launch.ready_for_teacher_enabled_run ? "ready" : "check"} / candidate ${launch.ready_for_candidate_promotion ? "promote" : "not yet"}`;
  const rows = [
    ...blockers.map((item) => ({
      title: item.gate,
      tag: "blocker",
      meta: item.evidence || ""
    })),
    ...warnings.map((item) => ({
      title: item.gate,
      tag: "warning",
      meta: item.evidence || ""
    })),
    ...((launch.candidate_blockers || []).map((gate) => ({
      title: gate,
      tag: "candidate gate",
      meta: "Candidate promotion remains blocked; autonomous training can still continue."
    })))
  ];
  $("readiness-list").innerHTML = rows.slice(0, 18).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "readiness")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No launch readiness report yet. Press Check.</div></div>`;
}

function renderArmLifecycle() {
  const governance = state.arm_lifecycle_governance || {};
  const summary = governance.summary || {};
  const proposals = governance.proposals || [];
  $("arm-summary").textContent = `ready ${governance.ready_for_long_autonomy ? "yes" : "no"} / arms ${summary.arms || 0} / proposals ${summary.proposal_count || 0} / real traces ${summary.real_trace_count || 0}`;
  const rows = proposals.slice(0, 18).map((proposal) => ({
    title: proposal.arm_name || (proposal.arm_names || []).join(", ") || proposal.kind,
    tag: proposal.priority || proposal.kind || "proposal",
    meta: `${proposal.kind || ""}: ${proposal.action || proposal.reason || ""}`
  }));
  $("arm-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "arm")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Arm lifecycle governance has no proposals.</div></div>`;
}

function renderSelfEvolution() {
  if (!$("self-evolution-summary") || !$("self-evolution-list")) return;
  const report = state.self_evolution_governance || {};
  const apply = report.teacher_apply || {};
  const git = report.git_state || {};
  const trigger = apply.triggered_by || {};
  const teacherTraces = ((state.teacher || {}).self_edit_traces || []);
  $("self-evolution-summary").textContent =
    `teacher apply ${apply.allowed_now ? "allowed" : "blocked"} / reason ${trigger.primary_reason || "--"} / branch ${git.branch || "--"} / dirty ${git.dirty ? "yes" : "no"} / repair traces ${teacherTraces.length}`;
  const lanes = report.lanes || [];
  const blockers = (apply.blockers || []).map((item) => ({
    title: item,
    tag: "blocker",
    meta: "Teacher self-edit waits until this clears."
  }));
  const teacherRepairRows = teacherTraces.slice(-3).reverse().map((trace) => ({
    title: trace.trace_id || "teacher repair trace",
    tag: trace.reason || trace.status || "trace",
    meta: `success ${String(trace.success)} / ATTD ${(trace.attd_before || {}).trigger_state || "--"} -> ${(trace.attd_after || {}).trigger_state || "--"}`
  }));
  $("self-evolution-list").innerHTML = [
    ...blockers,
    ...teacherRepairRows,
    ...lanes.map((lane) => ({
      title: lane.id,
      tag: lane.status,
      meta: lane.next_step || lane.purpose || ""
    }))
  ].map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "lane")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No self-evolution governance report yet.</div></div>`;
}

function renderATTD() {
  if (!$("attd-summary") || !$("attd-list")) return;
  const report = state.attd || {};
  const packets = state.attd_maintenance_packets || {};
  const components = report.components || {};
  const governance = report.governance || {};
  $("attd-summary").textContent = report.trigger_state
    ? `${report.trigger_state} / score ${fmt(report.attd_score)} / packets ${packets.packet_count || 0} / growth ${governance.allows_long_autonomy === false ? "blocked" : "allowed"}`
    : "No ATTD report yet.";
  const componentRows = Object.entries(components)
    .filter(([key]) => !["motif_repeated_coverage", "duplicate_density"].includes(key))
    .map(([key, value]) => ({
      title: key,
      tag: fmt(value),
      meta: "ATTD component score"
    }));
  const packetRows = (packets.packets || []).slice(0, 8).map((item) => ({
    title: item.packet_id || item.component,
    tag: item.priority || item.component,
    meta: `${item.bounded_action || ""} ${((item.scope || []).slice(0, 3).join(" / "))}`
  }));
  $("attd-list").innerHTML = [...packetRows, ...componentRows].map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "attd")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">ATTD maintenance packets will appear here.</div></div>`;
}

function renderArchitectureExperiments() {
  if (!$("architecture-summary") || !$("architecture-list")) return;
  const report = state.architecture_experiments || {};
  const rec = report.recommended_next_experiment || {};
  $("architecture-summary").textContent = `change allowed ${report.architecture_change_allowed ? "yes" : "no"} / next ${rec.id || "--"}`;
  const experiments = report.experiments || [];
  $("architecture-list").innerHTML = experiments.slice(0, 12).map((exp) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(exp.id || "experiment")}</span>
        <span class="pill">${escapeHtml(exp.status || "")} ${escapeHtml(exp.kind || "")}</span>
      </div>
      <div class="item-meta">
        score ${escapeHtml(exp.rank_score)} / teacher ${String(exp.teacher_needed)}<br>
        ${escapeHtml(exp.hypothesis || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No architecture experiment governance report yet.</div></div>`;
}

function renderArmTransferPlan() {
  if (!$("arm-transfer-summary") || !$("arm-transfer-list")) return;
  const report = state.arm_transfer_plan || {};
  const summary = report.summary || {};
  $("arm-transfer-summary").textContent = report.policy
    ? `${summary.frontier_family || "--"} / ready ${summary.ready_edges || 0} / blocked ${summary.blocked_edges || 0}`
    : "No arm transfer plan yet.";
  const rows = (report.transfer_plan || []).slice(0, 8);
  $("arm-transfer-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.source_arm || "source")} -> ${escapeHtml(item.target_arm || "target")}</span>
        <span class="pill">${escapeHtml(item.status || "")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(item.hypothesis || "")}<br>
        ${escapeHtml(item.next_action || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Transfer edges will appear after the next autonomy cycle.</div></div>`;
}

function renderAutoresearchAudit() {
  if (!$("autoresearch-summary") || !$("autoresearch-list")) return;
  const report = state.autoresearch_gap_audit || {};
  const summary = report.summary || {};
  $("autoresearch-summary").textContent = report.policy
    ? `${summary.trigger_state || "--"} / ${summary.passed || 0}/${summary.total || 0} checks / gaps ${summary.gap_count || 0} / ledger ${summary.ledger_entries || 0}`
    : "No Autoresearch audit report yet.";
  const gaps = (report.gaps || []).slice(0, 8).map((item) => ({
    title: item.id || "gap",
    tag: item.severity || "gap",
    meta: `${item.description || ""} ${item.next_action || ""}`
  }));
  const checks = (report.checks || [])
    .filter((item) => item.passed === false)
    .slice(0, 6)
    .map((item) => ({
      title: item.gate || "check",
      tag: item.severity || "check",
      meta: item.evidence || ""
    }));
  $("autoresearch-list").innerHTML = [...checks, ...gaps].map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "autoresearch")}</span>
        <span class="pill">${escapeHtml(item.tag || "")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">Autoresearch loop invariants are currently green.</div></div>`;
}

function renderBenchmarkAdapterFactory() {
  if (!$("adapter-summary") || !$("adapter-list")) return;
  const report = state.benchmark_adapter_factory || {};
  const summary = report.summary || {};
  $("adapter-summary").textContent = `${summary.cards || 0} cards / ${summary.ready_cards || 0} ready / ${summary.needs_smoke || 0} need smoke / ${summary.blocked || 0} blocked`;
  const cards = report.cards || [];
  $("adapter-list").innerHTML = cards.slice(0, 20).map((card) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(card.name || card.id || "adapter")}</span>
        <span class="pill">${escapeHtml(card.status || "")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(card.adapter_type || "")} / ${escapeHtml(card.runner_family || "")} / ${escapeHtml(card.priority || "")}<br>
        ${escapeHtml((card.smoke_steps || []).slice(0, 3).join(" / "))}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No adapter factory report yet.</div></div>`;
}

function renderLoopClosure() {
  if (!$("loop-closure-summary") || !$("loop-closure-list")) return;
  const report = state.loop_closure_harvester || {};
  const summary = report.summary || {};
  $("loop-closure-summary").textContent = `${summary.candidates || 0} candidates / ${summary.ready_for_tool_synthesis || 0} ready / traces ${summary.workflow_traces || 0}`;
  const candidates = report.candidates || [];
  $("loop-closure-list").innerHTML = candidates.slice(0, 18).map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.tool_name || "tool")}</span>
        <span class="pill">${escapeHtml(item.status || "")}</span>
      </div>
      <div class="item-meta">
        recurrence ${escapeHtml(item.recurrence_count)} / success ${fmt(item.success_rate)} / runtime ${escapeHtml(item.mean_runtime_ms)}ms<br>
        ${escapeHtml(item.next_action || "")}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No loop closure report yet.</div></div>`;
}

function renderAutonomousGoal() {
  const goal = state.autonomous_goal || {};
  const ledger = (((state.autonomy || {}).goal_ledger_tail) || []).slice(-8).reverse();
  $("goal-result").textContent = goal.goal ? JSON.stringify({
    goal: goal.goal,
    selected_arms: goal.selected_arms,
    outcome: goal.outcome,
    teacher_needed: goal.teacher_needed
  }, null, 2) : "Route a goal to see its arm plan and resource envelope.";
  $("goal-list").innerHTML = ledger.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.goal || item.goal_id || "goal")}</span>
        <span class="pill">${item.ok ? "ok" : "check"}</span>
      </div>
      <div class="item-meta">
        arms ${(item.selected_arms || []).join(", ")}<br>
        efficiency ${fmt(item.efficiency_score)} / teacher ${String(item.teacher_needed)}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No routed goals yet.</div></div>`;
}

function renderCheckpoints() {
  const list = $("checkpoint-list");
  const checkpoints = (((state.checkpoints || {}).checkpoints) || []).slice().reverse();
  const backup = state.checkpoint_backup || {};
  if ($("checkpoint-backup-summary")) {
    $("checkpoint-backup-summary").textContent = backup.status
      ? `backup ${backup.status} / checkpoint ${backup.checkpoint_id || "--"} / promote ${String(backup.candidate_promote)}`
      : "Accepted-candidate backups run only when promotion gates pass.";
  }
  list.innerHTML = checkpoints.slice(0, 20).map((c) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(c.label || c.checkpoint_id)}</span>
        <span class="pill">${escapeHtml(c.snapshot_kind || c.status || "")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(c.checkpoint_id || "")}<br>
        profile ${escapeHtml(c.profile || "")} / promote ${String(c.promote)} / depth ${escapeHtml(c.chain_depth)}
      </div>
      <div class="item-actions">
        <button data-checkpoint-fill="a" data-checkpoint="${escapeHtml(c.checkpoint_id || "")}">A</button>
        <button data-checkpoint-fill="b" data-checkpoint="${escapeHtml(c.checkpoint_id || "")}">B</button>
        <button data-checkpoint-chat="${escapeHtml(c.checkpoint_id || "")}">Chat</button>
        <button data-checkpoint-materialize="${escapeHtml(c.checkpoint_id || "")}">Materialize</button>
        <button data-checkpoint-backup="${escapeHtml(c.checkpoint_id || "")}">Backup</button>
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No checkpoints yet.</div></div>`;
}

function renderCheckpointChat() {
  const chat = state.checkpoint_chat || {};
  const response = chat.response || {};
  const text = response.answer || "";
  $("chat-result").textContent = text ? JSON.stringify(chat, null, 2) : "Ask live state or a checkpoint.";
}

function renderTeacher() {
  const last = ((state.teacher || {}).last || {});
  const response = last.response_json || last.response_text || last.status || "";
  $("teacher-last").textContent = typeof response === "string" ? response : JSON.stringify(response, null, 2);
}

function renderJobs() {
  const list = $("jobs-list");
  const jobs = state.jobs || [];
  list.innerHTML = jobs.map((job) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(job.name || job.job_id)}</span>
        <span class="pill">${job.running ? "running" : "done"}</span>
      </div>
      <div class="item-meta">pid ${job.pid || ""} / return ${job.returncode === null ? "--" : job.returncode}<br>${escapeHtml((job.command || []).join(" "))}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No active jobs.</div></div>`;
}

function renderDiscovery() {
  const seeker = state.benchmark_seeker || {};
  const discovered = seeker.discovered_external_candidates || [];
  const queued = seeker.queued_external_candidates || [];
  const recs = seeker.recommendations || [];
  $("discovered-list").innerHTML = discovered.slice(-12).reverse().map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.name || item.url || "source")}</span>
        <span class="pill">${escapeHtml(item.source_kind || "source")}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.url || "")}<br>${escapeHtml(item.status || "")}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No discovered sources yet.</div></div>`;
  $("seeker-list").innerHTML = [
    ...recs.map((rec) => ({
      title: rec.benchmark || rec.kind || "recommendation",
      meta: rec.action || rec.wall_type || "",
      tag: rec.priority || "rec"
    })),
    ...queued.slice(-8).reverse().map((item) => ({
      title: item.name || item.url,
      meta: item.url,
      tag: item.status || "queued"
    }))
  ].map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title)}</span>
        <span class="pill">${escapeHtml(item.tag)}</span>
      </div>
      <div class="item-meta">${escapeHtml(item.meta)}</div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No seeker recommendations.</div></div>`;
}

function renderKnowledgeSources() {
  if (!$("knowledge-summary") || !$("knowledge-list")) return;
  const registry = state.knowledge_sources || {};
  const sources = registry.sources || [];
  const request = registry.lookup_request || {};
  $("knowledge-summary").textContent = `${sources.length || 0} sources / ${request.status || "idle"}`;
  $("knowledge-list").innerHTML = sources.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.name || "source")}</span>
        <span class="pill">${escapeHtml(item.status || "pending")}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(item.url || "")}<br>
        training ${item.training_use_allowed ? "allowed" : "blocked pending audit"} / fetch ${item.autonomous_fetch_allowed ? "allowed" : "manual gated"}<br>
        ${escapeHtml((item.allowed_uses || []).slice(0, 4).join(" / "))}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No knowledge-source registry yet.</div></div>`;
}

function renderContextPackets() {
  if (!$("context-summary") || !$("context-list")) return;
  const report = state.context_packets || {};
  const summary = report.summary || {};
  $("context-summary").textContent = `${summary.active_packet_count || 0} active / ${summary.summary_packet_count || 0} summaries / ${summary.drop_candidate_count || 0} drop candidates / top ${fmt(summary.top_score)}`;
  const rows = [
    ...((report.summary_packets || []).slice(0, 6).map((item) => ({...item, tag: "summary"}))),
    ...((report.active_packets || []).slice(0, 10).map((item) => ({...item, tag: item.packet_type || "packet"})))
  ];
  $("context-list").innerHTML = rows.map((item) => `
    <div class="item">
      <div class="item-title">
        <span>${escapeHtml(item.title || "packet")}</span>
        <span class="pill">${escapeHtml(item.tag || "")} ${fmt(((item.importance || {}).score))}</span>
      </div>
      <div class="item-meta">
        ${escapeHtml(item.source_path || "")}<br>
        ${escapeHtml(String(item.text || "").slice(0, 420))}
      </div>
    </div>
  `).join("") || `<div class="item"><div class="item-meta">No context packet report yet.</div></div>`;
}

function renderReports() {
  const list = $("reports-list");
  const reports = state.reports || [];
  list.innerHTML = reports.slice(0, 40).map((r) => `
    <div class="item">
      <div class="item-title"><span>${escapeHtml(r.name)}</span></div>
      <div class="item-meta">${Math.round((r.bytes || 0) / 1024)} KB</div>
    </div>
  `).join("");
}

function drawBenchmarkChart() {
  const canvas = $("benchmark-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 1200;
  const height = canvas.clientHeight || 360;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101214";
  ctx.fillRect(0, 0, width, height);
  const benches = (Array.isArray(state.benchmarks) ? state.benchmarks : []).slice(0, 14);
  const pad = 36;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;
  ctx.strokeStyle = "#303942";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad + chartH - (chartH * i / 5);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
    ctx.fillStyle = "#98a4af";
    ctx.fillText((i / 5).toFixed(1), 6, y + 4);
  }
  if (!benches.length) {
    ctx.fillStyle = "#98a4af";
    ctx.fillText("No benchmark data", pad, pad + 20);
    return;
  }
  const gap = 10;
  const barW = Math.max(16, (chartW - gap * (benches.length - 1)) / benches.length);
  benches.forEach((b, i) => {
    const score = Math.max(0, Math.min(1, Number(b.score || 0)));
    const threshold = Math.max(0, Math.min(1, Number(((b.graduation_policy || {}).current_threshold) || 0)));
    const x = pad + i * (barW + gap);
    const barH = chartH * score;
    const y = pad + chartH - barH;
    ctx.fillStyle = b.lifecycle === "frontier" ? "#78d4c8" : "#6f8fb0";
    ctx.fillRect(x, y, barW, barH);
    if (threshold > 0) {
      const ty = pad + chartH - chartH * threshold;
      ctx.strokeStyle = "#f2c14e";
      ctx.beginPath();
      ctx.moveTo(x, ty);
      ctx.lineTo(x + barW, ty);
      ctx.stroke();
    }
    ctx.fillStyle = "#e8edf2";
    ctx.save();
    ctx.translate(x + 4, height - 8);
    ctx.rotate(-0.5);
    ctx.fillText(String(b.benchmark_name || "").slice(0, 24), 0, 0);
    ctx.restore();
  });
}

function drawHistoryChart() {
  const canvas = $("history-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 1200;
  const height = canvas.clientHeight || 320;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101214";
  ctx.fillRect(0, 0, width, height);
  const history = state.history || {};
  const series = history.benchmark_series || {};
  const names = Object.keys(series).slice(0, 6);
  $("history-summary").textContent = `${((history.summary || {}).points) || 0} points`;
  const pad = 36;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;
  ctx.strokeStyle = "#303942";
  for (let i = 0; i <= 5; i++) {
    const y = pad + chartH - (chartH * i / 5);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
    ctx.fillStyle = "#98a4af";
    ctx.fillText((i / 5).toFixed(1), 6, y + 4);
  }
  if (!names.length) {
    ctx.fillStyle = "#98a4af";
    ctx.fillText("No history yet. Run a cycle to append metrics.", pad, pad + 20);
    return;
  }
  const colors = ["#78d4c8", "#f2c14e", "#7ad67a", "#6f8fb0", "#f06b6b", "#c78bd8"];
  names.forEach((name, idx) => {
    const points = (series[name] || []).filter((p) => Number.isFinite(Number(p.score))).slice(-80);
    if (!points.length) return;
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = pad + (points.length === 1 ? chartW : chartW * i / (points.length - 1));
      const y = pad + chartH - chartH * Math.max(0, Math.min(1, Number(p.score || 0)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillText(name.slice(0, 28), pad + 8, pad + 16 + idx * 16);
  });
}

function fmt(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(3) : "--";
}

function fmtMicro(value) {
  const number = Number(value || 0) / 1000000;
  if (!Number.isFinite(number)) return "0";
  if (Math.abs(number) >= 1) return number.toFixed(3);
  return number.toFixed(6).replace(/0+$/, "").replace(/\.$/, "") || "0";
}

function fmtBytes(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  if (num >= 1024 * 1024 * 1024) return `${(num / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  if (num >= 1024 * 1024) return `${(num / (1024 * 1024)).toFixed(1)} MB`;
  if (num >= 1024) return `${(num / 1024).toFixed(1)} KB`;
  return `${num} B`;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

document.addEventListener("click", async (event) => {
  const dataset = event.target && event.target.dataset ? event.target.dataset : {};
  const fill = dataset.checkpointFill || "";
  if (fill) {
    $(fill === "a" ? "checkpoint-a" : "checkpoint-b").value = dataset.checkpoint || "";
    return;
  }
  const checkpointChat = dataset.checkpointChat || "";
  if (checkpointChat) {
    $("chat-checkpoint").value = checkpointChat;
    $("chat-prompt").focus();
    return;
  }
  const materializeId = dataset.checkpointMaterialize || "";
  if (materializeId) {
    const force = window.confirm("Materialize this checkpoint into checkpoints/materialized. Replace that folder if it already exists?");
    const payload = await post("/api/checkpoints/materialize", {
      checkpoint_id: materializeId,
      force
    });
    $("checkpoint-materialize-result").textContent = JSON.stringify(payload, null, 2);
    return;
  }
  const backupId = dataset.checkpointBackup || "";
  if (backupId) {
    const payload = await post("/api/checkpoints/backup", {
      checkpoint_id: backupId,
      provider: "all",
      execute: true
    });
    $("checkpoint-backup-result").textContent = JSON.stringify(payload, null, 2);
    return;
  }
  const action = dataset.action || "";
  if (!action) return;
  const profile = $("profile").value;
  const execute = $("execute").checked;
  const allowNetworkFetch = $("allow-network-fetch").checked;
  const passiveControl = action === "stop_daemon" || action === "pause_daemon" || action === "resume_daemon";
  let confirmLongRun = false;
  let confirmExternalFetch = false;
  if (!passiveControl && execute && (profile === "candidate" || profile === "seed_sweep")) {
    confirmLongRun = window.confirm(`${profile} can run for hours. Start it now?`);
    if (!confirmLongRun) return;
  }
  if (!passiveControl && allowNetworkFetch) {
    confirmExternalFetch = window.confirm("Allow this command to fetch benchmark/data URLs from the network?");
    if (!confirmExternalFetch) return;
  }
  await post("/api/control", {
    action,
    profile,
    execute,
    allow_teacher: $("allow-teacher").checked,
    allow_network_fetch: allowNetworkFetch,
    confirm_long_run: confirmLongRun,
    confirm_external_fetch: confirmExternalFetch,
    duration_hours: $("daemon-duration-hours").value
  });
});

$("checkpoint-create").addEventListener("click", async () => {
  await post("/api/checkpoints/create", {
    kind: $("checkpoint-kind").value,
    label: "dashboard_checkpoint",
    reason: "dashboard_manual",
    profile: $("profile").value,
    status: "recorded"
  });
});

$("checkpoint-compare").addEventListener("click", async () => {
  const payload = await post("/api/checkpoints/compare", {
    a: $("checkpoint-a").value,
    b: $("checkpoint-b").value
  });
  $("checkpoint-compare-result").textContent = JSON.stringify(payload, null, 2);
});

$("checkpoint-backup").addEventListener("click", async () => {
  const payload = await post("/api/checkpoints/backup", {
    provider: "all",
    execute: true
  });
  $("checkpoint-backup-result").textContent = JSON.stringify(payload, null, 2);
});

$("readiness-refresh").addEventListener("click", async () => {
  await post("/api/readiness/run", {
    profile: $("profile").value
  });
});

if ($("viea-action-run")) {
  $("viea-action-run").addEventListener("click", async () => {
    await post("/api/viea/action-executor/run", {
      max_actions: 3,
      max_steps: 8,
      timeout_seconds: 7200,
      allow_teacher: $("allow-teacher").checked
    });
  });
}

if ($("viea-action-pause")) {
  $("viea-action-pause").addEventListener("click", async () => {
    await post("/api/viea/action-executor/pause", {});
  });
}

if ($("viea-action-resume")) {
  $("viea-action-resume").addEventListener("click", async () => {
    await post("/api/viea/action-executor/resume", {});
  });
}

if ($("viea-action-block")) {
  $("viea-action-block").addEventListener("click", async () => {
    const actionId = $("viea-action-block-id").value.trim();
    if (!actionId) {
      alert("Paste an action id first.");
      return;
    }
    await post("/api/viea/action-executor/block", {
      action_id: actionId,
      reason: "dashboard_blocked"
    });
  });
}

if ($("viea-calibration-run")) {
  $("viea-calibration-run").addEventListener("click", async () => {
    await post("/api/viea/broad-calibration/run", {
      min_public_tasks: 32
    });
  });
}

if ($("viea-repo-refresh")) {
  $("viea-repo-refresh").addEventListener("click", async () => {
    await post("/api/viea/repo-repair/refresh", {});
  });
}

if ($("viea-sym-refresh")) {
  $("viea-sym-refresh").addEventListener("click", async () => {
    await post("/api/viea/symliquid/refresh", {});
  });
}

if ($("viea-teacher-request")) {
  $("viea-teacher-request").addEventListener("click", async () => {
    await post("/api/viea/teacher/request", {
      max_experiments: 1,
      max_steps: 2,
      timeout_seconds: 7200,
      allow_teacher: $("allow-teacher").checked
    });
  });
}

$("teacher-submit").addEventListener("click", async () => {
  await post("/api/teacher/ask", {
    reason: $("teacher-reason").value,
    prompt: $("teacher-prompt").value,
    allow_teacher: $("allow-teacher").checked
  });
});

$("chat-live").addEventListener("click", () => {
  $("chat-checkpoint").value = "live";
});

$("chat-submit").addEventListener("click", async () => {
  const payload = await post("/api/chat/checkpoint", {
    checkpoint_id: $("chat-checkpoint").value || "live",
    prompt: $("chat-prompt").value,
    allow_teacher: $("allow-teacher").checked
  });
  $("chat-result").textContent = JSON.stringify(payload, null, 2);
});

$("goal-submit").addEventListener("click", async () => {
  const allowNetworkFetch = $("allow-network-fetch").checked;
  let confirmExternalFetch = false;
  if (allowNetworkFetch) {
    confirmExternalFetch = window.confirm("Allow this goal to use network discovery/import if its route asks for it?");
    if (!confirmExternalFetch) return;
  }
  const payload = await post("/api/goals/run", {
    goal: $("goal-text").value,
    profile: $("profile").value,
    execute: $("execute").checked,
    allow_teacher: $("allow-teacher").checked,
    allow_network_fetch: allowNetworkFetch,
    confirm_external_fetch: confirmExternalFetch
  });
  $("goal-result").textContent = JSON.stringify(payload, null, 2);
});

$("benchmark-add").addEventListener("click", async () => {
  const allowNetworkFetch = $("allow-network-fetch").checked;
  let confirmExternalFetch = false;
  if (allowNetworkFetch) {
    confirmExternalFetch = window.confirm("Fetch this source from the network now? Cancel keeps it queued only.");
    if (!confirmExternalFetch) return;
  }
  await post("/api/benchmarks/add", {
    url: $("benchmark-url").value,
    name: $("benchmark-name").value,
    notes: $("benchmark-notes").value,
    allow_network_fetch: allowNetworkFetch,
    confirm_external_fetch: confirmExternalFetch
  });
});

$("discover-submit").addEventListener("click", async () => {
  const allowNetworkFetch = $("allow-network-fetch").checked;
  if (!allowNetworkFetch) {
    alert("Enable network fetch first. Discovery only queries public sources when you explicitly allow it.");
    return;
  }
  const confirmExternalFetch = window.confirm("Search public dataset sources now? Results stay pending audit.");
  if (!confirmExternalFetch) return;
  await post("/api/benchmarks/discover", {
    query: $("discover-query").value,
    limit: $("discover-limit").value,
    allow_network_fetch: allowNetworkFetch,
    confirm_external_fetch: confirmExternalFetch
  });
});

$("rl-discover-submit").addEventListener("click", async () => {
  const allowNetworkFetch = $("allow-network-fetch").checked;
  if (!allowNetworkFetch) {
    alert("Enable network fetch first. RL discovery only searches public sources when explicitly allowed.");
    return;
  }
  const confirmExternalFetch = window.confirm("Search public RL source repositories now? Results remain license-audited candidates.");
  if (!confirmExternalFetch) return;
  await post("/api/rl/discover", {
    query: $("rl-discover-query").value,
    limit: $("rl-discover-limit").value,
    allow_network_fetch: allowNetworkFetch,
    confirm_external_fetch: confirmExternalFetch
  });
});

if ($("source-catalog-refresh")) {
  $("source-catalog-refresh").addEventListener("click", async () => {
    await post("/api/sources/catalog", {
      import_sources: false
    });
  });
}

if ($("source-catalog-import")) {
  $("source-catalog-import").addEventListener("click", async () => {
    const allowNetworkFetch = $("allow-network-fetch").checked;
    if (!allowNetworkFetch) {
      alert("Enable network fetch first. Catalog imports stage source archives/metadata under ignored data folders.");
      return;
    }
    const confirmExternalFetch = window.confirm("Import approved source archives and dataset metadata now? Unknown licenses and ROM-like assets stay blocked.");
    if (!confirmExternalFetch) return;
    await post("/api/sources/catalog", {
      import_sources: true,
      allow_network_fetch: allowNetworkFetch,
      confirm_external_fetch: confirmExternalFetch,
      max_imports: 12
    });
  });
}

if ($("capability-refresh")) {
  $("capability-refresh").addEventListener("click", async () => {
    await post("/api/capabilities/refresh", {});
  });
}

if ($("license-refresh")) {
  $("license-refresh").addEventListener("click", async () => {
    await post("/api/license/status", {});
  });
}

if ($("license-register")) {
  $("license-register").addEventListener("click", async () => {
    await post("/api/license/register", {
      name: $("license-name").value,
      email: $("license-email").value,
      organization: $("license-organization").value,
      usage: $("license-usage").value,
      seats: Number($("license-seats").value || 1),
      commercial: $("license-commercial").checked,
      accept_terms: $("license-terms").checked
    });
  });
}

if ($("license-request")) {
  $("license-request").addEventListener("click", async () => {
    await post("/api/license/request", {
      features: ["company_hive", "distributed_worker_chunks", "multi_network_company_relay"]
    });
  });
}

if ($("license-import")) {
  $("license-import").addEventListener("click", async () => {
    await post("/api/license/import", {
      license_json: $("license-import-json").value
    });
  });
}

if ($("hive-probe")) {
  $("hive-probe").addEventListener("click", async () => {
    await post("/api/hive/probe", {});
  });
}

if ($("hive-start")) {
  $("hive-start").addEventListener("click", async () => {
    await post("/api/hive/start", {
      port: 8791
    });
  });
}

if ($("hive-relay-start")) {
  $("hive-relay-start").addEventListener("click", async () => {
    await post("/api/hive/relay/start", {
      port: 8793
    });
  });
}

if ($("hive-schedule")) {
  $("hive-schedule").addEventListener("click", async () => {
    await post("/api/hive/schedule", {
      execute: false,
      probe_peers: false,
      worker_chunks: true
    });
  });
}

if ($("hive-operator-refresh")) {
  $("hive-operator-refresh").addEventListener("click", async () => {
    await post("/api/hive/operator-os/refresh", {});
  });
}

if ($("hive-work-board-run")) {
  $("hive-work-board-run").addEventListener("click", async () => {
    await post("/api/hive/work-board/run", {
      max_tasks: 1,
      max_steps: 1,
      timeout_seconds: 21600
    });
  });
}

if ($("hive-high-transfer-schedule")) {
  $("hive-high-transfer-schedule").addEventListener("click", async () => {
    await post("/api/hive/high-transfer/schedule", {});
  });
}

if ($("hive-command-send")) {
  $("hive-command-send").addEventListener("click", async () => {
    const command = ($("hive-command-text").value || "").trim();
    if (!command) return;
    await post("/api/hive/command", {
      command,
      source_channel: "dashboard",
      execute: true,
      max_tasks: 1,
      max_steps: 1,
      timeout_seconds: 21600
    });
  });
}

if ($("market-check")) {
  $("market-check").addEventListener("click", async () => {
    const payload = await post("/api/market/status", {});
    $("market-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("market-quote")) {
  $("market-quote").addEventListener("click", async () => {
    const payload = await post("/api/market/quote", {
      task_kind: "cuda_eval_chunk",
      payload: {
        profile: $("profile").value || "smoke",
        cases_per_task: 4,
        epochs: 1,
        samples_per_launch: 64,
        hv_dim: 512
      },
      provider_node: (((state.hive || {}).status) || {})
    });
    $("market-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("market-settle")) {
  $("market-settle").addEventListener("click", async () => {
    const payload = await post("/api/market/settle", {limit: 50});
    $("market-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("openai-save")) {
  $("openai-save").addEventListener("click", async () => {
    await post("/api/openai/configure", {
      enabled: $("openai-enabled").checked,
      host: $("openai-host").value || "127.0.0.1",
      port: Number($("openai-port").value || 8789),
      model: $("openai-model").value || "theseus-live",
      checkpoint_id: $("openai-checkpoint").value || "live",
      require_token: false
    });
  });
}

if ($("openai-start")) {
  $("openai-start").addEventListener("click", async () => {
    await post("/api/openai/start", {
      host: $("openai-host").value || "127.0.0.1",
      port: Number($("openai-port").value || 8789),
      model: $("openai-model").value || "theseus-live",
      checkpoint_id: $("openai-checkpoint").value || "live"
    });
  });
}

if ($("openai-stop")) {
  $("openai-stop").addEventListener("click", async () => {
    await post("/api/openai/stop", {});
  });
}

if ($("updates-check")) {
  $("updates-check").addEventListener("click", async () => {
    const payload = await post("/api/updates/check", {
      catalog_url: $("updates-catalog-url") ? $("updates-catalog-url").value : "",
      update_id: $("updates-offer") ? $("updates-offer").value : ""
    });
    $("updates-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("updates-save")) {
  $("updates-save").addEventListener("click", async () => {
    const payload = await post("/api/updates/configure", {
      mode: $("updates-mode").value,
      channel: $("updates-channel").value,
      track: $("updates-track").value,
      catalog_url: $("updates-catalog-url").value,
      check_on_start: $("updates-check-start").checked,
      no_check_on_start: !$("updates-check-start").checked,
      auto_install_soft: $("updates-auto-soft").checked,
      no_auto_install_soft: !$("updates-auto-soft").checked,
      auto_install_hard: $("updates-auto-hard").checked,
      no_auto_install_hard: !$("updates-auto-hard").checked,
      allow_prerelease: $("updates-prerelease").checked,
      no_allow_prerelease: !$("updates-prerelease").checked
    });
    $("updates-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("updates-create")) {
  $("updates-create").addEventListener("click", async () => {
    const payload = await post("/api/updates/create", {
      checkpoint_id: $("checkpoint-a").value || "",
      if_promoted: true
    });
    $("updates-result").textContent = JSON.stringify(payload, null, 2);
  });
}

if ($("updates-apply-soft")) {
  $("updates-apply-soft").addEventListener("click", async () => {
    const payload = await post("/api/updates/apply", {
      mode: "soft",
      execute: true
    });
    $("updates-result").textContent = JSON.stringify(payload, null, 2);
  });
}

window.addEventListener("resize", () => {
  drawBenchmarkChart();
  drawHistoryChart();
});
fetchStatus();
connectEvents();
