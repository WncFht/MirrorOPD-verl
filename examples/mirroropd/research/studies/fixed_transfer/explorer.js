const $ = (id) => document.getElementById(id);
const fields = ["student", "view", "metric", "size", "step"];
const RAW_METRICS = Object.fromEntries(
  ["rkl", "log_absorption", "relative", "absolute"].map((key) => [
    key,
    DATA.metrics[key],
  ]),
);
const state = {
  student: "r1",
  view: "phase",
  metric: "log_absorption",
  size: "all",
  step: 320,
  stages: ["sft", "rl", "other"],
  hidden: [],
};
const STAGES = { sft: "追加 SFT", rl: "RL，未另加 SFT", other: "其他对照" };
const epoch = { value: 0 };
let renderTimer;

try {
  const saved = JSON.parse(
    decodeURIComponent(location.hash.slice(1) || "%7B%7D"),
  );
  for (const key of [...fields, "stages", "hidden"])
    if (saved[key] !== undefined) state[key] = saved[key];
} catch {}
if (!DATA.students[state.student]) state.student = "r1";
if (!DATA.views[state.view]) state.view = "phase";
if (!RAW_METRICS[state.metric]) state.metric = "log_absorption";
if (!["all", "same", "cross"].includes(state.size)) state.size = "all";
state.step = Math.max(
  0,
  Math.min(320, Math.round((Number(state.step) || 0) / 16) * 16),
);
if (!Array.isArray(state.stages)) state.stages = ["sft", "rl", "other"];
if (!Array.isArray(state.hidden)) state.hidden = [];

function stage(run) {
  if (run.added_stage === "added SFT") return "sft";
  if (run.added_stage === "RL; no added SFT") return "rl";
  return "other";
}
function group() {
  return DATA.runs
    .filter((run) => run.student === state.student)
    .sort((a, b) => a.display_order - b.display_order);
}
function color(run) {
  return run.color;
}
function allowed(run) {
  return (
    (state.size === "all" || (state.size === "same") === run.same_size) &&
    state.stages.includes(stage(run))
  );
}
function selected() {
  return group().filter(
    (run) => allowed(run) && !state.hidden.includes(run.run_id),
  );
}
function panelName(panel) {
  return panel === "Training scored"
    ? "T"
    : panel === "All held-out"
      ? "Avg"
      : panel;
}
function safeLabel(value) {
  return String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
}
function save() {
  try {
    history.replaceState(
      null,
      "",
      "#" + encodeURIComponent(JSON.stringify(state)),
    );
  } catch {}
}
function setOptions(select, entries, value, label) {
  select.replaceChildren();
  for (const [key, item] of Object.entries(entries)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = label(item);
    select.append(option);
  }
  select.value = value;
}
function initialize() {
  setOptions($("student"), DATA.students, state.student, (label) => label);
  for (const label of [
    ...new Set(Object.values(DATA.views).map((view) => view.group)),
  ]) {
    const options = document.createElement("optgroup");
    options.label = label;
    setOptions(
      options,
      Object.fromEntries(
        Object.entries(DATA.views).filter(([, view]) => view.group === label),
      ),
      "",
      (view) => view.label,
    );
    $("view").append(options);
  }
  $("view").value = state.view;
  setOptions(
    $("metric"),
    DATA.metrics,
    state.metric,
    (metric) => `${metric.symbol} · ${metric.label} (${metric.unit})`,
  );
  $("dataset-badge").textContent = `${DATA.runs.length} 个 run · 单文件离线`;
}
function sync() {
  const fixed = DATA.views[state.view].fixed_metric;
  setOptions(
    $("metric"),
    fixed ? { ...RAW_METRICS, [fixed]: DATA.metrics[fixed] } : RAW_METRICS,
    fixed || state.metric,
    (metric) => `${metric.symbol} · ${metric.label} (${metric.unit})`,
  );
  for (const key of fields) if (key !== "metric") $(key).value = state[key];
  $("metric").disabled = Boolean(DATA.views[state.view].fixed_metric);
  if (DATA.views[state.view].fixed_metric)
    $("metric").value = DATA.views[state.view].fixed_metric;
  for (const input of document.querySelectorAll("[name=stage]"))
    input.checked = state.stages.includes(input.value);
  $("step-value").textContent = state.step;
}
function teacherList() {
  const list = $("teachers");
  list.replaceChildren();
  const term = $("search").value.trim().toLowerCase();
  const visible = group().filter(
    (run) =>
      allowed(run) &&
      `${run.label} ${run.teacher}`.toLowerCase().includes(term),
  );
  $("teacher-count").textContent = `(${selected().length}/${group().length})`;
  for (const run of visible) {
    const row = document.createElement("div");
    row.className =
      "teacher-row" + (state.hidden.includes(run.run_id) ? " hidden-row" : "");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !state.hidden.includes(run.run_id);
    input.setAttribute("aria-label", run.label);
    input.onchange = () => {
      state.hidden = state.hidden.filter((id) => id !== run.run_id);
      if (!input.checked) state.hidden.push(run.run_id);
      update();
    };
    const label = document.createElement("label");
    label.className = "teacher-name";
    const symbol = document.createElement("span");
    symbol.className = "symbol";
    symbol.style.color = color(run);
    symbol.textContent = run.same_size ? "●" : "◆";
    label.append(symbol, document.createTextNode(run.label));
    const meta = document.createElement("span");
    meta.className = "teacher-meta";
    meta.textContent = `${run.same_size ? "同尺寸" : "跨尺寸"} · ${STAGES[stage(run)]} · ${run.budget} step`;
    label.append(meta);
    label.onclick = () => input.click();
    const solo = document.createElement("button");
    solo.className = "solo";
    solo.textContent = "仅看";
    solo.setAttribute("aria-label", "仅看 " + run.label);
    solo.onclick = () => {
      state.hidden = group()
        .filter((item) => item.run_id !== run.run_id)
        .map((item) => item.run_id);
      update();
    };
    row.append(input, label, solo);
    list.append(row);
  }
}
function panelSet() {
  return state.view === "time" ? DATA.panels : DATA.panels.slice(1);
}
function buildCards(panels) {
  const container = $("plots");
  const identity = `${state.view}|${panels.join("|")}`;
  if (container.dataset.panels === identity) return;
  for (const plot of container.querySelectorAll(".plot")) Plotly.purge(plot);
  container.replaceChildren();
  for (const panel of panels) {
    const card = document.createElement("section");
    card.className = "plot-card";
    const heading = document.createElement("div");
    heading.className = "plot-head";
    heading.textContent = panelName(panel);
    const detail = document.createElement("span");
    heading.append(detail);
    const chart = document.createElement("div");
    chart.className = "plot";
    chart.id = `plot-${panels.indexOf(panel)}`;
    card.append(heading, chart);
    container.append(card);
  }
  container.dataset.panels = identity;
}
function bounds(values, includeZero = false) {
  if (!values.length) return [-1, 1];
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (includeZero) {
    low = Math.min(0, low);
    high = Math.max(0, high);
  }
  if (low === high) {
    low -= 0.1;
    high += 0.1;
  }
  const pad = (high - low) * 0.08;
  return [low - pad, high + pad];
}
function layoutBase() {
  return {
    showlegend: false,
    paper_bgcolor: "#fff",
    plot_bgcolor: "#fff",
    hovermode: "closest",
    hoverlabel: { font: { size: 12 }, namelength: -1 },
    font: { family: "Inter, Arial, sans-serif", size: 12, color: "#3b5968" },
    margin: { l: 78, r: 24, t: 20, b: 70 },
  };
}
function rangeAxis(type, range, title) {
  return {
    type,
    range,
    title: { text: title, font: { size: 12 } },
    gridcolor: "#edf1f4",
    zerolinecolor: "#d9e3e9",
    tickfont: { size: 11, color: "#687f8c" },
    automargin: true,
  };
}
function baseTrace(run, panel, points, time) {
  const training = new Map(
    run.panels["Training scored"].map((point) => [point.step, point]),
  );
  const custom = points.map((point) => {
    const q =
      panel === "Training scored"
        ? {
            gap: 0,
            efficiency: point.log_absorption > DATA.min_progress ? 1 : null,
          }
        : run.diagnostics[panel].find((row) => row.step === point.step);
    return [
      point.step,
      point.rkl,
      point.initial,
      point.relative,
      point.absolute,
      point.log_absorption,
      point.prompts,
      training.get(point.step).rkl,
      q.gap,
      q.efficiency,
    ];
  });
  const metric = DATA.metrics[state.metric];
  const suffix = panel === "Training scored" ? "T" : "D";
  return {
    type: "scatter",
    mode: state.view === "endpoint" ? "markers" : "lines+markers",
    name: run.label,
    x: points.map((point) =>
      time ? point.step : training.get(point.step)[state.metric],
    ),
    y: points.map((point) => point[state.metric]),
    customdata: custom,
    line: { color: color(run), width: 1.8, dash: "solid" },
    marker: {
      color: color(run),
      symbol: run.same_size ? "circle" : "diamond",
      size: points.map((_, index) => (index === points.length - 1 ? 11 : 5)),
      line: { color: "#fff", width: 0.6 },
    },
    hovertemplate:
      `<b>${safeLabel(run.label)}</b><br>${run.same_size ? "同尺寸" : "跨尺寸"} · ${STAGES[stage(run)]}` +
      `<br>step %{customdata[0]} / ${run.budget}<br>K_${suffix}: %{customdata[1]:.6g} · K_T: %{customdata[7]:.6g}` +
      `<br>K_${suffix}(0): %{customdata[2]:.6g}<br>R_${suffix}: %{customdata[3]:.3f}% · B_${suffix}: %{customdata[4]:.6g} nats/H1024` +
      `<br>A_${suffix}: %{customdata[5]:.4f} dex · Δ_${suffix}: %{customdata[8]:.4f} dex · η_${suffix}: %{customdata[9]:.4f}` +
      `<br>prompts: %{customdata[6]}<extra></extra>`,
    meta: { metric: metric.symbol },
  };
}
function exportRaw() {
  const keys = [
    "student",
    "teacher",
    "same_size",
    "stage",
    "budget",
    "step",
    "panel",
    "prompts",
    "initial",
    "rkl",
    "relative",
    "absolute",
    "log_absorption",
    "gap",
    "efficiency",
  ];
  const rows = [keys];
  for (const run of selected())
    for (const panel of DATA.panels) {
      let points = run.panels[panel].filter(
        (point) => point.step <= state.step,
      );
      if (state.view === "endpoint") points = points.slice(-1);
      for (const point of points) {
        const diagnostic =
          panel === "Training scored"
            ? {
                gap: 0,
                efficiency: point.log_absorption > DATA.min_progress ? 1 : null,
              }
            : run.diagnostics[panel].find((row) => row.step === point.step);
        rows.push([
          run.student,
          run.teacher,
          run.same_size,
          stage(run),
          run.budget,
          point.step,
          panel,
          point.prompts,
          point.initial,
          point.rkl,
          point.relative,
          point.absolute,
          point.log_absorption,
          diagnostic?.gap ?? "",
          diagnostic?.efficiency ?? "",
        ]);
      }
    }
  downloadCsv(rows, state.step);
}
function downloadCsv(rows, suffix) {
  const csv = rows
    .map((row) =>
      row
        .map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`)
        .join(","),
    )
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `opd-${state.student}-${state.view}-${suffix}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportDiagnostics() {
  const keys = Object.keys(DATA.runs[0].diagnostics[DATA.panels[1]][0]);
  const rows = [
    ["student", "teacher", "same_size", "stage", "budget", ...keys],
  ];
  for (const run of selected())
    for (const panel of DATA.panels.slice(1)) {
      let points = run.diagnostics[panel].filter(
        (point) => point.step <= state.step,
      );
      if (DATA.views[state.view].sampling === "end") points = points.slice(-1);
      for (const point of points)
        rows.push([
          run.student,
          run.teacher,
          run.same_size,
          stage(run),
          run.budget,
          ...keys.map((key) => point[key]),
        ]);
    }
  downloadCsv(rows, state.step);
}
async function standardView(runs, panels, stamp) {
  const time = state.view === "time";
  const log = DATA.metrics[state.metric].scale === "log";
  $("subtitle").textContent =
    state.view === "endpoint"
      ? "每个点 = 更新上限内的最后有效检查点"
      : "每条曲线按实际检查点连接；单条 run 内快照有时间依赖";
  $("view-title").textContent = DATA.metrics[state.metric].label;
  $("formula").textContent = DATA.metrics[state.metric].formula;
  $("view-guide").textContent =
    "原始指标视图。悬停同时显示 K、A、Δ 与 η；η 在训练吸收量不超过 0.01 dex 时留空。";
  const values = runs.flatMap((run) =>
    DATA.panels.flatMap((panel) =>
      run.panels[panel]
        .filter((point) => point.step <= state.step)
        .map((point) => point[state.metric])
        .filter(Number.isFinite),
    ),
  );
  if (!values.length) return;
  const yRange = log ? bounds(values.map(Math.log10)) : bounds(values, true);
  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: {
      format: "png",
      scale: 2,
      filename: `opd-${state.student}-${state.metric}`,
    },
  };
  for (let index = 0; index < panels.length; index++) {
    if (stamp !== epoch.value) return;
    const panel = panels[index];
    const traces = [];
    if (!time) {
      const range = log ? yRange.map((value) => 10 ** value) : yRange;
      traces.push({
        type: "scatter",
        mode: "lines",
        x: range,
        y: range,
        line: { color: "#a5b5bf", dash: "dot", width: 1 },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
    for (const run of runs) {
      let points = run.panels[panel].filter(
        (point) => point.step <= state.step,
      );
      if (state.view === "endpoint") points = points.slice(-1);
      if (points.length) traces.push(baseTrace(run, panel, points, time));
    }
    const metric = DATA.metrics[state.metric];
    const layout = layoutBase();
    layout.xaxis = time
      ? rangeAxis("linear", [0, Math.max(16, state.step)], "Student updates t")
      : rangeAxis(
          log ? "log" : "linear",
          yRange,
          `${metric.symbol}_T (${metric.unit})`,
        );
    layout.yaxis = rangeAxis(
      log ? "log" : "linear",
      yRange,
      `${panelName(panel)} · ${metric.symbol}_D (${metric.unit})`,
    );
    if (time && panel === "Training scored")
      layout.yaxis.title.text = `Training · ${metric.symbol}_T (${metric.unit})`;
    await Plotly.react(`plot-${index}`, traces, layout, config);
    if (stamp !== epoch.value) return;
    $(`plot-${index}`).parentElement.querySelector(
      ".plot-head span",
    ).textContent = time
      ? "固定状态风险 K_D(t)"
      : "同一指标的训练池 / 留出域配对";
  }
}
function diagnosticPoint(run, panel) {
  const rows = run.diagnostics[panel].filter((row) => row.step <= state.step);
  return rows.length ? rows[rows.length - 1] : null;
}
async function diagnosticView(runs, panels, stamp) {
  const view = state.view;
  const metric = DATA.views[view].fixed_metric;
  const meta = DATA.metrics[metric];
  const isDrift = view === "drift";
  const title =
    view === "absorption"
      ? "终点吸收量"
      : view === "efficiency"
        ? "累计迁移效率"
        : view === "windows"
          ? "早期—末窗迁移效率"
          : "累计吸收差随更新";
  $("subtitle").textContent =
    "筛选和步数上限生效；终点取各 run 上限内的最后实际检查点";
  $("view-title").textContent = title;
  $("formula").textContent =
    view === "absorption"
      ? "A_T(e) = log₁₀[K_T(0)/K_T(e)]   ·   A_D(e) = log₁₀[K_D(0)/K_D(e)]"
      : view === "windows"
        ? "η_D[0,64] = ΔA_D[0,64] / ΔA_T[0,64]   ·   η_D[e−64,e] = ΔA_D[e−64,e] / ΔA_T[e−64,e]"
        : meta.formula;
  $("view-guide").textContent =
    view === "absorption"
      ? "每个点为一个 run。距 A_D = A_T 参考线越近，累计吸收越同步。"
      : view === "efficiency"
        ? "每个点为一个 run。η_D = 1 表示两池吸收倍数相同；Δ_D = A_T × (1 − η_D)。"
        : view === "windows"
          ? `早期固定 step 0–64，末窗取实际终点前64步。仅训练改善 > ${DATA.min_progress} dex 时显示比值。`
          : "Δ_D > 0 表示训练累计吸收领先；Δ_D < 0 表示留出累计吸收领先。变化率 dΔ_D/dt 表示两池当前吸收速度之差。";
  const points = [];
  for (const run of runs)
    for (const panel of panels) {
      const rows = isDrift
        ? run.diagnostics[panel].filter((row) => row.step <= state.step)
        : [diagnosticPoint(run, panel)];
      for (const row of rows) if (row) points.push(row);
    }
  const xValues =
    view === "absorption"
      ? points.map((row) => row.train_absorption)
      : view === "windows"
        ? points.map((row) => row.early_efficiency).filter(Number.isFinite)
        : isDrift
          ? points.map((row) => row.step)
          : runs.map((run) => run.display_order + 1);
  const yValues =
    view === "absorption"
      ? points.map((row) => row.held_absorption)
      : view === "efficiency"
        ? points.map((row) => row.efficiency).filter(Number.isFinite)
        : view === "windows"
          ? points.map((row) => row.late_efficiency).filter(Number.isFinite)
          : points.map((row) => row.gap);
  let xRange = bounds(xValues, view !== "drift");
  let yRange = bounds(yValues, view === "drift");
  if (view === "efficiency")
    xRange = [
      0.3,
      Math.max(1.7, ...runs.map((run) => run.display_order + 1.7)),
    ];
  if (view === "drift") xRange = [0, Math.max(16, state.step)];
  if (view === "absorption") {
    const common = bounds([...xValues, ...yValues], true);
    xRange = common;
    yRange = common;
  }
  for (let index = 0; index < panels.length; index++) {
    if (stamp !== epoch.value) return;
    const panel = panels[index];
    const traces = [];
    const missing = [];
    const reference = (x, y) =>
      traces.push({
        type: "scatter",
        mode: "lines",
        x,
        y,
        line: { color: "#a5b5bf", dash: "dot", width: 1 },
        hoverinfo: "skip",
        showlegend: false,
      });
    if (view === "absorption") reference(xRange, xRange);
    if (view === "efficiency" || view === "windows") reference(xRange, [1, 1]);
    if (view === "windows") reference([1, 1], yRange);
    if (isDrift) reference(xRange, [0, 0]);
    for (const run of runs) {
      const rows = isDrift
        ? run.diagnostics[panel].filter((row) => row.step <= state.step)
        : [diagnosticPoint(run, panel)];
      const valid = rows.filter(Boolean);
      const x = [];
      const y = [];
      for (const row of valid) {
        let xv;
        let yv;
        if (view === "absorption") {
          xv = row.train_absorption;
          yv = row.held_absorption;
        } else if (view === "efficiency") {
          xv = run.display_order + 1;
          yv = row.efficiency;
        } else if (view === "windows") {
          xv = row.early_efficiency;
          yv = row.late_efficiency;
        } else {
          xv = row.step;
          yv = row.gap;
        }
        x.push(xv ?? null);
        y.push(yv ?? null);
      }
      const row = valid.at(-1);
      if (
        x.some((value) => value === null) ||
        y.some((value) => value === null)
      )
        missing.push({ run, row });
      const fmt = (value) => (value == null ? "NA" : Number(value).toFixed(4));
      const hoverFields = [
        "train_rkl",
        "held_rkl",
        "train_initial",
        "held_initial",
        "train_absorption",
        "held_absorption",
        "gap",
        "efficiency",
        "early_train",
        "early_held",
        "early_efficiency",
        "late_train",
        "late_held",
        "late_efficiency",
      ];
      const hover = valid.map(
        (q) =>
          `<b>${safeLabel(run.label)}</b><br>${run.same_size ? "同尺寸" : "跨尺寸"} · ${STAGES[stage(run)]}` +
          `<br>step ${q.step}/${run.budget}<br>` +
          hoverFields
            .map((key) => `${DATA.symbols[key]} = ${fmt(q[key])}`)
            .join("<br>") +
          `<br>累计：${DATA.statuses[q.efficiency_status]}<br>末窗：${DATA.statuses[q.late_status]} · 训练题数 ${q.train_prompts}`,
      );
      traces.push({
        type: "scatter",
        mode: isDrift ? "lines+markers" : "markers",
        name: run.label,
        x,
        y,
        connectgaps: false,
        text: hover,
        hovertemplate: "%{text}<extra></extra>",
        line: { color: color(run), dash: "solid", width: 1.8 },
        marker: {
          color: color(run),
          symbol: run.same_size ? "circle" : "diamond",
          size: isDrift ? 5 : 10,
        },
      });
    }
    const card = $(`plot-${index}`).parentElement;
    let note = card.querySelector(".diagnostic-note");
    if (!note) {
      note = document.createElement("details");
      note.className = "diagnostic-note small";
      card.append(note);
    }
    note.replaceChildren();
    note.hidden = !missing.length;
    if (missing.length) {
      const summary = document.createElement("summary");
      summary.textContent = `${missing.length} 个缺失比值 · 展开看原因`;
      note.append(summary);
      for (const { run, row } of missing) {
        const item = document.createElement("p");
        const status =
          view === "efficiency"
            ? row.efficiency_status
            : row.early_status !== "eligible"
              ? row.early_status
              : row.late_status;
        item.textContent =
          `${run.label} · step ${row.step}：${DATA.statuses[status]}；` +
          `A_T=${row.train_absorption.toFixed(4)}；早窗 ΔA_T/ΔA_D=${row.early_train?.toFixed(4) ?? "NA"}/${row.early_held?.toFixed(4) ?? "NA"}；` +
          `末窗 ΔA_T/ΔA_D=${row.late_train?.toFixed(4) ?? "NA"}/${row.late_held?.toFixed(4) ?? "NA"}`;
        note.append(item);
      }
    }
    const layout = layoutBase();
    layout.margin.b = view === "efficiency" ? 110 : 68;
    layout.xaxis = rangeAxis(
      "linear",
      xRange,
      view === "absorption"
        ? "Training · A_T (dex)"
        : view === "efficiency"
          ? "Teacher (legend order)"
          : view === "windows"
            ? "Early · η_D[0,64]"
            : "Student updates t",
    );
    layout.yaxis = rangeAxis(
      "linear",
      yRange,
      view === "absorption"
        ? `${panelName(panel)} · A_D (dex)`
        : view === "efficiency"
          ? `${panelName(panel)} · η_D`
          : view === "windows"
            ? `${panelName(panel)} · η_D[e−64,e]`
            : `${panelName(panel)} · Δ_D (dex)`,
    );
    if (view === "efficiency") {
      layout.xaxis.tickmode = "array";
      layout.xaxis.tickvals = runs.map((run) => run.display_order + 1);
      layout.xaxis.ticktext = runs.map((run) => run.teacher);
      layout.xaxis.tickangle = -35;
    }
    await Plotly.react(`plot-${index}`, traces, layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
      toImageButtonOptions: {
        format: "png",
        scale: 2,
        filename: `opd-${state.student}-${view}-${panel}`,
      },
    });
    if (stamp !== epoch.value) return;
    card.querySelector(".plot-head span").textContent = title;
  }
}
async function draw() {
  const stamp = ++epoch.value;
  const runs = selected();
  const panels = panelSet();
  buildCards(panels);
  $("empty").hidden = runs.length > 0;
  $("plots").hidden = runs.length === 0;
  $("run-count").textContent = runs.length;
  $("snapshot-count").textContent = runs.reduce(
    (n, run) =>
      n +
      run.panels["Training scored"].filter((point) => point.step <= state.step)
        .length,
    0,
  );
  $("heading").textContent = DATA.students[state.student];
  const short = runs.filter((run) => run.budget < state.step).length;
  $("reading").textContent =
    `A 与 Δ 的单位是 dex；η 无量纲，不是知识吸收百分比。` +
    `比值要求相应训练吸收增量 > ${DATA.min_progress} dex。当前 ${short} 条 run 的实际终点早于更新上限；` +
    "比较相同预算可将滑块设为64。原始 CSV 字段名保持不变，符号对应关系见说明文档。";
  if (DATA.views[state.view].diagnostic)
    await diagnosticView(runs, panels, stamp);
  else await standardView(runs, panels, stamp);
}
function update() {
  ++epoch.value;
  sync();
  teacherList();
  save();
  clearTimeout(renderTimer);
  renderTimer = setTimeout(
    () =>
      draw().catch((error) => {
        console.error(error);
        $("empty").hidden = false;
        $("empty").textContent = "绘图失败：" + error.message;
      }),
    35,
  );
}
for (const field of fields)
  $(field).addEventListener(field === "step" ? "input" : "change", () => {
    state[field] = field === "step" ? Number($(field).value) : $(field).value;
    update();
  });
for (const input of document.querySelectorAll("[name=stage]"))
  input.onchange = () => {
    state.stages = [...document.querySelectorAll("[name=stage]:checked")].map(
      (item) => item.value,
    );
    update();
  };
$("search").oninput = teacherList;
$("show-all").onclick = () => {
  state.hidden = [];
  update();
};
$("hide-all").onclick = () => {
  state.hidden = group().map((run) => run.run_id);
  update();
};
$("reset").onclick = () => {
  Object.assign(state, {
    size: "all",
    step: 320,
    stages: ["sft", "rl", "other"],
    hidden: [],
  });
  $("search").value = "";
  update();
};
$("export").onclick = () =>
  DATA.views[state.view].diagnostic ? exportDiagnostics() : exportRaw();
initialize();
update();
