/* Standalone report controller: measured data -> controls, SVG charts and generic canvas views. */
(async () => {
  "use strict";
  const embedded = document.getElementById("report-data");
  const D = embedded
      ? JSON.parse(embedded.textContent)
      : await fetch("api/report").then((response) => {
          if (!response.ok) throw Error(`Report request failed: ${response.status}`);
          return response.json();
        }),
    ids = Object.keys(D.pairs);
  const $ = (id) => document.getElementById(id),
    esc = (s) =>
      String(s).replace(
        /[&<>"']/g,
        (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
      );
  const fmt = (v, n = 3) => Number(v).toFixed(n),
    pct = (v, n = 1) => `${fmt(100 * v, n)}%`,
    color = ["#4F8FA5", "#EE995B", "#C95B5B"];
  let pair = ids.includes(location.hash.slice(1)) ? location.hash.slice(1) : ids[0],
    view = null,
    explorer = null,
    occupancyView = null,
    overviewView = null,
    hoverPoints = [];
  const spec = () => D.pairs[pair],
    group = (id) => spec().groups.find((g) => g.id === id),
    value = (domain, bank, which) => {
      const rows = D.risks
        .filter((r) => r.pair === pair && r.domain === domain && r.bank === bank)
        .sort((a, b) => a.steps - b.steps);
      return rows[which === "first" ? 0 : rows.length - 1];
    };
  const cell = (label, body) => `<dl class="setting-item"><dt>${esc(label)}</dt><dd>${esc(body)}</dd></dl>`;
  const stat = (label, number, sub) =>
    `<div class="summary-card"><div class="label">${esc(label)}</div><div class="value">${esc(number)}</div><div class="sub">${esc(sub)}</div></div>`;
  const swatch = (g) => `<span class="swatch" style="background:${g.color}"></span>${esc(g.label)}`;
  const table = (headers, rows) =>
    `<thead><tr>${headers.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody>`;
  const download = (name, body, type) => {
    const url = URL.createObjectURL(new Blob([body], { type })),
      a = document.createElement("a");
    a.download = name;
    a.href = url;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  function selectPair(next) {
    if (!ids.includes(next)) return;
    pair = next;
    history.replaceState(null, "", `#${pair}`);
    for (const tab of $("pair-tabs").children) {
      const active = tab.dataset.pair === pair;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    const s = spec(),
      c = s.settings,
      m = s.metrics,
      initial = value("math", "public_initial", "last"),
      teacher = value("math", "public_teacher", "last");
    $("pair-content").setAttribute("aria-labelledby", `tab-${pair}`);
    $("pair-label").textContent = `${pair} / STUDENT → TEACHER`;
    $("pair-name").textContent = `${c.student.split("/").pop()} → ${c.teacher.split("/").pop()}`;
    $("pair-context").textContent = `40 outer · 640 updates · seed ${c.seed}`;
    $("summary-cards").innerHTML =
      stat("Math RKL 降低 · Initial 回答", pct(initial.reduction_percent / 100, 2), `step${initial.steps}`) +
      stat("Math RKL 降低 · Teacher 回答", pct(teacher.reduction_percent / 100, 2), `step${teacher.steps}`) +
      stat(
        "状态参考",
        m.reference_states.toLocaleString(),
        `${s.selected_training.toLocaleString()} 条训练回答 · PCA50`,
      );
    const template = Object.hasOwn(c.template, "enable_thinking")
      ? `enable_thinking=${c.template.enable_thinking}`
      : `${c.family} 原模板；无 thinking 开关`;
    const audit = s.prompt_audit
      ? `128 题；与 ${s.prompt_audit.train_unique_prompts.toLocaleString()} 道训练候选精确 prompt 重叠 ${s.prompt_audit.overlap}`
      : "128 题，来自已有固定评估面板";
    $("settings").innerHTML = [
      cell("Student", c.student),
      cell("Teacher / 状态编码器", c.teacher),
      cell("输入模板", template),
      cell("训练问题", `DAPO Math；${c.training_questions} 次新题访问；current-student rollout`),
      cell(
        "更新预算",
        `${c.outer_steps} outer × reuse${c.reuse} = ${c.updates} optimizer updates；batch${c.batch}`,
      ),
      cell("优化器", `${c.optimizer} · LR ${Number(c.lr).toExponential(0)} constant · wd ${c.weight_decay}`),
      cell(
        "训练目标",
        `${c.loss.objective.toUpperCase()} · full vocabulary · κ=${c.loss.kappa} · η=${c.loss.eta}`,
      ),
      cell("其余优化设置", `β=(${c.betas.join(", ")})；clip ${c.clip_grad}；seed ${c.seed}；单 seed`),
      cell(
        "长度与生成",
        `prompt≤${c.prompt_cap} · response≤${c.response_cap}；T=${c.temperature} / top-p=1 / n=1`,
      ),
      cell("Held-out Math", audit),
      cell("RKL 评估时点", "outer0 基线 + 每 outer 结束；41 个节点；固定回答全长评分"),
      cell("任务得分", "未测 accuracy；RKL 降低率不是答对率"),
    ].join("");
    $("map-count").textContent = `${s.points.length.toLocaleString()} 个展示点 · action0 只展示`;
    if (view) view.destroy();
    if (explorer) explorer.destroy();
    if (occupancyView) occupancyView.destroy();
    if (overviewView) overviewView.destroy();
    view = new LatentScatter($("scatter"), {
      points: s.points,
      groups: s.groups,
      onSelect: (p) => explorer?.select(p.trajectory, p.order),
      tooltip: (p) => {
        const g = group(p.group),
          meta = p.metadata;
        return `${g.label} · ${meta.split}\nouter ${meta.outer} · 簇 ${meta.clusters[$("clusters").value]}（K${$("clusters").value}）\naction ${meta.action} / ${meta.response_tokens} tokens\n${p.trajectory}`;
      },
    });
    explorer = new TrajectoryExplorer($("trajectory"), {
      points: s.points,
      groups: s.groups,
      texts: s.trajectories,
      onStep: (p) => view.update({ selected: p.trajectory, activePoint: p }),
    });
    occupancyView = new ClusterOccupancy($("occupancy"), {
      points: s.points,
      groups: [...s.groups, { id: "training_test", label: "Training test · 基线", color: "#3A758B" }],
      onSelect: (cluster) => {
        const k = $("clusters").value;
        view.update({ cluster, k });
        explorer.setCluster(k, cluster);
      },
      onPoint: (p) => {
        view.update({ split: "all", includeStart: true });
        view.splitSelect.value = "all";
        explorer.select(p.trajectory, p.order);
        activatePane("trajectory");
        $("trajectory-section").scrollIntoView({
          behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
          block: "start",
        });
      },
    });
    overviewView = new StateOverview($("overview"), {
      overview: s.overview,
      onCluster: (id) => {
        activatePane("clusters");
        occupancyView.reveal(id);
      },
    });
    $("source-table").innerHTML = table(
      ["来源", "回答数", "几何 test 回答", "展示状态", "指标候选状态", "平均回答长度 / tokens"],
      s.groups.map((g) => {
        const v = s.stats[g.id];
        return `<tr><td>${swatch(g)}</td><td>${v.trajectories}</td><td>${v.test_trajectories}</td><td>${v.states}</td><td>${v.metric_states}</td><td>${fmt(v.mean_tokens, 0)}</td></tr>`;
      }),
    );
    $("budget").innerHTML = [
      cell("已有训练 rollout", `${s.available_training} 条；40 outer × 64`),
      cell(
        "本版保留",
        `${s.selected_training} 条训练（${pct(s.selected_training / s.available_training)}）；每 outer ${D.per_outer} 条`,
      ),
      cell("每回答取点", "最多 8 个相对位置；排除 action0 后最多 7 个指标状态"),
      cell(
        "训练 reference",
        `${m.reference_states} 状态 / ${s.reference_trajectories} 条回答；其余 ${m.training_test_trajectories} 条训练回答测试`,
      ),
      cell(
        "K200 的拟合密度",
        `平均 ${(m.reference_states / 200).toFixed(1)} 个参考状态 / 簇；同回答状态相关`,
      ),
      cell("独立性限制", "训练题与 held-out bank 题不同；8 个位置不等于 8 个独立样本"),
    ].join("");
    const e = s.extraction;
    $("infra").innerHTML =
      `<div class="settings-grid">${cell("本版执行", `每 pair 一张 H800，三 pair 并行；${e.settings.dtype} / ${e.settings.attention}`)}${cell("批处理", `padded-token budget ${e.settings.token_budget}；max batch ${e.settings.max_batch}`)}${cell("实测有效 tokens/s", Math.round(e.tokens_per_second).toLocaleString())}${cell("分片前向总耗时", `${fmt(e.forward_seconds, 1)} 秒`)}${cell("峰值已分配显存", `${fmt(e.peak_gib, 2)} GiB`)}${cell("因果 prefix 校验", `cosine ${fmt(e.alignment_cosine, 6)}`)}</div>`;
    $("provenance").textContent = JSON.stringify(
      { plan_sha256: D.provenance.plan_sha256, ...s.provenance, prompt_audit: s.prompt_audit },
      null,
      2,
    );
    drawRisk();
    drawMetrics();
  }
  function svgText(x, y, text, extra = "") {
    return `<text x="${x}" y="${y}" fill="var(--muted)" font-size="10" ${extra}>${esc(text)}</text>`;
  }
  function drawRisk() {
    const bank = $("risk-bank").value,
      domain = $("risk-domain").value,
      mode = $("risk-scale").value,
      compare = $("risk-compare").checked;
    const series = (compare ? ids : [pair]).map((id) => ({
      id,
      color: color[ids.indexOf(id) % color.length],
      rows: D.risks
        .filter((r) => r.pair === id && r.bank === bank && r.domain === domain)
        .sort((a, b) => a.steps - b.steps),
    }));
    const W = 1040,
      H = 300,
      L = 65,
      R = 24,
      T = 24,
      B = 42,
      maxStep = Math.max(...series.flatMap((s) => s.rows.map((r) => r.steps)));
    const read = (r, s) => (mode === "remaining" ? r.rkl / s.rows[0].rkl : r.rkl);
    const all = series.flatMap((s) => s.rows.map((r) => read(r, s))),
      min = mode === "log" ? Math.floor(Math.log10(Math.min(...all))) : 0;
    const max = mode === "log" ? Math.ceil(Math.log10(Math.max(...all))) : Math.max(...all) * 1.12;
    const x = (step) => L + (step / maxStep) * (W - L - R),
      y = (v) => H - B - (((mode === "log" ? Math.log10(v) : v) - min) / (max - min || 1)) * (H - T - B);
    let markup = "";
    const ticks =
      mode === "log"
        ? Array.from({ length: max - min + 1 }, (_, i) => 10 ** (min + i))
        : Array.from({ length: 5 }, (_, i) => (max * i) / 4);
    for (const tick of ticks) {
      const yy = y(tick);
      markup +=
        `<path d="M${L} ${yy}H${W - R}" stroke="var(--line)" stroke-dasharray="3 4"/>` +
        svgText(
          L - 10,
          yy + 3,
          mode === "remaining"
            ? pct(tick, 0)
            : tick < 0.01
              ? tick.toExponential(0)
              : fmt(tick, tick < 1 ? 3 : 1),
          'text-anchor="end"',
        );
    }
    for (const step of [0, 160, 320, 480, 640].filter((s) => s <= maxStep))
      markup += svgText(x(step), H - B + 20, step, 'text-anchor="middle"');
    markup += svgText(W - R, H - 5, "optimizer updates", 'text-anchor="end"');
    markup += svgText(
      L,
      T - 10,
      mode === "remaining" ? "RKL / RKL(step0)" : "question-mean RKL · nats/token",
    );
    hoverPoints = [];
    for (const s of series) {
      const coords = s.rows.map((r) => [x(r.steps), y(read(r, s))]);
      markup += `<path d="${coords.map(([xx, yy], i) => `${i ? "L" : "M"}${xx} ${yy}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2.4"/>`;
      s.rows.forEach((r, i) => {
        const [xx, yy] = coords[i];
        markup += `<circle cx="${xx}" cy="${yy}" r="3" fill="${s.color}"/>`;
        hoverPoints.push({ x: xx, y: yy, ...r, color: s.color });
      });
    }
    $("risk-chart").innerHTML = markup;
    $("risk-chart").setAttribute("aria-label", `${domain} ${bank} ${mode} RKL curve`);
    $("risk-legend").innerHTML = series
      .map(
        (s) =>
          `<span><i style="background:${s.color}"></i>${s.id}${s.id === pair ? " · 当前配对" : ""}</span>`,
      )
      .join("");
    $("risk-caption").textContent =
      "41 个 checkpoint · KL(student ‖ teacher)，每题先平均 token，再对题平均 · 越低越接近。";
    $("risk-table").innerHTML = table(
      ["Pair · bank", "step0 RKL", "step640 RKL", "绝对降低", "相对降低"],
      (compare ? ids : [pair]).flatMap((id) =>
        ["public_initial", "public_teacher"].map((b) => {
          const rows = D.risks
              .filter((r) => r.pair === id && r.bank === b && r.domain === domain)
              .sort((a, b) => a.steps - b.steps),
            first = rows[0],
            last = rows.at(-1);
          return `<tr><td>${id} · ${b === "public_initial" ? "initial" : "teacher"}</td><td>${fmt(first.rkl, 6)}</td><td>${fmt(last.rkl, 6)}</td><td>${fmt(first.rkl - last.rkl, 6)}</td><td>${fmt(last.reduction_percent, 2)}%</td></tr>`;
        }),
      ),
    );
  }
  function metricChart(key, id) {
    const s = spec(),
      k = $("clusters").value,
      m = s.metrics.metrics[k],
      width = 500,
      L = 122,
      R = 60,
      T = 18,
      row = 30;
    const chartGroups = [...s.groups, { id: "training_test", label: "Training test 基线", color: "#3A758B" }];
    const entries = chartGroups.map((g) => ({ g, v: key === "js" ? m[g.id].balanced_js : m[g.id].coverage }));
    const max = key === "js" ? Math.max(0.15, ...entries.map((e) => e.v.mean + e.v.sd)) * 1.12 : 1;
    const x = (v) => L + (v / max) * (width - L - R),
      baseline = key === "js" ? m.training_test.balanced_js.mean : null;
    let svg = "";
    for (let t = 0; t <= 4; t++) {
      const v = (max * t) / 4,
        xx = x(v);
      svg +=
        `<path d="M${xx} ${T}V${T + row * 7}" stroke="var(--line)"/>` +
        svgText(xx, T + row * 7 + 20, key === "js" ? fmt(v, 2) : pct(v, 0), 'text-anchor="middle"');
    }
    if (baseline !== null)
      svg += `<path d="M${x(baseline)} ${T}V${T + row * 7}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
    entries.forEach(({ g, v }, i) => {
      const yy = T + row * i + row / 2,
        col = g.color === "#3A3F45" ? "var(--ink)" : g.color;
      svg +=
        svgText(L - 10, yy + 4, g.label, 'text-anchor="end"') +
        `<rect x="${L}" y="${yy - 5}" width="${Math.max(0, x(v.mean) - L)}" height="10" rx="3" fill="${col}" fill-opacity=".75"/>`;
      svg +=
        `<path d="M${x(Math.max(0, v.mean - v.sd))} ${yy}H${x(Math.min(max, v.mean + v.sd))}" stroke="var(--ink)" stroke-width="1.2"/>` +
        svgText(width - R + 10, yy + 4, key === "js" ? fmt(v.mean) : pct(v.mean, 1));
    });
    $(id).innerHTML = svg;
  }
  function drawMetrics() {
    const s = spec(),
      m = s.metrics,
      k = $("clusters").value,
      values = m.metrics[k];
    $("metric-budget").textContent =
      `K=${k} · JS 每组 ${m.js_trajectory_budget} 条查询与等量训练 test 回答比较，两侧回答不相交；coverage 每组 ${m.coverage_trajectory_budget} 条。均重复 30 次。虚线是训练 test 自身的有限样本基线。`;
    metricChart("js", "js-chart");
    metricChart("coverage", "coverage-chart");
    $("metric-table").innerHTML = table(
      [
        "来源",
        "测试回答 / 状态",
        "等预算 JS ± SD · bits",
        "Coverage ± SD",
        "训练 reference 10NN 距离",
        "超出训练 test p95",
      ],
      [...s.groups, { id: "training_test", label: "Training test · 基线", color: "#3A758B" }].map((g) => {
        const v = values[g.id],
          j = v.balanced_js,
          c = v.coverage;
        return `<tr${g.id === "training_test" ? ' class="baseline-row"' : ""}><td>${swatch(g)}${g.id === "training_test" ? " · 基线" : ""}</td><td>${v.trajectories} / ${v.states}</td><td>${fmt(j.mean)} ± ${fmt(j.sd)}</td><td>${fmt(c.mean)} ± ${fmt(c.sd)}</td><td>${fmt(v.reference_knn_cosine, 4)}</td><td>${pct(v.beyond_reference_p95)}</td></tr>`;
      }),
    );
    occupancyView.update(k, values);
    overviewView.update(k);
  }
  function activatePane(name) {
    for (const panel of document.querySelectorAll("[data-panel]"))
      panel.hidden = panel.dataset.panel !== name;
    for (const button of document.querySelectorAll("[data-pane]"))
      button.setAttribute("aria-pressed", String(button.dataset.pane === name));
    view?.draw();
    if (name === "overview") overviewView?.drawHeat();
  }
  for (const button of document.querySelectorAll("[data-pane]"))
    button.addEventListener("click", () => activatePane(button.dataset.pane));
  for (const id of ids) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = id;
    button.dataset.pair = id;
    button.setAttribute("role", "tab");
    button.id = `tab-${id}`;
    button.setAttribute("aria-controls", "pair-content");
    button.addEventListener("click", () => selectPair(id));
    button.addEventListener("keydown", (e) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
        e.preventDefault();
        const next =
          e.key === "Home"
            ? ids[0]
            : e.key === "End"
              ? ids.at(-1)
              : ids[(ids.indexOf(pair) + (e.key === "ArrowLeft" ? -1 : 1) + ids.length) % ids.length];
        selectPair(next);
        $(`tab-${next}`).focus();
      }
    });
    $("pair-tabs").append(button);
  }
  $("risk-chart").addEventListener("pointermove", (e) => {
    const svg = $("risk-chart"),
      point = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    let nearest = null,
      distance = Infinity;
    for (const p of hoverPoints) {
      const d = Math.hypot(p.x - point.x, p.y - point.y);
      if (d < distance) {
        nearest = p;
        distance = d;
      }
    }
    $("risk-tip").hidden = distance > 45;
    if (nearest && distance <= 45)
      $("risk-tip").textContent =
        `${nearest.pair} · update ${nearest.steps}\nRKL ${fmt(nearest.rkl, 6)} · 降低 ${fmt(nearest.reduction_percent, 2)}%\n${nearest.rows} 条回答 · ${nearest.tokens.toLocaleString()} 有效 tokens`;
  });
  $("risk-chart").addEventListener("pointerleave", () => {
    $("risk-tip").hidden = true;
  });
  for (const id of ["risk-bank", "risk-domain", "risk-scale", "risk-compare"])
    $(id).addEventListener("change", () => {
      $("risk-tip").hidden = true;
      drawRisk();
    });
  $("clusters").addEventListener("change", drawMetrics);
  $("theme").addEventListener("click", () => {
    document.documentElement.dataset.theme =
      document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    $("theme").textContent = document.documentElement.dataset.theme === "dark" ? "浅色" : "深色";
    view.draw();
    overviewView.drawHeat();
  });
  $("download-data").addEventListener("click", () => {
    const { risk_figure, ...data } = D;
    download("latent-state-results.json", JSON.stringify(data, null, 2), "application/json");
  });
  $("download-risk").addEventListener("click", () => {
    const clone = $("risk-chart").cloneNode(true),
      style = getComputedStyle(document.documentElement);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute(
      "style",
      `--ink:${style.getPropertyValue("--ink")};--muted:${style.getPropertyValue("--muted")};--line:${style.getPropertyValue("--line")};background:${style.getPropertyValue("--surface")};font-family:system-ui`,
    );
    download(`${pair}-heldout-rkl.svg`, new XMLSerializer().serializeToString(clone), "image/svg+xml");
  });
  window.addEventListener("hashchange", () => {
    if (ids.includes(location.hash.slice(1))) selectPair(location.hash.slice(1));
  });
  selectPair(pair);
})().catch((error) => {
  console.error(error);
  const message = document.createElement("p");
  message.setAttribute("role", "alert");
  message.textContent = `报告加载失败：${error.message}。请刷新页面或检查服务日志。`;
  document.querySelector("main").prepend(message);
});
