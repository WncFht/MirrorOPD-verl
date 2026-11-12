// Copyright 2026 MirrorOPD-verl contributors
// SPDX-License-Identifier: Apache-2.0
/* Offline, read-only comparison. Matrix statistics are recomputed after filtering. */
"use strict";
(async function () {
  const $ = id => document.getElementById(id);
  const palette = ["#237e9b", "#c96b30", "#715ab0", "#36915e", "#c24e60", "#8a7435", "#317981", "#b65b98", "#5c719a", "#797c42", "#9a6451"];
  const controls = ["view", "metric", "rank", "module", "layer", "step", "bank", "domain", "relative", "bands", "log", "figure"];
  const modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
  let data, csvRows = [], busy = false, pending = false;
  const opt = (parent, value, label) => { const e = document.createElement("option"); e.value = value; e.textContent = label; parent.appendChild(e); };
  const finite = values => values.filter(x => typeof x === "number" && Number.isFinite(x));
  function quantile(values, q) {
    const v = finite(values).sort((a, b) => a - b); if (!v.length) return null;
    const x = (v.length - 1) * q, lo = Math.floor(x), hi = Math.ceil(x);
    return v[lo] + (v[hi] - v[lo]) * (x - lo);
  }
  const median = values => quantile(values, .5);
  const selected = () => [...document.querySelectorAll("#runs input:checked")].map(e => e.value);
  function subset(pair) {
    return pair.matrices.filter(m => ($("layer").value === "all" || m.layer === Number($("layer").value)) &&
      ($("module").value === "all" || m.module === $("module").value ||
        ($("module").value === "Attention" && modules.slice(0, 4).includes(m.module)) ||
        ($("module").value === "MLP" && modules.slice(4).includes(m.module))));
  }
  const key = () => $("metric").value + (data.specs[$("metric").value]?.[2] ? ":" + $("rank").value : "");
  const riskKey = () => $("bank").value + "/" + $("domain").value;
  function riskValue(pair, record) {
    const v = record?.values[riskKey()], base = pair.risks[0].values[riskKey()];
    return typeof v === "number" ? ($("relative").checked ? (base > 0 ? v / base : null) : v) : null;
  }
  function summary(pair, ms) {
    const steps = pair.matrices[0].steps;
    return { steps, values: steps.map((t, i) => ms.map(m => m.series[key()]?.[i])) };
  }
  function newPlot(title) {
    const node = document.createElement("div"); node.className = "card"; $("plots").appendChild(node);
    return { node, layout: { title: {text: title, font: {size: 16}}, margin: {l: 90, r: 30, t: 65, b: 70},
      paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: {family: "system-ui", color: "#193344"},
      xaxis: {title: {text: "Optimizer updates"}, gridcolor: "#e4eaf0"},
      yaxis: {title: {text: data.specs[$("metric").value]?.[1]}, type: $("log").checked ? "log" : "linear", gridcolor: "#e4eaf0"},
      legend: {orientation: "h", y: -0.24}, hovermode: "closest"} };
  }
  function draw(plot, traces) { return Plotly.newPlot(plot.node, traces, plot.layout, {responsive: true, displaylogo: false, toImageButtonOptions: {format: "svg", filename: "checkpoint-comparison"}}); }
  function clear() {
    for (const node of $("plots").children) { if (node.data) Plotly.purge(node); }
    $("plots").replaceChildren(); csvRows = []; $("failure").textContent = "";
  }
  async function render() {
    if (busy) { pending = true; return; } busy = true;
    try {
      clear(); const names = selected(), metric = $("metric").value, view = $("view").value;
      const isRkl = metric === "rkl", title = isRkl ? `RKL · ${riskKey()}${$("relative").checked ? " / initial" : " (absolute)"}` : data.specs[metric][1];
      $("description").textContent = view === "gallery" ? "原始图对照：请选择图类型；下方按所选 pair 展示原图。rank、层和模块筛选仅用于交互视图，不会重新绘制这些固定图片。" :
        `${title}。${isRkl ? "固定 bank 的完整有效响应；层／模块筛选不影响 RKL。" : data.specs[metric][3]} ${data.specs[metric]?.[2] ? "k=" + $("rank").value + "。" : ""}`;
      $("rank").disabled = !(data.specs[metric]?.[2] || view === "pairwise");
      $("status").textContent = names.map(n => `${n}: ${subset(data.pairs[n]).length}/${data.pairs[n].expected} 个矩阵 · ${data.pairs[n].student} → ${data.pairs[n].teacher}`).join("\n");
      if (!names.length) { $("failure").textContent = "请至少选择一个 pair。"; return; }
      if (view === "gallery") {
        for (const n of names) {
          const card = document.createElement("div"); card.className = "card";
          const h = document.createElement("h2"); h.textContent = n + " · " + $("figure").selectedOptions[0].textContent; card.appendChild(h);
          const image = document.createElement("img"); image.className = "gallery-image"; image.loading = "lazy";
          image.alt = h.textContent; image.src = `figures/${n}-${$("figure").value}.png`;
          image.onerror = () => { image.replaceWith(document.createTextNode("该 pair 无此图；跨尺寸 teacher 参数对照不适用。")); };
          card.appendChild(image); $("plots").appendChild(card);
        } return;
      }
      if (view === "trajectory") {
        const traces = [], plot = newPlot(title);
        for (const n of names) {
          const p = data.pairs[n], color = palette[Object.keys(data.pairs).indexOf(n) % palette.length];
          let x, y, lo, hi, counts;
          if (isRkl) { x = p.risks.map(r => r.step); y = p.risks.map(r => riskValue(p, r)); counts = x.map(() => null); }
          else { const s = summary(p, subset(p)); x = s.steps; y = s.values.map(median); lo = s.values.map(v => quantile(v, .25)); hi = s.values.map(v => quantile(v, .75)); counts = s.values.map(v => finite(v).length); }
          if (!y.some(v => v !== null)) { $("failure").textContent += `${n}：此指标不适用或无覆盖。 `; continue; }
          if (!isRkl && $("bands").checked) {
            traces.push({x, y: lo, type: "scatter", mode: "lines", line: {width: 0}, showlegend: false, hoverinfo: "skip", legendgroup: n});
            traces.push({x, y: hi, type: "scatter", mode: "lines", line: {width: 0}, fill: "tonexty", fillcolor: color + "22", showlegend: false, hoverinfo: "skip", legendgroup: n});
          }
          traces.push({x, y, mode: "lines+markers", name: n, line: {color, width: 2.5}, marker: {size: 6}, legendgroup: n,
            customdata: counts, hovertemplate: `${n}<br>step=%{x}<br>value=%{y:.6g}<br>matrices=%{customdata}<extra></extra>`, connectgaps: false});
          x.forEach((step, i) => csvRows.push({pair: n, metric: isRkl ? riskKey() : key(), step, median: y[i], p25: lo?.[i], p75: hi?.[i], matrices: counts[i]}));
        }
        plot.layout.yaxis.title = {text: title}; plot.layout.hovermode = "x unified"; await draw(plot, traces);
      } else if (view === "scatter") {
        if (isRkl) { $("failure").textContent = "请选择一个几何指标作为横轴；纵轴固定为所选 bank / 域的 RKL。"; return; }
        const plot = newPlot(`${title} × ${riskKey()} · step ${$("step").value}`), traces = [];
        for (const n of names) {
          const p = data.pairs[n], s = summary(p, subset(p)), idx = s.steps.indexOf(Number($("step").value));
          const x = idx >= 0 ? median(s.values[idx]) : null, y = riskValue(p, p.risks.find(r => r.step === Number($("step").value)));
          if (x === null || y === null) continue;
          traces.push({x: [x], y: [y], mode: "markers+text", name: n, text: [n], textposition: "top center", marker: {size: 11, color: palette[Object.keys(data.pairs).indexOf(n) % palette.length]}});
          csvRows.push({pair: n, step: Number($("step").value), metric: key(), x, rkl: y});
        }
        plot.layout.xaxis.title = {text: title}; plot.layout.yaxis.title = {text: "RKL" + ($("relative").checked ? " / initial" : " (absolute)")}; await draw(plot, traces);
      } else {
        if (view === "layers" && isRkl) { $("failure").textContent = "RKL 是整模型输出指标，没有逐层 RKL；请选择几何指标。"; return; }
        const panels = [];
        for (const n of names) {
          const p = data.pairs[n], ms = subset(p); let z, x, y;
          if (view === "pairwise") {
            x = p.matrices[0].steps; y = x;
            z = y.map((a, i) => x.map((b, j) => median(ms.map(m => m.pairwise[$("rank").value][i][j]))));
          } else {
            x = modules.filter(module => ms.some(m => m.module === module)); y = [...new Set(ms.map(m => m.layer))].sort((a, b) => a - b);
            z = y.map(layer => x.map(module => median(ms.filter(m => m.layer === layer && m.module === module).map(m => m.series[key()]?.[m.steps.indexOf(Number($("step").value))]))));
          }
          panels.push({n, z, x, y});
        }
        const values = finite(panels.flatMap(p => p.z.flat()));
        const min = view === "pairwise" ? -.05 : Math.min(...values), max = view === "pairwise" ? 1 : Math.max(...values);
        if (!values.length) { $("failure").textContent = "所选指标没有有效矩阵。"; return; }
        for (const p of panels) {
          const plot = newPlot(`${p.n} · ${view === "pairwise" ? "两两 frame overlap · k=" + $("rank").value : title + " · step " + $("step").value}`);
          plot.layout.xaxis = {title: {text: view === "pairwise" ? "Optimizer updates（等距类别）" : "模块"}, type: "category"};
          plot.layout.yaxis = {title: {text: view === "pairwise" ? "Optimizer updates（等距类别）" : "层"}, type: "category", autorange: "reversed"};
          await draw(plot, [{type: "heatmap", x: p.x, y: p.y, z: p.z, zmin: min, zmax: max, colorscale: "Cividis", hoverongaps: false, colorbar: {title: {text: "共同色标"}}}]);
          p.y.forEach((y, i) => p.x.forEach((x, j) => csvRows.push({pair: p.n, metric: view === "pairwise" ? "overlap:" + $("rank").value : key(), step: Number($("step").value), x, y, value: p.z[i][j]})));
        }
      }
      if ($("log").checked && csvRows.some(r => [r.median, r.rkl].some(v => typeof v === "number" && v <= 0))) $("failure").textContent += " 对数轴隐藏零值和负值；完整数值保留在 CSV，查看终点零距离请关闭对数轴。";
    } catch (error) { $("failure").textContent = error.stack; console.error(error); }
    finally { busy = false; if (pending) { pending = false; void render(); } }
  }
  try {
    if (!window.DecompressionStream) throw new Error("请使用支持 DecompressionStream 的现代 Chrome / Edge / Firefox 打开离线页面。");
    const bytes = Uint8Array.from(atob($("payload").textContent.trim()), c => c.charCodeAt(0));
    data = JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text());
    for (const [group, label, value] of [["输出", "Held-out RKL", "rkl"], ...Object.entries(data.specs).map(([k, v]) => [v[0], v[1], k])]) {
      let parent = [...$("metric").children].find(e => e.label === group);
      if (!parent) { parent = document.createElement("optgroup"); parent.label = group; $("metric").appendChild(parent); }
      opt(parent, value, label);
    }
    $("metric").value = "overlap";
    modules.forEach(m => opt($("module"), m, m)); for (let i = 0; i < 28; i++) opt($("layer"), String(i), "Layer " + i);
    [16, 32, 48, 64, 128, 192, 320, 480, 640].forEach(s => opt($("step"), String(s), String(s))); $("step").value = "640";
    data.figures.forEach(([v, l], i) => opt($("figure"), v, String(i + 1).padStart(2, "0") + " " + l));
    for (const [n, p] of Object.entries(data.pairs)) {
      const label = document.createElement("label"), input = document.createElement("input"), small = document.createElement("small");
      input.type = "checkbox"; input.value = n; input.checked = ["E94", "E99", "E101", "E104"].includes(n); input.onchange = render;
      label.appendChild(input); label.appendChild(document.createTextNode(" " + n)); small.textContent = p.student + " → " + p.teacher; label.appendChild(small); $("runs").appendChild(label);
    }
    try { const state = JSON.parse(decodeURIComponent(location.hash.slice(1))); for (const id of controls) if (id in state) { const e = $(id); if (e.type === "checkbox") e.checked = !!state[id]; else if ([...e.options].some(o => o.value === state[id])) e.value = state[id]; } if (Array.isArray(state.pairs)) document.querySelectorAll("#runs input").forEach(e => e.checked = state.pairs.includes(e.value)); } catch (_) { /* Empty or outdated fragment: use defaults. */ }
    controls.forEach(id => $(id).onchange = render);
    for (const [id, fn] of [["all", () => true], ["none", () => false], ["preset", n => ["E99", "E100", "E101"].includes(n)]]) $(id).onclick = () => { document.querySelectorAll("#runs input").forEach(e => e.checked = fn(e.value)); render(); };
    $("share").onclick = () => { const s = {pairs: selected()}; controls.forEach(id => s[id] = $(id).type === "checkbox" ? $(id).checked : $(id).value); location.hash = encodeURIComponent(JSON.stringify(s)); };
    $("csv").onclick = () => {
      if (!csvRows.length) { $("failure").textContent = "当前视图没有可导出的交互数值；静态图的精确数据见相邻 data 目录。"; return; }
      const keys = [...new Set(csvRows.flatMap(Object.keys))], cell = v => '"' + String(v ?? "").replaceAll('"', '""') + '"';
      const text = "\ufeff" + [keys.map(cell).join(","), ...csvRows.map(r => keys.map(k => cell(r[k])).join(","))].join("\n");
      const url = URL.createObjectURL(new Blob([text], {type: "text/csv;charset=utf-8"})), a = document.createElement("a"); a.href = url; a.download = "checkpoint-comparison.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    window.checkpointExplorer = {data, render, selected}; await render();
  } catch (error) { $("failure").textContent = error.stack; console.error(error); }
})();
