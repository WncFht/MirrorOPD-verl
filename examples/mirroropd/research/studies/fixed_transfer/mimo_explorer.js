const colors = { E71: "#4F8FA5", E72: "#EE995B", E73: "#C95B5B" };
const controls = Object.fromEntries(
  ["run", "bank", "domain", "metric", "view"].map((k) => [
    k,
    document.getElementById(k),
  ]),
);
const metricNames = {
  rkl: "RKL 残差 K（nat）",
  log_absorption: "对数吸收量 A（dex）",
  relative_decrease: "相对下降（%）",
  absolute_decrease: "绝对下降（nat）",
  gap_proxy: "累计吸收差 Δ（rollout 参照，dex）",
  efficiency_proxy: "累计效率 η（rollout 参照）",
};
for (const [k, v] of Object.entries(DATA.teachers)) {
  const option = document.createElement("option");
  option.value = k;
  option.textContent = k + " · " + v;
  controls.run.append(option);
}
for (const value of DATA.domains) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent =
    value === "all"
      ? "Avg（四域等权）"
      : value[0].toUpperCase() + value.slice(1);
  controls.domain.append(option);
}
function draw() {
  const key = controls.metric.value;
  const proxy = ["gap_proxy", "efficiency_proxy"].includes(key);
  if (proxy) controls.view.value = "time";
  controls.view.disabled = proxy;
  const phase = controls.view.value === "phase";
  const traces = [];
  for (const [run, teacher] of Object.entries(DATA.teachers)) {
    if (controls.run.value !== "all" && controls.run.value !== run) continue;
    const rows = DATA.rows
      .filter(
        (row) =>
          row.run === run &&
          row.bank === controls.bank.value &&
          row.domain === controls.domain.value,
      )
      .sort((a, b) => a.step - b.step);
    const trainMetric = (row) =>
      key === "rkl"
        ? row.train_rkl_proxy
        : key === "log_absorption"
          ? row.train_absorption_proxy
          : key === "relative_decrease"
            ? 100 * (1 - row.train_rkl_proxy / row.train_initial_rkl)
            : row.train_initial_rkl - row.train_rkl_proxy;
    traces.push({
      x: rows.map((row) => (phase && !proxy ? trainMetric(row) : row.step)),
      y: rows.map((row) => row[key]),
      type: "scatter",
      mode: "lines+markers",
      name: run + " · " + teacher,
      line: { color: colors[run], width: 3 },
      marker: { size: 8 },
      connectgaps: false,
      customdata: rows.map((row) => [
        row.outer,
        row.questions,
        row.rkl,
        row.initial_rkl,
        row.step,
        row.train_parameter_step,
      ]),
      hovertemplate:
        "x=%{x:.5g} / y=%{y:.5g}<br>outer %{customdata[0]} · n=%{customdata[1]}<br>K=%{customdata[2]:.5g} / K0=%{customdata[3]:.5g}<br>bank 参数步 %{customdata[4]} / rollout 参数步 %{customdata[5]}<extra>%{fullData.name}</extra>",
    });
    if (key === "rkl" && !phase) {
      traces.push({
        x: rows.map((row) => row.train_parameter_step),
        y: rows.map((row) => row.train_rkl_proxy),
        type: "scatter",
        mode: "lines+markers",
        name: run + " · rollout（更新前）",
        line: { color: colors[run], dash: "dash" },
        marker: { size: 6, symbol: "x" },
        hovertemplate:
          "参数步 %{x}<br>训练 rollout K=%{y:.5g}<extra>%{fullData.name}</extra>",
      });
    }
  }
  const logY = key === "rkl";
  const shapes = [];
  if (phase && !proxy) {
    const values = traces.flatMap((trace) => [...trace.x, ...trace.y]);
    const low = Math.min(...values),
      high = Math.max(...values);
    shapes.push({
      type: "line",
      x0: low,
      y0: low,
      x1: high,
      y1: high,
      line: { color: "#97a6ae", dash: "dot" },
      layer: "below",
    });
  }
  Plotly.react(
    "chart",
    traces,
    {
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      font: { family: "Inter,Arial,sans-serif", color: "#344954" },
      margin: { l: 80, r: 25, t: 70, b: 65 },
      xaxis: {
        title: {
          text:
            phase && !proxy
              ? "训练 rollout · " + metricNames[key] + "（近似配对）"
              : "Optimizer updates",
        },
        type: phase && logY ? "log" : "linear",
        gridcolor: "#e9eef0",
      },
      yaxis: {
        title: { text: metricNames[key] },
        type: logY ? "log" : "linear",
        gridcolor: "#e9eef0",
      },
      legend: { orientation: "h", y: 1.2 },
      hovermode: "closest",
      shapes,
    },
    { responsive: true, displaylogo: false },
  );
}
for (const control of Object.values(controls))
  control.addEventListener("change", draw);
draw();
