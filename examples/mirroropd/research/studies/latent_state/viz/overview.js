/* Global teacher comparison. Metrics are precomputed from whole-trajectory draws. */
(function (global) {
  "use strict";
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  const stages = ["outer01_10", "outer11_20", "outer21_30", "outer31_40"],
    names = ["01–10", "11–20", "21–30", "31–40"];
  class StateOverview {
    constructor(root, { overview, onCluster = () => {} }) {
      this.root = root;
      this.data = overview;
      this.onCluster = onCluster;
      this.selected = null;
      const header = el("div", undefined, "section-head");
      header.append(el("h2", "训练中的状态分布"));
      this.budget = el("span", undefined, "muted");
      header.append(this.budget);
      root.append(header);
      this.charts = el("div", undefined, "overview-charts");
      root.append(this.charts);
      this.readout = el("div", undefined, "overview-readout");
      this.readout.setAttribute("aria-live", "polite");
      root.append(this.readout);
      root.append(
        el(
          "p",
          "每阶段与 Teacher 使用相同回答预算；线与误差棒为 30 次抽样的均值 ± SD。不同阶段题目不同。",
          "caption",
        ),
      );
      const title = el("div", undefined, "section-head overview-heat-heading");
      title.append(el("h2", "全部簇 · 与 Teacher 的占比差"));
      const wrap = el("label", "簇排序");
      this.sort = el("select");
      this.sort.setAttribute("aria-label", "全局热图排序");
      for (const [value, label] of [
        ["teacher", "Teacher 占比"],
        ["gap", "末段差异"],
        ["id", "簇编号"],
      ]) {
        const o = el("option", label);
        o.value = value;
        this.sort.append(o);
      }
      wrap.append(this.sort);
      title.append(wrap);
      root.append(title);
      const legend = el("div", undefined, "cluster-scale");
      legend.append(
        el("span", "少于 Teacher"),
        el("i", undefined, "cluster-diverging"),
        el("span", "多于 Teacher"),
      );
      this.scale = el("span");
      legend.append(this.scale);
      root.append(legend);
      const heat = el("div", undefined, "overview-heat-layout"),
        plot = el("div", undefined, "overview-heat-plot");
      this.canvas = el("canvas");
      this.canvas.setAttribute("aria-label", "全部簇相对 Teacher 的差值热图；点击选择簇");
      plot.append(this.canvas);
      this.detail = el("aside", undefined, "overview-heat-detail");
      heat.append(plot, this.detail);
      root.append(heat);
      this.sort.addEventListener("change", () => this.drawHeat());
      this.canvas.addEventListener("pointermove", (e) => {
        if (this.selected === null) {
          const i = this.hit(e);
          if (i !== null) this.inspect(i);
        }
      });
      this.canvas.addEventListener("click", (e) => {
        const i = this.hit(e);
        if (i !== null) {
          this.selected = i;
          this.drawHeat();
          this.inspect(i);
        }
      });
      this.observer = new ResizeObserver(() => this.drawHeat());
      this.observer.observe(plot);
    }
    update(k) {
      this.k = String(k);
      this.values = this.data.metrics[this.k];
      this.selected = null;
      this.budget.textContent = `K${k} · 每组 ${this.data.budget} 条回答`;
      this.renderCharts();
      this.drawHeat();
    }
    renderCharts() {
      this.charts.replaceChildren();
      for (const metric of ["js", "coverage"]) {
        const card = el("div", undefined, "overview-chart"),
          title = el("h3", metric === "js" ? "状态分布 → Teacher" : "状态访问范围");
        const caption = el("span", metric === "js" ? "JS · bits · 越低越接近" : "Coverage · 相同回答数");
        title.append(caption);
        card.append(title);
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 580 300");
        svg.setAttribute(
          "aria-label",
          metric === "js" ? "训练阶段到 Teacher 的 JS 曲线" : "训练阶段 coverage 曲线",
        );
        svg.setAttribute("role", "img");
        const W = 580,
          H = 300,
          L = 54,
          R = 20,
          T = 24,
          B = 54,
          initial = this.values.initial[metric],
          baseline = this.values.teacher[metric];
        const rows = stages.map((g) => this.values[g][metric]);
        const max =
          metric === "coverage"
            ? 1
            : Math.max(
                0.03,
                initial.mean + initial.sd,
                baseline.mean + baseline.sd,
                ...rows.map((r) => r.mean + r.sd),
              ) * 1.12;
        const x = (i) => L + (i * (W - L - R)) / 3,
          y = (v) => H - B - (v / max) * (H - T - B);
        const node = (tag, attrs, text) => {
          const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
          for (const [key, v] of Object.entries(attrs)) n.setAttribute(key, v);
          if (text !== undefined) n.textContent = text;
          svg.append(n);
          return n;
        };
        const line = (v, color, dash) =>
          node("path", {
            d: `M${L} ${y(v)}H${W - R}`,
            stroke: color,
            "stroke-dasharray": dash,
            "stroke-width": 1.4,
            fill: "none",
          });
        for (let i = 0; i <= 4; i++) {
          const v = (max * i) / 4;
          line(v, "var(--line)", "2 4");
          node(
            "text",
            { x: L - 9, y: y(v) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 },
            metric === "coverage" ? `${Math.round(v * 100)}%` : v.toFixed(2),
          );
        }
        const low = Math.max(0, baseline.mean - baseline.sd),
          high = Math.min(max, baseline.mean + baseline.sd);
        node("rect", {
          x: L,
          y: y(high),
          width: W - L - R,
          height: y(low) - y(high),
          fill: "var(--accent)",
          opacity: 0.1,
        });
        line(baseline.mean, "var(--accent)", "5 4");
        line(initial.mean, "var(--muted)", "2 5");
        node("path", {
          d: rows.map((r, i) => `${i ? "L" : "M"}${x(i)} ${y(r.mean)}`).join(" "),
          fill: "none",
          stroke: "var(--overview-line)",
          "stroke-width": 2.6,
        });
        rows.forEach((r, i) => {
          const xx = x(i),
            yy = y(r.mean);
          node("path", {
            d: `M${xx} ${y(Math.max(0, r.mean - r.sd))}V${y(Math.min(max, r.mean + r.sd))}`,
            stroke: "var(--overview-line)",
            "stroke-width": 2,
          });
          node("circle", { cx: xx, cy: yy, r: 4.5, fill: "var(--overview-line)" });
          node(
            "text",
            { x: xx, y: H - B + 24, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12 },
            names[i],
          );
          const target = node("circle", {
            cx: xx,
            cy: yy,
            r: 16,
            fill: "transparent",
            tabindex: 0,
            role: "button",
            "aria-label": `${names[i]} ${metric} ${r.mean.toFixed(4)}`,
          });
          const inspect = () => {
            this.readout.textContent = `Outer ${names[i]} · JS ${this.values[stages[i]].js.mean.toFixed(4)} ± ${this.values[stages[i]].js.sd.toFixed(4)} bits · Coverage ${(100 * this.values[stages[i]].coverage.mean).toFixed(1)}%`;
          };
          target.addEventListener("pointerenter", inspect);
          target.addEventListener("focus", inspect);
          target.addEventListener("click", inspect);
        });
        card.append(svg);
        const legend = el("div", undefined, "overview-chart-legend");
        for (const [label, cls] of [
          ["训练阶段", "stage"],
          [
            `Initial ${metric === "js" ? initial.mean.toFixed(3) : `${(100 * initial.mean).toFixed(1)}%`}`,
            "initial",
          ],
          [
            `${metric === "js" ? "Teacher 拆半基线" : "Teacher"} ${metric === "js" ? baseline.mean.toFixed(3) : `${(100 * baseline.mean).toFixed(1)}%`}`,
            "teacher",
          ],
        ]) {
          const item = el("span", label);
          item.prepend(el("i", undefined, cls));
          legend.append(item);
        }
        card.append(legend);
        this.charts.append(card);
      }
      const first = this.values[stages[0]].js,
        last = this.values[stages[3]].js;
      this.readout.textContent = `JS → Teacher：首段 ${first.mean.toFixed(4)} → 末段 ${last.mean.toFixed(4)} · Teacher 拆半基线 ${this.values.teacher.js.mean.toFixed(4)}`;
    }
    drawHeat() {
      if (!this.values) return;
      const k = Number(this.k),
        m = this.values,
        columns = ["initial", ...stages];
      this.order = Array.from({ length: k }, (_, i) => i).sort((a, b) => {
        const score = (i) =>
          this.sort.value === "teacher"
            ? m.teacher.occupancy[i]
            : this.sort.value === "gap"
              ? Math.abs(m.outer31_40.occupancy[i] - m.teacher.occupancy[i])
              : -i;
        return score(b) - score(a) || a - b;
      });
      this.limit = Math.max(
        0.00001,
        ...columns.flatMap((g) => m[g].occupancy.map((v, i) => Math.abs(v - m.teacher.occupancy[i]))),
      );
      this.scale.textContent = `±${(100 * this.limit).toFixed(2)} 个百分点 · 平方根色阶`;
      const width = Math.max(280, this.canvas.parentElement.clientWidth),
        row = 4.5,
        top = 32,
        left = 45,
        height = top + k * row + 8,
        dpr = devicePixelRatio || 1;
      Object.assign(this, { heatWidth: width, heatRow: row, heatTop: top, heatLeft: left });
      this.canvas.style.height = `${height}px`;
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      const c = this.canvas.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const style = getComputedStyle(this.root),
        ink = style.getPropertyValue("--ink").trim(),
        cell = (width - left) / columns.length;
      c.font = "11px system-ui";
      c.fillStyle = ink;
      c.textAlign = "center";
      ["Initial", ...names].forEach((label, i) => c.fillText(label, left + (i + 0.5) * cell, 18));
      this.order.forEach((id, r) => {
        const yy = top + r * row;
        columns.forEach((g, col) => {
          const diff = m[g].occupancy[id] - m.teacher.occupancy[id],
            strength = Math.sqrt(Math.abs(diff) / this.limit);
          c.fillStyle = style.getPropertyValue("--surface").trim();
          c.fillRect(left + col * cell, yy, cell - 2, row - 0.5);
          c.fillStyle = diff < 0 ? `rgba(67,134,197,${strength})` : `rgba(229,146,69,${strength})`;
          c.fillRect(left + col * cell, yy, cell - 2, row - 0.5);
        });
        if (r % 10 === 0 || id === this.selected) {
          c.fillStyle = ink;
          c.textAlign = "right";
          c.fillText(`#${id}`, left - 7, yy + 4);
        }
        if (id === this.selected) {
          c.strokeStyle = ink;
          c.lineWidth = 1.5;
          c.strokeRect(left, yy, width - left - 2, row);
        }
      });
      this.inspect(this.selected ?? this.order[0]);
    }
    hit(e) {
      const r = this.canvas.getBoundingClientRect(),
        row = Math.floor((e.clientY - r.top - this.heatTop) / this.heatRow);
      return row >= 0 && row < this.order.length ? this.order[row] : null;
    }
    inspect(id) {
      this.detail.replaceChildren();
      this.detail.append(
        el("div", this.selected === null ? "悬停查看 · 点击固定" : "已选择", "eyebrow"),
        el("h3", `Cluster ${id}`),
      );
      const table = el("table"),
        head = el("tr");
      for (const title of ["来源", "占比", "差值 / pp"]) head.append(el("th", title));
      table.append(head);
      for (const [g, label] of [
        ["initial", "Initial"],
        ...stages.map((g, i) => [g, names[i]]),
        ["teacher", "Teacher"],
      ]) {
        const v = this.values[g].occupancy[id],
          diff = 100 * (v - this.values.teacher.occupancy[id]),
          r = el("tr");
        r.append(
          el("td", label),
          el("td", `${(100 * v).toFixed(2)}%`),
          el("td", g === "teacher" ? "—" : `${diff > 0 ? "+" : ""}${diff.toFixed(2)}`),
        );
        table.append(r);
      }
      this.detail.append(table);
      const button = el("button", "查看簇明细 →");
      button.addEventListener("click", () => this.onCluster(id));
      const clear = el("button", "取消固定");
      clear.disabled = this.selected === null;
      clear.addEventListener("click", () => {
        this.selected = null;
        this.drawHeat();
      });
      this.detail.append(button, clear);
    }
    destroy() {
      this.observer.disconnect();
      this.root.replaceChildren();
    }
  }
  global.StateOverview = StateOverview;
})(globalThis);
