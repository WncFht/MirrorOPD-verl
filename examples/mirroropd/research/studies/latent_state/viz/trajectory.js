/* One saved response at a time. Coordinates and labels are supplied; no analysis or network calls. */
(function (global) {
  "use strict";
  const el = (tag, text) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  class TrajectoryExplorer {
    constructor(root, { points, groups, texts = {}, onStep = () => {} }) {
      this.texts = texts;
      this.root = root;
      this.groups = groups;
      this.onStep = onStep;
      this.k = "200";
      this.step = 0;
      this.traces = new Map();
      this.cluster = null;
      for (const p of points) {
        if (!this.traces.has(p.trajectory)) this.traces.set(p.trajectory, []);
        this.traces.get(p.trajectory).push(p);
      }
      for (const trace of this.traces.values()) trace.sort((a, b) => a.order - b.order);
      const controls = el("div");
      controls.className = "control-row";
      const field = (name, input) => {
        const label = el("label", name);
        label.append(input);
        controls.append(label);
        return input;
      };
      this.source = field("回答来源", el("select"));
      this.source.setAttribute("aria-label", "轨迹来源");
      for (const g of groups) {
        const o = el("option", g.label);
        o.value = g.id;
        this.source.append(o);
      }
      this.search = field("查找", el("input"));
      this.search.type = "search";
      this.search.placeholder = "题目文本 / 回答 ID";
      this.search.setAttribute("aria-label", "查找轨迹");
      this.selector = field("回答", el("select"));
      this.selector.setAttribute("aria-label", "选择回答轨迹");
      this.source.addEventListener("change", () => this.populate(this.current()?.metadata.prompt));
      this.search.addEventListener("input", () => this.populate());
      this.selector.addEventListener("change", () => this.select(this.selector.value));
      for (const [label, action] of [
        ["上一条", () => this.move(-1)],
        ["下一条", () => this.move(1)],
      ]) {
        const b = el("button", label);
        b.addEventListener("click", action);
        controls.append(b);
      }
      root.append(controls);
      this.empty = el("p", "没有匹配的回答，请修改查找条件。");
      this.empty.hidden = true;
      root.append(this.empty);
      this.body = el("div");
      this.body.className = "trajectory-body";
      const plot = el("div");
      plot.className = "trajectory-plot";
      this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      this.svg.setAttribute("viewBox", "0 0 600 340");
      this.svg.setAttribute("aria-label", "选定回答的有序状态轨迹");
      plot.append(
        this.svg,
        el("p", "同一张 t-SNE 地图的局部视图；连线表示回答中的先后顺序，线长不表示推理距离。"),
      );
      const detail = el("div");
      detail.className = "trajectory-detail";
      this.title = el("div");
      this.title.className = "eyebrow";
      this.prompt = el("pre");
      this.prompt.className = "trajectory-prompt";
      this.info = el("div");
      this.info.className = "state-info";
      this.context = el("pre");
      this.context.className = "trajectory-context";
      detail.append(
        this.title,
        el("h3", "题目 / 输入尾部"),
        this.prompt,
        this.info,
        el("h3", "当前状态已读取的回答片段"),
        this.context,
      );
      this.body.append(plot, detail);
      root.append(this.body);
      this.transport = el("div");
      this.transport.className = "control-row trajectory-transport";
      this.play = el("button", "播放 ▶");
      this.play.setAttribute("aria-label", "播放轨迹");
      this.play.addEventListener("click", () => this.toggle());
      this.slider = el("input");
      this.slider.type = "range";
      this.slider.min = 0;
      this.slider.step = 1;
      this.slider.setAttribute("aria-label", "轨迹状态步骤");
      this.slider.addEventListener("input", () => {
        this.stop();
        this.setStep(Number(this.slider.value));
      });
      this.progress = el("span");
      const prev = el("button", "← 上一状态"),
        next = el("button", "下一状态 →");
      prev.addEventListener("click", () => {
        this.stop();
        this.setStep(this.step - 1);
      });
      next.addEventListener("click", () => {
        this.stop();
        this.setStep(this.step + 1);
      });
      this.transport.append(this.play, prev, this.slider, next, this.progress);
      root.append(this.transport);
      this.steps = el("div");
      this.steps.className = "trajectory-steps";
      root.append(this.steps);
      this.populate();
    }
    current() {
      return this.traces.get(this.id)?.[this.step];
    }
    populate(prompt) {
      const query = this.search.value.trim().toLowerCase();
      this.selector.replaceChildren();
      for (const [id, trace] of this.traces) {
        const p = trace[0];
        if (
          p.group !== this.source.value ||
          (query && !`${id} ${this.texts[id]?.prompt || ""}`.toLowerCase().includes(query))
        )
          continue;
        const o = el(
          "option",
          `outer ${p.metadata.outer} · ${p.metadata.split} · ${id.split(":").slice(1).join(":").slice(-35)}`,
        );
        o.value = id;
        this.selector.append(o);
      }
      const match = [...this.selector.options].find(
        (o) => this.traces.get(o.value)[0].metadata.prompt === prompt,
      );
      this.empty.hidden = this.selector.options.length > 0;
      this.body.hidden = this.transport.hidden = this.steps.hidden = !this.empty.hidden;
      if (this.empty.hidden) this.select(match?.value || this.selector.value);
      else this.stop();
    }
    select(id, order = 0) {
      if (!this.traces.has(id)) return;
      this.stop();
      const p = this.traces.get(id)[0];
      if (this.source.value !== p.group || ![...this.selector.options].some((o) => o.value === id)) {
        this.source.value = p.group;
        this.search.value = "";
        this.populate();
      }
      this.id = id;
      this.selector.value = id;
      this.setStep(order);
    }
    move(offset) {
      const n = this.selector.options.length;
      if (!n) return;
      this.selector.selectedIndex = (this.selector.selectedIndex + offset + n) % n;
      this.select(this.selector.value);
    }
    setStep(step) {
      const trace = this.traces.get(this.id);
      if (!trace) return;
      this.step = Math.max(0, Math.min(trace.length - 1, step));
      this.render();
      this.onStep(this.current());
    }
    setCluster(k, cluster = null) {
      this.k = String(k);
      this.cluster = cluster;
      this.render();
    }
    toggle() {
      if (this.timer) {
        this.stop();
        return;
      }
      if (this.step === this.traces.get(this.id).length - 1) this.setStep(0);
      this.play.textContent = "暂停 Ⅱ";
      this.play.setAttribute("aria-pressed", "true");
      this.timer = setInterval(() => {
        if (this.step >= this.traces.get(this.id).length - 1) this.stop();
        else this.setStep(this.step + 1);
      }, 1000);
    }
    stop() {
      clearInterval(this.timer);
      this.timer = null;
      this.play.textContent = "播放 ▶";
      this.play.setAttribute("aria-pressed", "false");
    }
    render() {
      const trace = this.traces.get(this.id);
      if (!trace) return;
      const p = this.current(),
        g = this.groups.find((g) => g.id === p.group),
        cluster = p.metadata.clusters?.[this.k];
      this.title.textContent = `${g.label} · outer ${p.metadata.outer} · ${p.metadata.split}`;
      this.prompt.textContent = this.texts[this.id]?.prompt || p.metadata.prompt || this.id;
      this.info.textContent = `状态 ${this.step + 1}/${trace.length} · action ${p.metadata.action}/${p.metadata.response_tokens} · cluster ${cluster} (K${this.k})`;
      this.context.textContent =
        p.metadata.action === 0
          ? "回答尚未开始（prompt-end state）"
          : this.texts[this.id]?.contexts?.[String(p.order)] || "此缓存没有文本片段";
      this.slider.max = trace.length - 1;
      this.slider.value = this.step;
      this.progress.textContent = `${Math.round(p.position * 100)}% · ${this.step + 1}/${trace.length}`;
      const xs = trace.map((p) => p.x),
        ys = trace.map((p) => p.y),
        xmin = Math.min(...xs),
        ymin = Math.min(...ys),
        xmax = Math.max(...xs),
        ymax = Math.max(...ys);
      const scale = Math.min(500 / Math.max(xmax - xmin, 1), 240 / Math.max(ymax - ymin, 1));
      const pos = (p) => [300 + (p.x - (xmin + xmax) / 2) * scale, 170 - (p.y - (ymin + ymax) / 2) * scale];
      this.svg.replaceChildren();
      const node = (tag, attrs) => {
        const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
        this.svg.append(n);
        return n;
      };
      trace.slice(1).forEach((p, i) => {
        const [x, y] = pos(trace[i]),
          [xx, yy] = pos(p),
          angle = Math.atan2(yy - y, xx - x),
          ax = xx - 13 * Math.cos(angle),
          ay = yy - 13 * Math.sin(angle);
        node("path", {
          d: `M${x} ${y}L${xx} ${yy}`,
          stroke: "var(--muted)",
          "stroke-width": 1.5,
          opacity: i < this.step ? 0.85 : 0.25,
        });
        node("path", {
          d: `M${ax - 7 * Math.cos(angle - 0.5)} ${ay - 7 * Math.sin(angle - 0.5)}L${ax} ${ay}L${ax - 7 * Math.cos(angle + 0.5)} ${ay - 7 * Math.sin(angle + 0.5)}`,
          fill: "none",
          stroke: "var(--muted)",
        });
      });
      this.steps.replaceChildren();
      trace.forEach((p, i) => {
        const [x, y] = pos(p),
          active = i === this.step,
          c = p.metadata.clusters?.[this.k],
          matched = this.cluster === null || this.cluster === c;
        const circle = node("circle", {
          cx: x,
          cy: y,
          r: active ? 15 : 11,
          fill: active ? "var(--ink)" : "var(--surface)",
          stroke: matched ? g.color : "var(--line)",
          "stroke-width": active ? 3 : 2,
          tabindex: 0,
          role: "button",
          "aria-label": `状态 ${i + 1} 簇 ${c}`,
        });
        circle.addEventListener("click", () => {
          this.stop();
          this.setStep(i);
        });
        circle.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            this.stop();
            this.setStep(i);
          }
        });
        const text = node("text", {
          x,
          y: y + 4,
          "text-anchor": "middle",
          fill: active ? "var(--surface)" : "var(--ink)",
          "font-size": 11,
          "pointer-events": "none",
        });
        text.textContent = i + 1;
        const b = el("button");
        b.className = "trajectory-state";
        b.setAttribute("aria-pressed", String(active));
        b.textContent = `${i + 1} · ${Math.round(p.position * 100)}%\ncluster ${c}`;
        if (this.cluster !== null && c === this.cluster) b.classList.add("cluster-match");
        b.addEventListener("click", () => {
          this.stop();
          this.setStep(i);
        });
        this.steps.append(b);
      });
    }
    destroy() {
      this.stop();
      this.root.replaceChildren();
    }
  }
  global.TrajectoryExplorer = TrajectoryExplorer;
})(globalThis);
