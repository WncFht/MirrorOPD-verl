/* Reusable equal-scale small multiples, adapted from opd-latent-1109v2.html.
 * points: {x,y,group,trajectory?,order?,position?,metadata?}; groups: {id,label,color}.
 * Filtering changes display only. No fitting, metrics, experiment IDs or network calls.
 */
(function (global) {
  "use strict";
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  class LatentScatter {
    constructor(
      root,
      {
        points,
        groups,
        onSelect = () => {},
        tooltip = (p) => `${p.group}\n${p.trajectory}\n${JSON.stringify(p.metadata || {}, null, 2)}`,
      },
    ) {
      if (!points.length || points.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y)))
        throw Error("Finite nonempty coordinates required");
      if (points.some((p) => !groups.some((g) => g.id === p.group))) throw Error("Unknown scatter group");
      this.root = root;
      this.points = points;
      this.groups = groups;
      this.tooltip = tooltip;
      this.onSelect = onSelect;
      this.cluster = null;
      this.k = "200";
      this.activePoint = null;
      this.mode = "group";
      this.selected = null;
      this.split = "all";
      this.includeStart = true;
      this.background = true;
      this.pointSize = 1.6;
      this.focus = "all";
      this.isolate = false;
      this.overlay = "";
      this.view = { zoom: 1, dx: 0, dy: 0 };
      this.panels = [];
      this.destroyed = false;
      this.bounds = points.reduce(
        (b, p) => [Math.min(b[0], p.x), Math.max(b[1], p.x), Math.min(b[2], p.y), Math.max(b[3], p.y)],
        [Infinity, -Infinity, Infinity, -Infinity],
      );
      this.extent = Math.max(this.bounds[1] - this.bounds[0], this.bounds[3] - this.bounds[2], 1e-8);
      root.classList.add("latent-scatter");
      const controls = element("div", "ls-controls");
      this.buttons = {};
      for (const mode of ["group", "position"]) {
        const button = element("button", "", mode === "group" ? "来源" : "响应位置");
        button.type = "button";
        button.addEventListener("click", () => this.update({ mode }));
        controls.append(button);
        this.buttons[mode] = button;
      }
      const select = (label, entries, changed) => {
        const wrapper = element("label", "", label),
          input = element("select");
        for (const [value, text] of entries) {
          const option = element("option", "", text);
          option.value = value;
          input.append(option);
        }
        input.addEventListener("change", () => changed(input.value));
        wrapper.append(input);
        controls.append(wrapper);
        return input;
      };
      const splits = [...new Set(points.map((p) => p.metadata?.split).filter(Boolean))];
      this.splitSelect = select(
        "展示集",
        [
          ["all", "全部"],
          ...splits.map((s) => [
            s,
            s === "reference"
              ? "拟合候选 reference"
              : s === "test"
                ? "训练测试 test"
                : s === "query"
                  ? "Held-out bank query"
                  : s,
          ]),
        ],
        (split) => this.update({ split }),
      );
      select("叠加来源", [["", "无"], ...groups.map((g) => [g.id, g.label])], (overlay) =>
        this.update({ overlay }),
      );
      const checkbox = (label, initial, change) => {
        const wrapper = element("label", "ls-check"),
          input = element("input");
        input.type = "checkbox";
        input.checked = initial;
        input.addEventListener("change", () => change(input.checked));
        wrapper.append(input, document.createTextNode(label));
        controls.append(wrapper);
        return input;
      };
      checkbox("灰色背景", true, (background) => this.update({ background }));
      this.startCheck = checkbox("含起点", true, (includeStart) => this.update({ includeStart }));
      for (const [text, action] of [
        ["重置视图", () => this.reset()],
        ["导出地图 PNG ↓", () => this.download()],
      ]) {
        const button = element("button", "ls-quiet", text);
        button.type = "button";
        button.addEventListener("click", action);
        controls.append(button);
      }
      this.focusSelect = select(
        "查看",
        [["all", "六图总览"], ...groups.map((g) => [g.id, g.label + " · 放大"])],
        (focus) => this.update({ focus }),
      );
      const sizeLabel = element("label", "", "点大小"),
        size = element("input");
      size.type = "range";
      size.min = "0.7";
      size.max = "3";
      size.step = "0.1";
      size.value = this.pointSize;
      size.setAttribute("aria-label", "地图点大小");
      size.addEventListener("input", () => this.update({ pointSize: Number(size.value) }));
      sizeLabel.append(size);
      controls.append(sizeLabel);
      checkbox("突出选中轨迹", false, (isolate) => this.update({ isolate }));
      root.append(controls);
      this.legend = element("div", "ls-color-legend");
      root.append(this.legend);
      const trajectoryBar = element("div", "ls-trajectory");
      this.selection = element(
        "div",
        "ls-selection",
        "悬停查看状态；点击某个点后，该条回答的状态按顺序连接。",
      );
      this.selection.style.minHeight = "58px";
      this.selection.setAttribute("aria-live", "polite");
      const clear = element("button", "ls-quiet", "清除轨迹");
      clear.type = "button";
      clear.addEventListener("click", () => this.update({ selected: null, activePoint: null }));
      trajectoryBar.append(this.selection, clear);
      root.append(trajectoryBar);
      const grid = element("div", "ls-grid");
      root.append(grid);
      for (const group of groups) {
        const panel = element("section", "ls-panel"),
          header = element("div", "ls-header"),
          title = element("h4", "", group.label);
        title.style.setProperty("--group-color", group.color);
        const count = element("span", "ls-count"),
          canvas = element("canvas");
        canvas.tabIndex = 0;
        canvas.setAttribute("aria-label", group.label + " 状态地图；+/- 缩放，0 重置，Esc 清除轨迹");
        const expand = element("button", "ls-expand", "放大");
        expand.addEventListener("click", () => {
          this.update({ focus: this.focus === group.id ? "all" : group.id });
          panel.scrollIntoView({ block: "nearest" });
        });
        header.append(title, count, expand);
        panel.append(header, canvas);
        grid.append(panel);
        const item = {
          panel,
          expand,
          canvas,
          count,
          group,
          indices: points.map((p, i) => (p.group === group.id ? i : -1)).filter((i) => i >= 0),
        };
        let drag = null;
        canvas.addEventListener("pointerdown", (e) => {
          if (e.button !== 0) return;
          drag = { x: e.clientX, y: e.clientY, dx: this.view.dx, dy: this.view.dy, moved: false };
          canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener("pointermove", (e) => {
          if (drag) {
            const rect = canvas.getBoundingClientRect(),
              scale = this.scale(rect.width, rect.height);
            drag.moved ||= Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4;
            if (drag.moved) {
              this.view.dx = drag.dx + (e.clientX - drag.x) / scale;
              this.view.dy = drag.dy - (e.clientY - drag.y) / scale;
              this.draw();
            }
            return;
          }
          this.hover(item, e);
        });
        canvas.addEventListener("pointerup", (e) => {
          if (!drag) return;
          const moved = drag.moved;
          drag = null;
          if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
          if (!moved) {
            const point = this.hit(item, e);
            this.update({ selected: point?.trajectory || null });
            if (point) {
              this.selection.textContent = this.tooltip(point);
              this.onSelect(point);
            }
          }
        });
        canvas.addEventListener("pointercancel", () => {
          drag = null;
        });
        canvas.addEventListener(
          "wheel",
          (e) => {
            e.preventDefault();
            this.view.zoom = Math.max(1, Math.min(12, this.view.zoom * Math.exp(-e.deltaY * 0.001)));
            this.draw();
          },
          { passive: false },
        );
        canvas.addEventListener("dblclick", () => this.reset());
        canvas.addEventListener("keydown", (e) => {
          if (e.key === "Escape") this.update({ selected: null });
          else if (e.key === "0") this.reset();
          else if (["+", "=", "-"].includes(e.key)) {
            e.preventDefault();
            this.view.zoom = Math.max(1, Math.min(12, this.view.zoom * (e.key === "-" ? 0.8 : 1.25)));
            this.draw();
          }
        });
        this.panels.push(item);
      }
      this.observer = new ResizeObserver(() => this.draw());
      this.observer.observe(root);
      this.draw();
    }
    visible(p) {
      return (
        (this.split === "all" || p.metadata?.split === this.split) &&
        (this.includeStart || (p.metadata?.action ?? p.position ?? 1) > 0)
      );
    }
    scale(width, height) {
      return (Math.max(1, Math.min(width - 24, height - 24)) / this.extent) * this.view.zoom;
    }
    project(width, height) {
      const [x0, x1, y0, y1] = this.bounds,
        scale = this.scale(width, height);
      return (p) => [
        width / 2 + (p.x - (x0 + x1) / 2 + this.view.dx) * scale,
        height / 2 - (p.y - (y0 + y1) / 2 + this.view.dy) * scale,
      ];
    }
    hit(item, event) {
      const rect = item.canvas.getBoundingClientRect(),
        project = this.project(rect.width, rect.height),
        mx = event.clientX - rect.left,
        my = event.clientY - rect.top;
      let best = null,
        distance = 81;
      for (const i of item.indices) {
        const point = this.points[i];
        if (!this.visible(point)) continue;
        const [x, y] = project(point),
          d = (x - mx) ** 2 + (y - my) ** 2;
        if (d < distance) {
          best = point;
          distance = d;
        }
      }
      return best;
    }
    hover(item, event) {
      const point = this.hit(item, event);
      if (point) this.selection.textContent = this.tooltip(point);
    }
    update(options = {}) {
      for (const key of [
        "mode",
        "selected",
        "split",
        "includeStart",
        "background",
        "overlay",
        "cluster",
        "k",
        "activePoint",
        "focus",
        "pointSize",
        "isolate",
      ])
        if (key in options) this[key] = options[key];
      if (this.startCheck) this.startCheck.checked = this.includeStart;
      this.splitSelect.value = this.split;
      this.focusSelect.value = this.focus;
      this.root.classList.toggle("ls-focused", this.focus !== "all");
      for (const item of this.panels) {
        item.panel.hidden = this.focus !== "all" && item.group.id !== this.focus;
        item.expand.textContent = this.focus === item.group.id ? "返回六图" : "放大";
      }
      if (!this.selected)
        this.selection.textContent = "悬停查看状态；点击某个点后，该条回答的状态按顺序连接。";
      this.draw();
    }
    reset() {
      this.view = { zoom: 1, dx: 0, dy: 0 };
      this.draw();
    }
    draw() {
      if (this.destroyed) return;
      cancelAnimationFrame(this.frame);
      this.frame = requestAnimationFrame(() => this.render());
    }
    render() {
      const style = getComputedStyle(this.root),
        cloud = style.getPropertyValue("--cloud").trim() || "#cbd3dc",
        dark = document.documentElement.dataset.theme === "dark";
      for (const [mode, button] of Object.entries(this.buttons))
        button.setAttribute("aria-pressed", String(this.mode === mode));
      this.legend.replaceChildren();
      if (this.mode === "position") {
        this.legend.append(
          element("span", "", "回答开头 0%"),
          element("i", "ls-position-gradient"),
          element("span", "", "回答末尾 100%"),
        );
      } else this.legend.textContent = "彩色点：当前来源 · 淡灰点：其他来源背景 · 图中圆圈：当前轨迹状态";
      for (const { canvas, group, indices, count, panel } of this.panels) {
        if (panel.hidden) continue;
        const { width, height } = canvas.getBoundingClientRect();
        if (width <= 0 || height <= 0) continue;
        const dpr = global.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(width * dpr));
        canvas.height = Math.max(1, Math.round(height * dpr));
        const ctx = canvas.getContext("2d"),
          project = this.project(width, height);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const dot = (point, radius, color, alpha = 1) => {
          const [x, y] = project(point);
          if (x < 0 || y < 0 || x > width || y > height) return;
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.globalAlpha = alpha;
          ctx.fill();
        };
        if (this.background)
          for (const point of this.points) if (this.visible(point)) dot(point, 0.6, cloud, dark ? 0.1 : 0.16);
        if (this.overlay && this.overlay !== group.id)
          for (const point of this.points)
            if (point.group === this.overlay && this.visible(point))
              dot(point, 1.5, dark ? "#e3eaf0" : "#3A3F45", 0.28);
        const visible = indices.filter((i) => this.visible(this.points[i]));
        for (const i of visible) {
          const point = this.points[i],
            fraction = Math.min(1, Math.max(0, point.position || 0));
          const color =
            this.mode === "position"
              ? `hsl(${220 - 195 * fraction} 85% ${dark ? 64 : 43}%)`
              : dark && ["#3A3F45", "#85898F"].includes(group.color)
                ? "#c0cad4"
                : dark
                  ? ["#cbd5e1", "#53c4dc", "#ffb060", "#ff7b83", "#b49aff", "#f1f5f9"][
                      this.groups.indexOf(group) % 6
                    ]
                  : group.color;
          const matches = this.cluster === null || point.metadata?.clusters?.[this.k] === this.cluster;
          dot(
            point,
            matches && this.cluster !== null ? this.pointSize + 1.2 : this.pointSize,
            color,
            !matches
              ? 0.08
              : this.isolate && this.selected && point.trajectory !== this.selected
                ? 0.12
                : 0.85,
          );
        }
        if (this.selected) {
          const trajectory = visible
            .map((i) => this.points[i])
            .filter((p) => p.trajectory === this.selected)
            .sort((a, b) => a.order - b.order);
          ctx.beginPath();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = dark ? "#e3eaf0" : group.color;
          ctx.lineWidth = 1.7;
          trajectory.forEach((p, i) => {
            const [x, y] = project(p);
            if (i) ctx.lineTo(x, y);
            else ctx.moveTo(x, y);
          });
          ctx.stroke();
          for (const [index, point] of trajectory.entries()) {
            dot(point, 5, dark ? "#fff" : "#172d3a");
            const [x, y] = project(point);
            ctx.font = "bold 12px system-ui";
            ctx.textAlign = "center";
            ctx.lineWidth = 3;
            ctx.strokeStyle = dark ? "#11171e" : "#fff";
            ctx.strokeText(String(index + 1), x, y - 10);
            ctx.fillStyle = dark ? "#fff" : "#172d3a";
            ctx.fillText(String(index + 1), x, y - 10);
          }
        }
        if (this.activePoint?.group === group.id) {
          const [x, y] = project(this.activePoint);
          ctx.globalAlpha = 1;
          ctx.strokeStyle = dark ? "#fff" : "#172d3a";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        count.textContent = `${new Set(visible.map((i) => this.points[i].trajectory)).size} 回答 · ${visible.length} 点`;
      }
    }
    download() {
      this.render();
      const canvas = document.createElement("canvas"),
        width = 900,
        height = 350;
      canvas.width = 3 * width;
      canvas.height = Math.ceil(this.panels.length / 3) * (height + 50);
      const ctx = canvas.getContext("2d"),
        style = getComputedStyle(this.root);
      ctx.fillStyle = style.getPropertyValue("--surface").trim() || "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      this.panels.forEach((p, i) => {
        const x = (i % 3) * width,
          y = Math.floor(i / 3) * (height + 50);
        ctx.fillStyle = style.getPropertyValue("--ink").trim() || "#202933";
        ctx.font = "22px system-ui";
        ctx.fillText(p.group.label, x + 20, y + 30);
        const ratio = Math.min((width - 20) / p.canvas.width, height / p.canvas.height),
          w = p.canvas.width * ratio,
          h = p.canvas.height * ratio;
        ctx.drawImage(p.canvas, x + (width - w) / 2, y + 45, w, h);
      });
      const anchor = document.createElement("a");
      anchor.download = "latent-state-map.png";
      anchor.href = canvas.toDataURL("image/png");
      anchor.click();
    }
    destroy() {
      this.destroyed = true;
      cancelAnimationFrame(this.frame);
      this.observer.disconnect();
      this.root.replaceChildren();
      this.root.classList.remove("latent-scatter");
    }
  }
  global.LatentScatter = LatentScatter;
})(globalThis);
