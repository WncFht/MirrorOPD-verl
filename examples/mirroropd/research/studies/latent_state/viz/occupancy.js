/* Cluster comparison table: common scales, exact values, teacher differences and linked examples. */
(function (global) {
  "use strict";
  const el = (tag, text, className) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (className) n.className = className;
    return n;
  };
  const names = {
    initial: "Initial",
    outer01_10: "01–10",
    outer11_20: "11–20",
    outer21_30: "21–30",
    outer31_40: "31–40",
    teacher: "Teacher",
  };
  class ClusterOccupancy {
    constructor(root, { groups, points, onSelect = () => {}, onPoint = () => {} }) {
      Object.assign(this, {
        root,
        groups: groups.filter((g) => g.id !== "training_test"),
        points,
        onSelect,
        onPoint,
        selected: null,
        page: 0,
        pageSize: 20,
      });
      const controls = el("div", undefined, "control-row cluster-controls");
      const select = (label, options, change) => {
        const wrap = el("label", label),
          input = el("select");
        input.setAttribute("aria-label", label);
        for (const [value, name] of options) {
          const o = el("option", name);
          o.value = value;
          input.append(o);
        }
        input.addEventListener("change", () => {
          this.page = 0;
          change();
          this.render();
        });
        wrap.append(input);
        controls.append(wrap);
        return input;
      };
      this.mode = select(
        "显示",
        [
          ["delta", "与 Teacher 的差值"],
          ["share", "状态占比"],
        ],
        () => {},
      );
      this.sort = select(
        "排序",
        [
          ["gap", "末段与 Teacher 差异"],
          ["teacher", "Teacher 占比"],
          ["change", "训练前后变化"],
          ["id", "簇编号"],
        ],
        () => {},
      );
      this.nonempty = el("input");
      this.nonempty.type = "checkbox";
      this.nonempty.checked = true;
      const check = el("label", "仅显示访问过的簇");
      check.prepend(this.nonempty);
      controls.append(check);
      this.nonempty.addEventListener("change", () => {
        this.page = 0;
        this.render();
      });
      const clear = el("button", "清除选择");
      clear.addEventListener("click", () => this.select(null));
      controls.append(clear);
      root.append(controls);
      this.legend = el("div", undefined, "cluster-scale");
      root.append(this.legend);
      const layout = el("div", undefined, "cluster-layout"),
        wrap = el("div", undefined, "cluster-table-wrap");
      this.table = el("table", undefined, "cluster-comparison");
      wrap.append(this.table);
      this.info = el("aside", undefined, "cluster-detail");
      layout.append(wrap, this.info);
      root.append(layout);
      this.pager = el("div", undefined, "cluster-pagination");
      root.append(this.pager);
    }
    update(k, metrics) {
      this.k = String(k);
      this.metrics = metrics;
      this.page = 0;
      this.selected = null;
      this.onSelect(null);
      this.render();
    }
    reveal(id) {
      this.sort.value = "id";
      this.nonempty.checked = false;
      this.page = Math.floor(id / this.pageSize);
      this.select(id);
    }
    select(id) {
      this.selected = id;
      this.onSelect(id);
      this.render();
    }
    render() {
      if (!this.metrics) return;
      const m = this.metrics,
        ids = Array.from({ length: Number(this.k) }, (_, i) => i).filter(
          (i) => !this.nonempty.checked || this.groups.some((g) => m[g.id].occupancy[i] > 0),
        );
      const score = (i) =>
        this.sort.value === "gap"
          ? Math.abs(m.outer31_40.occupancy[i] - m.teacher.occupancy[i])
          : this.sort.value === "teacher"
            ? m.teacher.occupancy[i]
            : this.sort.value === "change"
              ? Math.abs(m.outer31_40.occupancy[i] - m.outer01_10.occupancy[i])
              : -i;
      ids.sort((a, b) => score(b) - score(a) || a - b);
      const delta = this.mode.value === "delta";
      const max = Math.max(
        0.0001,
        ...ids.flatMap((i) =>
          this.groups.map((g) =>
            delta ? Math.abs(m[g.id].occupancy[i] - m.teacher.occupancy[i]) : m[g.id].occupancy[i],
          ),
        ),
      );
      this.legend.replaceChildren();
      this.legend.append(
        el("span", delta ? "蓝：少于 Teacher" : "每格占该来源状态的比例"),
        el("i", undefined, delta ? "cluster-diverging" : "cluster-sequential"),
        el("span", delta ? "橙：多于 Teacher · 单位：百分点" : "白：0% · 深蓝：占比高"),
      );
      this.table.replaceChildren();
      const head = el("thead"),
        tr = el("tr");
      tr.append(el("th", "簇"));
      for (const g of this.groups) {
        const th = el("th", g.id === "teacher" ? "Teacher %" : names[g.id] || g.label);
        if (g.id === "teacher") th.className = "teacher-column";
        tr.append(th);
      }
      head.append(tr);
      this.table.append(head);
      const body = el("tbody");
      for (const i of ids.slice(this.page * this.pageSize, (this.page + 1) * this.pageSize)) {
        const row = el("tr");
        row.classList.toggle("selected", i === this.selected);
        const label = el("th"),
          pick = el("button", `#${i}`);
        pick.setAttribute("aria-label", `选择簇 ${i}`);
        pick.setAttribute("aria-pressed", String(i === this.selected));
        pick.addEventListener("click", () => this.select(i === this.selected ? null : i));
        label.append(pick);
        row.append(label);
        for (const g of this.groups) {
          const share = m[g.id].occupancy[i],
            diff = share - m.teacher.occupancy[i],
            teacher = g.id === "teacher";
          const td = el("td");
          if (teacher) td.className = "teacher-column";
          const b = el(
            "button",
            delta && !teacher
              ? `${diff > 0 ? "+" : ""}${(100 * diff).toFixed(2)}`
              : `${(100 * share).toFixed(2)}%`,
            "cluster-value",
          );
          const strength = delta && teacher ? 0 : Math.abs(delta ? diff : share) / max;
          b.style.setProperty("--fill", `${Math.min(100, strength * 100)}%`);
          b.style.setProperty(
            "--tone",
            delta && !teacher
              ? diff < 0
                ? "var(--cluster-blue)"
                : "var(--cluster-orange)"
              : "var(--cluster-blue)",
          );
          b.setAttribute(
            "aria-label",
            `${names[g.id]} 簇 ${i}，占比 ${(100 * share).toFixed(2)}%，与 Teacher 差 ${(100 * diff).toFixed(2)} 个百分点`,
          );
          b.addEventListener("click", () => this.select(i));
          td.append(b);
          row.append(td);
        }
        body.append(row);
      }
      this.table.append(body);
      this.pager.replaceChildren();
      const pages = Math.max(1, Math.ceil(ids.length / this.pageSize));
      const prev = el("button", "← 上一页"),
        next = el("button", "下一页 →");
      prev.disabled = this.page === 0;
      next.disabled = this.page >= pages - 1;
      prev.addEventListener("click", () => {
        this.page--;
        this.render();
      });
      next.addEventListener("click", () => {
        this.page++;
        this.render();
      });
      this.pager.append(prev, el("span", `${this.page + 1} / ${pages} · ${ids.length} 个簇`), next);
      this.inspect(this.selected ?? ids[this.page * this.pageSize]);
    }
    inspect(id) {
      this.info.replaceChildren();
      if (id === undefined) return;
      this.info.append(
        el("div", this.selected === null ? "当前页示例" : "已选择", "eyebrow"),
        el("h3", `Cluster ${id}`),
      );
      const values = this.groups.map((g) => this.metrics[g.id].occupancy[id]),
        max = Math.max(0.0001, ...values),
        teacher = this.metrics.teacher.occupancy[id];
      const bars = el("div", undefined, "cluster-bars");
      for (const g of this.groups) {
        const value = this.metrics[g.id].occupancy[id],
          row = el("div", undefined, "cluster-bar-row");
        row.append(el("span", names[g.id]));
        const track = el("div", undefined, "cluster-bar-track"),
          fill = el("i");
        fill.style.width = `${(100 * value) / max}%`;
        fill.style.background = g.id === "teacher" ? "var(--ink)" : "var(--accent)";
        track.append(fill);
        row.append(track, el("b", `${(100 * value).toFixed(2)}%`));
        bars.append(row);
      }
      this.info.append(bars);
      const delta = 100 * (this.metrics.outer31_40.occupancy[id] - teacher);
      this.info.append(
        el("p", `末段 − Teacher：${delta > 0 ? "+" : ""}${delta.toFixed(2)} 个百分点`, "caption"),
      );
      const examples = el("div", undefined, "cluster-examples");
      for (const g of this.groups) {
        const point = this.points.find(
          (p) =>
            p.group === g.id &&
            p.metadata.action > 0 &&
            ["test", "query"].includes(p.metadata.split) &&
            p.metadata.clusters?.[this.k] === id,
        );
        if (!point) continue;
        const b = el("button", `${names[g.id]} 回答 ↗`);
        b.addEventListener("click", () => {
          this.selected = id;
          this.onSelect(id);
          this.onPoint(point);
        });
        examples.append(b);
      }
      this.info.append(examples);
    }
    destroy() {
      this.root.replaceChildren();
    }
  }
  global.ClusterOccupancy = ClusterOccupancy;
})(globalThis);
