# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Build auditable per-run figure bundles from the curated overview sources.

Each run owns a stable directory, normalized CSVs, PNG/PDF pairs, source hashes,
coverage and a manifest. Rendering is CPU-only and does not import a trainer.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv, write_text
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.fixed_transfer.figure_data import GROUPS, load_run, registry
from examples.mirroropd.research.studies.fixed_transfer.figure_render import Renderer

BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism"
TABLES = ["training", "curves", "effects", "forgetting", "questions", "positions"]


def build(spec, output, force=False):
    data = load_run(spec)
    code = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ["figure_data.py", "figure_render.py", "figure_pipeline.py", "overview.py", "display_policy.py"]
    }
    fingerprint = hashlib.sha256(
        json.dumps(dict(spec=spec, sources=data["sources"], code=code), sort_keys=True).encode()
    ).hexdigest()
    target = output / spec["scope"] / spec["run_id"]
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force:
        previous = read(manifest_path)
        if previous["fingerprint"] == fingerprint and all(
            (target / p).exists() and file_hash(target / p) == h for p, h in previous["artifacts"].items()
        ):
            return previous
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".render-", dir=target.parent) as temp:
        stage = Path(temp)
        for name in TABLES:
            if data[name]:
                write_csv(stage / f"{name}.csv", data[name])
        dump(stage / "sources.json", data["sources"])
        renderer = Renderer(data, stage)
        figures = renderer.render()
        seen = {r["group"] for r in data["snapshots"][0]}
        coverage = dict(
            snapshot_steps=list(data["snapshots"]),
            training_steps=[r["step"] for r in data["training"]],
            missing_snapshot_steps=sorted(set(range(0, spec["budget"] + 1, data["reuse"])) - data["snapshots"].keys()),
            missing_training_steps=sorted(set(range(1, spec["budget"] + 1)) - {r["step"] for r in data["training"]}),
            unscored_training_batches=[g for g in GROUPS if g not in seen],
            panels=data["panels"],
            horizon=data["horizon"],
            reuse=data["reuse"],
            unavailable=["per-token trajectory", "teacher-rollout evaluation", "Adam moment history"],
        )
        if not data["positions"]:
            coverage["unavailable"].append("position-band statistics")
        if not any("parameter_step_norm" in r for r in data["training"]):
            coverage["unavailable"].append("parameter displacement history")
        dump(stage / "coverage.json", coverage)
        held = [
            r
            for r in data["curves"]
            if r["panel"] == "All held-out" and r["metric"] == "rkl" and r["step"] == spec["budget"]
        ]
        summary = dict(
            run_id=spec["run_id"],
            scope=spec["scope"],
            student=spec["student"],
            teacher=spec["teacher"],
            student_label=renderer.student[0],
            teacher_label=renderer.teacher[0],
            mode=renderer.student[2],
            budget=spec["budget"],
            support=spec["support"],
            figures=figures,
            held_reduction=held[0]["reduction"] if held else None,
            fingerprint=fingerprint,
            generator=code,
            coverage=coverage,
        )
        lines = [
            f"# {renderer.teacher[0]} → {renderer.student[0]}",
            "",
            f"输入：{renderer.student[2]}。教师属性：{renderer.teacher[2]}。"
            f"同尺寸：{'是' if renderer.style == '-' else '否'}。",
            "",
            f"显示范围：0–{spec['budget']} 更新；支持：`{spec['support']}`；冻结初始 student state。",
            "",
            "图中实线表示同尺寸、虚线表示跨尺寸；散点与热图通过本页身份说明区分尺寸。曲线仅连接已有观测，不生成中间数据。",
            "",
            "训练日志为更新前 loss，固定评分为更新后读出；两者不拼成一条曲线。旧批 A–D 在第 64 步后才全部训练过。",
            "",
            f"评分覆盖 {len(data['snapshots'])} 个快照，训练日志 {len(data['training'])} 步。"
            f"未评分训练批：{', '.join(coverage['unscored_training_batches']) or '无'}。",
            "",
            "held-out 总体按题数合并 Math／Code／IF（64／32／32），不是三个相对降幅的算术平均。"
            "每个实验只有一个训练 seed。",
            "",
            "来源目录可能带历史延长预算；本页仅读取 0–320 范围内的快照和训练日志。",
            "",
            "[返回索引](../../README.md) · [来源哈希](sources.json)"
            " · [覆盖清单](coverage.json) · [产物清单](manifest.json)",
            "",
            "## 归一化与可用性",
            "",
            "主风险为各题累计指标除以题数 × H1024。"
            "位置图用实际观测 token 数归一化，并在 CSV 记录覆盖；早停响应的后段覆盖较少。未评分的训练批不以零补齐。",
            "",
            "未补算逐 token 轨迹、teacher rollout 指标或 Adam 历史。"
            "teacher-copy 参考仅在已有独立实验中解释，不伪造为每个 run 的零线。",
            "",
            "批次影响是每段更新前后风险之差，包含优化器历史；遗忘以本批刚训练完的读出为锚点。"
            "工程图仅描述既有日志，不据此比较跨实现的 GPU 利用率。",
            "",
            "## 数据表",
            "",
        ]
        lines += [f"- [{name}.csv]({name}.csv)" for name in TABLES if data[name]]
        for fig in figures:
            lines += [
                "",
                f"## {fig['title']}",
                "",
                f"![{fig['title']}]({fig['key']}.png)",
                "",
                f"[矢量 PDF]({fig['key']}.pdf) · [图形审计]({fig['key']}.audit.json)",
            ]
        write_text(stage / "README.md", "\n".join(lines) + "\n")
        summary["artifacts"] = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        dump(stage / "manifest.json", summary)
        target.mkdir(parents=True, exist_ok=True)
        # Publish the receipt last; interrupted builds cannot be mistaken for valid cache hits.
        for p in sorted(stage.iterdir(), key=lambda p: p.name == "manifest.json"):
            os.replace(p, target / p.name)
    return summary


def index(results, output, registry_path):
    rows = []
    lines = [
        "# 每个 run 的诊断图谱",
        "",
        "每个目录保存同一 run 的图、数值表、来源哈希和覆盖清单。仅展示 0–320 更新；"
        "排序按全部 held-out 的 RKL 相对降幅，预算不同的行不视为等预算比较。",
        "",
        "[图谱与重绘说明](../../figure-plan.md) · [实验总览](../../../README.md)",
        "",
        f"共 {len(results)} 个条件、{sum(len(r['figures']) for r in results)} 张图，每张均有 PNG 与 PDF。",
        "",
    ]
    for scope, title in [("main", "主比较条件"), ("supplement", "独立附录条件")]:
        lines += [
            f"## {title}",
            "",
            "| Student／输入 | Teacher | 更新 | held-out 降幅 | 图谱 |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        selected = sorted(
            [r for r in results if r["scope"] == scope], key=lambda r: (r["student"], -(r["held_reduction"] or 0))
        )
        for r in selected:
            link = f"{scope}/{r['run_id']}/README.md"
            reduction = f"{r['held_reduction']:.2f}%" if r["held_reduction"] is not None else "缺失"
            lines.append(
                f"| {r['student_label']} / {r['mode']} | {r['teacher_label']} "
                f"| {r['budget']} | {reduction} | [打开]({link}) |"
            )
            rows.append(
                {k: r[k] for k in ["run_id", "scope", "student", "teacher", "budget", "held_reduction"]}
                | dict(figures=len(r["figures"]), path=link)
            )
        lines.append("")
    write_text(output / "README.md", "\n".join(lines) + "\n")
    write_csv(output / "runs.csv", rows)
    if output.resolve() == (BASE / "appendix/data/run-figures").resolve():
        write_csv(BASE / "appendix/run-figures.csv", rows)
    dump(
        output / "manifest.json", dict(registry=str(registry_path), runs=rows, figures=sum(r["figures"] for r in rows))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=BASE / "data/overview-20261029")
    parser.add_argument("--output", type=Path, default=BASE / "appendix/data/run-figures")
    parser.add_argument("--run", help="Exact run ID, or teacher key, for a selective rebuild")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    specs = registry(args.registry)
    if args.run:
        specs = [s for s in specs if args.run in [s["run_id"], s["teacher"]]]
    if not specs:
        raise ValueError("No runs match")
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build, spec, args.output, args.force): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"OK {spec['run_id']}: {len(result['figures'])} figures", flush=True)
            except Exception as exc:
                failures.append(dict(run=spec["run_id"], error=str(exc)))
                print(f"FAIL {spec['run_id']}: {exc}", flush=True)
    dump(args.output / "failures.json", failures)
    # Partial rebuilds retain other successful registered run receipts in the index.
    current = {s["run_id"] for s in registry(args.registry)}
    failed = {f["run"] for f in failures}
    receipts = [read(p) for p in args.output.glob("*/*/manifest.json") if p.parent.name in current - failed]
    if receipts:
        index(receipts, args.output, args.registry)
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed; see failures.json")


if __name__ == "__main__":
    main()
