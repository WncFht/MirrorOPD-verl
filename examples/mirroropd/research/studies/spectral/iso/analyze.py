# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only ISO coverage and scalar/paired summaries; never rank partial panels."""

import argparse
import gzip
import hashlib
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read


def queues(root):
    """Load explicitly chained queues, preserving each task's owning ledger."""
    root = Path(root)
    if (root / "queue-chain.json").exists():
        paths = [Path(p) for p in read(root / "queue-chain.json")["paths"]]
    else:
        active = root / "active-queue.json"
        paths = [Path(read(active)["path"]) if active.exists() else root / "offline-v1"]
    jobs = [dict(task, queue_root=str(path)) for path in paths for task in read(path / "queue.json")["jobs"]]
    return paths, jobs


def paired(rows, baseline, metric):
    lookup = {(r["origin"], r["row_id"]): r for r in baseline}
    differences = [r[metric] - lookup[r["origin"], r["row_id"]][metric] for r in rows]
    rng = np.random.default_rng(20261111)
    values = np.array(differences)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return dict(
        n=len(rows),
        absolute_mean=statistics.mean(r[metric] for r in rows),
        baseline_mean=statistics.mean(lookup[r["origin"], r["row_id"]][metric] for r in rows),
        delta=float(values.mean()),
        delta_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        inference="paired by question within origin/domain/split; one model intervention, not training seeds",
    )


def spectral_statistics(rows):
    """Report matrix medians separately from pooled Frobenius ratios."""
    valid = [r for r in rows if r["rho_sigma"] is not None]
    energy = sum(r["update_energy"] for r in rows)
    return dict(
        matrices=len(rows),
        delta_sigma_median=statistics.median(r["delta_sigma"] for r in rows),
        rho_sigma_median=statistics.median(r["rho_sigma"] for r in valid) if valid else None,
        pooled_rho_sigma=math.sqrt(sum(r["spectral_distance"] ** 2 for r in rows) / energy) if energy else None,
        kappa_median=statistics.median(r["kappa_spec"] for r in valid) if valid else None,
        absolute_update_norm=math.sqrt(energy),
        top16_median=statistics.median(r["update_spectrum"]["top_energy"]["16"] for r in valid) if valid else None,
        radial_update_fraction_median=statistics.median(r["radial_update_fraction"] for r in valid) if valid else None,
        spectrum_after_scale_fraction_median=statistics.median(r["spectrum_after_scale_fraction"] for r in valid)
        if valid
        else None,
    )


def radial_statistics(facts, base_singular):
    """Recover radial projection and uniform spectral scaling from exact saved norms.

    The polarization identity uses FP64 norms, without loading checkpoints or
    assuming that matching ordered spectra implies matching singular frames.
    """
    spectrum, base = np.asarray(facts["singular_values"]), np.asarray(base_singular)
    base_energy, update_energy = facts["base_energy"], facts["update_energy"]
    dot = (float(spectrum @ spectrum) - base_energy - update_energy) / 2
    radial = dot * dot / base_energy / update_energy if update_energy else None
    if radial is not None and radial > 1 + 1e-5:
        raise ValueError(f"Radial polarization identity violates Cauchy bound: {radial}")
    scale = float(spectrum @ base / (base @ base))
    drift = facts["spectral_distance"]
    return dict(
        facts,
        radial_update_fraction=min(radial, 1) if radial is not None else None,
        radial_scale_coefficient=dot / base_energy,
        best_spectrum_scale=scale,
        spectrum_after_scale_fraction=float(np.linalg.norm(spectrum - scale * base) / drift) if drift else 0.0,
    )


def distributions(values, fields):
    return [
        dict(
            zip(fields, key, strict=True),
            matrices=len(rows),
            median=statistics.median(rows),
            q25=float(np.quantile(rows, 0.25)),
            q75=float(np.quantile(rows, 0.75)),
            minimum=min(rows),
            maximum=max(rows),
        )
        for key, rows in values.items()
    ]


def summarize(root, output):
    root, output = Path(root), Path(output)
    manifest = read(root / "offline-manifest.json")
    labels = {j["job_id"]: j for j in manifest["jobs"]}
    queue_roots, tasks = queues(root)
    queue = dict(jobs=tasks)
    counts = defaultdict(int)
    for task in queue["jobs"]:
        counts[task["status"]] += 1
    result = dict(
        at=time.time(),
        queue_root=str(queue_roots[0]),
        queue_roots=[str(p) for p in queue_roots],
        queue_counts=dict(counts),
        coverage=queue["jobs"],
        spectra=[],
        functional=[],
        geometry=[],
        training=[],
        dapo_training=[],
        merger=[],
        paper_spectra=[],
    )
    groups = defaultdict(list)
    geometry_values, layer_values, subspace_values = defaultdict(list), defaultdict(list), defaultdict(list)
    geometry_tasks = [t for t in tasks if t["label"].startswith(("geometry-", "paper-geometry-"))]
    ready_geometry_jobs = {
        prefix: all(t["status"] == "complete" for t in geometry_tasks if t["label"].startswith(prefix))
        for prefix in ("geometry-", "paper-geometry-")
    }
    scan_done = all((root / f"scan-v1/shard-{i}/complete.json").exists() for i in range(8))
    result["scan_complete"] = scan_done
    if scan_done:
        paths = defaultdict(list)
        for job in manifest["jobs"]:
            paths[job["teacher_path"]].append((job, "teacher"))
            paths[job["endpoint"]].append((job, "endpoint"))
            for stage, path in job["gradient_stages"].items():
                if stage not in ("initial", "endpoint"):
                    paths[path].append((job, stage))
        for file in (root / "scan-v1").glob("shard-*/*.json"):
            record = read(file)
            if "targets" not in record:
                continue
            for path, facts in record["targets"].items():
                for job, role in paths[path]:
                    if job["base"] != record["base"]:
                        continue
                    groups[job["job_id"], role].append(
                        radial_statistics(facts, record["targets"][record["base"]]["singular_values"])
                    )
        for (job_id, role), rows in groups.items():
            result["spectra"].append(
                dict(
                    job_id=job_id,
                    family=labels[job_id]["family"],
                    teacher=labels[job_id]["teacher"],
                    role=role,
                    **spectral_statistics(rows),
                )
            )
        performance = [read(root / f"scan-v1/shard-{i}/performance.json") for i in range(8)]
        result["scan_performance"] = dict(
            shard_compute_makespan_seconds=max(p["wall_seconds"] for p in performance),
            total_gpu_worker_seconds=sum(p["wall_seconds"] for p in performance),
            completed=sum(p["completed"] for p in performance),
            phases_seconds={
                key: sum(p["phases_seconds"][key] for p in performance) for key in performance[0]["phases_seconds"]
            },
        )
    paper_tasks = [t for t in queue["jobs"] if t["label"].startswith("paper-scan-")]
    if paper_tasks and all(t["status"] == "complete" for t in paper_tasks):
        paper_jobs = read(root / "paper-manifest.json")["jobs"]
        paper_groups = defaultdict(list)
        for task in paper_tasks:
            dest = Path(task["output_attempt"]) / Path(task["complete_marker"]).parent
            for file in dest.glob("*.json"):
                record = read(file)
                if "targets" not in record:
                    continue
                for job in paper_jobs:
                    if job["base"] == record["base"]:
                        paper_groups[job["job_id"]].append(
                            radial_statistics(
                                record["targets"][job["teacher_path"]],
                                record["targets"][record["base"]]["singular_values"],
                            )
                        )
        for job in paper_jobs:
            result["paper_spectra"].append(
                dict(
                    job_id=job["job_id"],
                    teacher=job["teacher"],
                    interpretation=job["interpretation"],
                    **spectral_statistics(paper_groups[job["job_id"]]),
                )
            )
    for task in queue["jobs"]:
        if task["status"] != "complete":
            continue
        dest = Path(task["output_attempt"]) / Path(task.get("complete_marker", "complete.json")).parent
        label = task["label"]
        if "smoke" in label:
            continue
        if label.startswith("functional-"):
            for file in (dest / "scores").glob("*.json"):
                if file.name.endswith(".progress.json"):
                    continue
                scored = read(file)
                condition = scored["condition"]
                baseline = read(dest / "scores" / f"{condition['role']}-restore-alpha-1.0.json")["rows"]
                buckets = defaultdict(list)
                for row in scored["rows"]:
                    buckets[row["origin"], row["split"], row["domain"]].append(row)
                for identity, rows in buckets.items():
                    result["functional"].append(
                        dict(
                            task=label,
                            condition=condition,
                            origin=identity[0],
                            split=identity[1],
                            domain=identity[2],
                            rkl=paired(rows, baseline, "rkl"),
                            fkl=paired(rows, baseline, "fkl"),
                        )
                    )
        elif label.startswith("geometry-") or label.startswith("paper-geometry-"):
            values = defaultdict(list)
            for file in dest.glob("[0-9]*.json"):
                record = read(file)
                layer = int(re.search(r"\.layers\.(\d+)\.", record["tensor"])[1])
                panel_ready = ready_geometry_jobs["paper-geometry-" if label.startswith("paper-") else "geometry-"]
                for row in record["rows"]:
                    for transport in row["transport"]:
                        if transport["admissible"]:
                            for kind, value in transport["unexplained"].items():
                                if value is not None:
                                    values[row["job_id"], row["transition"], transport["fraction"], kind].append(value)
                                    if panel_ready:
                                        key = (row["job_id"], row["transition"], transport["fraction"], kind)
                                        geometry_values[key].append(value)
                                        layer_values[(*key, layer)].append(value)
                    if panel_ready:
                        for rank, overlap in row["subspaces"].items():
                            for side in ("left", "right"):
                                subspace_values[row["job_id"], row["transition"], int(rank), side, layer].append(
                                    overlap[f"{side}_overlap"]
                                )
            result["geometry"].extend(
                dict(
                    task=label,
                    job_id=k[0],
                    transition=k[1],
                    rank_fraction=k[2],
                    restriction=k[3],
                    matrices=len(v),
                    median=statistics.median(v),
                    q25=float(np.quantile(v, 0.25)),
                    q75=float(np.quantile(v, 0.75)),
                )
                for k, v in values.items()
            )
        elif label.startswith(("training-", "dapo-training-")):
            scores = []
            for file in sorted((dest / "scores").glob("step-*.json")):
                if file.name.endswith(".progress.json"):
                    continue
                rows = read(file)["rows"]
                for identity in sorted({(r["origin"], r["split"], r["domain"]) for r in rows}):
                    selected = [r for r in rows if (r["origin"], r["split"], r["domain"]) == identity][:2]
                    scores.append(
                        dict(
                            step=int(file.stem.split("-")[1]),
                            origin=identity[0],
                            split=identity[1],
                            domain=identity[2],
                            n=len(selected),
                            rkl=statistics.mean(r["rkl"] for r in selected),
                            fkl=statistics.mean(r["fkl"] for r in selected),
                        )
                    )
            result["dapo_training" if label.startswith("dapo-training-") else "training"].append(
                dict(
                    task=label,
                    protocol=read(dest / "protocol.json"),
                    curve=read(dest / "learning-curve.json"),
                    scores=scores,
                )
            )
        elif label.startswith("merge-"):
            result["merger"].append(
                dict(
                    task=label,
                    scores=[
                        dict(file=file.name, mean_rkl=statistics.mean(r["rkl"] for r in read(file)["rows"]))
                        for file in (dest / "scores").glob("*.json")
                        if not file.name.endswith(".progress.json")
                    ],
                )
            )
    result["geometry_all_layer"] = distributions(
        geometry_values, ["job_id", "transition", "rank_fraction", "restriction"]
    )
    result["geometry_by_layer"] = distributions(
        layer_values, ["job_id", "transition", "rank_fraction", "restriction", "layer"]
    )
    result["subspaces_by_layer"] = distributions(subspace_values, ["job_id", "transition", "rank", "side", "layer"])
    generations = []
    for task in queue["jobs"]:
        if task["status"] != "complete" or "smoke" in task["label"]:
            continue
        for file in (
            Path(task["output_attempt"]) / Path(task.get("complete_marker", "complete.json")).parent / "generation"
        ).glob("*.json"):
            if file.name.endswith(".progress.json"):
                continue
            rows = read(file)["records"]
            generations.append(
                dict(
                    task=task["label"],
                    condition=file.stem,
                    domains=[
                        dict(
                            domain=domain,
                            n=len(selected),
                            accuracy=statistics.mean(
                                float(r.get("acc", r.get("accuracy", r.get("reward", 0)))) for r in selected
                            ),
                            truncated=sum(r["truncated"] for r in selected),
                            mean_length=statistics.mean(r["response_length"] for r in selected),
                        )
                        for domain in sorted({r["data_source"] for r in rows})
                        if (selected := [r for r in rows if r["data_source"] == domain])
                    ],
                )
            )
    result["generation"] = generations
    result["all_tasks_complete"] = all(t["status"] == "complete" for t in queue["jobs"])
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "summary.json", result)
    return result


def publish(root, summary, docs):
    docs = Path(docs)
    data = docs / "data/iso-20261111"
    data.mkdir(parents=True, exist_ok=True)
    with gzip.open(data / "summary.json.gz", "wt", encoding="utf-8") as stream:
        import json

        json.dump(summary, stream, ensure_ascii=False)
    page = docs / "附录-定义与复算.md"
    text = page.read_text()
    a = "<!-- iso-live-start -->"
    b = "<!-- iso-live-end -->"
    count_text = "，".join(f"{k} {v}" for k, v in sorted(summary["queue_counts"].items()))
    scan_text = "已完成" if summary["scan_complete"] else "正在执行"
    body = "\n\n".join(
        [
            a,
            "## 全层与功能扩展的执行结果",
            "本轮范围是 16 个可执行配对条件（15 个连续条件与 1 个分支对照）、全部七类投影与现有阶段，"
            "另加原论文可取得的 1.5B 顺序案例和 14B SFT 过渡。"
            "训练 pilot 使用 12 个诊断 calibration prompt，已更正其证据归属；合并使用本地共享 base 的专家。",
            f"当前任务状态：{count_text}。全层扫描{scan_text}。"
            "已完成结果才进入统计和图，缺项及失败保留在"
            "[完整结果附件](data/iso-20261111/summary.json.gz)。"
            "实验设置与入口见[实现说明](../../../../research/studies/spectral/iso/README.md)。",
            "谱交换以同一模型 α=1 为基线，按题报告 RKL/FKL 配对差和 bootstrap 区间；"
            "谱几何分别汇总逐矩阵分布与 pooled 范数。"
            "训练记录每步采样轨迹、优化器耗时、显存和固定 heldout 读出，64 步终点预先指定。",
            b,
        ]
    )
    if a in text:
        text = text[: text.index(a)] + body + text[text.index(b) + len(b) :]
    else:
        text += "\n" + body + "\n"
    plots = []
    if summary["scan_complete"]:
        plots.append("![全层 teacher 与 student 谱统计](figures/iso-20261111/all-layer-spectra.png)")
        plots.append("![teacher 的整体尺度与谱形状](figures/iso-20261111/scale-and-shape.png)")
    if summary["training"]:
        plots.append("![12 题诊断 pilot，非 DAPO 训练](figures/iso-20261111/online-opd-curves.png)")
    if summary["functional"]:
        plots.append("![完整谱恢复与反向交换的功能差异](figures/iso-20261111/functional-spectrum-swap.png)")
    if summary.get("geometry_all_layer"):
        plots.append("![四类 rank-r 重构](figures/iso-20261111/rank90-transport.png)")
    if plots:
        text = text.replace(b, "\n" + "\n\n".join(plots) + "\n" + b)
    if summary["spectra"]:
        lookup = {(r["job_id"], r["role"]): r for r in summary["spectra"]}
        ids = sorted({r["job_id"] for r in summary["spectra"]})
        concentrated = sum(lookup[i, "endpoint"]["top16_median"] > lookup[i, "teacher"]["top16_median"] for i in ids)
        table = [
            "| 条件 | 更新绝对范数 teacher / student | ρΣ 中位数 teacher / student | κ 中位数 teacher / student |",
            "| --- | ---: | ---: | ---: |",
        ]
        for i in ids:
            teacher, student = lookup[i, "teacher"], lookup[i, "endpoint"]
            table.append(
                f"| {i} · {teacher['family']} · {teacher['teacher']} "
                f"| {teacher['absolute_update_norm']:.3f} / {student['absolute_update_norm']:.3f} "
                f"| {teacher['rho_sigma_median']:.4f} / {student['rho_sigma_median']:.4f} "
                f"| {teacher['kappa_median']:.2f} / {student['kappa_median']:.2f} |"
            )
        facts = (
            "### 全层扫描的观察\n\n"
            f"{concentrated}/{len(ids)} 个配对的 student 更新 top-16 能量中位数高于 teacher。"
            "这是逐矩阵 ΔW 的能量集中度，与完整权重的谱漂移分别测量。"
            "JustRL、Light、MiMo 的 teacher 对角富集明显高于多数其他条件，"
            "对应 OPD endpoint 的 κ 则接近 1–2；RL 标签本身没有统一的完整权重谱特征。\n\n"
            "下表的更新均相对该配对自己的 initial student。绝对范数汇总全部选中矩阵的平方范数后开根号；"
            "ρΣ 和 κ 为矩阵中位数，pooled ρΣ 另存结果附件。不同家族不做参数相减。\n\n" + "\n".join(table)
        )
        just = lookup[2, "teacher"]
        facts += (
            "\n\n完整谱漂移还包含整体尺度变化。"
            f"JustRL teacher 的径向更新能量中位数为 {100 * just['radial_update_fraction_median']:.2f}%；"
            "对奇异值向量拟合最佳统一缩放后，"
            f"残余谱距离中位数只剩原始距离的 {100 * just['spectrum_after_scale_fraction_median']:.2f}%。"
            "前者是 ΔW 在 W0 方向上的能量占比，后者是谱距离的比值，两个分母不同。"
            "这些量由已保存 FP64 范数和奇异值复算；统一缩放的训练来源尚未确定。"
            "新增全配对对照分别恢复每矩阵 Frobenius 尺度、或恢复归一化谱形状，"
            "保持目标模型的另一项属性，以区分两种变化对输出的影响。"
        )
        text = text.replace(b, facts + "\n\n" + b)
    if summary["paper_spectra"]:
        table = [
            "### 原论文 checkpoint 过渡\n",
            "| 过渡 | 矩阵数 | δΣ 中位数 | ρΣ 中位数 | κ 中位数 | 更新 top-16 中位数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary["paper_spectra"]:
            table.append(
                f"| {row['teacher']} | {row['matrices']} | {row['delta_sigma_median']:.5g} "
                f"| {row['rho_sigma_median']:.4f} | {row['kappa_median']:.2f} | {row['top16_median']:.3f} |"
            )
        table.append("\n这里的 outgoing 是 SFT/RL teacher checkpoint，没有对应的 OPD 训练终点。")
        text = text.replace(b, "\n".join(table) + "\n\n" + b)
    if summary["functional"]:
        rows = [
            r
            for r in summary["functional"]
            if r["origin"] == "initial-student"
            and r["split"] == "heldout"
            and r["domain"] == "math"
            and r["condition"]["role"] == "student"
            and "scale" not in r["task"]
        ]
        ids = sorted({int(r["task"].split("-")[-1]) for r in rows})
        lookup = {(int(r["task"].split("-")[-1]), r["condition"]["kind"], r["condition"]["alpha"]): r for r in rows}
        table = [
            "### 完整权重谱恢复的功能结果\n",
            "同一 student endpoint 分别恢复 base 谱、或恢复 base frames；其余参数保留 endpoint 值。"
            "下表为 initial-student prefix、heldout Math、H128 的绝对全词表 RKL，"
            "按 12 题均值报告。两 bank、三域、calibration/heldout 的逐题配对与区间保存在附件。\n",
            "| 配对 | 原始 endpoint | base 谱＋endpoint frames | endpoint 谱＋base frames |",
            "| --- | ---: | ---: | ---: |",
        ]
        for i in ids:
            values = [
                lookup[i, kind, alpha]["rkl"]["absolute_mean"]
                for kind, alpha in [("restore", 1.0), ("restore", 0.0), ("reverse", 1.0)]
            ]
            table.append(f"| {i} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
        table.append(
            "\n这些有限位移干预直接支持：已学输出主要随奇异框架保留，"
            "完整权重的奇异值变化贡献较小。它没有把 Qwen SFT 的残余平台解释成奇异值学不动；"
            "该条件保留 endpoint frames、恢复 base 谱后，残余 RKL 几乎不变。"
        )
        text = text.replace(b, "\n".join(table) + "\n\n" + b)
    if summary["training"]:
        text = text.replace(
            b,
            "### 12 题诊断 pilot 的结果与适用范围\n\n"
            "曲线只纳入完整 64 步任务，各面板标记完成数；训练中方法尚未进入图。"
            "在当前统一学习率 10⁻⁵、calibration 三域共 12 个 prompt 的 online OPD 配方下，"
            "仅谱优化变化很小，权重 AdamW 和固定谱 frame AdamW 均出现依条件而异的下降或反弹。"
            "JustRL 的固定谱训练从 0.0831 先下降，终点两 seed 为 0.1823、0.2480；"
            "权重训练终点为 0.0289、0.0373（同图固定 heldout Math 子集）。"
            "该 pilot 误用诊断 calibration prompts，不能代表原 DAPO 训练或 ISO optimizer 的正式比较；"
            "六个剩余任务已取消，原 attempt 和取消回执保留。"
            "相同名义学习率并不匹配权重与 U/V 坐标中的函数步长；"
            "下一项训练应在 calibration 上匹配初始更新的 logit/KL 位移，再固定配置比较 heldout 动态。\n\n" + b,
        )
    if summary.get("dapo_training"):
        table = [
            "### DAPO 参数化训练：固定 64 步对照\n",
            "![DAPO 训练参数化对照](figures/iso-20261111/dapo-opd-curves.png)\n",
            "真实 DAPO 数学训练池与三域固定 heldout 分开。每题 student 在线采样最多 128 token，"
            "全词表 RKL 按固定 H128 归约，学习率 10⁻⁵，seed42/43；"
            "下表只纳入完整 64 步结果，并列 seed，不按 heldout 选择阶段。"
            "三种坐标使用相同名义学习率，尚未匹配初始函数位移；该比较测参数化配方，"
            "不单独定义参数化的内在优劣。旧 12 题 pilot 保留在前节。\n",
            "| 配对 | 方法 | seed | 初始 heldout RKL | 终点 heldout RKL |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for record in summary["dapo_training"]:
            protocol = record["protocol"]
            if protocol["training_data"]["source_family"] != "dapo":
                raise ValueError("Non-DAPO training cannot enter formal DAPO results")
            scores = [r for r in record["scores"] if r["origin"] == "initial-student" and r["split"] == "heldout"]
            initial = statistics.mean(r["rkl"] for r in scores if r["step"] == 0)
            final = statistics.mean(r["rkl"] for r in scores if r["step"] == protocol["steps"])
            table.append(
                f"| {protocol['job']['teacher']} | {protocol['mode']} | "
                f"{protocol['seed']} | {initial:.6f} | {final:.6f} |"
            )
        text = text.replace(b, "\n".join(table) + "\n\n" + b)
    text = re.sub(r"(?<!\n)\n(?=#{1,6} )", "\n\n", text)
    page.write_text(re.sub(r"\n{3,}", "\n\n", text))
    from examples.mirroropd.tools.maintenance.pipeline import format_paths

    if format_paths(Path(__file__).resolve().parents[6], [page], write=True):
        raise RuntimeError(f"Published Markdown formatting failed: {page}")
    dump(
        data / "provenance.json",
        dict(
            root=str(Path(root).absolute()),
            updated=time.time(),
            analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--docs", type=Path)
    args = p.parse_args()
    summary = summarize(args.root, args.output)
    if args.docs:
        publish(args.root, summary, args.docs)


if __name__ == "__main__":
    main()
