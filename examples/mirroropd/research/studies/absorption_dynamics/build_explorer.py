# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Export matrix-level scalar trajectories to an offline comparison explorer.

CPU only; streaming reads keep full singular vectors out of the browser payload.
Plotly and ijson are build dependencies. The HTML embeds both data and Plotly;
opening it requires no server, network, or access to training run directories.
"""

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path

from examples.mirroropd.launchers.absorption import PAIRS

from .figure_catalog import FAMILIES

RANKS = (16, 32, 64, 128, 256)
SPECS = {
    "overlap": ("Frame", "到自身终点的调整后 overlap", True, "1=同空间，0=随机基线；不按能量加权"),
    "angle": ("Frame", "到自身终点的主夹角中位数（度）", True, "先在每个矩阵内取角度中位数，再跨矩阵汇总"),
    "angle90": ("Frame", "到自身终点的主夹角 P90（度）", True, "P90 是空间内角度分位，不是跨矩阵分位"),
    "anglemax": ("Frame", "到自身终点的最大主夹角（度）", True, "少数边界方向即可使最大角较大"),
    "capture": ("Frame", "当前空间捕获的最终总能量比例", True, "分母为最终累计更新的总能量"),
    "capture_optimal": ("Frame", "捕获能量／最终最优同 rank 能量", True, "1 不代表捕获全部更新能量"),
    "top_energy": ("Frame", "当前累计更新 top-k 能量比例", True, "当前更新自身的最优 rank-k 能量覆盖"),
    "weight_angle": ("Frame", "完整权重相对 W0 的主夹角中位数（度）", True, "完整权重 frame，与更新 frame 区分"),
    "teacher_overlap": ("Teacher", "与 teacher 更新 frame 的调整后 overlap", True, "仅同坐标 teacher；跨尺寸缺失"),
    "norm": ("更新", "累计更新 Frobenius 范数", False, "矩阵范数的中位数，不是全网范数"),
    "relative_norm": ("更新", "累计更新范数／初始权重范数", False, "先逐矩阵归一化，再汇总"),
    "stable_rank": ("更新", "累计更新 stable rank", False, "总能量／最大奇异值平方"),
    "rank90": ("更新", "累计更新 rank90", False, "覆盖 90% 能量所需方向数"),
    "rank95": ("更新", "累计更新 rank95", False, "覆盖 95% 能量所需方向数"),
    "entropy_rank": ("更新", "累计更新熵有效秩", False, "使用平方奇异值归一化的能量熵"),
    "norm_ratio": ("Teacher", "OPD／teacher 更新范数", False, "需要同坐标 teacher"),
    "cosine": ("Teacher", "OPD 与 teacher 更新余弦", False, "逐矩阵 Frobenius 余弦"),
    "residual_energy_ratio": ("Teacher", "参数残差能量／初始 teacher 差距能量", False, "1=残差与初始相同；大于1=更远"),
    "delta_sigma": ("谱", "完整权重 delta_sigma", False, "奇异值差范数／初始权重范数"),
    "rho_sigma": ("谱", "完整权重 rho_sigma", False, "奇异值差范数／参数更新范数"),
    "kappa_spec": ("谱", "完整权重 kappa_spec", False, "初始配对奇异方向能量集中度；不是条件数"),
    "weight_sd": ("谱", "完整权重到自身终点 SD 距离", False, "奇异值和归一化后 L2 距离；不含尺度与方向"),
    "update_sd": ("谱", "累计更新到自身终点 SD 距离", False, "奇异值和归一化后 L2 距离；不含尺度与方向"),
    "weight_shape": ("谱", "完整权重谱形状残差", False, "去除相对 step0 的最佳整体缩放"),
    "update_shape": ("谱", "累计更新谱形状残差", False, "去除相对 step16 的最佳整体缩放"),
    "weight_scale": ("谱", "完整权重谱最佳缩放", False, "参照 step0"),
    "update_scale": ("谱", "累计更新谱最佳缩放", False, "参照 step16"),
}


def matrices(path):
    """Yield one matrix at a time, avoiding a multi-GB in-memory readout."""
    import ijson

    builder, prefix, name = None, None, None
    with gzip.open(path, "rb") as stream:
        for key, event, value in ijson.parse(stream, use_float=True):
            parts = key.split(".")
            if builder is None and len(parts) == 4 and parts[0] == "pairs" and parts[2:] == ["matrices", "item"]:
                if event != "start_map":
                    raise ValueError("Expected matrix object")
                builder, prefix, name = ijson.ObjectBuilder(), key, parts[1]
            if builder is not None:
                builder.event(event, value)
                if key == prefix and event == "end_map":
                    yield name, builder.value
                    builder = None


def reduce_matrix(matrix):
    """Keep exact scalar trajectories and all-rank pairwise frame similarities."""
    stages = matrix["stages"]
    steps = [s["step"] for s in stages]
    final = steps[-1]
    series = {}

    def put(key, values):
        series[key] = [float(v) if v is not None and math.isfinite(v) else None for v in values]

    for key in ("stable_rank", "rank90", "rank95"):
        put(key, [s[key] for s in stages])
    put("norm", [math.sqrt(s["energy"]) for s in stages])
    put("relative_norm", [math.sqrt(s["energy"] / s["weight"]["base_energy"]) for s in stages])
    for key in ("delta_sigma", "rho_sigma", "kappa_spec"):
        put(key, [s["weight"][key] for s in stages])
    if "teacher_comparison" in stages[0]:
        for key in ("norm_ratio", "cosine", "residual_energy_ratio"):
            put(key, [s["teacher_comparison"][key] for s in stages])
    for kind in ("weight", "update"):
        points = matrix["sigma_variation"][kind]
        lookup = {p["step_a"]: p["sd_variation"] for p in points["pairs"] if p["step_b"] == final}
        put(f"{kind}_sd", [lookup[t] for t in steps])
        shape = {p["step"]: p for p in matrix["spectral_shape"][kind]}
        for target, source in (("shape", "shape_residual"), ("scale", "scale")):
            put(f"{kind}_{target}", [shape[t][source] for t in steps])
        if kind == "update":
            put("entropy_rank", [shape[t]["entropy_rank"] for t in steps])
    pairs = {}
    for rank in RANKS:
        rel = {p["step_a"]: p for p in matrix["pairs"] if p["step_b"] == final and p["requested_rank"] == rank}
        for target, source in (
            ("overlap", "adjusted_overlap"),
            ("angle", "median_angle_deg"),
            ("angle90", "p90_angle_deg"),
            ("anglemax", "max_angle_deg"),
            ("capture", "a_captures_b"),
        ):
            put(f"{target}:{rank}", [rel[t][source] for t in steps])
        optimal = next(r["captured_energy"] for r in stages[-1]["ranks"] if r["requested_rank"] == rank)
        put(f"capture_optimal:{rank}", [rel[t]["a_captures_b"] / optimal for t in steps])
        put(
            f"top_energy:{rank}",
            [next(r["captured_energy"] for r in s["ranks"] if r["requested_rank"] == rank) for s in stages],
        )
        put(
            f"weight_angle:{rank}",
            [next(r["median_angle_deg"] for r in s["weight"]["frames"] if r["requested_rank"] == rank) for s in stages],
        )
        if "teacher_comparison" in stages[0]:
            frames = [[r for r in s["teacher_comparison"]["frames"] if r["requested_rank"] == rank] for s in stages]
            if all(frames):
                put(f"teacher_overlap:{rank}", [r[0]["adjusted_overlap"] for r in frames])
        lookup = {
            (p["step_a"], p["step_b"]): p["adjusted_overlap"] for p in matrix["pairs"] if p["requested_rank"] == rank
        }
        pairs[str(rank)] = [[lookup[min(a, b), max(a, b)] for b in steps] for a in steps]
    return dict(layer=matrix["layer"], module=matrix["module"], steps=steps, series=series, pairwise=pairs)


def build(readout, synthesis, target):
    """Build a standalone page plus compressed, auditable browser data."""
    cohort = json.loads(synthesis.read_text())
    labels = {f"E{p[0]}": dict(student=p[1], teacher=p[2]) for p in PAIRS}
    data = dict(schema="checkpoint-explorer-v1", specs=SPECS, ranks=RANKS, pairs={})
    for name, m in matrices(readout):
        if name not in data["pairs"]:
            source = cohort["pairs"][name]
            risks = []
            for r in source["risks"]:
                risks.append(
                    dict(
                        step=r["optimizer_steps"],
                        values={
                            f"{bank}/{domain}": r["metrics"].get(f"eval/rkl/public_{bank}/{domain}/question_mean")
                            for bank in ("initial", "teacher")
                            for domain in ("math", "code", "if", "all")
                        },
                    )
                )
            data["pairs"][name] = dict(**labels[name], expected=source["expected"], matrices=[], risks=risks)
        data["pairs"][name]["matrices"].append(reduce_matrix(m))
        if len(data["pairs"][name]["matrices"]) % 196 == 0:
            print(name, len(data["pairs"][name]["matrices"]), flush=True)
    for name, pair in data["pairs"].items():
        identities = [(m["layer"], m["module"]) for m in pair["matrices"]]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate matrix in {name}")
        if len(identities) != pair["expected"]:
            raise ValueError(f"Incomplete coverage: {name}")
    write_outputs(data, readout, synthesis, target)


def write_outputs(data, readout, synthesis, target):
    """Render templates from reduced data without rereading full scientific arrays."""
    from plotly.offline import get_plotlyjs

    data["figures"] = [entry for entries in FAMILIES.values() for entry in entries]
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    compressed = gzip.compress(encoded, mtime=0)
    here = Path(__file__).parent
    html = (here / "checkpoint_explorer.html").read_text()
    html = html.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("__PAYLOAD__", base64.b64encode(compressed).decode())
    html = html.replace("/*__APP__*/", (here / "checkpoint_explorer.js").read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    (target.parent / "data").mkdir(exist_ok=True)
    (target.parent / "data/explorer-data.json.gz").write_bytes(compressed)
    receipt = dict(
        schema=data["schema"],
        pairs=list(data["pairs"]),
        matrices=sum(len(p["matrices"]) for p in data["pairs"].values()),
        html_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        payload_sha256=hashlib.sha256(encoded).hexdigest(),
        source=str(readout),
        source_bytes=readout.stat().st_size,
        synthesis_sha256=hashlib.sha256(synthesis.read_bytes()).hexdigest(),
        offline=True,
        aggregation="matrix median and IQR; no seed confidence interval",
    )
    (target.parent / "data/explorer-provenance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.readout, args.synthesis, args.output)


if __name__ == "__main__":
    main()
