# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: each row is one author-published Vanilla pair; indices are JSON positions.
Gold curves are author-rounded data. d**2 is initial-student KL, not teacher KL.
Teacher training/validation KL is unavailable in these JSONs and is never imputed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
SCALES = [.5, 1.5, 3, 7, 14]


def read(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw, out):
    curves = read(raw/"trajectories.json")["vanilla_opd"]
    peaks = read(raw/"peaks.json")
    rows = []
    mismatches = []
    for s in SCALES:
        for t in SCALES:
            pair = f"{s:g}<-{t:g}"
            r = curves[pair]
            assert len(r["d"]) == len(r["g"]) and len(r["g"]) >= 100
            g, d = r["g"], r["d"]
            maximum = max(g)
            published = peaks["vanilla_opd"][pair]
            if abs(maximum-published) > 1e-9:
                mismatches.append(dict(pair=pair, curve_max=maximum, published_peak=published,
                                       difference_pp=100*(maximum-published)))
            teacher = peaks["teacher_score"][f"{t:g}"]
            row = dict(pair=pair, student_b=s, teacher_b=t,
                       relation="same_size" if s==t else "small_teacher_large_student" if s>t else "large_teacher_small_student",
                       observation_count=len(g), last_observation_index=len(g)-1,
                       initial_accuracy=g[0], curve_peak_accuracy=maximum, peak_observation_index=g.index(maximum),
                       published_peak_accuracy=published, last_accuracy=g[-1], teacher_accuracy=teacher,
                       peak_gain_pp=100*(maximum-g[0]), last_gain_pp=100*(g[-1]-g[0]),
                       peak_to_last_drop_pp=100*(maximum-g[-1]), last_minus_teacher_pp=100*(g[-1]-teacher),
                       last_50_accuracy=statistics.mean(g[-50:]),
                       previous_50_accuracy=statistics.mean(g[-100:-50]),
                       late_window_accuracy_change_pp=100*(statistics.mean(g[-50:])-statistics.mean(g[-100:-50])),
                       last_d_to_initial=d[-1], last_k3_to_initial_approx=d[-1]**2,
                       last_50_k3_to_initial_approx=statistics.mean(v*v for v in d[-50:]),
                       late_window_k3_change_approx=statistics.mean(v*v for v in d[-50:])-statistics.mean(v*v for v in d[-100:-50]),
                       teacher_training_kl="", teacher_validation_kl="",
                       teacher_kl_status="not_in_downloaded_author_JSONs")
            rows.append(row)
    assert len(rows)==25
    write_csv(out/"vanilla-summary.csv",rows)
    dump(out/"discrepancies.json",mismatches)
    manifest = read(raw/"manifest.json")
    metric_names = sorted({v["name"] for r in manifest["checkpoints"] for v in r.get("metrics",[])})
    files = read(raw/"hub-info.json")["siblings"]
    suspicious = [f["rfilename"] for f in files if any(z in f["rfilename"].rsplit("/",1)[-1].lower()
                   for z in ["history","metrics","jsonl","csv","wandb"])]
    summary = dict(pairs=25, checkpoint_exports=len(manifest["checkpoints"]), hub_file_count=len(files),
                   hub_revision=read(raw/"hub-info.json")["sha"], checkpoint_metric_names=metric_names,
                   candidate_history_files_by_basename=suspicious,
                   below_initial_at_last=sum(r["last_gain_pp"]<0 for r in rows),
                   dropped_more_than_5pp_from_peak=sum(r["peak_to_last_drop_pp"]>5 for r in rows),
                   discrepancies=mismatches,
                   caveats=["Observation index has no explicit optimizer-step field.",
                            "d and g are author-rounded published curves.",
                            "Latest weight export and last logged curve value are separate endpoints.",
                            "Training proxy in PDF is not exported in these JSONs.",
                            "No teacher-validation KL is present in downloaded records."])
    dump(out/"summary.json",summary)
    evidence=[]
    for name in ["trajectories.json","peaks.json","law.json","manifest.json","hub-info.json","github-tree.json","project.js"]:
        p=raw/name
        evidence.append(dict(file=name,path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(out/"sources.json",dict(project="https://colored-dye.github.io/blog/2026/opd-scaling/",
         github_revision="7672d2edc36011cfcd2d2b5ea76dd6ee2787a5d0",hub_revision=summary["hub_revision"],files=evidence))
    return rows,curves


def plots(out,rows,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from examples.mirroropd.research.core.plotting import load_style,save_figure
    fv=load_style();fv.apply_rc("paper",fontsize=10)
    blue,orange,violet=[fv.HUES[k] for k in ["vblue","vorange","vviolet"]]
    paper=fv.NEUTRALS["paper"];ink=fv.NEUTRALS["vink"];gray=fv.NEUTRALS["baseline"]
    seq=LinearSegmentedColormap.from_list("accuracy",[paper,tuple(.45+.55*v for v in to_rgb(blue))])
    div=LinearSegmentedColormap.from_list("change",[orange,paper,blue])
    fig,axs=plt.subplots(2,2,figsize=(13,11))
    fig.subplots_adjust(left=.08,right=.94,bottom=.13,top=.90,wspace=.42,hspace=.38)
    specs=[("curve_peak_accuracy","Observed curve peak (%)",100,False),
           ("last_accuracy","Last published observation (%)",100,False),
           ("peak_to_last_drop_pp","Peak minus last (pp)",1,False),
           ("last_minus_teacher_pp","Last student minus teacher (pp)",1,True)]
    for ax,(field,title,mul,diverging) in zip(axs.flat,specs,strict=True):
        matrix=np.array([r[field]*mul for r in rows]).reshape(5,5)
        if diverging:
            bound=max(abs(matrix.min()),abs(matrix.max()))
            opts=dict(cmap=div,vmin=-bound,vmax=bound)
        else:opts=dict(cmap=seq,vmin=0,vmax=100 if mul==100 else max(matrix.max(),1))
        im=ax.imshow(matrix,**opts)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",color=ink,fontsize=9)
        ax.set_xticks(range(5),[f"{v:g}B" for v in SCALES])
        ax.set_yticks(range(5),[f"{v:g}B" for v in SCALES])
        ax.set(xlabel="Qwen2.5 GRPO teacher",ylabel="Qwen2.5 SFT student",title=title)
        fig.colorbar(im,ax=ax,fraction=.047,pad=.03).set_label("%" if mul==100 else "Percentage points")
    fig.suptitle("Author-published Vanilla-OPD curves: peaks and late observations",y=.96)
    save_figure(fig,out/"endpoint-matrix",
                "25 pairs; one run per pair. Curves may stop at different indices. Last observation is not a shared update budget.\n"
                "Peak is recomputed from trajectories.json; the 14B-to-14B peak differs from peaks.json by 0.05 pp.\n"
                "Values describe task accuracy, not teacher-distribution KL.",caption_size=8)

    selected=["14<-0.5","7<-1.5","7<-7"]
    fig,axs=plt.subplots(2,3,figsize=(15,9))
    fig.subplots_adjust(left=.07,right=.97,top=.87,bottom=.20,hspace=.42,wspace=.3)
    handles=[]
    for j,pair in enumerate(selected):
        r=curves[pair];row=next(x for x in rows if x["pair"]==pair);x=np.arange(len(r["g"]))
        ax=axs[0,j]
        line,=ax.plot(x,np.array(r["g"])*100,color=blue,label="Student held-out accuracy")
        teacher=ax.axhline(row["teacher_accuracy"]*100,color=orange,linestyle="--",label="Teacher accuracy")
        initial=ax.axhline(row["initial_accuracy"]*100,color=gray,linestyle=":",label="Student initial accuracy")
        if j==0:handles=[line,teacher,initial]
        ax.set(title=f'Teacher {row["teacher_b"]:g}B -> Student {row["student_b"]:g}B',
               xlabel="Published observation index",ylabel="Held-out accuracy (%)")
        ax=axs[1,j]
        ax.plot(x,np.square(r["d"]),color=violet)
        ax.set(xlabel="Published observation index",ylabel="Approx. token k3 to initial student",title="Distance from initial student")
    fig.suptitle("Qwen2.5 Vanilla-OPD: late accuracy versus displacement from initialization",y=.96)
    fig.legend(handles,[h.get_label() for h in handles],loc="lower center",bbox_to_anchor=(.5,.10),ncol=3,frameon=False)
    save_figure(fig,out/"selected-trajectories",
                "Published JSON values; d^2 reconstructs the rounded initial-student KL coordinate.\n"
                "No teacher training or teacher validation KL is supplied by these curves. Index is not a verified optimizer step.",
                caption_size=9)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",action="store_true")
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows,curves=analyze(a.raw,a.output)
    if a.plot:plots(a.output,rows,curves)
    print(json.dumps(read(a.output/"summary.json"),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
