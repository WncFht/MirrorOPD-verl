# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Consistent geometry plot layout, numerical reductions and publication metadata."""

import numpy as np

from examples.mirroropd.research.core.plotting import load_style, save_figure


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper")
    return plt, fv


def grid(rows, cols, width=None, height=None):
    plt, _ = setup()
    fig, axes = plt.subplots(
        rows, cols, figsize=(width or 4.3 * cols, height or 3.6 * rows), layout="constrained", squeeze=False
    )
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.85))
    return fig, axes


def band(ax, x, values, color, label=None):
    v = np.asarray(values, dtype=float)
    ax.plot(x, np.median(v, axis=0), color=color, label=label)
    ax.fill_between(x, *np.quantile(v, [0.25, 0.75], axis=0), color=color, alpha=0.14)


def legend(ax, ncol=3):
    if ax.get_title():
        ax.figure.get_layout_engine().set(rect=(0, 0.15, 1, 0.76))
        ax.figure.legend(
            *ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=ncol, fontsize=9
        )
    else:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, fontsize=9)


def heatmap(fig, ax, values, xlabels, ylabels, label, vmin=None, vmax=None, cmap="cividis"):
    data = np.ma.masked_invalid(np.asarray(values, dtype=float))
    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=60)
    stride = max(1, int(np.ceil(len(ylabels) / 12)))
    ticks = list(range(0, len(ylabels), stride))
    ax.set_yticks(ticks, [ylabels[i] for i in ticks])
    fig.colorbar(image, ax=ax, label=label, shrink=0.78)


def finish(fig, destination, name, suffix, pair, matrices, caption):
    fig.suptitle(f"{name}: {caption[0]}")
    coverage = f"{len(matrices)}/{pair['expected']} matrices; {len(pair['complete_layers'])} full layers"
    save_figure(fig, destination / f"{name}-{suffix}", f"{coverage}. {caption[1]}", caption_size=8)
