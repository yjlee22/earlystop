import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

DATASETS = ["dermamnist", "bloodmnist"]
SEEDS = [0, 1, 2]

SPLIT_RULE = "Dirichlet"
SPLIT_COEF = "0.1"

METHOD = "FedAvg"
OPTIMIZER = "sgd"
LOCAL_LR = "0.001"
TXT_THRESHOLD = "0.01"
PATIENCE = 10

MAX_ROUND = 500
T = MAX_ROUND

ACC_COL = 1
GROWTH_COL = 5

TASK_TITLES = {
    "dermamnist": "Skin lesion task",
    "bloodmnist": "Blood cell task",
}

ACC_CURVE_COLOR = "#000000"     # Test accuracy curve
GROWTH_CURVE_COLOR = "#009A66"  # Growth rate curve

MARKER_COLORS = {
    "val_loss": "#00B2A9",
    "val_acc":  "#007C97",
    "proposed": "#009A66",
}

GROWTH_THRESHOLD = 0.01
GROWTH_YTICKS = [GROWTH_THRESHOLD, 0.4, 0.8]

def candidate_run_roots(dataset, seed):
    cands = [
        os.path.join(dataset, f"seed_{seed}"),
        os.path.join("out", dataset, f"seed_{seed}"),
        os.path.join("out_file", dataset, f"seed_{seed}"),
        os.path.join(dataset, f"seed{seed}"),
    ]
    cands += sorted(glob.glob(os.path.join("**", dataset,
                    f"seed_{seed}"), recursive=True))

    uniq = []
    for p in cands:
        if p not in uniq and os.path.isdir(p):
            uniq.append(p)
    return uniq

def find_npy_file(dataset, seed):
    run_roots = candidate_run_roots(dataset, seed)
    if not run_roots:
        raise FileNotFoundError(
            f"Cannot find run root for {dataset} seed {seed}")

    subdir = f"{SPLIT_RULE}_{SPLIT_COEF}" if SPLIT_RULE else "IID"

    patterns = []
    for rr in run_roots:
        root = os.path.join(rr, subdir)
        pat = os.path.join(root, f"{METHOD}_*_*_{OPTIMIZER}_{LOCAL_LR}.npy")
        patterns.append(pat)

    hits = []
    for pat in patterns:
        hits += glob.glob(pat)

    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(
            "No npy matched. Tried:\n" + "\n".join(patterns))

    return hits[0]

def find_txt_file(dataset, seed, mode_folder):
    run_roots = candidate_run_roots(dataset, seed)
    if not run_roots:
        raise FileNotFoundError(
            f"Cannot find run root for {dataset} seed {seed}")

    subdir = f"{SPLIT_RULE}_{SPLIT_COEF}" if SPLIT_RULE else "IID"

    patterns = []
    for rr in run_roots:
        summary_root = os.path.join(rr, "summary", mode_folder, subdir)
        pat = os.path.join(
            summary_root,
            f"{METHOD}_*_*_{OPTIMIZER}_{LOCAL_LR}_{PATIENCE}_{TXT_THRESHOLD}.txt"
        )
        patterns.append(pat)

    hits = []
    for pat in patterns:
        hits += glob.glob(pat)

    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(
            f"No txt matched for mode='{mode_folder}'. Tried:\n" +
            "\n".join(patterns)
        )

    return hits[0]

def valid_prefix_len(acc):
    nz = np.nonzero(acc > 0)[0]
    return int(nz[-1] + 1) if len(nz) else 0

def pad_hold(x, T):
    x = np.asarray(x, dtype=float)
    if len(x) >= T:
        return x[:T]
    out = np.empty(T)
    out[:len(x)] = x
    out[len(x):] = x[-1]
    return out

def mean_std(mat):
    mat = np.asarray(mat, dtype=float)
    all_nan_mask = np.all(np.isnan(mat), axis=0)

    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)

    mean[all_nan_mask] = np.nan
    std[all_nan_mask] = np.nan
    return mean, std

def find_npy_early_stop(gro_mean, threshold, patience):
    below = np.array([(not np.isnan(v)) and (v < threshold) for v in gro_mean])
    for i in range(len(below) - patience + 1):
        if np.all(below[i:i + patience]):
            return i
    return None

def _spread_close_x(pts, min_sep=7.0):
    pts = [(k, float(x)) for k, x in pts if x is not None]
    if len(pts) <= 1:
        return {k: x for k, x in pts}

    pts_sorted = sorted(pts, key=lambda t: t[1])
    out = {}

    group = [pts_sorted[0]]
    for k, x in pts_sorted[1:]:
        if x - group[-1][1] < min_sep:
            group.append((k, x))
        else:
            if len(group) == 1:
                out[group[0][0]] = group[0][1]
            else:
                center = float(np.mean([gx for _, gx in group]))
                m = len(group)
                start = center - (m - 1) / 2.0 * min_sep
                for i, (gk, _) in enumerate(group):
                    out[gk] = start + i * min_sep
            group = [(k, x)]

    if len(group) == 1:
        out[group[0][0]] = group[0][1]
    else:
        center = float(np.mean([gx for _, gx in group]))
        m = len(group)
        start = center - (m - 1) / 2.0 * min_sep
        for i, (gk, _) in enumerate(group):
            out[gk] = start + i * min_sep

    return out

def marker_x_map_for_plot(markers, keys=("val_loss", "val_acc", "proposed"), min_sep=7.0):
    pts = []
    x_true = {}
    for k in keys:
        m = markers.get(k)
        if m is None:
            continue
        true_x = float(m) + 1.0
        x_true[k] = true_x
        pts.append((k, true_x))
    x_plot = _spread_close_x(pts, min_sep=min_sep)
    return x_plot, x_true

def parse_txt(txt_path, mode_folder):
    txt = open(txt_path).read()

    def grab(pat):
        m = re.search(pat, txt)
        if not m:
            return None
        s = m.group(1).strip()
        if s.lower().startswith("not triggered"):
            return None
        return int(s)

    if mode_folder == "validation":
        return {
            "val_loss": grab(r"Validation \(Loss\) Stop Round:\s*([^\n\r]+)"),
            "val_acc":  grab(r"Validation \(Acc\) Stop Round\s*:\s*([^\n\r]+)"),
            "proposed": None,
        }
    else:  # "proposed"
        return {
            "val_loss": None,
            "val_acc":  None,
            "proposed": grab(r"Proposed Early Stop Round\s*:\s*([^\n\r]+)"),
        }

def mean_marker(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None

def force_xticks_1_to_500(ax, step=50):
    ax.set_xlim(1, 500)
    ticks = [1] + list(range(step, 501, step))
    ax.set_xticks(sorted(set(ticks)))
    ax.margins(x=0)

def plot_mean_shadow(ax, x, mean, std, ylabel, curve_color):
    ax.plot(x, mean, linewidth=0.75, color=curve_color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=curve_color)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, color='grey', linestyle='--', linewidth=0.5)
    force_xticks_1_to_500(ax)

def add_accuracy_markers(ax, markers, min_sep=7.0):
    
    keys = ("val_loss", "val_acc", "proposed")
    x_plot, _ = marker_x_map_for_plot(markers, keys=keys, min_sep=min_sep)

    for k in keys:
        if k not in x_plot:
            continue
        ax.axvline(
            x_plot[k],
            linestyle=":",
            linewidth=1.0,
            color=MARKER_COLORS[k],
            zorder=4,
        )

def add_proposed_only(ax, proposed):
    if proposed is None:
        return
    ax.axvline(
        proposed + 1,
        linestyle=":",
        linewidth=0.75,
        color=MARKER_COLORS["proposed"],
    )

def apply_growth_yticks(ax, yticks):
    yticks = sorted(set(float(t) for t in yticks))
    ax.set_yticks(yticks)
    ymin, ymax = ax.get_ylim()
    ymin = min(ymin, yticks[0])
    ymax = max(ymax, yticks[-1])
    ax.set_ylim(ymin, ymax)

def draw_inset(parent_ax, x, mean, std, markers,
               xlims, ylims,
               curve_color,
               is_growth=False,
               loc_bounds=[0.3, 0.3, 0.45, 0.35],
               xtick_label_override=None,
               accuracy_min_sep=7.0):
   
    axins = parent_ax.inset_axes(loc_bounds)

    axins.plot(x, mean, linewidth=0.75, color=curve_color)
    axins.fill_between(x, mean - std, mean + std, alpha=0.1, color=curve_color)

    xticks_locs = []

    if is_growth:
        axins.axhline(GROWTH_THRESHOLD, linestyle="--",
                      linewidth=1.0, color=curve_color)
        p = markers.get("proposed")
        if p is not None:
            line_x = p + 1
            axins.axvline(line_x, linestyle=":",
                          linewidth=0.75, color=MARKER_COLORS["proposed"])
            xticks_locs.append(line_x)
        axins.set_yticks([GROWTH_THRESHOLD])

        if xticks_locs:
            axins.set_xticks(xticks_locs)
            if xtick_label_override:
                labels = [xtick_label_override.get(
                    t, f"{int(round(t))}") for t in xticks_locs]
            else:
                labels = [f"{int(round(t))}" for t in xticks_locs]
            axins.set_xticklabels(labels, fontsize=6)

    else:
        keys = ("val_loss", "val_acc", "proposed")
        x_plot, x_true = marker_x_map_for_plot(
            markers, keys=keys, min_sep=accuracy_min_sep)

        for k in keys:
            if k not in x_plot:
                continue
            line_x = x_plot[k]
            axins.axvline(line_x, linestyle=":", linewidth=1.0,
                          color=MARKER_COLORS[k])
            xticks_locs.append(line_x)

        if xticks_locs:
            axins.set_xticks(xticks_locs)
            labels = []
            for t in xticks_locs:
                matched = None
                for k in keys:
                    if k in x_plot and abs(x_plot[k] - t) < 1e-9:
                        matched = k
                        break
                labels.append(
                    f"{int(round(x_true[matched]))}" if matched else f"{int(round(t))}")
            axins.set_xticklabels(labels, fontsize=6)

    axins.set_xlim(*xlims)
    axins.set_ylim(*ylims)

    axins.tick_params(axis='both', which='major', labelsize=6)
    axins.grid(True, alpha=0.3, color='grey', linestyle='--', linewidth=0.5)
    return axins

x_axis = np.arange(1, MAX_ROUND + 1)
results = {}

for ds in DATASETS:
    acc_all, gro_all = [], []
    txt_vals = {"val_loss": [], "val_acc": [], "proposed": []}

    acc_seeds_list = []
    markers_seeds_list = []

    for seed in SEEDS:
        arr = np.load(find_npy_file(ds, seed))
        acc = arr[:, ACC_COL]
        gro = arr[:, GROWTH_COL]

        L = valid_prefix_len(acc)
        if L == 0:
            raise RuntimeError(f"No valid accuracy in {ds} seed {seed}")

        acc_500 = pad_hold(acc[:L], T)
        gro_500 = pad_hold(gro[:L], T)
        gro_500[:2] = np.nan

        acc_all.append(acc_500)
        gro_all.append(gro_500)

        info_val = parse_txt(find_txt_file(
            ds, seed, "validation"), "validation")
        info_prop = parse_txt(find_txt_file(ds, seed, "proposed"), "proposed")
        info = {
            "val_loss": info_val["val_loss"],
            "val_acc":  info_val["val_acc"],
            "proposed": info_prop["proposed"],
        }

        for k in txt_vals:
            txt_vals[k].append(info[k])

        acc_seeds_list.append(acc_500)
        markers_seeds_list.append(info)

    acc_mean, acc_std = mean_std(np.vstack(acc_all))
    gro_mean, gro_std = mean_std(np.vstack(gro_all))
    markers = {k: mean_marker(v) for k, v in txt_vals.items()}

    npy_stop = find_npy_early_stop(gro_mean, GROWTH_THRESHOLD, PATIENCE)

    if npy_stop is not None:
        end_idx = npy_stop + PATIENCE
        end_idx = max(0, min(end_idx, MAX_ROUND - 1))
        gro_mean[end_idx + 1:] = np.nan
        gro_std[end_idx + 1:] = np.nan

    results[ds] = dict(
        acc_mean=acc_mean, acc_std=acc_std,
        gro_mean=gro_mean, gro_std=gro_std,
        markers=markers,
        npy_stop=npy_stop,
        acc_seeds=acc_seeds_list,
        markers_seeds=markers_seeds_list,
    )

fig, axs = plt.subplots(2, 2, figsize=(4.8, 2.0), constrained_layout=True)

for i, ds in enumerate(DATASETS):
    ax = axs[0, i]
    res = results[ds]

    plot_mean_shadow(
        ax, x_axis,
        res["acc_mean"], res["acc_std"],
        ylabel="Test accuracy (%)" if i == 0 else "",
        curve_color=ACC_CURVE_COLOR
    )
    ax.set_xlabel("")
    ax.set_title(TASK_TITLES[ds])

    add_accuracy_markers(ax, res["markers"], min_sep=7.0)

    valid_m = [v for v in res["markers"].values() if v is not None]
    if valid_m:
        x1 = max(1, min(valid_m) - 25)
        x2 = min(MAX_ROUND, max(valid_m) + 25)

        idx1, idx2 = int(x1) - 1, int(x2) - 1
        y_segment = res["acc_mean"][idx1:idx2]
        if len(y_segment) > 0:
            y_min, y_max = np.min(y_segment), np.max(y_segment)
            y_margin = (y_max - y_min) * 0.2
            if y_margin == 0:
                y_margin = 1.0

            draw_inset(
                ax, x_axis, res["acc_mean"], res["acc_std"], res["markers"],
                xlims=(x1, x2),
                ylims=(y_min - y_margin, y_max + y_margin),
                curve_color=ACC_CURVE_COLOR,
                is_growth=False,
                loc_bounds=[0.3, 0.3, 0.45, 0.35],
                accuracy_min_sep=7.0
            )

for i, ds in enumerate(DATASETS):
    ax = axs[1, i]
    res = results[ds]

    plot_mean_shadow(
        ax, x_axis,
        res["gro_mean"], res["gro_std"],
        ylabel="Growth rate" if i == 0 else "",
        curve_color=GROWTH_CURVE_COLOR
    )
    ax.set_xlabel("Global round")
    ax.set_title("")

    ax.axhline(GROWTH_THRESHOLD, linestyle="--", linewidth=1.0,
               color=GROWTH_CURVE_COLOR, zorder=3)
    apply_growth_yticks(ax, GROWTH_YTICKS)
    add_proposed_only(ax, res["markers"]["proposed"])

    p_npy = res["npy_stop"]               
    p_txt = res["markers"]["proposed"]   

    if p_npy is not None:
        npy_x = p_npy + 1                
        x1 = max(1, npy_x - 3)
        x2 = min(MAX_ROUND, npy_x + 10)

        idx1 = max(0, int(x1) - 1)
        idx2 = min(len(res["gro_mean"]), int(x2))

        y_segment = res["gro_mean"][idx1:idx2]
        if len(y_segment) > 0:
            curr_min, curr_max = np.nanmin(y_segment), np.nanmax(y_segment)
            view_min = min(curr_min, GROWTH_THRESHOLD)
            view_max = max(curr_max, GROWTH_THRESHOLD)
            y_pad = (view_max - view_min) * \
                0.5 if view_max != view_min else 0.01

            npy_markers = {"proposed": p_npy}

            txt_label = f"{int(round(p_txt + 1))}" if p_txt is not None else f"{int(npy_x)}"
            label_override = {npy_x: txt_label}

            draw_inset(
                ax, x_axis, res["gro_mean"], res["gro_std"], npy_markers,
                xlims=(x1, x2),
                ylims=(view_min - y_pad, view_max + y_pad),
                curve_color=GROWTH_CURVE_COLOR,
                is_growth=True,
                loc_bounds=[0.3, 0.45, 0.45, 0.35],
                xtick_label_override=label_override,
            )

fig.align_ylabels(axs[:, 0])
fig.savefig('result1.svg', format='svg', dpi=300, bbox_inches='tight')