import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ["dermamnist", "bloodmnist"]
SEEDS = [0, 1, 2]
METHODS = [
    'FedAvg', 'FedProx', 'FedDyn', 'SCAFFOLD', 'FedSAM',
    'FedSpeed', 'FedSMOO', 'FedGamma', 'FedLESAM', 'FedWMSAM'
]

SPLIT_RULE = "Dirichlet"
SPLIT_COEF = "0.1"
PATIENCE = 10
OPTIMIZER = "sgd"

THRESHOLDS = [0.005, 0.01, 0.05, 0.1]
LOCAL_LR = 0.001

TASK_TITLES = {
    "dermamnist": "Skin lesion",
    "bloodmnist": "Blood cell",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "mathtext.fontset": "stix",
})

def candidate_run_roots(dataset, seed):
    """
    Locate actual experiment root that contains 'summary/'.
    """
    cands = [
        os.path.join(dataset, f"seed_{seed}"),
        os.path.join("out", dataset, f"seed_{seed}"),
        os.path.join("out_file", dataset, f"seed_{seed}"),
    ]

    cands += glob.glob(os.path.join("**", dataset,
                       f"seed_{seed}"), recursive=True)

    uniq = []
    for p in cands:
        if os.path.isdir(p) and p not in uniq:
            uniq.append(p)
    return uniq

def parse_txt_file(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read()

    def extract_acc(pattern):
        m = re.search(pattern, content)
        return float(m.group(1)) if m else None

    loss_acc = extract_acc(
        r"Validation \(Loss\) Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )
    val_acc = extract_acc(
        r"Validation \(Acc\) Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )
    prop_acc = extract_acc(
        r"Proposed Early Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )

    return {"val_loss": loss_acc, "val_acc": val_acc, "proposed": prop_acc}


def get_base_val(parsed):
    if parsed is None:
        return np.nan
    cands = [v for v in [parsed.get("val_loss"), parsed.get(
        "val_acc")] if v is not None]
    return max(cands) if cands else np.nan

records = []
print("Collecting curves from NEW summary structure...")

for dataset in DATASETS:
    for method in METHODS:
        for th in THRESHOLDS:
            for seed in SEEDS:

                run_roots = candidate_run_roots(dataset, seed)

                if not run_roots:
                    records.append({
                        "Dataset": dataset, "Method": method, "Threshold": th, "Seed": seed,
                        "Base": np.nan, "Prop": np.nan
                    })
                    continue

                subdir = f"{SPLIT_RULE}_{SPLIT_COEF}"

                txt_val = []
                txt_prop = []

                for rr in run_roots:
                    txt_val += glob.glob(
                        os.path.join(
                            rr, "summary", "validation", subdir,
                            f"{method}_*_{OPTIMIZER}_{LOCAL_LR}_{PATIENCE}_{th}.txt"
                        )
                    )

                    txt_prop += glob.glob(
                        os.path.join(
                            rr, "summary", "proposed", subdir,
                            f"{method}_*_{OPTIMIZER}_{LOCAL_LR}_{PATIENCE}_{th}.txt"
                        )
                    )

                if not txt_val and not txt_prop:
                    records.append({
                        "Dataset": dataset, "Method": method, "Threshold": th, "Seed": seed,
                        "Base": np.nan, "Prop": np.nan
                    })
                    continue

                parsed_val = parse_txt_file(txt_val[0]) if txt_val else None
                parsed_prop = parse_txt_file(txt_prop[0]) if txt_prop else None

                base_val = get_base_val(parsed_val)
                prop_val = parsed_prop["proposed"] if parsed_prop else np.nan

                records.append({
                    "Dataset": dataset, "Method": method, "Threshold": th, "Seed": seed,
                    "Base": base_val, "Prop": prop_val
                })

df = pd.DataFrame(records)

g = df.groupby(["Dataset", "Method", "Threshold"], as_index=False)[
    ["Base", "Prop"]].agg(['mean', 'std'])
g.columns = ['_'.join(col).strip() if col[1] else col[0]
             for col in g.columns.values]
g = g.reset_index()

g["Threshold"] = pd.Categorical(
    g["Threshold"], categories=THRESHOLDS, ordered=True)
g = g.sort_values(["Method", "Dataset", "Threshold"])
x_pos = np.arange(len(THRESHOLDS))

style_map = {
    "dermamnist": {
        "prop": {"color": "#007C97", "marker": "s", "linestyle": "-"},
        "base": {"color": "#009A66", "marker": "^", "linestyle": "--"},
    },
    "bloodmnist": {
        "prop": {"color": "#ed1c24", "marker": "s", "linestyle": "-"},
        "base": {"color": "#f26522", "marker": "^", "linestyle": "--"},
    },
}

fig, axes = plt.subplots(2, 5, figsize=(4.8, 2.8), constrained_layout=False)
plt.subplots_adjust(wspace=0.35, hspace=0.45, bottom=0.18,
                    top=0.92, left=0.10, right=0.98)
axes = axes.flatten()

for i, method in enumerate(METHODS):
    ax = axes[i]

    for dataset in DATASETS:
        sub = g[(g["Method"] == method) & (g["Dataset"] == dataset)].copy()

        y_prop_mean = sub["Prop_mean"].to_numpy(dtype=float)
        y_prop_std = sub["Prop_std"].fillna(0).to_numpy(dtype=float)

        y_base_mean = sub["Base_mean"].to_numpy(dtype=float)
        y_base_std = sub["Base_std"].fillna(0).to_numpy(dtype=float)

        title_ds = TASK_TITLES.get(dataset, dataset)
        style_prop = style_map[dataset]["prop"]
        style_base = style_map[dataset]["base"]

        ax.plot(x_pos, y_prop_mean,
                linestyle=style_prop["linestyle"],
                marker=style_prop["marker"],
                color=style_prop["color"],
                linewidth=0.8, markersize=2.8,
                label=f"{title_ds} (Proposed)")

        ax.fill_between(x_pos, y_prop_mean - y_prop_std, y_prop_mean + y_prop_std,
                        color=style_prop["color"], alpha=0.1)

        ax.plot(x_pos, y_base_mean,
                linestyle=style_base["linestyle"],
                marker=style_base["marker"],
                color=style_base["color"],
                linewidth=0.8, markersize=2.8,
                label=f"{title_ds} (w/ Real data)")

        ax.fill_between(x_pos, y_base_mean - y_base_std, y_base_mean + y_base_std,
                        color=style_base["color"], alpha=0.1)

    ax.set_title(method, pad=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t) for t in THRESHOLDS])

    if i % 5 == 0:
        ax.set_ylabel("Test accuracy (%)")
    if i >= 5:
        ax.set_xlabel("Threshold ($\\tau$)")

    ax.grid(True, alpha=0.3, color='grey', linestyle='--', linewidth=0.5)

for j in range(len(METHODS), len(axes)):
    axes[j].axis("off")

handles, labels = axes[0].get_legend_handles_labels()

fig.align_ylabels(axes)

fig.legend(handles, labels,
           loc="upper center",
           bbox_to_anchor=(0.5, 0.07),
           ncol=4,
           frameon=False,
           fontsize=6,
           columnspacing=0.8,
           handletextpad=0.4,
           handlelength=1.5)

savename = "result2.pdf"
plt.savefig(savename, format="pdf", dpi=300)
print(f"Saved figure to {savename}")
print("\n" + "="*80)
print("Test Accuracy Summary by Threshold (mean ± std)")
print(f"(Opt={OPTIMIZER}, LR={LOCAL_LR}, Pat={PATIENCE}, Split={SPLIT_RULE}_{SPLIT_COEF})")
print("="*80)

g_print = g.copy()
g_print["Threshold"] = g_print["Threshold"].astype(float)

g_print = g_print.sort_values(["Dataset", "Method", "Threshold"])

for dataset in DATASETS:
    print("\n" + "-"*80)
    print(f"[{dataset}]")
    print("-"*80)

    sub_ds = g_print[g_print["Dataset"] == dataset].copy()

    for method in METHODS:
        sub_m = sub_ds[sub_ds["Method"] == method].copy()
        if sub_m.empty:
            continue

        print(f"\nMethod: {method}")
        print("Threshold | Proposed (mean±std) | w/ Real data (mean±std)")
        print("-"*62)

        sub_m = sub_m.sort_values("Threshold")

        for th in THRESHOLDS:
            row = sub_m[np.isclose(
                sub_m["Threshold"].to_numpy(dtype=float), float(th))]
            if row.empty:
                prop_str = "nan"
                base_str = "nan"
            else:
                prop_mean = float(row["Prop_mean"].iloc[0])
                prop_std = float(row["Prop_std"].iloc[0]) if not pd.isna(
                    row["Prop_std"].iloc[0]) else 0.0
                base_mean = float(row["Base_mean"].iloc[0])
                base_std = float(row["Base_std"].iloc[0]) if not pd.isna(
                    row["Base_std"].iloc[0]) else 0.0

                prop_str = f"{prop_mean:.2f}±{prop_std:.2f}"
                base_str = f"{base_mean:.2f}±{base_std:.2f}"

            print(f"{th:<9} | {prop_str:<18} | {base_str:<18}")