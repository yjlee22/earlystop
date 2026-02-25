import os
import re
import glob
import numpy as np
import pandas as pd

DATASETS = ["dermamnist", "bloodmnist"]
SEEDS = [0, 1, 2]
METHODS = [
    'FedAvg', 'FedProx', 'FedDyn', 'SCAFFOLD', 'FedSAM',
    'FedSpeed', 'FedSMOO', 'FedGamma', 'FedLESAM', 'FedWMSAM'
]

SPLIT_CONFIGS = {
    'Dirichlet':    [0.01, 0.1, 1.0],
    'Pathological': [1.0, 2.0, 3.0],
    'Quantity':     [0.01, 0.1, 1.0]
}

TARGET_THRESHOLD = 0.01
PATIENCE = 10
OPTIMIZER = "sgd"
LOCAL_LR = 0.001

TASK_MAP = {
    "dermamnist": "Skin lesion task",
    "bloodmnist": "Blood cell task"
}

COLOR_POS = "007C97"  
COLOR_NEG = "ED1C24"  

def candidate_run_roots(dataset, seed):
    cands = [
        os.path.join(dataset, f"seed_{seed}"),
        os.path.join(dataset.lower(), f"seed_{seed}"),
        os.path.join("out", dataset, f"seed_{seed}"),
        os.path.join("out", dataset.lower(), f"seed_{seed}"),
        os.path.join("out_file", dataset, f"seed_{seed}"),
        os.path.join("out_file", dataset.lower(), f"seed_{seed}"),
    ]
    cands += glob.glob(os.path.join("**", dataset,
                       f"seed_{seed}"), recursive=True)
    cands += glob.glob(os.path.join("**", dataset.lower(),
                       f"seed_{seed}"), recursive=True)

    uniq = []
    for p in cands:
        if os.path.isdir(p) and p not in uniq:
            uniq.append(p)
    return uniq


def find_summary_txt(run_root, mode_folder, split_rule, coef_str, method, optimizer, lr, patience, threshold):
    subdir = f"{split_rule}_{coef_str}"
    pat = os.path.join(
        run_root, "summary", mode_folder, subdir,
        f"{method}_*_*_{optimizer}_{lr}_{patience}_{threshold}.txt"
    )
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None

def parse_validation_txt(filepath):
    if (filepath is None) or (not os.path.exists(filepath)):
        return {"val_loss": None, "val_acc": None}

    with open(filepath, "r") as f:
        content = f.read()

    def extract_acc(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    loss_acc = extract_acc(
        r"Validation \(Loss\) Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%",
        content
    )
    val_acc = extract_acc(
        r"Validation \(Acc\) Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%",
        content
    )
    return {"val_loss": loss_acc, "val_acc": val_acc}


def parse_proposed_txt(filepath):
    if (filepath is None) or (not os.path.exists(filepath)):
        return {"proposed": None}

    with open(filepath, "r") as f:
        content = f.read()

    m = re.search(
        r"Proposed Early Stop Round\s*:[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%",
        content
    )
    prop_acc = float(m.group(1)) if m else None
    return {"proposed": prop_acc}


def calculate_gain(parsed_val, parsed_prop):
    if parsed_val is None or parsed_prop is None:
        return np.nan

    candidates = [v for v in [parsed_val.get(
        "val_loss"), parsed_val.get("val_acc")] if v is not None]
    base_val = max(candidates) if candidates else np.nan
    prop_val = parsed_prop.get("proposed")

    if (prop_val is not None) and (not np.isnan(base_val)):
        return prop_val - base_val
    return np.nan

records = []
print("Collecting Data... (NEW summary path structure)")

for dataset in DATASETS:
    for split_rule, coefs in SPLIT_CONFIGS.items():
        for coef in coefs:
            coef_str = str(coef)
            for method in METHODS:
                gains = []

                for seed in SEEDS:
                    run_roots = candidate_run_roots(dataset, seed)
                    if not run_roots:
                        continue

                    val_txt = None
                    prop_txt = None

                    for rr in run_roots:
                        if val_txt is None:
                            val_txt = find_summary_txt(
                                rr, "validation", split_rule, coef_str,
                                method, OPTIMIZER, LOCAL_LR, PATIENCE, TARGET_THRESHOLD
                            )
                        if prop_txt is None:
                            prop_txt = find_summary_txt(
                                rr, "proposed", split_rule, coef_str,
                                method, OPTIMIZER, LOCAL_LR, PATIENCE, TARGET_THRESHOLD
                            )
                        if (val_txt is not None) and (prop_txt is not None):
                            break

                    parsed_val = parse_validation_txt(
                        val_txt) if val_txt else None
                    parsed_prop = parse_proposed_txt(
                        prop_txt) if prop_txt else None

                    gain = calculate_gain(parsed_val, parsed_prop)
                    if not np.isnan(gain):
                        gains.append(gain)

                mean_gain = np.mean(gains) if gains else np.nan
                records.append({
                    "Dataset": dataset,
                    "SplitRule": split_rule,
                    "Coef": coef,
                    "Method": method,
                    "Gain": mean_gain
                })

df = pd.DataFrame(records)

if not df["Gain"].isna().all():
    MAX_ABS_VAL = df["Gain"].abs().max()
    if MAX_ABS_VAL == 0:
        MAX_ABS_VAL = 1.0
else:
    MAX_ABS_VAL = 1.0

def format_heatmap_cell(val, is_best=False):
    if pd.isna(val):
        return "-"

    if val == 0:
        txt = "0.00"
    else:
        txt = f"{val:+.2f}"

    if is_best:
        txt = f"\\textbf{{\\underline{{{txt}}}}}"

    if val == 0:
        return txt

    color_name = "myteal" if val > 0 else "myred"
    ratio = abs(val) / MAX_ABS_VAL
    percent = int(ratio * 100)

    if percent < 5:
        percent = 5
    if percent > 100:
        percent = 100

    text_color_cmd = r"\textcolor{white}" if percent > 60 else r"\textcolor{black}"
    return f"\\cellcolor{{{color_name}!{percent}}}{{{text_color_cmd}{{{txt}}}}}"

def generate_latex(df):
    RULE_NAME_MAP = {
        "Dirichlet":    r"\emph{Label skew} (Dirichlet)",
        "Pathological": r"\emph{Label skew} (Pathological)",
        "Quantity":     r"\emph{Quantity skew}",
    }

    rules_ordered = list(SPLIT_CONFIGS.keys())
    cols_per_rule = {rule: len(SPLIT_CONFIGS[rule]) for rule in rules_ordered}
    total_data_cols = sum(cols_per_rule.values())

    align_str = "c l" + ("c" * total_data_cols)

    latex = []
    latex.append(r"% Add \usepackage[table]{xcolor} to your preamble")
    latex.append(r"% Defining custom colors for heatmap")
    latex.append(f"\\definecolor{{myteal}}{{HTML}}{{{COLOR_POS}}}")
    latex.append(f"\\definecolor{{myred}}{{HTML}}{{{COLOR_NEG}}}")

    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Test Accuracy Gain ($\%$) heatmap. Positive(Teal), Negative(Red).}")
    latex.append(r"\label{tab:gain_heatmap}")
    latex.append(r"\resizebox{\textwidth}{!}{")
    latex.append(r"\scriptsize")
    latex.append(r"\begin{tabular}{" + align_str + r"}")
    latex.append(r"\toprule")
    row1 = ["", r"\multirow{2}{*}{\textbf{Method}}"]
    for rule in rules_ordered:
        pretty_name = RULE_NAME_MAP.get(rule, rule)
        row1.append(
            r"\multicolumn{" +
            str(cols_per_rule[rule]) + r"}{c}{" + pretty_name + r"}"
        )
    latex.append(" & ".join(row1) + r" \\")

    cmid_parts = []
    cur = 3
    for rule in rules_ordered:
        end = cur + cols_per_rule[rule] - 1
        cmid_parts.append(f"\\cmidrule(lr){{{cur}-{end}}}")
        cur = end + 1
    latex.append(" ".join(cmid_parts))
    row2 = ["", ""]
    for rule in rules_ordered:
        for coef in SPLIT_CONFIGS[rule]:
            val_str = str(int(coef)) if rule == 'Pathological' else str(coef)
            row2.append(r"$c=" + val_str + r"$")
    latex.append(" & ".join(row2) + r" \\")
    latex.append(r"\midrule")
    num_methods = len(METHODS)
    for idx, ds in enumerate(DATASETS):
        task_name = TASK_MAP.get(ds, ds)

        best_map = {}
        for rule in rules_ordered:
            for coef in SPLIT_CONFIGS[rule]:
                sub = df[(df["Dataset"] == ds) &
                         (df["SplitRule"] == rule) &
                         (df["Coef"] == coef)]
                best_map[(rule, coef)] = sub["Gain"].max() if (
                    not sub.empty and not sub["Gain"].isna().all()) else -999

        for m_idx, method in enumerate(METHODS):
            row = []
            if m_idx == 0:
                row.append(r"\multirow{" + str(num_methods) +
                           r"}{*}{\rotatebox[origin=c]{90}{" + task_name + r"}}")
            else:
                row.append("")

            row.append(method)

            for rule in rules_ordered:
                for coef in SPLIT_CONFIGS[rule]:
                    rec = df[(df["Dataset"] == ds) &
                             (df["SplitRule"] == rule) &
                             (df["Coef"] == coef) &
                             (df["Method"] == method)]

                    if rec.empty or pd.isna(rec.iloc[0]["Gain"]):
                        row.append("-")
                    else:
                        val = float(rec.iloc[0]["Gain"])
                        is_best = (val >= best_map[(rule, coef)] - 1e-9)
                        row.append(format_heatmap_cell(val, is_best))

            latex.append(" & ".join(row) + r" \\")

        latex.append(r"\midrule" if idx < len(
            DATASETS) - 1 else r"\bottomrule")

    latex.append(r"\end{tabular}")
    latex.append(r"}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)

output_filename = "table1.tex"
if df.empty or df["Gain"].isna().all():
    print("No valid data.")
else:
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(generate_latex(df))
        print(f"Saved Heatmap Table to '{output_filename}'")
        print(
            "IMPORTANT: Include \\usepackage[table]{xcolor} in your LaTeX preamble.")
    except Exception as e:
        print(f"Error: {e}")