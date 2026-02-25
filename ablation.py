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

LRS = [0.1]
THRESHOLD = 0.1

SPLIT_RULE = "Dirichlet"
SPLIT_COEF = "0.1"
PATIENCE = 10
OPTIMIZER = "sgd"

R_FIXED = 500

TASK_TITLES = {
    "dermamnist": "Skin lesion task",
    "bloodmnist": "Blood cell task",
}

def candidate_run_roots(dataset, seed):
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

    with open(filepath, 'r') as f:
        content = f.read()

    def extract(pattern):
        m = re.search(pattern, content)
        if m:
            return int(m.group(1)), float(m.group(2))
        return None, None

    r_loss, acc_loss = extract(
        r"Validation \(Loss\) Stop Round\s*:\s*(\d+)[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )
    r_val, acc_val = extract(
        r"Validation \(Acc\) Stop Round\s*:\s*(\d+)[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )
    r_prop, acc_prop = extract(
        r"Proposed Early Stop Round\s*:\s*(\d+)[^\n]*\n\s*-\s*Test Accuracy at Stop\s*:\s*([0-9\.]+)%"
    )

    return {"loss": (r_loss, acc_loss), "val": (r_val, acc_val), "prop": (r_prop, acc_prop)}

records = []
print(f"Collecting Data...")

for dataset in DATASETS:
    for method in METHODS:

        prop_accs, prop_rounds = [], []
        delta_accs, delta_rounds = [], []
        eff_joint_vals = []

        for seed in SEEDS:

            run_roots = candidate_run_roots(dataset, seed)

            if not run_roots:
                prop_accs.append(np.nan)
                prop_rounds.append(np.nan)
                delta_accs.append(np.nan)
                delta_rounds.append(np.nan)
                eff_joint_vals.append(np.nan)
                continue

            subdir = f"{SPLIT_RULE}_{SPLIT_COEF}"

            txt_val, txt_prop = [], []

            for rr in run_roots:
                txt_val += glob.glob(
                    os.path.join(
                        rr, "summary", "validation", subdir,
                        f"{method}_*_{OPTIMIZER}_{LRS[0]}_{PATIENCE}_{THRESHOLD}.txt"
                    )
                )
                txt_prop += glob.glob(
                    os.path.join(
                        rr, "summary", "proposed", subdir,
                        f"{method}_*_{OPTIMIZER}_{LRS[0]}_{PATIENCE}_{THRESHOLD}.txt"
                    )
                )

            if not txt_prop:
                prop_accs.append(np.nan)
                prop_rounds.append(np.nan)
                delta_accs.append(np.nan)
                delta_rounds.append(np.nan)
                eff_joint_vals.append(np.nan)
                continue

            parsed_prop = parse_txt_file(txt_prop[0])
            parsed_val = parse_txt_file(txt_val[0]) if txt_val else None

            r_prop, acc_prop = parsed_prop["prop"]
            r_loss, acc_loss = parsed_val["loss"] if parsed_val else (
                None, None)
            r_val, acc_val = parsed_val["val"] if parsed_val else (None, None)

            valid_prop = (r_prop is not None) and (acc_prop is not None)
            valid_loss = (r_loss is not None) and (acc_loss is not None)
            valid_val = (r_val is not None) and (acc_val is not None)

            # Proposed
            prop_accs.append(acc_prop if valid_prop else np.nan)
            prop_rounds.append(r_prop if valid_prop else np.nan)

            # Delta
            if valid_prop and (valid_loss or valid_val):
                base_acc = max(
                    [v for v in [acc_loss, acc_val] if v is not None])
                base_round = min([v for v in [r_loss, r_val] if v is not None])

                d_acc = acc_prop - base_acc
                d_round = r_prop - base_round
            else:
                d_acc, d_round = np.nan, np.nan

            delta_accs.append(d_acc)
            delta_rounds.append(d_round)

            if valid_prop and r_prop > 0:
                acc_adj = 0.0 if np.isnan(d_acc) else (d_acc / 100.0)
                eff_joint = (R_FIXED / float(r_prop)) * (1.0 + acc_adj)
            else:
                eff_joint = np.nan

            eff_joint_vals.append(eff_joint)

        records.append({
            "Dataset": dataset,
            "Method": method,
            "Prop_Acc_Mean": np.nanmean(prop_accs),
            "Prop_Round_Mean": np.nanmean(prop_rounds),
            "Delta_Acc_Mean": np.nanmean(delta_accs),
            "Delta_Round_Mean": np.nanmean(delta_rounds),
            "EffJoint_Mean": np.nanmean(eff_joint_vals)
        })

df = pd.DataFrame(records)

latex_code = []

latex_code.append(r"\begin{table*}[t]")
latex_code.append(r"\centering")
latex_code.append(
    r"\caption{Performance and stopping behavior of the proposed approach in terms of the stopping round ($r^*$) "
    r"and corresponding test accuracy (\textit{Acc.}), under $\tau=0.1$ and $\rho=10$. "
    r"The $\Delta_{Acc.}$ and $\Delta_{r}$ denote the accuracy gain and round difference, "
    r"respectively, relative to the best validation-based baseline at its stopping point.}"
)
latex_code.append(r"\vspace{-0.38cm}")
latex_code.append(r"\begin{threeparttable}")
latex_code.append(r"\label{tab:1}")

latex_code.append(r"\begin{tabular}{lcccccccc}")
latex_code.append(r"\toprule")

header1 = r"\multirow{2}{*}{\textbf{Method}} & "
header2 = " & "
cmid = ""

for i, dataset in enumerate(DATASETS):
    title = TASK_TITLES.get(dataset, dataset)

    header1 += rf"\multicolumn{{4}}{{c}}{{{title}}}"

    header2 += r"\textit{Acc.} (\%) & $r^*$ & $\Delta_{Acc.}$ & $\Delta_{r}$"

    start = 2 + i * 4
    end = start + 3
    cmid += rf"\cmidrule(lr){{{start}-{end}}} "

    if i < len(DATASETS) - 1:
        header1 += " & "
        header2 += " & "

header1 += r" \\"
header2 += r" \\"

latex_code.append(header1)
latex_code.append(cmid.strip())
latex_code.append(header2)
latex_code.append(r"\midrule")

def fmt_delta(val, fmt="{:+.2f}"):
    if np.isnan(val):
        return "N/A"
    if abs(val) < 0.005:
        return "-"
    return fmt.format(val)

for method in METHODS:
    row = method

    for dataset in DATASETS:
        sub = df[(df["Dataset"] == dataset) & (df["Method"] == method)]

        if sub.empty or np.isnan(sub["Prop_Acc_Mean"].item()):
            row += " & N/A & N/A & N/A & N/A"
        else:
            acc = sub["Prop_Acc_Mean"].item()
            rnd = sub["Prop_Round_Mean"].item()
            dacc = sub["Delta_Acc_Mean"].item()
            drnd = sub["Delta_Round_Mean"].item()

            s_acc = f"{acc:.2f}"
            s_rnd = f"{rnd:.1f}"
            s_dacc = fmt_delta(dacc, "{:+.2f}")
            s_drnd = fmt_delta(drnd, "{:+.1f}")

            row += f" & {s_acc} & {s_rnd} & {s_dacc} & {s_drnd}"

    row += r" \\"
    latex_code.append(row)

latex_code.append(r"\bottomrule")
latex_code.append(r"\end{tabular}")
latex_code.append(r"\begin{tablenotes}[flushleft]")
latex_code.append(r"\footnotesize")
latex_code.append(
    r"\item \textit{Note.} The symbol `-' denotes no difference.")
latex_code.append(r"\end{tablenotes}")
latex_code.append(r"\end{threeparttable}")
latex_code.append(r"\vspace{-0.38cm}")
latex_code.append(r"\end{table*}")

output_path = "ablation.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_code))

print(f"LaTeX table saved to {output_path}")
