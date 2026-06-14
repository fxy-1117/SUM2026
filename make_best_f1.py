"""Build the final Best F1 table from parameter sweep metrics."""

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = os.environ.get("BEST_F1_EXPERIMENT_NAME", "best_f1")
REPORTS_DIR = ROOT / "results" / os.environ.get("BEST_F1_REPORTS_SUBDIR", "reports")
METRICS_CSV = ROOT / "results" / os.environ.get("BEST_F1_METRICS_INPUT", "metrics.csv")
OUTPUT_CSV = ROOT / "results" / os.environ.get("BEST_F1_OUTPUT", f"{EXPERIMENT_NAME}.csv")
OUTPUT_TEX = ROOT / "results" / os.environ.get("BEST_F1_TEX_OUTPUT", f"{EXPERIMENT_NAME}.tex")
USE_PAPER_TIE_BREAKS = os.environ.get("BEST_F1_USE_PAPER_TIES", "1") == "1"
RUN_DATASETS = [
    part.strip()
    for part in os.environ.get("BEST_F1_DATASETS", "anli,arct").split(",")
    if part.strip()
]


# The table reports F1 rounded to two decimals. When several parameter settings
# share the same displayed F1, these preferences choose the final table entry.
PAPER_TIE_BREAKS: Dict[Tuple[str, str, str], Tuple[str, str]] = {
    ("anli", "0", "original"): ("0.8", "80"),
    ("anli", "0", "one"): ("0.6", "80"),
    ("anli", "0", "two"): ("0.5", "80"),
    ("anli", "0", "three"): ("0.6", "80"),
    ("anli", "1", "original"): ("0.55", "100"),
    ("anli", "1", "one"): ("0.55", "90"),
    ("anli", "1", "two"): ("0.6", "100"),
    ("anli", "1", "three"): ("0.55", "80"),
    ("arct", "0", "original"): ("0.65", "80"),
    ("arct", "0", "one"): ("0.7", "80"),
    ("arct", "0", "two"): ("0.75", "80"),
    ("arct", "0", "three"): ("0.7", "90"),
    ("arct", "1", "original"): ("0.6", "80"),
    ("arct", "1", "one"): ("0.65", "90"),
    ("arct", "1", "two"): ("0.6", "90"),
    ("arct", "1", "three"): ("0.65", "90"),
}


Record = Dict[str, str]


def read_records() -> List[Record]:
    if METRICS_CSV.exists():
        with METRICS_CSV.open("r", newline="", encoding="utf-8") as fp:
            return [
                {
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "tau_m": row["tau_m"],
                    "tau_c": row["tau_c"],
                    "class": row["class"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": f"{float(row['f1']):.2f}",
                    "class_support": row["support"],
                    "accuracy": row["accuracy"],
                }
                for row in csv.DictReader(fp)
            ]

    records: List[Record] = []
    for path in sorted(REPORTS_DIR.glob("*ptxt")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        meta: Dict[str, str] = {}
        for line in lines:
            if "=" in line and line.split("=", 1)[0] in {
                "dataset",
                "variant",
                "tau_m",
                "tau_c",
                "accuracy",
                "support",
                "counted_rows",
            }:
                key, value = line.split("=", 1)
                meta[key] = value

        for line in lines:
            match = re.match(r"\s*([01])\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
            if match and {"dataset", "variant", "tau_m", "tau_c"} <= set(meta):
                class_id, precision, recall, f1, support = match.groups()
                records.append(
                    {
                        **meta,
                        "class": class_id,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "class_support": support,
                    }
                )
    return records


def choose_best(records: List[Record], dataset: str, class_id: str, variant: str) -> Record:
    candidates = [
        record
        for record in records
        if record["dataset"] == dataset and record["class"] == class_id and record["variant"] == variant
    ]
    if not candidates:
        raise RuntimeError(f"Missing reports for {dataset}/{variant} class {class_id}")

    max_f1 = max(float(record["f1"]) for record in candidates)
    bests = [record for record in candidates if float(record["f1"]) == max_f1]

    preferred = PAPER_TIE_BREAKS.get((dataset, class_id, variant)) if USE_PAPER_TIE_BREAKS else None
    if preferred is not None:
        tau_m, tau_c = preferred
        for record in bests:
            if f"{float(record['tau_m']):g}" == tau_m and record["tau_c"] == tau_c:
                return record

    return sorted(bests, key=lambda record: (float(record["tau_m"]), int(record["tau_c"])))[0]


def step_label(variant: str) -> str:
    return {
        "original": "original",
        "one": "1-step",
        "two": "2-step",
        "three": "3-step",
    }[variant]


def table_rows(records: List[Record]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for dataset in RUN_DATASETS:
        for class_id in ("0", "1"):
            for variant in ("original", "one", "two", "three"):
                record = choose_best(records, dataset, class_id, variant)
                rows.append(
                    {
                        "dataset": dataset.upper(),
                        "class": class_id,
                        "step_type": step_label(variant),
                        "best_f1": f"{float(record['f1']):.2f}",
                        "tau_m": f"{float(record['tau_m']):g}",
                        "tau_c": record["tau_c"],
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, str]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset", "class", "step_type", "best_f1", "tau_m", "tau_c"])
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: List[Dict[str, str]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\begin{tabular}{lllllr}",
        "    \\toprule",
        "    Dataset & Class & Step type & Best F1 & $\\tau_m$ & $\\tau_c$ \\\\",
        "    \\midrule",
    ]

    for dataset in [dataset.upper() for dataset in RUN_DATASETS]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        lines.append(f"    \\multirow{{8}}{{*}}{{{dataset}}}")
        for class_id in ("0", "1"):
            class_rows = [row for row in dataset_rows if row["class"] == class_id]
            first = class_rows[0]
            lines.append(
                f"    & \\multirow{{4}}{{*}}{{{class_id}}} & {first['step_type']} & "
                f"{first['best_f1']} & {first['tau_m']} & {first['tau_c']} \\\\"
            )
            for row in class_rows[1:]:
                lines.append(f"    & & {row['step_type']} & {row['best_f1']} & {row['tau_m']} & {row['tau_c']} \\\\")
            if class_id == "0":
                lines.append("    \\cmidrule(l){2-6}")
        if dataset != [item.upper() for item in RUN_DATASETS][-1]:
            lines.append("    \\midrule")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Best F1-Score by Dataset, Class (non-entailment is 0 and entailment is 1), and Step type.}",
            "  \\label{tab:best_f1_scores}",
            "\\end{table}",
            "",
        ]
    )
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = table_rows(read_records())
    write_csv(rows)
    write_tex(rows)
    for row in rows:
        print(
            f"{row['dataset']:4s} class={row['class']} {row['step_type']:8s} "
            f"F1={row['best_f1']} tau_m={row['tau_m']} tau_c={row['tau_c']}"
        )
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
