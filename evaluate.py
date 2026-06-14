"""Run the final parameter sweep reports.

The Best F1 table is extracted from this same parameter sweep.  Defaults match
the final reproducibility setup: FIX_NUMBER=150, tau_m in {0.5, ..., 1.0},
tau_c in {80, 90, 100}, generated data under ``data/*/generated``, and reports
under ``results/reports``.

By default, all paper settings are enabled: ARCT/ANLI original plus generated
one/two/three-step variants.
"""

import contextlib
import csv
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tqdm.auto import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ("transformers", "sentence_transformers", "transition_amr_parser"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from enthymeme_eval import logic
from enthymeme_eval.config import FIX_NUMBER, TAU_C_VALUES, TAU_M_VALUES, Setting
from enthymeme_eval.datasets import load_rows
from enthymeme_eval.models import ModelBundle
from enthymeme_eval.runner import EvaluationRunner, ThresholdResult


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache_eval"
RESULTS_DIR = ROOT / "results"
EXPERIMENT_NAME = os.environ.get("PAPER_EXPERIMENT_NAME", "sweep")
REPORTS_DIR = RESULTS_DIR / os.environ.get("PAPER_REPORTS_SUBDIR", "reports")
OUTPUT_CSV = RESULTS_DIR / os.environ.get("PAPER_OUTPUT", f"{EXPERIMENT_NAME}.csv")
METRICS_CSV = RESULTS_DIR / os.environ.get("PAPER_METRICS_OUTPUT", "metrics.csv")


# Comment/uncomment settings here to choose which runs to execute.
RUN_SETTINGS = [
    Setting("arct", "original", None, 1),
    Setting("arct", "one", "one.csv", 1),
    Setting("arct", "two", "two.csv", 2),
    Setting("arct", "three", "three.csv", 3),
    Setting("anli", "original", None, 1),
    Setting("anli", "one", "one.csv", 1),
    Setting("anli", "two", "two.csv", 2),
    Setting("anli", "three", "three.csv", 3),
]


def _env_list(name: str) -> Optional[List[str]]:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _float_values(name: str, default: List[float]) -> List[float]:
    values = _env_list(name)
    return [float(value) for value in values] if values else default


def _int_values(name: str, default: List[int]) -> List[int]:
    values = _env_list(name)
    return [int(value) for value in values] if values else default


def _filter_settings(settings: List[Setting]) -> List[Setting]:
    datasets = set(_env_list("PAPER_RUN_DATASETS") or [])
    variants = set(_env_list("PAPER_RUN_VARIANTS") or [])
    return [
        setting
        for setting in settings
        if (not datasets or setting.dataset in datasets)
        and (not variants or setting.variant in variants)
    ]


RUN_SETTINGS = _filter_settings(RUN_SETTINGS)
RUN_TAU_M = _float_values("PAPER_TAU_M_VALUES", TAU_M_VALUES)
RUN_TAU_C = _int_values("PAPER_TAU_C_VALUES", TAU_C_VALUES)
RUN_FIX_NUMBER = int(os.environ.get("PAPER_FIX_NUMBER", str(FIX_NUMBER)))
LOGIC_BATCH_SIZE = 32
PAPER_SUBSET = os.environ.get("PAPER_SUBSET", "1") != "0"

ResultKey = Tuple[str, str, str, str]


@contextlib.contextmanager
def quiet_output():
    """Suppress model chatter while keeping this script's tqdm bars."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def result_key(dataset: str, variant: str, tau_m: float, tau_c: int) -> ResultKey:
    return dataset, variant, f"{tau_m:g}", str(tau_c)


def report_path(dataset: str, variant: str, tau_m: float, tau_c: int) -> Path:
    name = f"{dataset}_{variant}_tm{tau_m:g}_tc{tau_c}.txt".replace(".", "p")
    return REPORTS_DIR / name


def existing_csv_keys(path: Path) -> Set[ResultKey]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as fp:
        return {
            (row["dataset"], row["variant"], row["tau_m"], row["tau_c"])
            for row in csv.DictReader(fp)
        }


def existing_report_keys() -> Set[ResultKey]:
    keys: Set[ResultKey] = set()
    for path in REPORTS_DIR.glob("*ptxt"):
        stem = path.name[:-4] if path.name.endswith("ptxt") else path.stem
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        dataset = parts[0]
        variant = parts[1]
        tau_m = parts[2][2:].replace("p", ".") if parts[2].startswith("tm") else parts[2].replace("p", ".")
        tau_c = parts[3][2:] if parts[3].startswith("tc") else parts[3]
        keys.add((dataset, variant, f"{float(tau_m):g}", tau_c))
    return keys


def ensure_writer(path: Path, fields: Iterable[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fp = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=list(fields))
    writer._output_file = fp  # type: ignore[attr-defined]
    if not exists:
        writer.writeheader()
        fp.flush()
    return writer


def close_writer(writer: csv.DictWriter) -> None:
    getattr(writer, "_output_file").close()


def write_result(result_writer: csv.DictWriter, metric_writer: csv.DictWriter, result: ThresholdResult) -> None:
    result_writer.writerow(result.as_row())
    getattr(result_writer, "_output_file").flush()

    for class_id in ("0", "1"):
        class_metrics = result.metrics[class_id]
        metric_writer.writerow(
            {
                "dataset": result.dataset,
                "variant": result.variant,
                "tau_m": f"{result.tau_m:g}",
                "tau_c": str(result.tau_c),
                "class": class_id,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1": class_metrics["f1-score"],
                "support": int(class_metrics["support"]),
                "accuracy": result.accuracy,
            }
        )
    getattr(metric_writer, "_output_file").flush()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path(result.dataset, result.variant, result.tau_m, result.tau_c).write_text(
        (
            f"dataset={result.dataset}\n"
            f"variant={result.variant}\n"
            f"tau_m={result.tau_m:g}\n"
            f"tau_c={result.tau_c}\n"
            f"accuracy={result.accuracy:.6f}\n"
            f"support={result.support}\n"
            f"counted_rows={result.counted_rows}\n\n"
            f"skipped_invalid={result.skipped_invalid}\n"
            f"{result.report}\n"
        ),
        encoding="utf-8",
    )


def evaluate() -> None:
    # A run is complete only when both the report and the summary CSV contain
    # it. This keeps resume safe after targeted runs that wrote reports to the
    # shared report directory but used a separate output CSV.
    done = existing_report_keys() & existing_csv_keys(OUTPUT_CSV)
    models = ModelBundle()
    runner = EvaluationRunner(cache_dir=CACHE_DIR, models=models, core=logic, fix_number=RUN_FIX_NUMBER)
    result_writer = ensure_writer(
        OUTPUT_CSV,
        [
            "dataset",
            "variant",
            "tau_m",
            "tau_c",
            "accuracy",
            "support",
            "counted_rows",
            "skipped_invalid",
            "skipped_ent",
            "skipped_both",
            "exceptions",
            "elapsed_sec",
        ],
    )
    metric_writer = ensure_writer(
        METRICS_CSV,
        ["dataset", "variant", "tau_m", "tau_c", "class", "precision", "recall", "f1", "support", "accuracy"],
    )
    rows_by_setting: Dict[str, List[list]] = {}
    logic_caches = {}
    pending_runs = sum(
        1
        for setting in RUN_SETTINGS
        for tau_m in RUN_TAU_M
        for tau_c in RUN_TAU_C
        if result_key(setting.dataset, setting.variant, tau_m, tau_c) not in done
    )

    try:
        with tqdm(total=len(RUN_SETTINGS), desc="logic warmup", unit="setting", dynamic_ncols=True) as warmup:
            for setting in RUN_SETTINGS:
                rows = load_rows(ROOT, setting, paper_subset=PAPER_SUBSET)
                with quiet_output():
                    logic_cache = runner.build_logic_cache(setting)
                logic_cache.prepare_rows(rows, batch_size=LOGIC_BATCH_SIZE, row_compat=runner.row_compat)
                rows_by_setting[setting.key] = rows
                logic_caches[setting.key] = logic_cache
                warmup.update(1)

        with tqdm(total=pending_runs, desc="paper reports", unit="run", dynamic_ncols=True) as overall:
            for setting in RUN_SETTINGS:
                rows = rows_by_setting[setting.key]
                logic_cache = logic_caches[setting.key]
                for tau_m in RUN_TAU_M:
                    for tau_c in RUN_TAU_C:
                        key = result_key(setting.dataset, setting.variant, tau_m, tau_c)
                        if key in done:
                            continue
                        result = runner.run_threshold(setting, rows, tau_m, tau_c, logic_cache)
                        write_result(result_writer, metric_writer, result)
                        done.add(key)
                        overall.update(1)
                        overall.set_postfix(
                            last=f"{result.dataset}/{result.variant}",
                            acc=f"{result.accuracy:.4f}",
                            support=result.support,
                            refresh=False,
                        )
    finally:
        close_writer(result_writer)
        close_writer(metric_writer)
        runner.close()


if __name__ == "__main__":
    evaluate()
