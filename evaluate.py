"""Run the final parameter sweep.

The Best F1 table is extracted from this same parameter sweep. Defaults match
the final reproducibility setup: FIX_NUMBER=150, tau_m in {0.5, ..., 1.0},
tau_c in {80, 90, 100}, generated data under ``data/*/generated``, and summary
files under ``results/``.

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
from typing import Dict, Iterable, List

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
OUTPUT_CSV = RESULTS_DIR / "sweep.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"


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


RUN_TAU_M = TAU_M_VALUES
RUN_TAU_C = TAU_C_VALUES
RUN_FIX_NUMBER = FIX_NUMBER
LOGIC_BATCH_SIZE = 32

@contextlib.contextmanager
def quiet_output():
    """Suppress model chatter while keeping this script's tqdm bars."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def ensure_writer(path: Path, fields: Iterable[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=list(fields))
    writer._output_file = fp  # type: ignore[attr-defined]
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


def evaluate() -> None:
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
    total_runs = len(RUN_SETTINGS) * len(RUN_TAU_M) * len(RUN_TAU_C)

    try:
        with tqdm(total=len(RUN_SETTINGS), desc="logic warmup", unit="setting", dynamic_ncols=True) as warmup:
            for setting in RUN_SETTINGS:
                rows = load_rows(ROOT, setting)
                with quiet_output():
                    logic_cache = runner.build_logic_cache(setting)
                logic_cache.prepare_rows(rows, batch_size=LOGIC_BATCH_SIZE)
                rows_by_setting[setting.key] = rows
                logic_caches[setting.key] = logic_cache
                warmup.update(1)

        with tqdm(total=total_runs, desc="parameter sweep", unit="run", dynamic_ncols=True) as overall:
            for setting in RUN_SETTINGS:
                rows = rows_by_setting[setting.key]
                logic_cache = logic_caches[setting.key]
                for tau_m in RUN_TAU_M:
                    for tau_c in RUN_TAU_C:
                        result = runner.run_threshold(setting, rows, tau_m, tau_c, logic_cache)
                        write_result(result_writer, metric_writer, result)
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
