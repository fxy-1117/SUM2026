"""Threshold evaluation logic shared by CLI scripts."""

import contextlib
import io
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

from .cache import LogicCache, NeuralCache
from .config import FIX_NUMBER, Setting


@dataclass
class ThresholdResult:
    dataset: str
    variant: str
    tau_m: float
    tau_c: int
    accuracy: float
    support: int
    counted_rows: int
    skipped_invalid: int
    skipped_ent: int
    skipped_both: int
    exceptions: int
    elapsed_sec: float
    metrics: Dict[str, Any]

    def as_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "variant": self.variant,
            "tau_m": self.tau_m,
            "tau_c": self.tau_c,
            "accuracy": self.accuracy,
            "support": self.support,
            "counted_rows": self.counted_rows,
            "skipped_invalid": self.skipped_invalid,
            "skipped_ent": self.skipped_ent,
            "skipped_both": self.skipped_both,
            "exceptions": self.exceptions,
            "elapsed_sec": self.elapsed_sec,
        }


class EvaluationRunner:
    """Evaluate one setting over all rows for each tau_m/tau_c threshold."""

    def __init__(self, cache_dir: Path, models: Any, core: Any, fix_number: int = FIX_NUMBER) -> None:
        self.cache_dir = cache_dir
        self.models = models
        self.core = core
        self.fix_number = fix_number
        self.neural_caches: Dict[str, NeuralCache] = {}
        self.core.nli_tokenizer = None
        self.core.model_nli = None

    def close(self) -> None:
        for neural_cache in self.neural_caches.values():
            neural_cache.close()

    def _use_neural_cache(self, dataset: str) -> None:
        neural_cache = self.neural_caches.get(dataset)
        if neural_cache is None:
            db_path = self.cache_dir / "neural_cache" / f"{dataset}.sqlite"
            neural_cache = NeuralCache(db_path, self.models)
            self.neural_caches[dataset] = neural_cache
        self.core.score = neural_cache.score
        self.core.NLI = neural_cache.nli

    def build_logic_cache(self, setting: Setting) -> LogicCache:
        parser, converter = self.models.load_amr()
        return LogicCache(self.cache_dir, setting.key, self.core, parser, converter)

    def row_outcome(
        self,
        setting: Setting,
        row: list,
        tau_m: float,
        tau_c: int,
        logic_cache: LogicCache,
    ) -> Tuple[str, List[int], List[int]]:
        """Evaluate one row with the final sweep skip rules."""
        self._use_neural_cache(setting.dataset)
        with self._quiet():
            pre_data = logic_cache.get_row(row[:-1])
            temcheck = self.core.prove([pre_data[0], pre_data[1]], tau_m, tau_c)
        if temcheck and temcheck[0] == "ent":
            return "ent", [], []

        helpful_start = 2
        helpful_end = helpful_start + setting.steps
        non_helpful_end = helpful_end + setting.steps
        helpful_logic = pre_data[helpful_start:helpful_end]
        non_helpful_logic = pre_data[helpful_end:non_helpful_end]

        with self._quiet():
            label1 = self.core.prove([pre_data[0], pre_data[1]] + helpful_logic, tau_m, tau_c)
            label2 = self.core.prove([pre_data[0], pre_data[1]] + non_helpful_logic, tau_m, tau_c)

        label1_name = label1[0]
        label2_name = label2[0]
        if label1_name == "both" or label2_name == "both":
            return "both", [], []

        if label1_name == "ent" and label2_name != "ent":
            predicted = [1, 0]
        elif label1_name != "ent" and label2_name == "ent":
            predicted = [0, 1]
        elif label1_name == "ent" and label2_name == "ent":
            predicted = [1, 1]
        else:
            predicted = [0, 0]

        gold = [1, 0] if row[-1] == 1 else [0, 1]
        return "counted", predicted, gold

    def run_threshold(
        self,
        setting: Setting,
        rows: List[list],
        tau_m: float,
        tau_c: int,
        logic_cache: LogicCache,
    ) -> ThresholdResult:
        """Run the proof pipeline until FIX_NUMBER valid rows are counted."""
        self._use_neural_cache(setting.dataset)
        started = time.time()
        ll_stsb: List[int] = []
        gl_stsb: List[int] = []
        counted = 0
        skipped_invalid = 0
        skipped_ent = 0
        skipped_both = 0
        exceptions = 0

        label = f"{setting.dataset}/{setting.variant} tm={tau_m:g} tc={tau_c}"
        with tqdm(total=len(rows), desc=label, unit="row", dynamic_ncols=True, leave=False) as progress:
            for row in rows:
                try:
                    status, predicted, gold = self.row_outcome(setting, row, tau_m, tau_c, logic_cache)
                    if status == "ent":
                        skipped_ent += 1
                        continue
                    if status == "both":
                        skipped_both += 1
                        continue

                    ll_stsb.extend(predicted)
                    gl_stsb.extend(gold)
                    counted += 1
                    if counted == self.fix_number:
                        break
                except Exception:
                    exceptions += 1
                    continue
                finally:
                    progress.update(1)
                    progress.set_postfix(
                        valid=counted,
                        invalid=skipped_invalid,
                        ent=skipped_ent,
                        both=skipped_both,
                        err=exceptions,
                        refresh=False,
                    )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = classification_report(gl_stsb, ll_stsb, zero_division=0, output_dict=True)
        accuracy = accuracy_score(gl_stsb, ll_stsb)
        return ThresholdResult(
            dataset=setting.dataset,
            variant=setting.variant,
            tau_m=tau_m,
            tau_c=tau_c,
            accuracy=float(accuracy),
            support=len(gl_stsb),
            counted_rows=counted,
            skipped_invalid=skipped_invalid,
            skipped_ent=skipped_ent,
            skipped_both=skipped_both,
            exceptions=exceptions,
            elapsed_sec=round(time.time() - started, 3),
            metrics=metrics,
        )

    @staticmethod
    @contextlib.contextmanager
    def _quiet():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
