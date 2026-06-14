"""Evaluate the final entailment-only table."""

from __future__ import annotations

import contextlib
import csv
import io
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ("transformers", "sentence_transformers", "transition_amr_parser"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from enthymeme_eval import logic as logic_core
from enthymeme_eval.cache import LogicCache, NeuralCache
from enthymeme_eval.config import SHUFFLE_SEED
from enthymeme_eval.models import ModelBundle


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache_eval"
RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "entailment.csv"
OUTPUT_TEX = RESULTS_DIR / "entailment.tex"

FIX_NUMBER = 250
LOGIC_BATCH_SIZE = 8
ANLI_SOURCE_LIMIT = 1000

# Comment entries here if you want to run only part of the entailment table.
DATASETS = ["anli", "arct"]
STEP_TYPES = ["none", "original", "one", "two", "three"]

PARAMS = {
    "anli": (0.55, 90),
    "arct": (0.65, 90),
}

EXPECTED = {
    ("anli", "none"): 0.530,
    ("anli", "original"): 0.558,
    ("anli", "one"): 0.645,
    ("anli", "two"): 0.673,
    ("anli", "three"): 0.733,
    ("arct", "none"): 0.293,
    ("arct", "original"): 0.303,
    ("arct", "one"): 0.478,
    ("arct", "two"): 0.518,
    ("arct", "three"): 0.563,
}

STEP_COUNT = {
    "one": 1,
    "two": 2,
    "three": 3,
}

STEP_LABELS = {
    "none": "none",
    "original": "original",
    "one": "1-step",
    "two": "2-step",
    "three": "3-step",
}


@dataclass
class EntailmentItem:
    dataset: str
    step_type: str
    premise: str
    claim: str
    implicit: List[str]


@dataclass
class EntailmentResult:
    dataset: str
    step_type: str
    tau_m: float
    tau_c: int
    fix_number: int
    entailment_predictions: int
    valid_items: int
    skipped_false: int
    exceptions: int
    seen_items: int
    available_items: int
    exhausted: bool
    accuracy: float
    expected: float
    elapsed_sec: float


def split_steps(text: str, steps: int) -> List[str]:
    parts = [part.strip() for part in str(text).split(". ") if part.strip()]
    return parts[:steps]


def seeded_shuffle(items: List[EntailmentItem]) -> List[EntailmentItem]:
    shuffled = list(items)
    np.random.seed(SHUFFLE_SEED)
    np.random.shuffle(shuffled)
    return shuffled


def dedup_premise_claim(items: List[EntailmentItem]) -> List[EntailmentItem]:
    deduped: List[EntailmentItem] = []
    seen = set()
    for item in items:
        key = (str(item.premise), str(item.claim))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def load_items(dataset: str, step_type: str) -> List[EntailmentItem]:
    if step_type in ("none", "original"):
        items = load_original_items(dataset, step_type)
    else:
        items = load_generated_items(dataset, step_type)
    if dataset == "arct":
        items = dedup_premise_claim(items)
    return seeded_shuffle(items)


def load_original_items(dataset: str, step_type: str) -> List[EntailmentItem]:
    if dataset == "anli":
        return load_anli_original_items(step_type)
    if dataset == "arct":
        return load_arct_original_items(step_type)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_anli_original_items(step_type: str) -> List[EntailmentItem]:
    anli_dir = ROOT / "data" / "anli" / "original"
    df = pd.read_json(anli_dir / "test.jsonl", lines=True).iloc[:ANLI_SOURCE_LIMIT]
    labels = pd.read_csv(anli_dir / "test-labels.lst", header=None).iloc[:ANLI_SOURCE_LIMIT]
    items: List[EntailmentItem] = []
    for idx, row in df.iterrows():
        correct_hypothesis = row["hyp1"] if int(labels.iloc[idx][0]) == 1 else row["hyp2"]
        implicit = [] if step_type == "none" else [correct_hypothesis]
        items.append(EntailmentItem("anli", step_type, row["obs1"], row["obs2"], implicit))
    return items


def load_arct_original_items(step_type: str) -> List[EntailmentItem]:
    df = pd.read_csv(ROOT / "data" / "arct" / "original" / "test.tsv", sep="\t")
    items: List[EntailmentItem] = []
    for _, row in df.iterrows():
        implicit = [] if step_type == "none" else [correct_arct_warrant(row)]
        items.append(EntailmentItem("arct", step_type, row["reason"], row["claim"], implicit))
    return items


def correct_arct_warrant(row) -> str:
    return str(row["warrant0"] if int(row["correctLabelW0orW1"]) == 0 else row["warrant1"])


def load_generated_items(dataset: str, step_type: str) -> List[EntailmentItem]:
    steps = STEP_COUNT[step_type]
    df = pd.read_csv(ROOT / "data" / dataset / "generated" / f"{step_type}.csv")
    items: List[EntailmentItem] = []
    for _, row in df.iterrows():
        items.append(
            EntailmentItem(
                dataset=dataset,
                step_type=step_type,
                premise=row["Premise"],
                claim=row["Claim"],
                implicit=split_steps(row["Helpful"], steps),
            )
        )
    return items


class LazyLogic:
    """Dataset-level sentence cache that loads AMR only when a miss appears."""

    def __init__(self, models: ModelBundle) -> None:
        self.models = models
        self.parser = None
        self.converter = None
        self.caches: Dict[str, LogicCache] = {}

    def get(self, dataset: str, texts: Iterable[str]) -> List[object]:
        text_list = [str(text) for text in texts]
        self.warm(dataset, text_list)
        cache = self._cache_for_dataset(dataset)
        return [cache.sentence_cache[text] for text in text_list]

    def warm(self, dataset: str, texts: Iterable[str]) -> None:
        text_list = [str(text) for text in texts]
        cache = self._cache_for_dataset(dataset)
        missing = [text for text in text_list if text not in cache.sentence_cache]
        if missing:
            self._ensure_amr()
            self._warm_missing(dataset, missing)

    def _cache_for_dataset(self, dataset: str) -> LogicCache:
        cache = self.caches.get(dataset)
        if cache is None:
            cache = LogicCache(CACHE_DIR, f"{dataset}_entailment_fix{FIX_NUMBER}", logic_core, self.parser, self.converter)
            self.caches[dataset] = cache
        return cache

    def _ensure_amr(self) -> None:
        if self.parser is not None and self.converter is not None:
            return
        self.parser, self.converter = self.models.load_amr()
        logic_core.parser = self.parser
        logic_core.converter = self.converter

    def _warm_missing(self, dataset: str, texts: List[str]) -> None:
        cache = self._cache_for_dataset(dataset)
        seen = set()
        missing = []
        for text in texts:
            if text in cache.sentence_cache or text in seen:
                continue
            seen.add(text)
            missing.append(text)

        for start in range(0, len(missing), max(1, LOGIC_BATCH_SIZE)):
            batch = missing[start : start + LOGIC_BATCH_SIZE]
            try:
                for text, logic in zip(batch, cache._generate_sentences(batch)):
                    cache.sentence_cache[text] = logic
                cache._save_pickle(cache.sentence_path, cache.sentence_cache)
            except Exception:
                for text in batch:
                    cache.sentence_cache[text] = cache._generate_sentences([text])[0]
                    cache._save_pickle(cache.sentence_path, cache.sentence_cache)


def texts_for_item(item: EntailmentItem) -> List[str]:
    return [item.premise, item.claim] + item.implicit


def prove_texts(logic_cache: LazyLogic, dataset: str, texts: Iterable[str], tau_m: float, tau_c: int):
    logic = logic_cache.get(dataset, texts)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return logic_core.prove(logic, tau_m, tau_c)


def evaluate_setting(
    logic_cache: LazyLogic,
    items: List[EntailmentItem],
    dataset: str,
    step_type: str,
) -> EntailmentResult:
    tau_m, tau_c = PARAMS[dataset]
    started = time.time()
    ent = 0
    valid = 0
    skipped_false = 0
    exceptions = 0
    seen = 0

    label = f"{dataset}/{step_type} fix{FIX_NUMBER} tm={tau_m:g} tc={tau_c}"
    with tqdm(total=FIX_NUMBER, desc=label, unit="valid", dynamic_ncols=True) as progress:
        for item in items:
            seen += 1
            try:
                result = prove_texts(logic_cache, dataset, texts_for_item(item), tau_m, tau_c)
                if result is False:
                    skipped_false += 1
                    continue
                valid += 1
                ent += int(result[0] == "ent")
                progress.update(1)
                progress.set_postfix(ent=ent, seen=seen, skip=skipped_false, err=exceptions, refresh=False)
                if valid >= FIX_NUMBER:
                    break
            except Exception:
                exceptions += 1
                continue

    accuracy = ent / valid if valid else 0.0
    return EntailmentResult(
        dataset=dataset,
        step_type=step_type,
        tau_m=tau_m,
        tau_c=tau_c,
        fix_number=FIX_NUMBER,
        entailment_predictions=ent,
        valid_items=valid,
        skipped_false=skipped_false,
        exceptions=exceptions,
        seen_items=seen,
        available_items=len(items),
        exhausted=valid < FIX_NUMBER,
        accuracy=accuracy,
        expected=EXPECTED[(dataset, step_type)],
        elapsed_sec=round(time.time() - started, 3),
    )


def write_csv(results: List[EntailmentResult]) -> None:
    fields = [
        "dataset",
        "step_type",
        "tau_m",
        "tau_c",
        "fix_number",
        "entailment_predictions",
        "valid_items",
        "skipped_false",
        "exceptions",
        "seen_items",
        "available_items",
        "exhausted",
        "accuracy",
        "expected",
        "elapsed_sec",
        "delta",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = result.__dict__.copy()
            row["delta"] = result.accuracy - result.expected
            writer.writerow(row)


def write_tex(results: List[EntailmentResult]) -> None:
    lookup = {(result.dataset, result.step_type): result.accuracy for result in results}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabularx}{\\linewidth}{>{\\RaggedRight}p{0.25\\linewidth}>{\\RaggedRight}p{0.35\\linewidth}>{\\RaggedRight}p{0.35\\linewidth}}",
        "\\toprule",
        "\\textbf{Step type} & \\textbf{ANLI dataset} & \\textbf{ARCT dataset} \\\\",
        "\\midrule",
    ]
    for step_type in STEP_TYPES:
        anli = lookup.get(("anli", step_type))
        arct = lookup.get(("arct", step_type))
        anli_text = "---" if anli is None else f"{anli:.3f}"
        arct_text = "---" if arct is None else f"{arct:.3f}"
        lines.append(f"{STEP_LABELS[step_type]} & {anli_text} & {arct_text}\\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\caption{Accuracy for entailment with different options for dealing with implicit premises where none means no intermediate premises were used, original means using the helpful intermediate premise given in the dataset, and 1- (resp. 2- and 3-) step means using the response from prompting the LLM for one (resp. two and three) steps of intermediate premises.}",
            "\\label{tab:ent}",
            "\\end{table}",
            "",
        ]
    )
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    models = ModelBundle()
    logic_cache = LazyLogic(models)
    logic_core.nli_tokenizer = None
    logic_core.model_nli = None
    neural_caches: Dict[str, NeuralCache] = {}
    results: List[EntailmentResult] = []

    try:
        for dataset in DATASETS:
            neural_cache = NeuralCache(CACHE_DIR / "neural_cache" / f"{dataset}.sqlite", models)
            neural_caches[dataset] = neural_cache
            logic_core.score = neural_cache.score
            logic_core.NLI = neural_cache.nli

            for step_type in STEP_TYPES:
                items = load_items(dataset, step_type)
                if step_type in STEP_COUNT:
                    warm_texts: List[str] = []
                    for item in items:
                        warm_texts.extend(texts_for_item(item))
                    logic_cache.warm(dataset, warm_texts)
                results.append(evaluate_setting(logic_cache, items, dataset, step_type))
    finally:
        for neural_cache in neural_caches.values():
            neural_cache.close()

    write_csv(results)
    write_tex(results)

    print(f"\nEntailment fix{FIX_NUMBER} reproduction")
    print("dataset step       actual expected delta   ent/valid seen/avail skip err exhausted")
    for result in results:
        delta = result.accuracy - result.expected
        print(
            f"{result.dataset:<7} {result.step_type:<9} "
            f"{result.accuracy:>6.3f} {result.expected:>8.3f} {delta:>+7.3f} "
            f"{result.entailment_predictions:>4}/{result.valid_items:<4} "
            f"{result.seen_items:>4}/{result.available_items:<4} "
            f"{result.skipped_false:>4} {result.exceptions:>3} {result.exhausted}"
        )


if __name__ == "__main__":
    main()
