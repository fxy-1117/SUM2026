"""Evaluate the final entailment-only table.

The final protocol uses seed 1129, shuffles each fixed data pool, and keeps
evaluating rows until FIX_NUMBER=250 valid entailment-side examples are
collected.  For this entailment-only table, ARCT settings are deduplicated by
exact Premise/Claim pairs.  Logic and neural caches are shared with the
parameter sweep through dataset-specific cache files.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
OUTPUT_CSV = RESULTS_DIR / os.environ.get("ENT_FIX400_OUTPUT", "entailment.csv")
OUTPUT_TEX = RESULTS_DIR / os.environ.get("ENT_FIX400_TEX_OUTPUT", "entailment.tex")
FAILED_LOGIC_PATH = RESULTS_DIR / "entailment_fix400_failed_logic.txt"

FIX_NUMBER = int(os.environ.get("ENT_FIX400_FIX_NUMBER", "250"))
LOGIC_BATCH_SIZE = int(os.environ.get("ENT_FIX400_LOGIC_BATCH", "8"))
GENERATED_FORMAT = os.environ.get("ENT_FIX400_GENERATED_FORMAT", "csv")
GENERATED_SUBDIR = os.environ.get("ENT_FIX400_GENERATED_SUBDIR", "generated")
GENERATED_JSON_DIR = Path(os.environ.get("ENT_FIX400_GENERATED_JSON_DIR", "llm_generated_data"))
DATA_SPLITS = [part.strip() for part in os.environ.get("ENT_FIX400_DATA_SPLITS", "test").split(",") if part.strip()]
ALIGN_ORIGINAL_TO_GENERATED = os.environ.get("ENT_FIX400_ALIGN_ORIGINAL_TO_GENERATED", "0") == "1"
ROW_COMPAT = os.environ.get("ENT_FIX400_ROW_COMPAT", "0") == "1"
ARCT_ORIGINAL_MODE = os.environ.get("ENT_FIX400_ARCT_ORIGINAL_MODE", "correct")
ANLI_SOURCE_LIMIT = int(os.environ.get("ENT_FIX400_ANLI_SOURCE_LIMIT", "1000"))
DEDUP_PC_MODE = os.environ.get("ENT_FIX400_DEDUP_PC", "auto").lower()

PARAMS = {
    "anli": (0.55, 90),
    "arct": (0.65, 90),
}

for _dataset, _params in list(PARAMS.items()):
    _prefix = f"ENT_FIX400_{_dataset.upper()}"
    PARAMS[_dataset] = (
        float(os.environ.get(f"{_prefix}_TAU_M", _params[0])),
        int(os.environ.get(f"{_prefix}_TAU_C", _params[1])),
    )

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

GENERATED_FILES = {
    ("anli", "one"): "one.csv",
    ("anli", "two"): "two.csv",
    ("anli", "three"): "three.csv",
    ("arct", "one"): "one.csv",
    ("arct", "two"): "two.csv",
    ("arct", "three"): "three.csv",
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
class Fix400Result:
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


def env_list(name: str, default: List[str]) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def split_steps(text: str, steps: Optional[int] = None) -> List[str]:
    if steps is None:
        parts = [part.strip() for part in str(text).split(". ") if part.strip()]
    else:
        parts = [part.strip() for part in str(text).split(". ")[:steps] if part.strip()]
    if not parts:
        parts = [str(text).strip()]
    return parts


def split_steps_safe(text: str, steps: int) -> List[str]:
    protected = {
        "U.S.": "U<DOT>S<DOT>",
        "C.I.A.": "C<DOT>I<DOT>A<DOT>",
        "Mr.": "Mr<DOT>",
        "Mrs.": "Mrs<DOT>",
        "Ms.": "Ms<DOT>",
        "Dr.": "Dr<DOT>",
        "Prof.": "Prof<DOT>",
        "e.g.": "e<DOT>g<DOT>",
        "i.e.": "i<DOT>e<DOT>",
    }
    restored = {value: key for key, value in protected.items()}
    safe_text = str(text)
    for source, target in protected.items():
        safe_text = safe_text.replace(source, target)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", safe_text) if part.strip()]
    output = []
    for part in parts[:steps]:
        for source, target in restored.items():
            part = part.replace(source, target)
        output.append(part)
    while output and len(output) < steps:
        output.append(output[-1])
    if not output:
        output = [str(text).strip()] * steps
    return output


def seeded_shuffle(items: List[EntailmentItem]) -> List[EntailmentItem]:
    """Shuffle full rows with the fixed project seed."""

    shuffled = list(items)
    np.random.seed(SHUFFLE_SEED)
    np.random.shuffle(shuffled)
    return shuffled


def dedup_premise_claim(items: List[EntailmentItem]) -> List[EntailmentItem]:
    """Keep the first item for each exact Premise/Claim pair."""

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
        items = load_original_entailment_side(dataset, step_type)
    else:
        items = load_generated_helpful_side(dataset, step_type)
    if should_dedup_premise_claim(dataset, step_type):
        items = dedup_premise_claim(items)
    return seeded_shuffle(items)


def should_dedup_premise_claim(dataset: str, step_type: str) -> bool:
    if DEDUP_PC_MODE in {"1", "true", "yes"}:
        return True
    if DEDUP_PC_MODE in {"0", "false", "no"}:
        return False
    return dataset == "arct"


def load_original_entailment_side(dataset: str, step_type: str) -> List[EntailmentItem]:
    if ALIGN_ORIGINAL_TO_GENERATED:
        return load_aligned_original_entailment_side(dataset, step_type)

    items: List[EntailmentItem] = []
    if dataset == "anli":
        anli_dir = ROOT / "data" / "anli" / "original"
        for split in DATA_SPLITS:
            df = pd.read_json(anli_dir / f"{split}.jsonl", lines=True)
            labels = pd.read_csv(anli_dir / f"{split}-labels.lst", header=None)
            if split == "test" and ANLI_SOURCE_LIMIT > 0:
                df = df.iloc[:ANLI_SOURCE_LIMIT]
                labels = labels.iloc[:ANLI_SOURCE_LIMIT]
            for idx, row in df.iterrows():
                correct_hypothesis = row["hyp1"] if int(labels.iloc[idx][0]) == 1 else row["hyp2"]
                implicit = [] if step_type == "none" else [correct_hypothesis]
                items.append(EntailmentItem(dataset, step_type, row["obs1"], row["obs2"], implicit))
        return items

    for split in DATA_SPLITS:
        df = pd.read_csv(ROOT / "data" / "arct" / "original" / f"{split}.tsv", sep="\t")
        for _, row in df.iterrows():
            implicit = [] if step_type == "none" else [arct_original_implicit(row)]
            items.append(EntailmentItem(dataset, step_type, row["reason"], row["claim"], implicit))
    return items


def load_generated_helpful_side(dataset: str, step_type: str) -> List[EntailmentItem]:
    if GENERATED_FORMAT == "json":
        return load_generated_json_helpful_side(dataset, step_type)

    source = GENERATED_FILES[(dataset, step_type)]
    df = pd.read_csv(resolve_generated_csv_path(dataset, source))
    steps = STEP_COUNT[step_type]
    items = []
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


def resolve_generated_csv_path(dataset: str, source: str) -> Path:
    """Resolve generated CSV names, including final unique-key subset files."""

    generated_dir = ROOT / "data" / dataset / GENERATED_SUBDIR
    candidates = [
        generated_dir / source,
        generated_dir / source.replace(".csv", "_unique.csv"),
        generated_dir / source.replace("_ds_444.csv", "_ds_289_unique.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"Could not find generated CSV for {dataset}/{source}; tried {tried}")


def generated_key_order(dataset: str, steps: int = 1) -> List[Tuple[str, str]]:
    selected_order = selected_key_order(dataset)
    if selected_order is not None:
        return selected_order

    keys: List[Tuple[str, str]] = []
    seen = set()
    for split in DATA_SPLITS:
        source_path = ROOT / GENERATED_JSON_DIR / dataset / f"{split}_{steps}hops.json"
        data = json.loads(source_path.read_text(encoding="utf-8"))
        for key in data:
            premise, claim = key.split("###", 1)
            pair = (premise, claim)
            if pair in seen:
                continue
            seen.add(pair)
            keys.append(pair)
    return keys


def selected_key_order(dataset: str) -> Optional[List[Tuple[str, str]]]:
    if DATA_SPLITS != ["selected"]:
        return None
    path = ROOT / GENERATED_JSON_DIR / dataset / "selected_keys.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    keys: List[Tuple[str, str]] = []
    seen = set()
    for row in rows:
        pair = (str(row["premise"]), str(row["claim"]))
        if pair in seen:
            continue
        seen.add(pair)
        keys.append(pair)
    return keys


def original_lookup_splits(dataset: str) -> List[str]:
    splits = list(DATA_SPLITS)
    if "selected" in splits:
        return ["test", "dev", "train"]
    if dataset == "arct" and "train" not in splits:
        splits.append("train")
    return splits


def load_aligned_original_entailment_side(dataset: str, step_type: str) -> List[EntailmentItem]:
    key_order = generated_key_order(dataset, steps=1)
    if dataset == "anli":
        lookup = anli_original_entailment_lookup()
    elif dataset == "arct":
        lookup = arct_original_entailment_lookup()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    missing = [pair for pair in key_order if pair not in lookup]
    if missing:
        example = "###".join(missing[0])
        raise ValueError(f"{dataset} original is missing {len(missing)} generated keys; first={example}")
    items = []
    for premise, claim in key_order:
        implicit_text = lookup[(premise, claim)]
        implicit = [] if step_type == "none" else [implicit_text]
        items.append(EntailmentItem(dataset, step_type, premise, claim, implicit))
    return items


def anli_original_entailment_lookup() -> Dict[Tuple[str, str], str]:
    anli_dir = ROOT / "data" / "anli" / "original"
    lookup: Dict[Tuple[str, str], str] = {}
    for split in original_lookup_splits("anli"):
        df = pd.read_json(anli_dir / f"{split}.jsonl", lines=True)
        labels = pd.read_csv(anli_dir / f"{split}-labels.lst", header=None)
        for idx, row in df.iterrows():
            key = (str(row["obs1"]), str(row["obs2"]))
            correct_hypothesis = row["hyp1"] if int(labels.iloc[idx][0]) == 1 else row["hyp2"]
            lookup.setdefault(key, str(correct_hypothesis))
    return lookup


def arct_original_implicit(row) -> str:
    if ARCT_ORIGINAL_MODE == "correct":
        return str(row["warrant0"] if int(row["correctLabelW0orW1"]) == 0 else row["warrant1"])
    return str(row["warrant0"])


def arct_original_entailment_lookup() -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for split in original_lookup_splits("arct"):
        df = pd.read_csv(ROOT / "data" / "arct" / "original" / f"{split}.tsv", sep="\t")
        for _, row in df.iterrows():
            key = (str(row["reason"]), str(row["claim"]))
            lookup.setdefault(key, arct_original_implicit(row))
    return lookup


def load_generated_json_helpful_side(dataset: str, step_type: str) -> List[EntailmentItem]:
    steps = STEP_COUNT[step_type]
    items: List[EntailmentItem] = []
    for split in DATA_SPLITS:
        source_path = ROOT / GENERATED_JSON_DIR / dataset / f"{split}_{steps}hops.json"
        data = json.loads(source_path.read_text(encoding="utf-8"))
        selected_order = selected_key_order(dataset)
        if selected_order is None:
            key_order = list(data)
        else:
            key_order = [f"{premise}###{claim}" for premise, claim in selected_order if f"{premise}###{claim}" in data]
        for key in key_order:
            value = data[key]
            if not value.get("valid", True):
                continue
            premise, claim = key.split("###", 1)
            items.append(
                EntailmentItem(
                    dataset=dataset,
                    step_type=step_type,
                    premise=premise,
                    claim=claim,
                    implicit=split_steps_safe(value["helpful"], steps),
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
        self.failed: List[str] = []

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
            cache = LogicCache(
                CACHE_DIR,
                f"{dataset}_entailment_fix{FIX_NUMBER}",
                logic_core,
                self.parser,
                self.converter,
            )
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
                    try:
                        cache.sentence_cache[text] = cache._generate_sentences([text])[0]
                        cache._save_pickle(cache.sentence_path, cache.sentence_cache)
                    except Exception:
                        self.failed.append(text)
                        raise

    def save_failures(self) -> None:
        if self.failed:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            FAILED_LOGIC_PATH.write_text("\n".join(self.failed) + "\n", encoding="utf-8")


def texts_for_item(item: EntailmentItem) -> List[str]:
    return [item.premise, item.claim] + item.implicit


def row_cache_key(item: EntailmentItem) -> str:
    return item.premise + item.claim


def prove_texts(logic_cache: LazyLogic, dataset: str, texts: Iterable[str], tau_m: float, tau_c: int):
    logic = logic_cache.get(dataset, texts)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return logic_core.prove(logic, tau_m, tau_c)


def evaluate_setting(
    logic_cache: LazyLogic,
    items: List[EntailmentItem],
    dataset: str,
    step_type: str,
) -> Fix400Result:
    tau_m, tau_c = PARAMS[dataset]
    started = time.time()
    ent = 0
    valid = 0
    skipped_false = 0
    exceptions = 0
    seen = 0
    row_logic_texts: dict[str, List[str]] = {}

    label = f"{dataset}/{step_type} fix{FIX_NUMBER} tm={tau_m:g} tc={tau_c}"
    with tqdm(total=FIX_NUMBER, desc=label, unit="valid", dynamic_ncols=True) as progress:
        for item in items:
            seen += 1
            try:
                if ROW_COMPAT:
                    logic_texts = row_logic_texts.setdefault(row_cache_key(item), texts_for_item(item))
                else:
                    logic_texts = texts_for_item(item)
                result = prove_texts(logic_cache, dataset, logic_texts, tau_m, tau_c)
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
    return Fix400Result(
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


def write_header_if_needed(path: Path, fields: List[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    append = os.environ.get("ENT_FIX400_APPEND", "0") == "1" and path.exists()
    fp = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=fields)
    if not append:
        writer.writeheader()
    writer._output_file = fp  # type: ignore[attr-defined]
    return writer


def write_result(writer: csv.DictWriter, result: Fix400Result) -> None:
    row = result.__dict__.copy()
    row["delta"] = result.accuracy - result.expected
    writer.writerow(row)
    getattr(writer, "_output_file").flush()


def close_writer(writer: csv.DictWriter) -> None:
    getattr(writer, "_output_file").close()


def write_tex_from_csv() -> None:
    """Write the final entailment table from the current CSV results."""

    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        return
    df = df.drop_duplicates(["dataset", "step_type"], keep="last")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabularx}{\\linewidth}{>{\\RaggedRight}p{0.25\\linewidth}>{\\RaggedRight}p{0.35\\linewidth}>{\\RaggedRight}p{0.35\\linewidth}}",
        "\\toprule",
        "\\textbf{Step type} & \\textbf{ANLI dataset} & \\textbf{ARCT dataset} \\\\",
        "\\midrule",
    ]
    for step_type in ("none", "original", "one", "two", "three"):
        values = {}
        for dataset in ("anli", "arct"):
            match = df[(df["dataset"] == dataset) & (df["step_type"] == step_type)]
            values[dataset] = "---" if match.empty else f"{float(match.iloc[-1]['accuracy']):.3f}"
        lines.append(f"{STEP_LABELS[step_type]} & {values['anli']} & {values['arct']}\\\\")
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
    datasets = env_list("ENT_FIX400_DATASETS", ["anli", "arct"])
    steps = env_list("ENT_FIX400_STEPS", ["none", "original", "one", "two", "three"])

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

    models = ModelBundle()
    logic_cache = LazyLogic(models)
    logic_core.nli_tokenizer = None
    logic_core.model_nli = None
    neural_caches: Dict[str, NeuralCache] = {}

    def use_neural_cache(dataset: str) -> None:
        neural_cache = neural_caches.get(dataset)
        if neural_cache is None:
            neural_cache = NeuralCache(CACHE_DIR / "neural_cache" / f"{dataset}.sqlite", models)
            neural_caches[dataset] = neural_cache
        logic_core.score = neural_cache.score
        logic_core.NLI = neural_cache.nli

    writer = write_header_if_needed(OUTPUT_CSV, fields)
    results: List[Fix400Result] = []
    try:
        for dataset in datasets:
            use_neural_cache(dataset)
            for step_type in steps:
                items = load_items(dataset, step_type)
                if step_type in STEP_COUNT:
                    warm_texts = []
                    for item in items:
                        warm_texts.extend(texts_for_item(item))
                    logic_cache.warm(dataset, warm_texts)
                result = evaluate_setting(logic_cache, items, dataset, step_type)
                results.append(result)
                write_result(writer, result)
    finally:
        close_writer(writer)
        for neural_cache in neural_caches.values():
            neural_cache.close()
        logic_cache.save_failures()
    write_tex_from_csv()

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
