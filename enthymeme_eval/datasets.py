import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SHUFFLE_SEED, Setting


DATA_DIR = Path("data")
GENERATED_SUBDIR = os.environ.get("ENTHYMEME_GENERATED_SUBDIR", "generated")
GENERATED_FORMAT = os.environ.get("ENTHYMEME_GENERATED_FORMAT", "csv")
GENERATED_JSON_DIR = Path(os.environ.get("ENTHYMEME_GENERATED_JSON_DIR", "llm_generated_data"))
DATA_SPLITS = [part.strip() for part in os.environ.get("ENTHYMEME_DATA_SPLITS", "test").split(",") if part.strip()]
ALIGN_ORIGINAL_TO_GENERATED = os.environ.get("ENTHYMEME_ALIGN_ORIGINAL_TO_GENERATED", "0") == "1"
DEDUP_PC_MODE = os.environ.get("ENTHYMEME_DEDUP_PC", "auto").lower()


class EvalRow(list):
    """List row with evaluation-only metadata.

    The legacy pipeline expects plain list indexing, so this keeps list
    behavior while letting the runner skip invalid generated chains after the
    1,000-row shuffle.
    """

    def __init__(self, values: List[object], eval_valid: bool = True, validation_error: str = "") -> None:
        super().__init__(values)
        self.eval_valid = eval_valid
        self.validation_error = validation_error


# The ANLI generated one/two/three-step files were produced from the first
# 1,000 ANLI test examples in the notebook. The paper's ANLI/original table
# follows the same source subset, so the default loader applies that cap before
# shuffling. Pass paper_subset=False to evaluate the full ANLI test split.
PAPER_SOURCE_LIMITS: Dict[Tuple[str, str], int] = {
    ("anli", "original"): 1000,
}


def _split_steps(text: str, steps: int) -> List[str]:
    parts = [p.strip() for p in str(text).split(". ") if p.strip()]
    return parts[:steps]


def _split_steps_safe(text: str, steps: int) -> List[str]:
    """Split generated chains while preserving common abbreviations."""

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


def _shuffle(rows: List[list]) -> List[list]:
    rows = list(rows)
    np.random.seed(SHUFFLE_SEED)
    np.random.shuffle(rows)
    return rows


def _dedup_premise_claim(rows: List[list]) -> List[list]:
    """Keep the first row for each exact Premise/Claim pair."""
    deduped = []
    seen = set()
    for row in rows:
        key = (str(row[0]), str(row[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def load_rows(
    root: Path,
    setting: Setting,
    paper_subset: bool = True,
) -> List[list]:
    if setting.dataset == "anli" and setting.variant == "original":
        rows = _load_aligned_original(root, setting.dataset) if ALIGN_ORIGINAL_TO_GENERATED else _load_anli_original(root, DATA_SPLITS)
    elif setting.dataset == "arct" and setting.variant == "original":
        rows = _load_aligned_original(root, setting.dataset) if ALIGN_ORIGINAL_TO_GENERATED else _load_arct_original(root, DATA_SPLITS)
    else:
        rows = _load_generated(root, setting)

    if paper_subset and DATA_SPLITS == ["test"] and (setting.dataset, setting.variant) in PAPER_SOURCE_LIMITS:
        rows = rows[: PAPER_SOURCE_LIMITS[(setting.dataset, setting.variant)]]

    if should_dedup_premise_claim(setting):
        rows = _dedup_premise_claim(rows)

    return _shuffle(rows)


def should_dedup_premise_claim(setting: Setting) -> bool:
    """Decide whether to collapse repeated exact Premise/Claim keys.

    The final ARCT setup keeps generated 1-step top-up rows, so global de-dupe
    would remove useful extra chains.  Only ARCT original needs automatic
    de-dupe because the raw ARCT test split contains repeated premise/claim
    keys with different warrants.
    """

    if DEDUP_PC_MODE in {"1", "true", "yes"}:
        return True
    if DEDUP_PC_MODE in {"0", "false", "no"}:
        return False
    return setting.dataset == "arct" and setting.variant == "original"


def generated_key_order(root: Path, dataset: str, steps: int = 1) -> List[Tuple[str, str]]:
    """Return generated JSON premise/claim keys in experiment order."""

    selected_order = _selected_key_order(root, dataset)
    if selected_order is not None:
        return selected_order

    keys: List[Tuple[str, str]] = []
    seen = set()
    for split in DATA_SPLITS:
        source_path = root / GENERATED_JSON_DIR / dataset / f"{split}_{steps}hops.json"
        data = json.loads(source_path.read_text(encoding="utf-8"))
        for key in data:
            premise, claim = key.split("###", 1)
            pair = (premise, claim)
            if pair in seen:
                continue
            seen.add(pair)
            keys.append(pair)
    return keys


def _selected_key_order(root: Path, dataset: str) -> Optional[List[Tuple[str, str]]]:
    """Use selected_keys.json as the canonical order for selected experiments."""

    if DATA_SPLITS != ["selected"]:
        return None
    path = root / GENERATED_JSON_DIR / dataset / "selected_keys.json"
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


def _original_lookup_splits(dataset: str) -> List[str]:
    splits = list(DATA_SPLITS)
    if "selected" in splits:
        return ["test", "dev", "train"]
    if dataset == "arct" and "train" not in splits:
        # The new ARCT dev generated file contains 195 train keys.
        splits.append("train")
    return splits


def _load_aligned_original(root: Path, dataset: str) -> List[list]:
    if GENERATED_FORMAT != "json":
        raise ValueError("Original alignment currently expects generated JSON data.")

    key_order = generated_key_order(root, dataset, steps=1)
    if dataset == "anli":
        lookup = _anli_original_lookup(root, _original_lookup_splits(dataset))
    elif dataset == "arct":
        lookup = _arct_original_lookup(root, _original_lookup_splits(dataset))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    missing = [pair for pair in key_order if pair not in lookup]
    if missing:
        example = "###".join(missing[0])
        raise ValueError(f"{dataset} original is missing {len(missing)} generated keys; first={example}")
    return [lookup[pair] for pair in key_order]


def _load_anli_original(root: Path, splits: List[str]) -> List[list]:
    anli_dir = root / DATA_DIR / "anli" / "original"
    rows = []
    for split in splits:
        df = pd.read_json(anli_dir / f"{split}.jsonl", lines=True)
        labels = pd.read_csv(anli_dir / f"{split}-labels.lst", header=None)
        for idx, row in df.iterrows():
            rows.append([row["obs1"], row["obs2"], row["hyp1"], row["hyp2"], labels.loc[idx][0]])
    return rows


def _anli_original_lookup(root: Path, splits: List[str]) -> Dict[Tuple[str, str], list]:
    anli_dir = root / DATA_DIR / "anli" / "original"
    lookup: Dict[Tuple[str, str], list] = {}
    for split in splits:
        df = pd.read_json(anli_dir / f"{split}.jsonl", lines=True)
        labels = pd.read_csv(anli_dir / f"{split}-labels.lst", header=None)
        for idx, row in df.iterrows():
            key = (str(row["obs1"]), str(row["obs2"]))
            lookup.setdefault(key, [row["obs1"], row["obs2"], row["hyp1"], row["hyp2"], labels.loc[idx][0]])
    return lookup


def _load_arct_original(root: Path, splits: List[str]) -> List[list]:
    rows = []
    for split in splits:
        df = pd.read_csv(root / DATA_DIR / "arct" / "original" / f"{split}.tsv", sep="\t")
        for _, row in df.iterrows():
            rows.append(
                [
                    row["reason"],
                    row["claim"],
                    row["warrant0"],
                    row["warrant1"],
                    int(row["correctLabelW0orW1"]) + 1,
                ]
            )
    return rows


def _arct_original_lookup(root: Path, splits: List[str]) -> Dict[Tuple[str, str], list]:
    lookup: Dict[Tuple[str, str], list] = {}
    for split in splits:
        df = pd.read_csv(root / DATA_DIR / "arct" / "original" / f"{split}.tsv", sep="\t")
        for _, row in df.iterrows():
            key = (str(row["reason"]), str(row["claim"]))
            lookup.setdefault(
                key,
                [
                    row["reason"],
                    row["claim"],
                    row["warrant0"],
                    row["warrant1"],
                    int(row["correctLabelW0orW1"]) + 1,
                ],
            )
    return lookup


def _load_generated(root: Path, setting: Setting) -> List[list]:
    if GENERATED_FORMAT == "json":
        return _load_generated_json(root, setting)

    if setting.source is None:
        raise ValueError(f"Generated setting {setting.key} has no source CSV")
    df = pd.read_csv(_resolve_generated_csv_path(root, setting.dataset, setting.source))
    rows = []
    for _, row in df.iterrows():
        if setting.steps == 1:
            helpful = _split_steps(row["Helpful"], 1)
            non_helpful = _split_steps(row["Non-Helpful"], 1)
            rows.append([row["Premise"], row["Claim"], helpful[0], non_helpful[0], 1])
        else:
            rows.append(
                [row["Premise"], row["Claim"]]
                + _split_steps(row["Helpful"], setting.steps)
                + _split_steps(row["Non-Helpful"], setting.steps)
                + [1]
            )
    return rows


def _resolve_generated_csv_path(root: Path, dataset: str, source: str) -> Path:
    """Resolve generated CSV names, including final unique-key subset files."""

    generated_dir = root / DATA_DIR / dataset / GENERATED_SUBDIR
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


def _load_generated_json(root: Path, setting: Setting) -> List[list]:
    rows = []
    for split in DATA_SPLITS:
        source_path = root / GENERATED_JSON_DIR / setting.dataset / f"{split}_{setting.steps}hops.json"
        data = json.loads(source_path.read_text(encoding="utf-8"))
        selected_order = _selected_key_order(root, setting.dataset)
        if selected_order is None:
            key_order = list(data)
        else:
            key_order = [f"{premise}###{claim}" for premise, claim in selected_order if f"{premise}###{claim}" in data]

        for key in key_order:
            value = data[key]
            premise, claim = key.split("###", 1)
            helpful = _split_steps_safe(value["helpful"], setting.steps)
            non_helpful = _split_steps_safe(value["non_helpful"], setting.steps)
            valid = bool(value.get("valid", True))
            validation_error = str(value.get("validation_error", ""))
            if setting.steps == 1:
                row = [premise, claim, helpful[0], non_helpful[0], 1]
            else:
                row = [premise, claim] + helpful + non_helpful + [1]
            rows.append(EvalRow(row, eval_valid=valid, validation_error=validation_error))
    return rows
