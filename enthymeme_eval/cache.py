"""Persistent caches for AMR logic and neural model calls."""

import contextlib
import hashlib
import io
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm.auto import tqdm


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class LogicCache:
    """Cache AMR-to-logic outputs by exact sentence text."""

    def __init__(self, cache_dir: Path, setting_key: str, core: Any, parser: Any, converter: Any) -> None:
        self.cache_dir = cache_dir
        self.setting_key = setting_key
        self.dataset = setting_key.split("_", 1)[0]
        self.core = core
        self.core.parser = parser
        self.core.converter = converter

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sentence_cache_dir = self.cache_dir / "logic_sentences"
        self.sentence_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sentence_path = self.sentence_cache_dir / f"{self.dataset}.pkl"
        self.sentence_cache = self._load_pickle(self.sentence_path)

    @staticmethod
    def _load_pickle(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("rb") as fp:
            return pickle.load(fp)

    @staticmethod
    def _save_pickle(path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        for attempt in range(20):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)

    def get_row(self, texts: Iterable[str]) -> List[Any]:
        """Return cached logic for each sentence in a row."""

        text_list = [str(text) for text in texts]
        missing_texts = [text for text in text_list if text not in self.sentence_cache]
        if missing_texts:
            self.warm_sentences(missing_texts, batch_size=len(missing_texts))
        return [self.sentence_cache[text] for text in text_list]

    def prepare_rows(self, rows: Iterable[Iterable[Any]], batch_size: int) -> None:
        """Warm sentence cache for this setting."""

        all_texts: List[str] = []
        for row in rows:
            row_list = list(row)
            all_texts.extend(str(text) for text in row_list[:-1])
        self.warm_sentences(all_texts, batch_size=batch_size)

    def warm_sentences(self, texts: Iterable[str], batch_size: int) -> int:
        """Generate and persist missing sentence logic in batches."""
        missing_texts: List[str] = []
        seen = set()
        for text in texts:
            text = str(text)
            if text in self.sentence_cache or text in seen:
                continue
            seen.add(text)
            missing_texts.append(text)

        if not missing_texts:
            return 0

        safe_batch_size = max(1, batch_size)
        with tqdm(
            total=len(missing_texts),
            desc=f"logic warmup {self.setting_key}",
            unit="sent",
            dynamic_ncols=True,
            leave=False,
        ) as progress:
            for start in range(0, len(missing_texts), safe_batch_size):
                batch = missing_texts[start : start + safe_batch_size]
                for text, logic in zip(batch, self._generate_sentences(batch)):
                    self.sentence_cache[text] = logic
                self._save_pickle(self.sentence_path, self.sentence_cache)
                progress.update(len(batch))
        return len(missing_texts)

    def _generate_sentences(self, texts: List[str]) -> List[Any]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_logic = self.core.generate_logic(texts)[-2]
        return [self.core.transform_logic(x) for x in raw_logic]


class NeuralCache:
    """SQLite cache for deterministic neural similarity and NLI calls."""

    def __init__(self, db_path: Path, models: Any) -> None:
        self.db_path = db_path
        self.models = models
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS similarity (
                key TEXT PRIMARY KEY,
                s1 TEXT NOT NULL,
                s2 TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nli (
                key TEXT PRIMARY KEY,
                premise TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def score(self, s1: str, s2: str) -> float:
        key = stable_hash("score", s1, s2)
        row = self.conn.execute("SELECT score FROM similarity WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return float(row[0])

        score = float(self.models.compute_similarity(s1, s2))
        self.conn.execute(
            "INSERT OR REPLACE INTO similarity(key, s1, s2, score) VALUES (?, ?, ?, ?)",
            (key, s1, s2, score),
        )
        self.conn.commit()
        return score

    def nli(self, premise: str, hypothesis: str, *_args: Any) -> Tuple[str, float]:
        key = stable_hash("nli", premise, hypothesis)
        row = self.conn.execute("SELECT label, confidence FROM nli WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return str(row[0]), float(row[1])

        label, confidence = self.models.compute_nli(premise, hypothesis)
        self.conn.execute(
            "INSERT OR REPLACE INTO nli(key, premise, hypothesis, label, confidence) VALUES (?, ?, ?, ?, ?)",
            (key, premise, hypothesis, label, float(confidence)),
        )
        self.conn.commit()
        return label, float(confidence)
