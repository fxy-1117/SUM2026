"""Lazy loaders for AMR, sentence similarity, and NLI models."""

from typing import Any, Tuple

import torch


class ModelBundle:
    """Owns heavyweight model instances so each process loads them once."""

    def __init__(self) -> None:
        self.parser = None
        self.converter = None
        self.sentence_model = None
        self.nli_tokenizer = None
        self.nli_model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load_amr(self) -> Tuple[Any, Any]:
        if self.parser is None or self.converter is None:
            self._patch_torch_hub_for_cached_fairseq()

            from amr_logic_converter import AmrLogicConverter
            from transition_amr_parser.parse import AMRParser

            self.parser = AMRParser.from_pretrained("AMR3-joint-ontowiki-seed42")
            self.converter = AmrLogicConverter(
                existentially_quantify_instances=False,
                invert_relations=True,
            )
        return self.parser, self.converter

    @staticmethod
    def _patch_torch_hub_for_cached_fairseq() -> None:
        if getattr(torch.hub.load, "_enthymeme_eval_offline_patch", False):
            return

        original_load = torch.hub.load

        def load(repo_or_dir: Any, *args: Any, **kwargs: Any) -> Any:
            if repo_or_dir == "pytorch/fairseq":
                repo_or_dir = "pytorch/fairseq:main"
            return original_load(repo_or_dir, *args, **kwargs)

        load._enthymeme_eval_offline_patch = True  # type: ignore[attr-defined]
        torch.hub.load = load

    def load_sentence_model(self) -> Any:
        if self.sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self.sentence_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self.sentence_model

    def load_nli(self) -> Tuple[Any, Any]:
        if self.nli_tokenizer is None or self.nli_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        return self.nli_tokenizer, self.nli_model

    def compute_similarity(self, s1: str, s2: str) -> float:
        from sentence_transformers import util

        model = self.load_sentence_model()
        embedding_1 = model.encode(s1, convert_to_tensor=True, show_progress_bar=False)
        embedding_2 = model.encode(s2, convert_to_tensor=True, show_progress_bar=False)
        return float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0].detach().cpu())

    def compute_nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        tokenizer, model = self.load_nli()
        inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        scores = {name: round(float(prob) * 100, 1) for prob, name in zip(prediction, label_names)}
        label = max(scores, key=scores.get)
        return label, scores[label]
