from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol


DEFAULT_RERANK_TOP_K = 8
DEFAULT_MAX_LENGTH = 512


class RerankerError(ValueError):
    """Raised when reranker inputs or outputs cannot be used safely."""


class RerankBackend(Protocol):
    model_name: str
    normalized_scores: bool

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Return one relevance score for every passage, in the same order."""


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


class LocalCrossEncoderReranker:
    """Load a reranker from a local Hugging Face directory only.

    Unlike the embedding model, a cross-encoder reads the query and passage
    together and emits a direct relevance logit. Network model identifiers are
    intentionally rejected so production runs cannot download a model silently.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize: bool = True,
        use_fp16: bool = True,
    ) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise RerankerError(
                "Reranker model_path must be an existing local directory: "
                f"{path}"
            )
        if batch_size <= 0:
            raise RerankerError("batch_size must be greater than zero")
        if max_length <= 0:
            raise RerankerError("max_length must be greater than zero")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RerankerError(
                "Local BGE reranking requires torch and transformers."
            ) from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RerankerError("CUDA was requested but is not available")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        )
        self._device = resolved_device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize
        self._model.to(resolved_device)
        if use_fp16 and resolved_device.startswith("cuda"):
            self._model.half()
        self._model.eval()

        self.model_name = path.name
        self.normalized_scores = normalize

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query = _clean(query)
        if not query:
            raise RerankerError("query must not be empty")
        if not passages:
            return []

        cleaned_passages = [_clean(passage) for passage in passages]
        if any(not passage for passage in cleaned_passages):
            raise RerankerError("passages must not contain empty text")

        scores: list[float] = []
        for start in range(0, len(cleaned_passages), self._batch_size):
            batch = cleaned_passages[start : start + self._batch_size]
            pairs = [[query, passage] for passage in batch]
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_length,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded, return_dict=True).logits.reshape(-1)
                logits = logits.float()
                if self._normalize:
                    logits = self._torch.sigmoid(logits)
            batch_scores = logits.detach().cpu().tolist()
            if len(batch_scores) != len(batch):
                raise RerankerError(
                    "Reranker model must return exactly one score per query-passage pair"
                )
            scores.extend(float(score) for score in batch_scores)
        return scores


def rerank_evidence(
    query: str,
    evidence: Sequence[Mapping[str, object]],
    *,
    backend: RerankBackend,
    top_k: int | None = DEFAULT_RERANK_TOP_K,
    min_score: float | None = None,
) -> dict[str, object]:
    """Score Hybrid evidence and return a new, auditable ranking.

    The input mappings are never mutated. Hybrid `rank` remains the retrieval
    rank; `retrieval_rank`, `rerank_rank`, and `rerank_score` make the two stages
    explicit.
    """

    query = _clean(query)
    if not query:
        raise RerankerError("query must not be empty")
    if top_k is not None and top_k <= 0:
        raise RerankerError("top_k must be greater than zero or None")
    if min_score is not None and not math.isfinite(float(min_score)):
        raise RerankerError("min_score must be finite")

    prepared: list[dict[str, object]] = []
    passages: list[str] = []
    seen_ids: set[str] = set()
    for position, source_item in enumerate(evidence, start=1):
        item = copy.deepcopy(dict(source_item))
        evidence_id = _clean(item.get("evidence_id"))
        content = _clean(item.get("content"))
        if not evidence_id or not content:
            raise RerankerError("Every evidence item requires evidence_id and content")
        if evidence_id in seen_ids:
            raise RerankerError(f"Duplicate evidence_id before reranking: {evidence_id}")
        seen_ids.add(evidence_id)

        rank_value = item.get("rank", position)
        try:
            retrieval_rank = int(rank_value)
        except (TypeError, ValueError) as exc:
            raise RerankerError("Evidence rank must be an integer") from exc
        if retrieval_rank <= 0:
            raise RerankerError("Evidence rank must be greater than zero")

        item["retrieval_rank"] = retrieval_rank
        prepared.append(item)
        passages.append(content)

    raw_scores = list(backend.score(query, passages)) if passages else []
    if len(raw_scores) != len(prepared):
        raise RerankerError(
            "Reranker backend returned a different number of scores than passages"
        )

    for item, score_value in zip(prepared, raw_scores, strict=True):
        try:
            score = float(score_value)
        except (TypeError, ValueError) as exc:
            raise RerankerError("Reranker scores must be numeric") from exc
        if not math.isfinite(score):
            raise RerankerError("Reranker scores must be finite")
        item["rerank_score"] = score
        item["reranker_model"] = backend.model_name

    prepared.sort(
        key=lambda item: (
            -float(item["rerank_score"]),
            int(item["retrieval_rank"]),
            str(item["evidence_id"]),
        )
    )

    eligible = [
        item
        for item in prepared
        if min_score is None or float(item["rerank_score"]) >= float(min_score)
    ]
    selected = eligible if top_k is None else eligible[:top_k]
    selected_ids = {str(item["evidence_id"]) for item in selected}
    for rerank_rank, item in enumerate(selected, start=1):
        item["rerank_rank"] = rerank_rank
        item["rerank_score"] = round(float(item["rerank_score"]), 8)

    excluded: list[dict[str, object]] = []
    for item in prepared:
        evidence_id = str(item["evidence_id"])
        if evidence_id in selected_ids:
            continue
        reason = (
            "below_min_score"
            if min_score is not None
            and float(item["rerank_score"]) < float(min_score)
            else "top_k"
        )
        excluded.append(
            {
                "evidence_id": evidence_id,
                "retrieval_rank": item["retrieval_rank"],
                "rerank_score": round(float(item["rerank_score"]), 8),
                "reason": reason,
            }
        )

    normalized = bool(getattr(backend, "normalized_scores", False))
    return {
        "query": query,
        "reranker": {
            "model": backend.model_name,
            "score_scale": "sigmoid_0_1" if normalized else "raw_logit",
            "top_k": top_k,
            "min_score": min_score,
        },
        "stats": {
            "candidate_evidence": len(prepared),
            "below_min_score": len(prepared) - len(eligible),
            "excluded_by_top_k": max(0, len(eligible) - len(selected)),
            "returned_evidence": len(selected),
        },
        "evidence": selected,
        "excluded_evidence": excluded,
    }

