"""임베딩 후보의 설정을 읽고 질의·문서 벡터를 생성한다.

모델은 처음 encode를 호출할 때 로드하며, 후보별 길이 상한과 질의 접두어를 적용한다.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from rag import settings as C


class Embedder:
    def __init__(self, model_key: str = C.DEFAULT_EMBEDDING, device: Optional[str] = None):
        if model_key not in C.EMBEDDING_CANDIDATES:
            raise KeyError(f"unknown embedding model: {model_key}")
        self.spec = C.EMBEDDING_CANDIDATES[model_key]
        self.model_key = model_key
        self._device = device
        self._model = None

    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # 캐시가 있으면 HF 허브 확인 요청 없이 로드(허브 응답이 느리면 로드마다 재시도로 수십 초 지연),
            # 캐시에 없을 때만 내려받는다.
            kw = {}
            if getattr(self.spec, "torch_dtype", ""):
                import torch
                kw["model_kwargs"] = {"torch_dtype": getattr(torch, self.spec.torch_dtype)}
            try:
                self._model = SentenceTransformer(self.spec.hf_id, device=self._device, local_files_only=True, **kw)
            except (OSError, ValueError):
                self._model = SentenceTransformer(self.spec.hf_id, device=self._device, **kw)
            self._model.max_seq_length = self.spec.max_seq_length
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        batch_size: int = C.EMBED_BATCH_SIZE,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            payload,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=C.EMBED_NORMALIZE,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    def fingerprint(self) -> Dict[str, str]:
        """재현성 기록용 지문."""
        return {
            "model_key": self.model_key,
            "hf_id": self.spec.hf_id,
            "dim": str(self.spec.dim),
            "max_seq_length": str(self.spec.max_seq_length),
            "normalize": str(C.EMBED_NORMALIZE),
            "query_prefix": self.spec.query_prefix,
            "passage_prefix": self.spec.passage_prefix,
        }


def text_hash(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]
