"""검색기 — 인덱스 구성(scope) × 검색 방식(dense/bm25/hybrid) × 메타데이터 필터.

물리 인덱스는 `manual_ko`(사용설명서 + 자가수리 가이드) 하나와 스모크용 `pilot`이다.
"사용설명서 only / 가이드 only / 둘 다" 구성은 쿼리타임 `MetadataFilter(doc_types=...)`
(= `index.preset_filter`)로 만든다.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rag import settings as C
from rag.embeddings import Embedder
from rag.indexes import BM25Index, MetadataFilter, VectorIndex, mmr_rerank, reciprocal_rank_fusion, weighted_fusion, preset_filter
from rag.text import normalize_sparse, sparse_tokens
from rag.files import read_jsonl


@dataclass
class RetrievalHit:
    rank: int
    chunk_id: str
    score: float
    text: str
    meta: Dict[str, Any]
    retriever: str = "dense"

    def citation(self) -> Dict[str, Any]:
        m = self.meta
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "doc_type": m.get("doc_type"),
            "category": m.get("category"),
            "models": m.get("models"),
            "source_file": m.get("source_file"),
            "page_number": m.get("page_number"),
            "section_title": m.get("section_title"),
            "os_guide": m.get("os_guide"),
            "rev": m.get("rev"),
        }


class Retriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        vector_index: VectorIndex,
        bm25_index: Optional[BM25Index],
        embedder: Embedder,
        cfg: Optional[C.RetrievalConfig] = None,
    ):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.vi = vector_index
        self.bm = bm25_index
        self.embedder = embedder
        self.cfg = cfg or C.RetrievalConfig()
        self.row_of = {cid: i for i, cid in enumerate(vector_index.chunk_ids)}

    # ------------------------------------------------------------------
    def _allowed_rows(self, flt: Optional[MetadataFilter]) -> Optional[np.ndarray]:
        if flt is None:
            return None
        rows = [i for i, cid in enumerate(self.vi.chunk_ids) if flt.match(self.by_id[cid])]
        return np.asarray(rows, dtype=np.int64)

    def search(
        self,
        query: str,
        *,
        method: str = "dense",           # "dense" | "bm25" | "hybrid"
        top_k: Optional[int] = None,
        flt: Optional[MetadataFilter] = None,
        use_mmr: Optional[bool] = None,
    ) -> List[RetrievalHit]:
        top_k = top_k or self.cfg.top_k
        use_mmr = self.cfg.use_mmr if use_mmr is None else use_mmr
        allowed = self._allowed_rows(flt)
        cand_k = max(self.cfg.candidate_k, top_k)

        dense: List[Tuple[str, float]] = []
        sparse: List[Tuple[str, float]] = []
        if method in ("dense", "hybrid"):
            qv = self.embedder.encode([query], is_query=True)[0]
            dense = self.vi.search(qv, cand_k, allowed_rows=allowed)
        if method in ("bm25", "hybrid"):
            if self.bm is None:
                raise RuntimeError("BM25 index is not built")
            qt = sparse_tokens(normalize_sparse(query))
            sparse = self.bm.search(qt, cand_k, allowed_rows=allowed)

        if method == "dense":
            fused = dense
        elif method == "bm25":
            fused = sparse
        elif self.cfg.hybrid_mode == "weighted":
            fused = weighted_fusion(dense, sparse, self.cfg.dense_weight)
        else:
            fused = reciprocal_rank_fusion([dense, sparse], k=self.cfg.rrf_k)

        if use_mmr:
            fused = mmr_rerank(
                fused[: cand_k],
                lambda cid: self.vi.emb[self.row_of[cid]],
                top_k,
                self.cfg.mmr_lambda,
            )

        out: List[RetrievalHit] = []
        for i, (cid, score) in enumerate(fused[:top_k], start=1):
            c = self.by_id[cid]
            out.append(
                RetrievalHit(rank=i, chunk_id=cid, score=float(score), text=c["text"],
                             meta=c, retriever=method)
            )
        return out

    # ------------------------------------------------------------------
    def run_config(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return {
            "retrieval": asdict(self.cfg),
            "embedding": self.embedder.fingerprint(),
            "n_chunks": len(self.chunks),
            "bm25": {"available": self.bm is not None,
                     "k1": getattr(self.bm, "k1", None), "b": getattr(self.bm, "b", None)},
        }


# ---------------------------------------------------------------- scope helpers
def select_scope(chunks: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    """물리 인덱스별 청크 부분집합(= 빌드 대상 선택).

    - "manual_ko": doc_type이 INDEX_SCOPES[scope]["doc_types"]에 있는 청크 전부.
    - "pilot": source_file이 config.PILOT_FILES인 청크(NFC 비교 — macOS 파일명은 NFD).
    문서유형별 구성은 여기서 만들지 않는다 — 검색 시점에 `index.preset_filter(...)`로 만든다.
    """
    if scope not in C.INDEX_SCOPES:
        raise KeyError(f"unknown index scope: {scope!r} (available: {list(C.INDEX_SCOPES)})")
    if scope == "pilot":
        pilot = {unicodedata.normalize("NFC", f) for f in C.PILOT_FILES}
        return [c for c in chunks
                if unicodedata.normalize("NFC", c.get("source_file") or "") in pilot]
    doc_types = set(C.INDEX_SCOPES[scope]["doc_types"])
    return [c for c in chunks if c["doc_type"] in doc_types]


def index_dir(corpus_scope: str, index_scope: str, model_key: str) -> Path:
    return C.DATA_INDEX / corpus_scope / index_scope / model_key


def load_retriever(
    corpus_scope: str = "pilot",
    index_scope: str = "manual_ko",
    model_key: str = C.DEFAULT_EMBEDDING,
    *,
    with_bm25: bool = True,
    device: Optional[str] = None,
    retrieval_cfg: Optional[C.RetrievalConfig] = None,
) -> Retriever:
    d = index_dir(corpus_scope, index_scope, model_key)
    if not d.exists():
        raise FileNotFoundError(f"index not found: {d}(로컬 data/ 필요)")
    chunks = read_jsonl(d / "chunks.jsonl")
    vi = VectorIndex.load(d)
    bm = None
    if with_bm25:
        by_id = {c["chunk_id"]: c for c in chunks}
        toks = [sparse_tokens(by_id[cid]["text_normalized"]) for cid in vi.chunk_ids]
        bm = BM25Index(toks, list(vi.chunk_ids))
    emb = Embedder(model_key, device=device)
    return Retriever(chunks, vi, bm, emb, retrieval_cfg or C.RetrievalConfig())


def load_scoped_retriever(
    corpus_scope: str = "pilot",
    index_request: str = "manual_ko",
    model_key: str = C.DEFAULT_EMBEDDING,
    *,
    filter_overrides: Optional[Dict[str, Any]] = None,
    **kw: Any,
) -> "tuple[Retriever, Optional[MetadataFilter]]":
    """인덱스 요청 문자열을 풀어 (Retriever, 기본 필터)를 돌려준다.

    "어떤 문서유형 구성인가"는 인덱스가 아니라 필터가 결정하므로,
    **로더가 필터를 함께 돌려주지 않으면 호출부가 필터를 빠뜨려 조용히 통합 검색을 해 버린다.**
    그 사고를 구조적으로 막으려고 두 값을 한 번에 반환한다.

      r, flt = load_scoped_retriever("full", "manual_ko")
      r, flt = load_scoped_retriever("full", "manual_ko:repair_guide_only",
                                     filter_overrides={"models": ["SM-S938N"]})
    """
    scope, preset = C.resolve_index_request(index_request)
    r = load_retriever(corpus_scope, scope, model_key, **kw)
    overrides = dict(filter_overrides or {})
    return r, preset_filter(preset, **overrides)


def index_manifest(corpus_scope: str, index_scope: str, model_key: str) -> Dict[str, Any]:
    p = index_dir(corpus_scope, index_scope, model_key) / "index_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))
