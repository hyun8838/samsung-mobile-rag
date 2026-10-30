"""벡터/스파스 인덱스 + 메타데이터 필터 + 하이브리드 융합.

벡터 백엔드 선택 근거 (재현성 기록 항목):
    numpy flat exact cosine을 쓴다. FAISS/Chroma의 ANN이 아니라 **완전탐색**이다.
    이유 ①현재 코퍼스가 수천~수만 청크 규모라 완전탐색이 밀리초 단위로 끝난다
        ②ANN 근사 오차가 Recall@k에 섞이면 "검색 방식(H1~H5)의 차이"와 "인덱스 근사 손실"이
          분리되지 않는다 — 평가가 목적인 프로젝트에서는 근사 오차를 0으로 두는 편이 낫다
        ③메타데이터 사전 필터링(H4)을 정확히 적용할 수 있다. ANN은 필터와 결합하면
          후보 부족/재탐색 문제가 생겨 결과가 필터 강도에 따라 흔들린다.
    인터페이스(`VectorIndex.search`)는 유지하므로 규모가 커지면 FAISS 백엔드로 교체 가능하다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from rag import settings as C


# ---------------------------------------------------------------- 필터
def _norm_model(x: Any) -> str:
    """모델코드 비교용 정규화: NFC·대문자·공백 제거, 'SM-' 접두 통일.

    요청이 "S938N"처럼 접두 없이 와도, 청크가 "SM-S938N"이어도 같은 값이 되도록 양쪽에 같은 규칙을 쓴다.
    북(NT940XJG)·워치/버즈 등 SM- 계열이 아닌 코드는 접두를 붙이지 않는다(영문 1글자+숫자로 시작할 때만 붙임).
    """
    t = unicodedata.normalize("NFC", str(x or "")).upper().replace(" ", "")
    if t and not t.startswith("SM-") and re.match(r"^[A-Z]\d{2,3}[A-Z0-9]*$", t):
        t = "SM-" + t
    return t


@dataclass
class MetadataFilter:
    """검색 전 후보 축소용 메타데이터 필터.

    H4(메타데이터 필터링 효과) 검증과, 같은 계열 다른 모델 매뉴얼이 distractor가 되는 문제
    (Power of Noise, SIGIR'24)의 통제 수단이 바로 이 클래스다.

    `models` 매칭은 **삼성 사이트가 문서에 배정한 `models` 목록만** 쓴다(요청 집합과 교집합이 있으면 통과).
    파일명 와일드카드를 펼친 `filename_models`는 기록용이며 필터에 절대 쓰지 않는다
    (공용 PDF 파일명이 실제 배정과 다를 수 있어, 쓰면 다른 모델 청크가 정답 후보로 새어 들어온다).
    doc_type 구분 없이 사용설명서·자가수리 가이드에 동일 적용된다.
    """

    models: Optional[Sequence[str]] = None          # 사이트 배정 기본 모델코드(교집합 매칭)
    doc_types: Optional[Sequence[str]] = None
    categories: Optional[Sequence[str]] = None      # phone | tablet | book | watch | buds
    languages: Optional[Sequence[str]] = None
    section_labels: Optional[Sequence[str]] = None
    is_official: Optional[bool] = None
    pilot_only: bool = False                        # config.PILOT_FILES 문서만

    def match(self, m: Dict[str, Any]) -> bool:
        def inset(val, allowed):
            return allowed is None or (val in set(allowed))

        if self.models is not None:
            want = {_norm_model(x) for x in self.models}
            have = {_norm_model(x) for x in (m.get("models") or [])}
            if not (want & have):
                return False
        if not inset(m.get("doc_type"), self.doc_types):
            return False
        if not inset(m.get("category"), self.categories):
            return False
        if not inset(m.get("language"), self.languages):
            return False
        if self.section_labels is not None:
            if m.get("section_label") not in set(self.section_labels):
                return False
        if self.is_official is not None and bool(m.get("is_official")) != self.is_official:
            return False
        if self.pilot_only:
            pilot = {unicodedata.normalize("NFC", f) for f in C.PILOT_FILES}
            if unicodedata.normalize("NFC", m.get("source_file") or "") not in pilot:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, False)}


def preset_filter(preset: Optional[str], **overrides: Any) -> Optional[MetadataFilter]:
    """문서유형 프리셋(+추가 조건) → `MetadataFilter`.

    "사용설명서 only / 자가수리 가이드 only / 둘 다" 구성을 만드는 **유일한 경로**다.
    물리 인덱스(`manual_ko`)는 하나이고, 구성은 이 필터를 쿼리타임에 건다.

      preset_filter("user_manual_only", models=["SM-S938N"])
        == MetadataFilter(doc_types=["user_manual"], models=["SM-S938N"])

    `doc_types`를 명시로 넘기면 프리셋보다 우선한다. 프리셋도 추가 조건도 없으면 `None`
    (= 필터 미적용)을 돌려주므로 호출부에서 조건 분기를 따로 쓸 필요가 없다.
    """
    kw = {k: v for k, v in overrides.items() if v is not None}
    if preset:
        kw.setdefault("doc_types", list(C.DOC_TYPE_PRESETS[preset]["doc_types"]))
    if not kw:
        return None
    return MetadataFilter(**kw)


# ---------------------------------------------------------------- 벡터 인덱스
class VectorIndex:
    def __init__(self, embeddings: np.ndarray, chunk_ids: List[str]):
        assert embeddings.shape[0] == len(chunk_ids)
        self.emb = embeddings.astype(np.float32)
        self.chunk_ids = chunk_ids
        self.pos = {cid: i for i, cid in enumerate(chunk_ids)}

    def search(
        self, query_vec: np.ndarray, top_k: int, allowed_rows: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        q = query_vec.reshape(-1).astype(np.float32)
        if allowed_rows is None:
            scores = self.emb @ q
            idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
            idx = idx[np.argsort(-scores[idx])]
            return [(self.chunk_ids[i], float(scores[i])) for i in idx]
        if allowed_rows.size == 0:
            return []
        sub = self.emb[allowed_rows] @ q
        k = min(top_k, sub.shape[0])
        idx = np.argpartition(-sub, k - 1)[:k]
        idx = idx[np.argsort(-sub[idx])]
        return [(self.chunk_ids[allowed_rows[i]], float(sub[i])) for i in idx]

    # ---- io ----
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.emb)
        (path / "chunk_ids.json").write_text(json.dumps(self.chunk_ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        emb = np.load(path / "embeddings.npy")
        ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
        return cls(emb, ids)


# ---------------------------------------------------------------- BM25
class BM25Index:
    """BM25Okapi. 입력은 `text_normalized`(형태소 기반 sparse 프로파일)."""

    def __init__(self, token_lists: List[List[str]], chunk_ids: List[str], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        self.k1, self.b = k1, b
        self.bm25 = BM25Okapi(token_lists, k1=k1, b=b)

    def search(
        self, query_tokens: List[str], top_k: int, allowed_rows: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if allowed_rows is not None:
            mask = np.full(scores.shape, -np.inf, dtype=np.float32)
            mask[allowed_rows] = scores[allowed_rows]
            scores = mask
        k = min(top_k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx if np.isfinite(scores[i])]


# ---------------------------------------------------------------- 융합
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]], k: int = 60
) -> List[Tuple[str, float]]:
    """RRF: score = Σ 1/(k + rank). 점수 스케일이 다른 dense/sparse를 정규화 없이 합칠 수 있다."""
    agg: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (cid, _s) in enumerate(lst, start=1):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda x: -x[1])


def weighted_fusion(
    dense: Sequence[Tuple[str, float]],
    sparse: Sequence[Tuple[str, float]],
    dense_weight: float = 0.5,
) -> List[Tuple[str, float]]:
    """min-max 정규화 후 가중합. RRF와 달리 점수 크기 정보를 유지하지만 스케일에 민감하다."""

    def mm(lst):
        if not lst:
            return {}
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {cid: (s - lo) / rng for cid, s in lst}

    d, s = mm(dense), mm(sparse)
    keys = set(d) | set(s)
    out = {k: dense_weight * d.get(k, 0.0) + (1 - dense_weight) * s.get(k, 0.0) for k in keys}
    return sorted(out.items(), key=lambda x: -x[1])


def mmr_rerank(
    candidates: Sequence[Tuple[str, float]],
    emb_lookup: Callable[[str], np.ndarray],
    top_k: int,
    lam: float = 0.7,
) -> List[Tuple[str, float]]:
    """MMR — 거의 같은 내용의 청크(모델 간 근접중복 매뉴얼)가 top-k를 독점하는 문제 대응."""
    if not candidates:
        return []
    selected: List[Tuple[str, float]] = []
    pool = list(candidates)
    sel_vecs: List[np.ndarray] = []
    while pool and len(selected) < top_k:
        best_i, best_val = 0, -1e9
        for i, (cid, base) in enumerate(pool):
            v = emb_lookup(cid)
            div = max((float(v @ sv) for sv in sel_vecs), default=0.0)
            val = lam * base - (1 - lam) * div
            if val > best_val:
                best_i, best_val = i, val
        cid, base = pool.pop(best_i)
        selected.append((cid, base))
        sel_vecs.append(emb_lookup(cid))
    return selected
