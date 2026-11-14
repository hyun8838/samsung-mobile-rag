"""앱·MCP 서버용 벡터 저장소 백엔드(Chroma 기본 / FAISS 선택). 평가 스크립트는 index.VectorIndex(NumPy)를 그대로 쓴다.

  GALAXY_VECTOR_BACKEND = chroma(기본) | faiss | numpy

세 백엔드는 같은 `search(query_vec, top_k, allowed_rows=None, flt=None)` 를 가진다(반환 [(chunk_id, 코사인 유사도)]).
  - allowed_rows: VectorIndex와 같은 의미(허용 행 번호). Chroma는 `ids=` 제한, FAISS는 IDSelectorBatch로 검색 안에서 적용.
  - flt(MetadataFilter): Chroma는 where(메타데이터 조건)로 검색 안에서 적용, FAISS는 행 번호로 바꿔 IDSelectorBatch.
    필터를 검색 *뒤*에 거르지 않는다. 허용 청크가 k개 이상이면 k개를 돌려주는 것은 FAISS·numpy(정확 탐색)에서만 보장된다.
    Chroma(HNSW+where)는 k개 보장이 없다(근사 그래프 탐색 중 필터 통과 후보가 모자랄 수 있음). 이 154문항 점검에서는 전 문항 10개였다.
  - Chroma 색인은 HNSW 근사 탐색이다(기본 M=16, efConstruction=100, efSearch=100, cosine). FAISS는 IndexFlatIP 정확 탐색.
    NumPy 정확 검색과의 일치 정도는 README의 검색 저장소 비교 참조.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rag.indexes import MetadataFilter, VectorIndex, _norm_model

BACKENDS = ("chroma", "faiss", "numpy")
ENV_NAME = "GALAXY_VECTOR_BACKEND"
CHROMA_DIRNAME = "chroma_v2"  # 본문(documents) 포함 색인
CHROMA_COLLECTION = "manual_ko_arctic"
MODEL_KEY_PREFIX = "m__"


def default_backend() -> str:
    b = (os.environ.get(ENV_NAME) or "chroma").strip().lower()
    if b not in BACKENDS:
        raise ValueError(f"{ENV_NAME}={b!r}: {BACKENDS} 중 하나여야 합니다")
    return b


def chroma_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """청크 → Chroma 메타데이터(스칼라만 허용). 모델 목록은 키 'm__<정규화 모델코드>': True 로 펼친다."""
    m: Dict[str, Any] = {"doc_type": chunk.get("doc_type") or "",
                         "category": chunk.get("category") or "",
                         "language": chunk.get("language") or "",
                         "section_label": chunk.get("section_label") or "",
                         "is_official": bool(chunk.get("is_official"))}
    for code in chunk.get("models") or []:
        m[MODEL_KEY_PREFIX + _norm_model(code)] = True
    return m


def chroma_where(flt: MetadataFilter) -> Optional[Dict[str, Any]]:
    """MetadataFilter → Chroma where. 지원하지 않는 조건(pilot_only)은 조용히 무시하지 않고 오류."""
    if flt.pilot_only:
        raise NotImplementedError("pilot_only 필터는 Chroma 백엔드에서 지원하지 않습니다")
    conds: List[Dict[str, Any]] = []
    if flt.models is not None:
        keys = sorted({MODEL_KEY_PREFIX + _norm_model(x) for x in flt.models})
        ors = [{k: True} for k in keys]
        if not ors:
            raise ValueError("models가 비어 있습니다")
        conds.append(ors[0] if len(ors) == 1 else {"$or": ors})
    for field, vals in (("doc_type", flt.doc_types), ("category", flt.categories),
                        ("language", flt.languages), ("section_label", flt.section_labels)):
        if vals is not None:
            conds.append({field: {"$in": list(vals)}})
    if flt.is_official is not None:
        conds.append({"is_official": bool(flt.is_official)})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class ChromaVectorIndex:
    """Chroma PersistentClient 위의 읽기 전용 검색기. 색인은 `build_chroma_index()`가 만든다."""

    backend = "chroma"

    def __init__(self, chroma_dir: Path, chunk_ids: List[str], embeddings: np.ndarray):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_collection(CHROMA_COLLECTION)
        if self.col.count() != len(chunk_ids):
            raise RuntimeError(f"Chroma 청크 수 {self.col.count()} != 인덱스 청크 수 {len(chunk_ids)}")
        self.chunk_ids = chunk_ids
        self.emb = embeddings  # MMR 등 호출부 호환용(검색에는 쓰지 않음)
        self.pos = {cid: i for i, cid in enumerate(chunk_ids)}

    def search(self, query_vec: np.ndarray, top_k: int, allowed_rows: Optional[np.ndarray] = None,
               flt: Optional[MetadataFilter] = None) -> List[Tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1).tolist()
        kw: Dict[str, Any] = {}
        if flt is not None:
            w = chroma_where(flt)
            if w is not None:
                kw["where"] = w
        if allowed_rows is not None:
            if allowed_rows.size == 0:
                return []
            kw["ids"] = [self.chunk_ids[int(i)] for i in allowed_rows]
        r = self.col.query(query_embeddings=[q], n_results=top_k, include=["distances"], **kw)
        return [(cid, 1.0 - float(d)) for cid, d in zip(r["ids"][0], r["distances"][0])]


class FaissVectorIndex:
    """FAISS IndexFlatIP(정확 내적 = 정규화 벡터의 코사인) + IDSelectorBatch 필터."""

    backend = "faiss"

    def __init__(self, embeddings: np.ndarray, chunk_ids: List[str], chunks_by_id: Optional[Dict[str, Dict[str, Any]]] = None):
        import faiss

        assert embeddings.shape[0] == len(chunk_ids)
        self._faiss = faiss
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.chunk_ids = chunk_ids
        self.pos = {cid: i for i, cid in enumerate(chunk_ids)}
        self._by_id = chunks_by_id
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def _rows_from_filter(self, flt: MetadataFilter) -> np.ndarray:
        if self._by_id is None:
            raise RuntimeError("flt 사용에는 chunks_by_id가 필요합니다")
        return np.asarray([i for i, cid in enumerate(self.chunk_ids) if flt.match(self._by_id[cid])], dtype=np.int64)

    def search(self, query_vec: np.ndarray, top_k: int, allowed_rows: Optional[np.ndarray] = None,
               flt: Optional[MetadataFilter] = None) -> List[Tuple[str, float]]:
        faiss = self._faiss
        q = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        if flt is not None:
            rows = self._rows_from_filter(flt)
            allowed_rows = rows if allowed_rows is None else np.intersect1d(allowed_rows, rows)
        if allowed_rows is None:
            D, I = self.index.search(q, top_k)
        else:
            if allowed_rows.size == 0:
                return []
            sel = faiss.IDSelectorBatch(np.ascontiguousarray(allowed_rows, dtype=np.int64))
            D, I = self.index.search(q, top_k, params=faiss.SearchParameters(sel=sel))
        return [(self.chunk_ids[int(i)], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]


class NumpyFilterAdapter:
    """VectorIndex(NumPy)에 flt= 인자를 붙인 얇은 어댑터. 앱에서 세 백엔드를 같은 호출로 쓰기 위한 것(검색 계산은 VectorIndex 그대로)."""

    backend = "numpy"

    def __init__(self, vi: VectorIndex, chunks_by_id: Dict[str, Dict[str, Any]]):
        self.vi, self._by_id = vi, chunks_by_id
        self.emb, self.chunk_ids, self.pos = vi.emb, vi.chunk_ids, vi.pos

    def search(self, query_vec, top_k, allowed_rows=None, flt=None):
        if flt is not None:
            rows = np.asarray([i for i, cid in enumerate(self.chunk_ids) if flt.match(self._by_id[cid])], dtype=np.int64)
            allowed_rows = rows if allowed_rows is None else np.intersect1d(allowed_rows, rows)
        return self.vi.search(query_vec, top_k, allowed_rows=allowed_rows)


def _verify_chroma_manifest(index_dir: Path, chroma_dir: Path) -> None:
    """chroma_manifest.json의 입력 sha256(embeddings.npy·chunk_ids.json·chunks.jsonl)이 현재 색인 파일과 같은지 대조. 다르면 오류."""
    import hashlib

    mf = chroma_dir / "chroma_manifest.json"
    if not mf.exists():
        return  # 색인 폴더 없음 오류는 호출부가 낸다
    want = json.loads(mf.read_text(encoding="utf-8")).get("inputs_sha256") or {}
    for rel, sha in want.items():
        p = index_dir / Path(rel).name
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != sha:
            raise RuntimeError(f"Chroma 색인 입력 불일치: {p.name} sha256 {h.hexdigest()[:12]} != chroma_manifest {sha[:12]} - Chroma 색인을 다시 만드세요")


def load_vector_backend(index_dir: Path, chunks: Sequence[Dict[str, Any]], backend: Optional[str] = None):
    """index_dir(embeddings.npy·chunk_ids.json가 있는 색인 폴더)에서 백엔드 객체를 만든다."""
    backend = (backend or default_backend()).lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend={backend!r}: {BACKENDS} 중 하나여야 합니다")
    ids: List[str] = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    if backend == "chroma":
        d = index_dir / CHROMA_DIRNAME
        _verify_chroma_manifest(index_dir, d)
        if not d.exists():
            raise FileNotFoundError(f"{d} 없음 - vector_backends.build_chroma_index()로 먼저 만드세요 "
                                    f"(또는 {ENV_NAME}=numpy|faiss)")
        return ChromaVectorIndex(d, ids, np.load(index_dir / "embeddings.npy", mmap_mode="r"))
    emb = np.load(index_dir / "embeddings.npy")
    if backend == "faiss":
        return FaissVectorIndex(emb, ids, by_id)
    return NumpyFilterAdapter(VectorIndex(emb, ids), by_id)


def build_chroma_index(index_dir: Path) -> Path:
    """index_dir(embeddings.npy·chunk_ids.json·chunks.jsonl)에서 chroma_v2/ 영구 색인(벡터+메타데이터+본문)을 새로 만든다. 이미 있으면 중단. 본문은 제조사 원문이라 로컬 전용."""
    import chromadb
    from chromadb.config import Settings

    out = index_dir / CHROMA_DIRNAME
    if out.exists():
        raise FileExistsError(f"이미 있음(덮어쓰기 금지): {out}")
    emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
    ids = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    by_id = {}
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            by_id[c["chunk_id"]] = c
    assert emb.shape[0] == len(ids) == len(by_id)
    out.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(out), settings=Settings(anonymized_telemetry=False))
    col = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    for s in range(0, len(ids), 4000):
        sl = ids[s:s + 4000]
        col.add(ids=sl, embeddings=emb[s:s + 4000], metadatas=[chroma_metadata(by_id[c]) for c in sl],
                documents=[by_id[c]["text"] for c in sl])
    return out
