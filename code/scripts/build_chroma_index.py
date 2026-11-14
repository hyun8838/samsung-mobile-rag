"""앱·MCP 서버가 쓰는 Chroma 색인(chroma_v2/)을 만든다.

입력: data/indexes/full_v3/manual_ko/arctic-ko/{embeddings.npy, chunk_ids.json, chunks.jsonl} (로컬 전용)
출력: 같은 폴더 아래 chroma_v2/ (Chroma PersistentClient, cosine HNSW) + chroma_manifest.json. 이미 있으면 중단.
실행(code/ 에서): python scripts/build_chroma_index.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.search import index_dir  # noqa: E402
from rag.vector_backends import CHROMA_COLLECTION, build_chroma_index  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    import chromadb

    idx = index_dir("full_v3", "manual_ko", "arctic-ko")
    t0 = time.perf_counter()
    out = build_chroma_index(idx)
    (out / "chroma_manifest.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "chromadb": chromadb.__version__,
        "collection": CHROMA_COLLECTION, "hnsw": {"space": "cosine", "M": 16, "construction_ef": 100, "search_ef": 100},
        "build_seconds": round(time.perf_counter() - t0, 1),
        "inputs_sha256": {n: sha(idx / n) for n in ("embeddings.npy", "chunk_ids.json", "chunks.jsonl")},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[완료] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
