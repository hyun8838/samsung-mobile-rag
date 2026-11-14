"""기기 필터를 검색 뒤에 거는 방식과 검색 안에서 거는 방식을 같은 임베딩(arctic-ko)·같은 질문으로 비교한다.

  langchain_faiss_fetch20  LangChain FAISS 기본 설정: 후보 20개(fetch_k 기본값)를 뽑은 뒤 기기 필터
  langchain_chroma_where   LangChain Chroma: filter(where)를 DB 검색 안에서 적용 (앱 기본)
두 구성 모두 dense top10을 jsonl로 쓰고, 결과가 5개 미만인 질문 수를 출력한다. R@5는 ragas_recall_at5.py로 계산.
입력: --questions jsonl(줄마다 {"qid", "question", "models": [모델코드]}), 로컬 색인(full_v3/manual_ko/arctic-ko, chroma_v2/)
실행(code/ 에서): python scripts/compare_langchain_filter.py --questions questions.jsonl --out runs_filter.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import settings as C  # noqa: E402
from rag.indexes import MetadataFilter  # noqa: E402
from rag.search import index_dir, load_retriever  # noqa: E402
from rag.vector_backends import CHROMA_COLLECTION, CHROMA_DIRNAME, chroma_where  # noqa: E402

K = 10
FETCH_K_DEFAULT = 20  # langchain_community FAISS.similarity_search_by_vector 기본값


def main() -> int:
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.embeddings import Embeddings

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r = load_retriever("full_v3", "manual_ko", "arctic-ko", with_bm25=False)
    idx = index_dir("full_v3", "manual_ko", "arctic-ko")
    emb = np.load(idx / "embeddings.npy").astype(np.float32)
    ids = r.vi.chunk_ids
    qs = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]

    class _Unused(Embeddings):  # 문서·질문 벡터를 직접 넘기므로 호출되지 않음
        def embed_documents(self, texts):
            raise RuntimeError("not used")

        def embed_query(self, text):
            raise RuntimeError("not used")

    faiss_store = FAISS.from_embeddings(
        text_embeddings=[(r.by_id[c]["text"], emb[i].tolist()) for i, c in enumerate(ids)],
        embedding=_Unused(), metadatas=[{"chunk_id": c} for c in ids], ids=list(ids),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    client = chromadb.PersistentClient(path=str(idx / CHROMA_DIRNAME), settings=Settings(anonymized_telemetry=False))
    chroma_store = Chroma(client=client, collection_name=CHROMA_COLLECTION, embedding_function=_Unused())

    short = {"langchain_faiss_fetch20": 0, "langchain_chroma_where": 0}
    with open(a.out, "x", encoding="utf-8") as fh:
        for q in qs:
            flt = MetadataFilter(models=q["models"], doc_types=list(C.DOC_TYPES))
            allowed = {ids[i] for i in r._allowed_rows(flt)}
            qv = r.embedder.encode([q["question"]], is_query=True)[0].tolist()
            docs = faiss_store.similarity_search_by_vector(qv, k=K, fetch_k=FETCH_K_DEFAULT,
                                                           filter=lambda md: md["chunk_id"] in allowed)
            hits_f = [d.metadata["chunk_id"] for d in docs]
            hits_c = [d.id for d in chroma_store.similarity_search_by_vector(qv, k=K, filter=chroma_where(flt))]
            for row, hits in (("langchain_faiss_fetch20", hits_f), ("langchain_chroma_where", hits_c)):
                short[row] += len(hits) < 5
                fh.write(json.dumps({"row": row, "qid": q["qid"], "hits": hits}, ensure_ascii=False) + "\n")
    for row, n in short.items():
        print(f"{row:28s} 결과 5개 미만 {n}/{len(qs)}문항")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
