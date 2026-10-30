"""앱 검색 단계의 LangChain 구현(검색 단계만. 답 생성·웹 검색은 건드리지 않는다).

  langchain_chroma.Chroma (chroma_v2 컬렉션: 벡터+메타데이터+본문 documents 저장, 공식 사용법)
      -> Chroma.similarity_search_with_score(k=10, filter=where): 기기 필터를 검색 안에서 적용
      -> ContextualCompressionRetriever + CrossEncoderReranker(top_n=5, FT final_all 점수 그대로)

자체 부품은 세 개이며 이유는 각 클래스 주석에 적었다(ArcticKoEmbeddings=기존 임베더 연결, FilteredDenseRetriever·_RecordingCrossEncoder=점수·순위 기록과 누수 검사).
환경변수 GALAXY_RETRIEVAL_STACK = langchain | native  (rag_service가 읽는다)
"""
from __future__ import annotations

import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

try:  # langchain 0.3.x (.venv, python3.9)
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:  # langchain-classic 1.x (.venv-mcp, python3.12)
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from rag.vector_backends import CHROMA_COLLECTION, chroma_where
from rag.indexes import MetadataFilter

_nfc = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731


class ArcticKoEmbeddings(Embeddings):
    """기존 Embedder(arctic-ko)를 감싼 LangChain Embeddings. embed_query = encode([q], is_query=True)[0] (질의 접두 포함, 기존과 동일 벡터)."""

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_query(self, text: str) -> List[float]:
        return np.asarray(self.embedder.encode([text], is_query=True)[0], dtype=np.float32).reshape(-1).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return np.asarray(self.embedder.encode(list(texts), is_query=False), dtype=np.float32).tolist()


def build_vectorstore(chroma_client, embedder) -> Chroma:
    """공식 Chroma 벡터스토어를 기존 컬렉션에 연결. 본문(documents)이 없는 색인이면 즉시 중단(조용한 빈 본문 방지)."""
    vs = Chroma(client=chroma_client, collection_name=CHROMA_COLLECTION, embedding_function=ArcticKoEmbeddings(embedder))
    probe = vs._collection.get(limit=1, include=["documents"])
    if not probe.get("documents") or not probe["documents"][0]:
        raise RuntimeError("Chroma 컬렉션에 본문(documents)이 없습니다 - chroma_v2 색인을 쓰세요(scripts/build_chroma_index_v2.py)")
    return vs


class FilteredDenseRetriever(BaseRetriever):
    """공식 Chroma.similarity_search_with_score(k, filter=where)를 부른다. as_retriever()는 점수·순위를 버리므로(앱이 dense_score·dense_rank를
    표시) 얇게 감쌌고, 필터 누수 검사와 '허용 청크 있는데 0건' 오류도 여기서 한다."""

    vectorstore: Any
    where: Optional[Dict[str, Any]] = None
    k: int = 10
    codes: List[str] = []
    models_by_id: Any = None  # chunk_id(NFC) -> models 목록 (누수 검사용, 필터 구현과 독립)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        kw: Dict[str, Any] = {"k": self.k}
        if self.where is not None:
            kw["filter"] = self.where
        res = self.vectorstore.similarity_search_with_score(query, **kw)
        if not res and any(set(self.codes) & {_nfc(m) for m in ms} for ms in self.models_by_id.values()):
            raise RuntimeError(f"검색 0건이지만 선택 기기 {self.codes} 의 허용 청크가 있음 - 색인·필터 이상(근거 없음으로 처리하지 않고 중단)")
        out = []
        for rank, (doc, dist) in enumerate(res, 1):
            models = self.models_by_id[_nfc(doc.id)]
            if not set(self.codes) & {_nfc(m) for m in models}:  # 필터 누수 방어
                raise RuntimeError(f"필터 누수: {doc.id} models={models} 선택={self.codes}")
            md = dict(doc.metadata)
            md.update({"chunk_id": doc.id, "dense_score": 1.0 - float(dist), "dense_rank": rank})
            out.append(Document(id=doc.id, page_content=doc.page_content, metadata=md))
        return out


class _RecordingCrossEncoder(BaseCrossEncoder):
    """기존 CrossEncoder.predict(pairs, batch_size=16)와 같은 점수를 내는 BaseCrossEncoder. CrossEncoderReranker는 점수를 버리므로 본문별로 기록(앱 표시용)하고 화면 단계 콜백을 부른다."""

    def __init__(self, ce, batch_size: int, on_start: Optional[Callable[[], None]] = None):
        self.ce, self.batch_size, self.on_start = ce, batch_size, on_start
        self.by_text: Dict[str, float] = {}

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        if self.on_start:
            self.on_start()
        scores = [float(s) for s in self.ce.predict(list(text_pairs), batch_size=self.batch_size)]
        for (_, t), s in zip(text_pairs, scores):
            self.by_text[t] = s
        return scores


class LangChainRetrievalStack:
    def __init__(self, chroma_client, embedder, cross_encoder, chunks: Sequence[Dict[str, Any]],
                 dense_k: int = 10, final_k: int = 5, batch_size: int = 16):
        self.dense_k, self.final_k, self.batch_size, self.ce = dense_k, final_k, batch_size, cross_encoder
        self.models_by_id = {_nfc(c["chunk_id"]): (c.get("models") or []) for c in chunks}
        self.vectorstore = build_vectorstore(chroma_client, embedder)

    def retrieve(self, question: str, flt: MetadataFilter, codes: List[str],
                 on_rerank_start: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
        """반환 [{chunk_id, text, rerank_score, dense_score, dense_rank}] (리랭크 점수 내림차순 top_n). 호출마다 새 객체라 스레드 안전."""
        base = FilteredDenseRetriever(vectorstore=self.vectorstore, where=chroma_where(flt), k=self.dense_k,
                                      codes=codes, models_by_id=self.models_by_id)
        scorer = _RecordingCrossEncoder(self.ce, self.batch_size, on_rerank_start)
        pipe = ContextualCompressionRetriever(base_compressor=CrossEncoderReranker(model=scorer, top_n=self.final_k),
                                              base_retriever=base)
        docs = pipe.invoke(question)
        return [{"chunk_id": d.id, "text": d.page_content, "rerank_score": scorer.by_text[d.page_content],
                 "dense_score": d.metadata["dense_score"], "dense_rank": d.metadata["dense_rank"]} for d in docs]
