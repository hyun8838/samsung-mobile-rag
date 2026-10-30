"""앱용 RAG 래퍼(Streamlit 무관). 챗봇 앱과 MCP 서버가 같은 검색 경로를 쓴다.

  검색 : full_v3 / manual_ko / arctic-ko dense top10 (질의 접두 "query: "는 Embedder가 붙임),
         MetadataFilter(models=[선택 기기 묶음], doc_types=전체), MMR 끔
  리랭크: FT 리랭커(final_all, CrossEncoder max_length 512, float32) -> top5
  생성 : rag.create_answer.answer(..., series_level=False, allow_external=True)
         SUFFICIENT·TYPE·EVIDENCE 줄은 stream_text가 화면에서 숨기고 answer.parse_output이 본문에서 뗀다.
기기 선택 없는(모델 필터 없는) 검색 경로는 만들지 않는다: ask()는 카탈로그에 있는 기기만 받는다.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from rag import create_answer as A  # noqa: E402
from rag import settings as C  # noqa: E402
from rag.indexes import MetadataFilter  # noqa: E402
from rag.search import load_retriever  # noqa: E402
import devices as DC  # noqa: E402
from stream_text import StreamingClient  # noqa: E402

EMBEDDER = "arctic-ko"
RERANKER_DIR = ROOT / "data/models/reranker_ft_arctic_ko_common_groupkfold5_v2/final_all"
DENSE_K, FINAL_K, MAX_LENGTH, BATCH = 10, 5, 512, 16


def load_resources() -> Dict[str, Any]:
    """무거운 자원 1회 로드(임베더, 인덱스 numpy, FT 리랭커, OpenAI 클라이언트)."""
    import hashlib

    import torch
    from openai import OpenAI
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    C.load_env_file()
    chunks_path = ROOT / "data/indexes/full_v3/manual_ko" / EMBEDDER / "chunks.jsonl"
    got = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    want = DC.catalog_input_sha().get("chunks.jsonl")
    if got != want:  # 기기 카탈로그가 만들어진 인덱스와 로드할 인덱스가 다르면 묶음 동일성 보증이 깨짐 -> 중단
        raise RuntimeError(f"chunks.jsonl sha256 불일치: 인덱스 {got[:12]} / 카탈로그 manifest {str(want)[:12]} - 카탈로그를 다시 빌드하세요")
    r = load_retriever("full_v3", "manual_ko", EMBEDDER)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ce = CrossEncoder(str(RERANKER_DIR), max_length=MAX_LENGTH, model_kwargs={"dtype": torch.float32}, device=dev)
    client = OpenAI()
    return {"retriever": r, "reranker": ce, "client": client, "device": dev, "embedder": EMBEDDER,
            "by_id": {DC.nfc(c["chunk_id"]): c for c in r.chunks}, "load_seconds": round(time.time() - t0, 1)}


# ---- 자원 싱글턴 + 백그라운드 미리 로드 (검색·생성 결과에는 영향 없음: 임베더 지연 로드/커널 워밍업을 앱 시작 때로 앞당길 뿐)
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"res": None, "thread": None}


def get_resources() -> Dict[str, Any]:
    """load_resources + 워밍업을 프로세스당 1회. 미리 로드 스레드와 첫 질문이 동시에 불러도 한 번만 로드(잠금)."""
    with _LOCK:
        if _STATE["res"] is None:
            res = load_resources()
            t0 = time.time()
            try:  # 워밍업 실패는 치명적이지 않다(첫 질문에서 지연될 뿐)
                res["retriever"].embedder.encode(["워밍업"], is_query=True)
                res["reranker"].predict([("워밍업", "워밍업")], batch_size=BATCH)
                res["warmup_seconds"] = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001
                res["warmup_error"] = f"{type(e).__name__}: {e}"
            _STATE["res"] = res
        return _STATE["res"]


def is_ready() -> bool:
    return _STATE["res"] is not None


def preload_async() -> None:
    """앱 시작 시 한 번 호출: 백그라운드 스레드에서 get_resources(). 이미 시작했으면 아무 일도 안 한다."""
    with _LOCK:
        if _STATE["thread"] is not None or _STATE["res"] is not None:
            return
        t = threading.Thread(target=lambda: _safe_preload(), name="rag-preload", daemon=True)
        _STATE["thread"] = t
    t.start()


def _safe_preload() -> None:
    try:
        get_resources()
    except Exception:  # noqa: BLE001 - 실패하면 첫 질문의 get_resources()가 같은 예외를 화면에 낸다
        _STATE["thread"] = None


class _StageClient:
    """OpenAI 클라이언트 위임 래퍼: 웹 검색(client.responses.create) 호출 직전에 on_stage("external")만 알린다. 요청·인자는 그대로."""

    def __init__(self, client, on_stage):
        self._c, self._cb = client, on_stage

    def __getattr__(self, name):
        attr = getattr(self._c, name)
        if name == "responses":
            cb = self._cb

            class _R:
                def __getattr__(_, n):
                    f = getattr(attr, n)
                    if n == "create":
                        def wrapped(*a, **kw):
                            cb("external")
                            return f(*a, **kw)
                        return wrapped
                    return f
            return _R()
        return attr


def retrieve_top5(res: Dict[str, Any], question: str, device: Dict, on_stage: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """선택 기기 필터 dense top10 -> FT 리랭크 -> top5. 각 항목 {chunk, rerank_score, dense_score, dense_rank}."""
    codes = [DC.nfc(m) for m in (device.get("model_codes") or [device["model_code"]])]  # 묶음 전체(청크 집합 동일 = 단일 코드와 후보 동일)
    flt = MetadataFilter(models=codes, doc_types=list(C.DOC_TYPES))
    if on_stage:
        on_stage("search")
    hits = res["retriever"].search(question, method="dense", top_k=DENSE_K, flt=flt, use_mmr=False)
    if not hits:
        return []
    chunks = [res["by_id"][DC.nfc(h.chunk_id)] for h in hits]
    for c in chunks:  # 필터 누수 방어(필터 구현과 독립 검사): 선택 묶음의 코드가 하나도 청크 models에 없으면 중단
        if not set(codes) & {DC.nfc(m) for m in (c.get("models") or [])}:
            raise RuntimeError(f"필터 누수: {c['chunk_id']} models={c.get('models')} 선택={codes}")
    if on_stage:
        on_stage("rerank")
    scores = res["reranker"].predict([(question, c["text"]) for c in chunks], batch_size=BATCH)
    order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))[:FINAL_K]
    return [{"chunk": chunks[i], "rerank_score": float(scores[i]), "dense_score": float(hits[i].score),
             "dense_rank": i + 1} for i in order]


def ask(res: Dict[str, Any], question: str, device: Dict, on_stage: Optional[Callable[[str], None]] = None,
        on_text: Optional[Callable[[str], None]] = None, on_reset: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """on_stage(name): "search" -> "rerank" -> "generate" -> ("external"). 표시 전용 콜백.
    on_text/on_reset을 주면 생성 응답을 점진 표시(stream_text가 스트림만 감쌈, 프롬프트·인자 불변). r["stream"]에 표시 기록."""
    question = (question or "").strip()
    if not question:
        raise ValueError("질문이 비어 있습니다")
    if not device or not device.get("model_code"):
        raise ValueError("기기 선택은 필수입니다")
    if DC.find_group(DC.load_groups(), device["model_code"]) is None:
        raise ValueError(f"카탈로그에 없는 기기입니다: {device.get('model_code')}")
    t0 = time.time()
    top5 = retrieve_top5(res, question, device, on_stage)
    t1 = time.time()
    if on_stage:
        on_stage("generate")
    client = res["client"]
    sc = StreamingClient(client, on_text, on_reset) if on_text else None
    if sc:
        client = sc
    if on_stage:
        client = _StageClient(client, on_stage)
    out = A.answer(client, question, DC.model_desc(device), [t["chunk"] for t in top5],
                   series_level=False, allow_external=True)
    t2 = time.time()
    r = {"embedder": res.get("embedder"), "question": question, "device": device, "model_desc": DC.model_desc(device), "top5": top5,
         "route": out["route"], "final_text": out["final_text"], "manual": out["manual"],
         "external": out.get("external"),
         "timing_s": {"retrieve_rerank": round(t1 - t0, 2), "generate": round(t2 - t1, 2), "total": round(t2 - t0, 2)}}
    if sc and sc.calls:
        r["stream"] = {"n_calls": len(sc.calls), "raw": sc.last["raw"], "parser_state": sc.last["state"], "stream_sufficient": sc.last.get("sufficient"), "finish_reason": sc.last.get("finish_reason"),
                       "first_visible_s_from_gen_call": sc.last["t_first_visible"], "gen_call_total_s": sc.last["t_total"]}
    return r
