"""파이프라인 전역 설정 + 재현성 파라미터.

여기 있는 값이 곧 "재현성 기록"의 원본이다(문서와 어긋나면 이 파일이 진실).
전신 프로젝트(robot-vacuum-rag)의 설정 모듈을 삼성 모바일 매뉴얼 전용으로 줄였다.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INDEX = PROJECT_ROOT / "data" / "indexes"
REPORTS_DIR = PROJECT_ROOT / "reports"

MANUALS_DIR = DATA_RAW / "samsung_mobile_manual"          # 사용설명서 PDF(로컬 전용, 저장소 제외)
REPAIR_GUIDES_DIR = MANUALS_DIR / "repair_guide"            # 자가수리 가이드 PDF
# 문서별 메타데이터(파일명·문서종류·품목·사이트 배정 모델 등). 별도 스크립트가
# 목록 엑셀에서 만든다.
MANUAL_MANIFEST = DATA_RAW / "manual_manifest_v4.jsonl"

DOC_TYPES = ("user_manual", "repair_guide")
CATEGORIES = ("phone", "tablet", "book", "watch", "buds")

# 생성(LLM 호출)용 seed. 통계(bootstrap·fold 분할·표본 추출) seed 20260916과는 별개다.
RANDOM_SEED = 20260821
STAT_SEED = 20260916
BOOTSTRAP_N = 10_000


# ---------------------------------------------------------------- .env 로딩
def load_env_file(path: Optional[Path] = None, *, override: bool = False) -> List[str]:
    """프로젝트 루트의 `.env`를 os.environ에 올린다.

    - **값은 절대 로그·반환하지 않는다.** 반환값은 '어떤 키가 로드됐는지' 이름 목록뿐이다.
    - python-dotenv가 설치돼 있으면 그것을 쓰고, 없으면 최소 파서로 폴백한다
      (의존성 하나 때문에 파이프라인이 못 도는 상황을 만들지 않는다).
    - 기본적으로 이미 설정된 환경변수를 덮어쓰지 않는다(셸에서 준 값이 우선).
    """
    env_path = Path(path) if path else (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return []

    loaded: List[str] = []
    try:
        from dotenv import dotenv_values  # type: ignore

        pairs = dotenv_values(env_path)
    except Exception:  # noqa: BLE001 — python-dotenv 미설치 시 최소 파서
        pairs = {}
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            pairs[k.strip()] = v

    for k, v in pairs.items():
        if v is None:
            continue
        if override or not os.environ.get(k):
            os.environ[k] = v
            loaded.append(k)
    return loaded


def has_openai_key() -> bool:
    """키 '존재 여부'만 알려준다. 값은 어디에도 노출하지 않는다."""
    load_env_file()
    return bool(os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------- 스모크(pilot) 범위
# smoke_test.py --scope pilot 이 쓰는 작은 코퍼스: 이미 받은 사용설명서 1 + 자가수리 가이드 1.
# 파일명 비교는 반드시 NFC 정규화 후(macOS 파일명은 NFD).
PILOT_FILES: List[str] = [
    "SM-S94X_S93X_S92X_S91X_UG_17_Kor_Rev.1.0_260918.pdf",
    "SM-S938N_RepairGuide_Kor_Rev.1.0_250218.pdf",
]
PILOT_FILTER_MODEL = "SM-S938N"   # 위 두 문서에 모두 배정된 모델(필터 누수 검사용)


# ---------------------------------------------------------------- 청킹 파라미터
@dataclass
class ManualChunkConfig:
    """매뉴얼(PDF) 청킹 — 섹션(목차/북마크) 단위가 1순위."""

    strategy: str = "toc_section_then_size_split"
    # 길이 단위는 '문자'다. 한국어는 토크나이저별 토큰 수 편차가 커서
    # 임베딩 모델을 바꿔도 청크 경계가 흔들리지 않도록 문자 기준으로 고정한다.
    target_chars: int = 700
    max_chars: int = 1200
    min_chars: int = 120          # 이보다 짧은 섹션은 같은 부모 아래 인접 섹션과 병합
    overlap_chars: int = 100      # 섹션이 max_chars를 넘어 분할될 때만 적용(섹션 경계는 넘지 않음)
    keep_heading_prefix: bool = True   # 청크 앞에 섹션 경로(H1 > H2 > H3)를 붙여 문맥 보존
    fallback_strategy: str = "page_window"  # 목차(북마크)가 없는 PDF용
    fallback_pages_per_chunk: int = 1


@dataclass
class ChunkConfig:
    manual: ManualChunkConfig = field(default_factory=ManualChunkConfig)


# ---------------------------------------------------------------- 임베딩 후보
@dataclass
class EmbeddingModelSpec:
    name: str
    hf_id: str
    dim: int
    max_seq_length: int
    query_prefix: str = ""
    passage_prefix: str = ""
    note: str = ""
    torch_dtype: str = ""  # 빈 값 = 라이브러리 기본(fp32). 큰 모델만 "float16".


EMBEDDING_CANDIDATES: Dict[str, EmbeddingModelSpec] = {
    "bge-m3": EmbeddingModelSpec(
        name="bge-m3",
        hf_id="BAAI/bge-m3",
        dim=1024,
        max_seq_length=1024,  # 실제 8192까지 가능하나 청크가 짧아 1024로 고정(속도)
        note="다국어 베이스라인(전신 프로젝트 사전 등록 임베딩).",
    ),
    "kure-v1": EmbeddingModelSpec(
        name="kure-v1",
        hf_id="nlpai-lab/KURE-v1",
        dim=1024,
        max_seq_length=1024,
        note="고려대 한국어 특화. bge-m3 파인튜닝이라 아키텍처/토크나이저 동일 → 통제된 비교 가능.",
    ),
    "bge-m3-ko": EmbeddingModelSpec(
        name="bge-m3-ko",
        hf_id="dragonkue/BGE-m3-ko",
        dim=1024,
        max_seq_length=1024,
        note="한국어 파인튜닝 bge-m3. 위와 동일 이유로 비교군에 포함.",
    ),
    "qwen3-emb-4b": EmbeddingModelSpec(
        name="qwen3-emb-4b",
        hf_id="Qwen/Qwen3-Embedding-4B",
        dim=2560,
        max_seq_length=1024,  # bge-m3와 같은 길이 상한(비교 조건 통일)
        # Qwen 공식 README 기본 지시문 그대로(튜닝하지 않음, 결과 보기 전 고정)
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        note="Qwen3 LLM 기반 계열. M1 16GB라 fp16 로드.",
        torch_dtype="float16",
    ),
    "arctic-ko": EmbeddingModelSpec(
        name="arctic-ko",
        hf_id="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        dim=1024,
        max_seq_length=1024,  # bge-m3와 같은 길이 상한(비교 조건 통일)
        query_prefix="query: ",  # 모델 config_sentence_transformers.json의 query 프롬프트 그대로
        note="arctic-l-v2.0의 한국어 미세조정, 0.6B.",
    ),
}
DEFAULT_EMBEDDING = "bge-m3"

EMBED_BATCH_SIZE = 8
EMBED_NORMALIZE = True  # 코사인 = 내적이 되도록 L2 정규화


# ---------------------------------------------------------------- 검색 파라미터
@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 50          # 하이브리드 융합 전 각 리트리버가 뽑는 후보 수
    rrf_k: int = 60                # Reciprocal Rank Fusion 상수(관례값)
    hybrid_mode: str = "rrf"       # "rrf" | "weighted"
    dense_weight: float = 0.5      # hybrid_mode="weighted"일 때만 사용
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    vector_backend: str = "numpy_flat_exact_cosine"


# ---------------------------------------------------------------- 생성 파라미터
@dataclass
class GenerationConfig:
    backend: str = "template"      # "openai" | "oss_api" | "template"(오프라인 폴백)
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1200
    top_k_context: int = 5
    max_context_chars: int = 6000
    # backend="oss_api" 전용 — OpenAI Chat Completions 호환 엔드포인트를 쓰는
    # 공개 가중치 OSS 모델에 공용. 최종 챗봇은 쓰지 않는 선택 경로.
    oss_base_url: str = "https://api.together.xyz/v1"  # 호스팅형 추론 API
    oss_api_key_env: str = "OSS_API_KEY"               # .env에 넣을 환경변수 이름
    # Qwen/Qwen3.5-9B(Apache 2.0)는 "thinking"
    # 모델이라 기본값으로는 추론만 하다 max_tokens를 전부 소진해 빈 답변을 낸다(실측:
    # max_tokens=3000에서도 11/11건 빈 답변). enable_thinking=False로 끄면 정상 작동
    # (실측 12/12건 정상, 평균 completion_tokens 399). 다른 OpenAI 호환 엔드포인트로
    # 바꿀 경우 이 플래그가 무시될 수 있으니(모델마다 chat_template_kwargs 지원 여부가
    # 다름) None으로 두면 아예 보내지 않는다.
    oss_enable_thinking: Optional[bool] = False
    # temperature=0도 결정론을 보장하지 않는다(OpenAI 공식). seed를 고정하고
    # 응답의 system_fingerprint를 함께 기록해 재현 조건을 남긴다.
    seed: Optional[int] = RANDOM_SEED


# ---------------------------------------------------------------- 인덱스 구성
# 물리 색인은 하나(`manual_ko`): 사용설명서 + 자가수리 가이드. 문서종류·모델은 쿼리타임 필터로 나눈다.
INDEX_SCOPES: Dict[str, Dict[str, Any]] = {
    "manual_ko": {
        "desc": "삼성 모바일 한국어 사용설명서 + 자가수리 가이드(2024년 이후 판)",
        "doc_types": list(DOC_TYPES),
    },
    "pilot": {
        "desc": "스모크 테스트용(PILOT_FILES 2개)",
        "doc_types": list(DOC_TYPES),
    },
}

# 문서종류 프리셋. 주 평가는 "both"(대상 모델의 사용설명서 + 가이드).
DOC_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "both": {"desc": "사용설명서 + 자가수리 가이드", "doc_types": ["user_manual", "repair_guide"]},
    "user_manual_only": {"desc": "사용설명서만", "doc_types": ["user_manual"]},
    "repair_guide_only": {"desc": "자가수리 가이드만", "doc_types": ["repair_guide"]},
}


def resolve_index_request(request: str) -> "tuple[str, Optional[str]]":
    """`"manual_ko"` → ("manual_ko", None), `"manual_ko:user_manual_only"` → ("manual_ko", "user_manual_only")."""
    req = (request or "").strip()
    scope, _, preset = req.partition(":")
    if scope not in INDEX_SCOPES:
        raise KeyError(f"unknown index scope: {scope!r} (available: {list(INDEX_SCOPES)})")
    if preset and preset not in DOC_TYPE_PRESETS:
        raise KeyError(f"unknown doc-type preset: {preset!r} (available: {list(DOC_TYPE_PRESETS)})")
    return scope, (preset or None)


def index_request_choices() -> List[str]:
    """CLI `--index` 인자에 노출할 값 목록."""
    return list(INDEX_SCOPES) + [f"{s}:{p}" for s in INDEX_SCOPES for p in DOC_TYPE_PRESETS]


# ---------------------------------------------------------------- 환경 스냅샷
def package_versions() -> Dict[str, str]:
    import importlib

    out: Dict[str, str] = {}
    for mod in [
        "numpy",
        "torch",
        "sentence_transformers",
        "transformers",
        "rank_bm25",
        "kiwipiepy",
        "fitz",
        "bs4",
    ]:
        try:
            m = importlib.import_module(mod)
            out[mod] = str(getattr(m, "__version__", getattr(m, "VersionBind", "unknown")))
        except Exception:  # noqa: BLE001
            out[mod] = "not installed"
    return out


def run_manifest(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """모든 산출물에 함께 저장하는 재현성 매니페스트."""
    man: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "packages": package_versions(),
        "project_root": str(PROJECT_ROOT),
        "stat_seed": STAT_SEED,
    }
    if extra:
        man.update(extra)
    return man


def config_snapshot() -> Dict[str, Any]:
    return {
        "chunking": asdict(ChunkConfig()),
        "retrieval": asdict(RetrievalConfig()),
        "generation": asdict(GenerationConfig()),
        "embedding_candidates": {k: asdict(v) for k, v in EMBEDDING_CANDIDATES.items()},
        "default_embedding": DEFAULT_EMBEDDING,
        "index_scopes": INDEX_SCOPES,
        "doc_type_presets": DOC_TYPE_PRESETS,
        "pilot_files": PILOT_FILES,
    }
