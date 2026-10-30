"""Document / Chunk 스키마.

전신 프로젝트(robot-vacuum-rag)의 스키마를 삼성 모바일 매뉴얼 전용으로 줄였다.

모델 귀속 필드 3종(혼동 주의):
- `models`          : 삼성 사이트가 이 문서에 배정한 기본 모델코드. **검색 필터는 이것만 쓴다.**
- `filename_models` : 파일명 와일드카드(SM-S93X 등)를 펼친 코드. 기록용이며 필터에는 쓰지 않는다.
- `os_guide`        : 안드로이드/OS 버전별 가이드 표지(UG_17, UM, Win11, Android ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


# 청크 단위 라벨.
MANUAL_SECTION_TYPES = (
    "usage",
    "troubleshooting",
    "maintenance",
    "safety",
    "spec",
    "error_code",
    "install",
    "other",
)

DOC_TYPES = ("user_manual", "repair_guide")


@dataclass
class Document:
    doc_id: str                                  # f"{doc_type}::{NFC(파일 stem)}"
    doc_type: str                                # "user_manual" | "repair_guide"
    category: Optional[str] = None               # phone | tablet | book | watch | buds
    models: List[str] = field(default_factory=list)            # 사이트 배정 기본 모델코드(필터용)
    filename_models: List[str] = field(default_factory=list)   # 파일명 와일드카드 전개(기록 전용)
    os_guide: Optional[str] = None               # "UG_17" | "UM" | "Win11" | "Android" | None
    rev: Optional[str] = None
    edition_date: Optional[str] = None           # "YYYYMMDD"
    source_file: str = ""                        # PDF 파일명(NFC)
    file_sha256: Optional[str] = None            # 바이트 동일 PDF 식별(고유 PDF 수 산출)
    source_url: Optional[str] = None
    title: Optional[str] = None
    language: str = "ko"
    is_official: bool = True

    # --- PDF 추출 정보 ---
    file_path: Optional[str] = None
    pages: Optional[int] = None                 # C11: notes 자유텍스트가 아니라 정식 필드
    chars: Optional[int] = None
    is_scanned: Optional[bool] = None
    chars_per_page: Optional[float] = None
    extraction_method: Optional[str] = None     # "pdftext"
    has_toc: Optional[bool] = None
    has_korean: Optional[bool] = None

    raw_text: Optional[str] = None              # jsonl 저장 시 포함
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    text: str                      # preprocess_profile="raw" — dense 임베딩 입력(원문 보존)
    text_normalized: str           # preprocess_profile="normalized" — BM25/sparse 입력
    text_hash: str = ""            # sha1(text_normalized) — 모델 간 동일 청크/문장 수 집계용
    preprocess_profile: str = "dual"   # 두 표현을 한 레코드가 함께 보유

    # --- 문서에서 복사한 메타데이터 ---
    category: Optional[str] = None
    models: List[str] = field(default_factory=list)            # 사이트 배정(필터용)
    filename_models: List[str] = field(default_factory=list)   # 기록 전용
    os_guide: Optional[str] = None
    rev: Optional[str] = None
    edition_date: Optional[str] = None
    source_file: str = ""
    file_sha256: Optional[str] = None
    source_url: Optional[str] = None
    language: str = "ko"
    is_official: bool = True

    n_chars: int = 0
    chunk_index_in_doc: int = 0

    # --- 섹션/페이지 ---
    section_title: Optional[str] = None
    section_path: Optional[str] = None          # "H1 > H2 > H3"
    section_label: Optional[str] = None         # MANUAL_SECTION_TYPES 중 하나(구 manual_section_type)
    page_number: Optional[int] = None
    page_range: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # 검색 결과 출력용
    def citation(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_type": self.doc_type,
            "category": self.category,
            "models": self.models,
            "source_file": self.source_file,
            "section_title": self.section_title,
            "section_label": self.section_label,
            "page_number": self.page_number,
            "os_guide": self.os_guide,
            "rev": self.rev,
            "source_url": self.source_url,
        }


@dataclass
class LoadedSection:
    """문서 안의 구조 단위 하나(매뉴얼 목차 섹션 또는 페이지 묶음)."""

    title: Optional[str]
    path: Optional[str]              # "H1 > H2 > H3"
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    level: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    document: Document
    sections: List[LoadedSection] = field(default_factory=list)


class BaseLoader(ABC):
    """모든 로더의 공통 계약."""

    #: 이 로더가 처리하는 doc_type 목록
    doc_types: tuple = ()

    @abstractmethod
    def load(self, source: Any, **kwargs: Any) -> Iterable[LoadedDocument]:
        """원천(파일 경로/JSON 레코드 등)을 LoadedDocument로 변환한다."""
        raise NotImplementedError
