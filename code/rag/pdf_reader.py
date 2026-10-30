"""PDF 매뉴얼 로더 (PyMuPDF).

핵심 설계:
1. **PDF 북마크(목차)를 1차 섹션 소스로 쓴다.** 레벨·페이지가
   붙은 북마크가 대부분의 PDF에 들어 있다. 목차를 재구성하는 휴리스틱(폰트 크기 추정)보다
   원천 구조를 그대로 쓰는 쪽이 정확하고 재현 가능하다.
2. 북마크가 없는 PDF는 `page_window` 폴백을 쓴다.
3. 페이지 번호는 **텍스트에서 제거하고 메타데이터로 옮긴다**.
4. 반복 머리말/꼬리말은 문서 단위 빈도로 탐지해 제거한다(L1 오염 방지).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag.text import detect_repeated_lines, normalize_raw, strip_page_boilerplate
from rag.documents import Document
from rag.documents import BaseLoader, LoadedDocument, LoadedSection

warnings.filterwarnings("ignore")

# 스캔본 판정 임계값. OHRBench(ICCV'25)가 지적한 "OCR 노이즈의 전 단계 전파"를 피하려면
# 텍스트 레이어 유무를 문서 단위로 기록해야 한다. 임계값은 경험값이다.
SCAN_CHARS_PER_PAGE_THRESHOLD = 150

_HANGUL_RE = re.compile(r"[가-힣]")


def _norm_head(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


class PdfManualLoader(BaseLoader):
    doc_types = ("user_manual", "repair_guide")

    def __init__(self, fallback_pages_per_chunk: int = 1) -> None:
        self.fallback_pages_per_chunk = fallback_pages_per_chunk

    # ------------------------------------------------------------------ public
    def load(self, source: Any, *, doc_meta: Optional[Dict[str, Any]] = None, **kwargs: Any):
        path = Path(source)
        doc_meta = doc_meta or {}
        import fitz

        pdf = fitz.open(path)
        pages_raw = [p.get_text() for p in pdf]
        toc = pdf.get_toc() or []
        page_count = pdf.page_count
        pdf.close()

        repeated = detect_repeated_lines(pages_raw)
        pages = [strip_page_boilerplate(t, repeated) for t in pages_raw]
        full_chars = sum(len(t) for t in pages_raw)
        cpp = full_chars / max(1, page_count)
        korean_chars = sum(len(_HANGUL_RE.findall(t)) for t in pages_raw)

        document = self._build_document(path, doc_meta, page_count, full_chars, cpp,
                                        bool(toc), korean_chars)

        if toc:
            sections = self._sections_from_toc(toc, pages)
            document.notes = (document.notes + " | sectioning=toc").strip(" |")
        else:
            sections = self._sections_from_pages(pages)
            document.notes = (document.notes + " | sectioning=page_window_fallback").strip(" |")

        return [LoadedDocument(document=document, sections=sections)]

    # ------------------------------------------------------------------ internals
    def _build_document(
        self,
        path: Path,
        m: Dict[str, Any],
        page_count: int,
        chars: int,
        cpp: float,
        has_toc: bool,
        korean_chars: int,
    ) -> Document:
        # 파일명은 macOS에서 NFD로 읽히므로 비교·ID에 쓰는 값은 반드시 NFC로 통일한다.
        sha = m.get("file_sha256") or hashlib.sha256(path.read_bytes()).hexdigest()
        stem = unicodedata.normalize("NFC", path.stem)
        doc_type = m.get("doc_type") or "user_manual"
        lang = m.get("language") or ("ko" if korean_chars > 200 else "en")
        return Document(
            doc_id=f"{doc_type}::{stem}",
            doc_type=doc_type,
            category=m.get("category"),
            models=list(m.get("models") or []),
            filename_models=list(m.get("filename_models") or []),
            os_guide=m.get("os_guide"),
            rev=m.get("rev"),
            edition_date=m.get("edition_date"),
            source_file=unicodedata.normalize("NFC", path.name),
            file_sha256=sha,
            source_url=m.get("source_url"),
            title=m.get("title") or stem,
            language=lang,
            is_official=bool(m.get("is_official", True)),
            file_path=str(path),
            pages=page_count,
            chars=chars,
            chars_per_page=round(cpp, 1),
            is_scanned=cpp < SCAN_CHARS_PER_PAGE_THRESHOLD,
            extraction_method="pdftext",
            has_toc=has_toc,
            has_korean=korean_chars > 200,
            notes=m.get("notes", ""),
        )

    def _sections_from_toc(
        self, toc: List[List[Any]], pages: List[str]
    ) -> List[LoadedSection]:
        """북마크 제목을 페이지 텍스트에서 찾아 섹션 경계를 잡는다.

        북마크는 '이 제목이 몇 페이지에 있다'만 알려주므로, 같은 페이지에 여러 섹션이 있으면
        본문 안에서 제목 문자열을 다시 찾아야 한다. 못 찾으면 그 섹션은 별도로 끊지 않고
        직전 섹션에 흡수시킨다(=경계를 잘못 자르느니 덜 자른다).
        """
        # 페이지 오프셋 있는 전체 텍스트
        offsets: List[int] = []
        buf: List[str] = []
        pos = 0
        for t in pages:
            offsets.append(pos)
            buf.append(t)
            pos += len(t) + 1
        full = "\n".join(buf)
        full_norm = _norm_head(full)
        # full_norm 인덱스 → full 인덱스 매핑
        map_norm_to_full: List[int] = []
        for i, ch in enumerate(full):
            if not ch.isspace():
                map_norm_to_full.append(i)

        def page_of(idx: int) -> int:
            lo, hi = 0, len(offsets) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if offsets[mid] <= idx:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1  # 1-based

        def norm_index_of(idx: int) -> int:
            # full 인덱스에 대응하는 full_norm 인덱스(근사) — bisect
            import bisect

            return bisect.bisect_left(map_norm_to_full, idx)

        marks: List[Tuple[int, int, str, List[str]]] = []  # (full_idx, level, title, path_parts)
        stack: List[str] = []
        for level, title, page in toc:
            level = max(1, int(level))
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            path_parts = [p for p in stack if p]

            p0 = max(0, int(page) - 1)
            search_from_full = offsets[min(p0, len(offsets) - 1)]
            # 북마크 페이지가 1 어긋나는 경우가 있어 한 페이지 앞에서부터 찾는다
            if p0 > 0:
                search_from_full = offsets[p0 - 1]
            key = _norm_head(title)
            hit = -1
            if key:
                nstart = norm_index_of(search_from_full)
                j = full_norm.find(key, nstart)
                if j != -1 and j < len(map_norm_to_full):
                    hit = map_norm_to_full[j]
            if hit == -1:
                continue
            marks.append((hit, level, title, list(path_parts)))

        marks.sort(key=lambda x: x[0])
        # 같은 위치 중복 제거
        dedup: List[Tuple[int, int, str, List[str]]] = []
        for mk in marks:
            if dedup and mk[0] == dedup[-1][0]:
                continue
            dedup.append(mk)

        sections: List[LoadedSection] = []
        for i, (idx, level, title, path_parts) in enumerate(dedup):
            end = dedup[i + 1][0] if i + 1 < len(dedup) else len(full)
            body = full[idx:end]
            # 제목 줄 자체는 본문에서 한 번만 남긴다(중복 방지)
            text = normalize_raw(body, drop_hashtag_tail=False)
            if not text:
                continue
            sections.append(
                LoadedSection(
                    title=title,
                    path=" > ".join(path_parts),
                    text=text,
                    page_start=page_of(idx),
                    page_end=page_of(max(idx, end - 1)),
                    level=level,
                )
            )
        return sections

    def _sections_from_pages(self, pages: List[str]) -> List[LoadedSection]:
        step = max(1, self.fallback_pages_per_chunk)
        out: List[LoadedSection] = []
        for i in range(0, len(pages), step):
            block = "\n".join(pages[i : i + step])
            text = normalize_raw(block, drop_hashtag_tail=False)
            if not text:
                continue
            out.append(
                LoadedSection(
                    title=None,
                    path=None,
                    text=text,
                    page_start=i + 1,
                    page_end=min(i + step, len(pages)),
                    level=1,
                    meta={"fallback": "page_window"},
                )
            )
        return out
