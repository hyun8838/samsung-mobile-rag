"""매뉴얼 청킹 — 섹션(목차) 단위가 1순위.

다음 4개 원칙을 구현한다.
  1. 목차·제목·소제목·문제해결 항목 단위로 나눈다  → TOC 섹션 경계 사용
  2. 에러코드와 해결방법을 같은 청크에 유지한다     → 섹션 내부 분할 시 '증상/원인' 블록 경계 우선
  3. 페이지 번호를 반드시 보존한다                  → Chunk.page_number / page_range
  4. 주의·경고·유지관리에 별도 메타데이터           → Chunk.section_label

크기 정책(문자 기준, config.ManualChunkConfig):
  - min_chars 미만 섹션은 **같은 부모(H1) 아래 인접 섹션과 병합**한다.
    (목차가 매우 잘게 쪼개진 매뉴얼에서 20자짜리 섹션이 대량 생기기 때문)
  - max_chars 초과 섹션만 분할하고, 분할할 때만 overlap을 준다.
    섹션 경계를 넘는 overlap은 만들지 않는다(서로 다른 절차가 한 청크에 섞이는 것을 막는다).
"""
from __future__ import annotations

import hashlib
import re
from typing import List

from rag.settings import ManualChunkConfig
from rag.documents import LoadedDocument, LoadedSection
from rag.documents import Chunk
from rag.text import normalize_sparse
from rag.section_names import label_section

# 매뉴얼 본문 안에서 우선적으로 끊고 싶은 경계(번호 절차, 불릿, 증상 블록)
_SOFT_BREAK = re.compile(r"\n(?=(?:\d+[\s\t]|[•‣·▪–-]\s|[가-힣A-Za-z][^\n]{0,30}\n))")


def _split_long(text: str, cfg: ManualChunkConfig) -> List[str]:
    """max_chars를 넘는 섹션을 문단/절차 경계 우선으로 분할."""
    if len(text) <= cfg.max_chars:
        return [text]
    # 1차: 빈 줄 문단
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) == 1:
        blocks = [b for b in _SOFT_BREAK.split(text) if b.strip()]
    if len(blocks) == 1:
        blocks = [b for b in text.split("\n") if b.strip()]

    out: List[str] = []
    cur = ""
    for b in blocks:
        if cur and len(cur) + len(b) + 1 > cfg.target_chars:
            out.append(cur.strip())
            tail = cur[-cfg.overlap_chars:] if cfg.overlap_chars > 0 else ""
            cur = (tail + "\n" + b) if tail else b
        else:
            cur = (cur + "\n" + b) if cur else b
    if cur.strip():
        out.append(cur.strip())

    # 그래도 max를 넘는 조각은 하드 컷
    final: List[str] = []
    for piece in out:
        while len(piece) > cfg.max_chars:
            final.append(piece[: cfg.max_chars])
            piece = piece[cfg.max_chars - cfg.overlap_chars :]
        if piece.strip():
            final.append(piece.strip())
    return final


def _merge_small(sections: List[LoadedSection], cfg: ManualChunkConfig) -> List[LoadedSection]:
    """min_chars 미만 섹션을 같은 최상위(H1) 그룹 안에서 뒤쪽과 병합."""
    merged: List[LoadedSection] = []
    for s in sections:
        top = (s.path or "").split(" > ")[0]
        if merged:
            prev = merged[-1]
            prev_top = (prev.path or "").split(" > ")[0]
            if len(prev.text) < cfg.min_chars and prev_top == top:
                prev.text = (prev.text + "\n" + s.text).strip()
                prev.page_end = s.page_end or prev.page_end
                # 제목은 상위(더 짧은 경로) 쪽을 유지
                if s.path and prev.path and len(s.path) < len(prev.path):
                    prev.title, prev.path, prev.level = s.title, s.path, s.level
                continue
        merged.append(
            LoadedSection(
                title=s.title, path=s.path, text=s.text,
                page_start=s.page_start, page_end=s.page_end, level=s.level, meta=dict(s.meta),
            )
        )
    # 마지막 조각이 너무 짧으면 직전과 합침
    if len(merged) >= 2 and len(merged[-1].text) < cfg.min_chars:
        merged[-2].text = (merged[-2].text + "\n" + merged[-1].text).strip()
        merged[-2].page_end = merged[-1].page_end or merged[-2].page_end
        merged.pop()
    return merged


def chunk_manual(loaded: LoadedDocument, cfg: ManualChunkConfig) -> List[Chunk]:
    doc = loaded.document
    sections = _merge_small(loaded.sections, cfg)
    chunks: List[Chunk] = []
    ci = 0
    for s in sections:
        pieces = _split_long(s.text, cfg)
        for piece in pieces:
            if len(piece.strip()) < 20:
                continue
            heading = s.path or s.title or ""
            body = piece
            if cfg.keep_heading_prefix and heading:
                # 청크 단독으로 검색됐을 때 "어느 섹션의 내용인지"가 남아야 근거 제시가 가능하다.
                if not (s.title and body.startswith(s.title)):
                    body = f"[{heading}]\n{piece}"
            sec_label = label_section(s.path, s.title, piece)
            cid = f"{doc.doc_id}#c{ci:04d}"
            normalized = normalize_sparse(body)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    text=body,
                    text_normalized=normalized,
                    # 모델이 다른 문서의 동일 문장 수를 세기 위한 해시(근접중복 실측용)
                    text_hash=hashlib.sha1(normalized.encode("utf-8")).hexdigest(),
                    category=doc.category,
                    models=list(doc.models),
                    filename_models=list(doc.filename_models),
                    os_guide=doc.os_guide,
                    rev=doc.rev,
                    edition_date=doc.edition_date,
                    source_file=doc.source_file,
                    file_sha256=doc.file_sha256,
                    source_url=doc.source_url,
                    language=doc.language,
                    is_official=doc.is_official,
                    n_chars=len(body),
                    chunk_index_in_doc=ci,
                    section_title=s.title,
                    section_path=s.path,
                    section_label=sec_label,
                    page_number=s.page_start,
                    page_range=[s.page_start, s.page_end] if s.page_start else None,
                )
            )
            ci += 1
    return chunks
