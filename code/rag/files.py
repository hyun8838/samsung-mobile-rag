"""manifest(JSONL) → Document / Chunk 코퍼스.

이 모듈이 "전처리·정규화" 단계의 진입점이다.
전신 프로젝트(robot-vacuum-rag)의 코퍼스 모듈에서
`read_jsonl`·`write_jsonl`·`load_manual_documents`·`build_corpus`만 남겼다.

입력 계약: `config.MANUAL_MANIFEST`(JSONL, PDF 1개 = 1행). 이 파일은 별도 스크립트가 만든다.
  {"file", "doc_type", "category", "models", "filename_models", "os_guide", "rev",
   "edition_date", "source_url"}
  - PDF 경로 = user_manual → MANUALS_DIR/file, repair_guide → REPAIR_GUIDES_DIR/file
  - 파일명 비교는 NFC 정규화 후(macOS 파일명은 NFD라 안 하면 조용히 0건)
  - 없는 파일은 stderr에 경고하고 건너뛴다(중단하지 않는다)
"""
from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag import settings as C
from rag.split_text import chunk_manual
from rag.documents import LoadedDocument
from rag.pdf_reader import PdfManualLoader


def _nfc(s: Optional[str]) -> str:
    return unicodedata.normalize("NFC", s or "")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- io
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- load
def _resolve_pdf(row: Dict[str, Any]) -> Optional[Path]:
    """manifest 행 → 실제 PDF 경로. NFC/NFD 어느 쪽 파일명이든 찾는다."""
    base = C.REPAIR_GUIDES_DIR if row.get("doc_type") == "repair_guide" else C.MANUALS_DIR
    want = _nfc(row["file"])
    direct = base / row["file"]
    if direct.exists():
        return direct
    if base.is_dir():
        for p in base.iterdir():
            if _nfc(p.name) == want:        # 디스크의 NFD 이름과 NFC로 맞춰 본다
                return p
    return None


def load_manual_documents(
    manifest_path: Path = C.MANUAL_MANIFEST,
    files: Optional[Sequence[str]] = None,
) -> Tuple[List[LoadedDocument], List[str]]:
    """manifest를 읽어 PDF를 로드한다. 반환: (LoadedDocument 목록, 파일이 없어 건너뛴 manifest 파일명 목록).

    `files`를 주면 그 파일명(NFC 비교)만 대상으로 한다.
    """
    rows = read_jsonl(Path(manifest_path))
    only = {_nfc(f) for f in files} if files is not None else None
    loader = PdfManualLoader(
        fallback_pages_per_chunk=C.ManualChunkConfig().fallback_pages_per_chunk
    )
    loaded: List[LoadedDocument] = []
    missing: List[str] = []
    seen_ids = set()
    for row in rows:
        if only is not None and _nfc(row.get("file")) not in only:
            continue
        pdf = _resolve_pdf(row)
        if pdf is None:
            missing.append(_nfc(row.get("file")))
            continue
        meta = dict(row)
        meta["file_sha256"] = _sha256_file(pdf)
        ld = loader.load(pdf, doc_meta=meta)[0]
        if ld.document.doc_id in seen_ids:
            print(f"[corpus] 경고: doc_id 중복, 건너뜀: {ld.document.doc_id}", file=sys.stderr)
            continue
        seen_ids.add(ld.document.doc_id)
        loaded.append(ld)
    if missing:
        print(f"[corpus] 경고: PDF 없음 {len(missing)}건 건너뜀:\n  " + "\n  ".join(missing),
              file=sys.stderr)
    return loaded, missing


# ---------------------------------------------------------------- build
def build_corpus(out_dir: Path, files: Optional[Sequence[str]] = None,
                 manifest_path: Path = C.MANUAL_MANIFEST) -> Dict[str, Any]:
    """PDF → 문서·청크 jsonl + 실행 manifest. 출력 폴더가 이미 있으면 덮어쓰지 않고 중단한다."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"출력 폴더가 이미 있음(덮어쓰기 금지): {out_dir}")

    loaded, missing = load_manual_documents(manifest_path, files)
    cfg = C.ChunkConfig().manual
    docs: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    for ld in loaded:
        docs.append(ld.document.to_dict())
        chunks.extend(c.to_dict() for c in chunk_manual(ld, cfg))

    out_dir.mkdir(parents=True)
    n_docs = write_jsonl(out_dir / "documents.jsonl", docs)
    n_chunks = write_jsonl(out_dir / "chunks.jsonl", chunks)
    man = C.run_manifest({
        "stage": "build_corpus",
        "input_manifest": str(Path(manifest_path)),
        "input_manifest_sha256": _sha256_file(Path(manifest_path)),
        "files_filter": [_nfc(f) for f in files] if files is not None else None,
        "n_documents": n_docs,
        "n_chunks": n_chunks,
        "n_missing_files": len(missing),
        "missing_files": missing,
        "chunk_config": C.config_snapshot()["chunking"],
    })
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return man
