"""final_text -> 표시용 구조(본문 / 참고 근거 / 외부 정보) 파서. Streamlit 무관, 단위 테스트 대상.

rag.create_answer.answer가 붙이는 고정 문자열에만 기댄다(프롬프트·생성 경로는 건드리지 않음):
  "[사용설명서에서 확인된 내용]\\n" (부분 답 + 외부 병기일 때 앞머리), "\\n\\n참고 근거\\n[n] ...",
  EXTERNAL_HEADER + 본문 + "\\n\\n출처\\n- URL...", 또는 EXTERNAL_HEADER_NONE(외부 폐기).
무손실 보장: 파싱 결과를 다시 이어붙여 원문과 정확히 같지 않으면 None을 돌려주고, 호출자는 원문 그대로 표시한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.create_answer import EXTERNAL_HEADER, EXTERNAL_HEADER_NONE, _BULLET, _EXTRA_HEAD  # noqa: E402  (추가 정보 추출과 같은 규칙)

MANUAL_MARK = "[사용설명서에서 확인된 내용]\n"
REF_SEP = "\n\n참고 근거\n"
SRC_SEP = "\n\n출처\n"
_REF = re.compile(r"^\[\d{1,2}\] \S.*$")
_SRC = re.compile(r"^- (https?://\S+)$")


def _rebuild(p: Dict) -> str:
    s = (MANUAL_MARK if p["manual_mark"] else "") + p["body"]
    if p["refs"]:
        s += REF_SEP + "\n".join(p["refs"])
    if p["ext_kind"]:
        ext = EXTERNAL_HEADER if p["ext_kind"] == "accepted" else EXTERNAL_HEADER_NONE
        if p["ext_kind"] == "accepted":
            ext += "\n" + p["ext_body"] + SRC_SEP + "\n".join(f"- {u}" for u in p["sources"])
        s += ("\n\n" if p["body"] or p["manual_mark"] else "") + ext
    return s


def parse_final_text(text: str) -> Optional[Dict]:
    """성공: {manual_mark, body, refs[list[str]], ext_kind(None|'accepted'|'discarded'), ext_body, sources[list[str]]}. 실패: None."""
    try:
        p = {"manual_mark": False, "body": "", "refs": [], "ext_kind": None, "ext_body": "", "sources": []}
        rest = text
        # 1) 외부 영역 분리
        idx = [(rest.find(h), k, h) for h, k in ((EXTERNAL_HEADER, "accepted"), (EXTERNAL_HEADER_NONE, "discarded")) if rest.find(h) >= 0]
        if idx:
            i, kind, h = min(idx)
            ext_part, rest = rest[i:], rest[:i]
            if kind == "discarded":
                if ext_part != h:
                    return None
            else:
                tail = ext_part[len(h) + 1:]
                if not ext_part.startswith(h + "\n") or SRC_SEP not in tail:
                    return None
                ext_body, src_txt = tail.rsplit(SRC_SEP, 1)
                srcs = []
                for ln in src_txt.split("\n"):
                    m = _SRC.match(ln)
                    if not m:
                        return None
                    srcs.append(m.group(1))
                p["ext_body"], p["sources"] = ext_body, srcs
            p["ext_kind"] = kind
            if rest.endswith("\n\n"):
                rest = rest[:-2]
            elif rest:
                return None  # 머리말 앞에 빈 줄 구분이 없으면 형식 불일치
        # 2) 매뉴얼 부분
        if rest.startswith(MANUAL_MARK):
            p["manual_mark"], rest = True, rest[len(MANUAL_MARK):]
        if REF_SEP in rest:
            body, ref_txt = rest.rsplit(REF_SEP, 1)
            refs = ref_txt.split("\n")
            if not all(_REF.match(r) for r in refs):
                return None
            p["body"], p["refs"] = body, refs
        else:
            p["body"] = rest
        if _rebuild(p) != text:
            return None
        return p
    except Exception:  # noqa: BLE001
        return None


_REF_PARTS = re.compile(r"^\[(\d{1,2})\] (.+?) p\.(\S+)$")


def group_refs(refs: List[str]) -> List[str]:
    """표시용: 같은 문서(라벨)의 참고 근거를 한 줄로 합친다. 예 "… 사용설명서 p.21 [1][2] · p.19 [3]".

    쪽마다 그 쪽을 인용한 번호를 붙여 [n]↔쪽 대응을 유지한다. 근거가 하나뿐인 문서와 형식이 다른 줄은 그대로 둔다.
    """
    groups: Dict[str, Dict] = {}
    order: List = []
    for r in refs:
        m = _REF_PARTS.match(r)
        if not m:
            order.append(("raw", r))
            continue
        n, label, page = m.groups()
        if label not in groups:
            groups[label] = {"raw": [], "pages": {}}
            order.append(("grp", label))
        g = groups[label]
        g["raw"].append(r)
        g["pages"].setdefault(page, []).append(n)
    out = []
    for kind, v in order:
        if kind == "raw":
            out.append(v)
        else:
            g = groups[v]
            if len(g["raw"]) == 1:
                out.append(g["raw"][0])
            else:
                out.append(f"{v} " + " · ".join(f"p.{p} " + "".join(f"[{n}]" for n in ns) for p, ns in g["pages"].items()))
    return out


def route_label(route: str) -> str:
    """응답 시간 caption 꼬리말."""
    return {"manual": "사용설명서 근거", "external": "공식 웹 검색 포함", "insufficient": "근거 부족"}.get(route, route)


def split_extra(body: str) -> Optional[Dict]:
    """본문의 '추가 정보' 블록 분리(표시 전용, 내용 불변). 머리줄 뒤에 이어지는 불릿(사이 빈 줄 허용)까지가 블록.
    성공: {before, head, bullets, after}; before + head + bullets + after 를 이으면 body와 정확히 같다(아니면 None -> 호출자는 본문 그대로 표시).
    머리줄이 없거나 뒤에 불릿이 없으면 None."""
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if not _EXTRA_HEAD.match(ln):
            continue
        j, last = i + 1, i   # last = 마지막 불릿 줄 위치
        while j < len(lines):
            if _BULLET.match(lines[j]):
                last = j
            elif lines[j].strip():
                break
            j += 1
        if last == i:
            return None
        pre = "\n".join(lines[:i])
        d = {"before": pre + "\n" if i else "", "head": lines[i] + "\n",
             "bullets": "\n".join(lines[i + 1:last + 1]), "after": ("\n" + "\n".join(lines[last + 1:])) if last + 1 < len(lines) else ""}
        return d if d["before"] + d["head"] + d["bullets"] + d["after"] == body else None
    return None
