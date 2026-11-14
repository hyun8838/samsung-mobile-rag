"""갤럭시 사용설명서 검색 MCP 서버(stdio). 실행: .venv-mcp/bin/python3 mcp_server/server.py

다른 AI 앱(호스트)이 이 서버의 도구로 갤럭시 설명서를 검색하고, 답변은 호스트가 만든다.
검색 경로 = 챗봇과 같다(app/chatbot_service.retrieve_top5): arctic-ko dense top10(선택 기기 필터) -> FT 리랭커 -> top5.
  벡터 저장소는 앱과 같다: 환경변수 GALAXY_VECTOR_BACKEND = chroma(기본) | faiss | numpy.
  - 근거 형식 = rag.create_answer.format_contexts와 같은 문자열(문서·쪽 / 섹션, 6000자 절단). 번호는 대화 안에서 이어 매기고 같은 청크는 같은 번호.
  - 서버 지침(MCP instructions) = 질문 유형별 규칙이 없는 지침(추가 정보·다른 모델 설명 제외 포함, SUFFICIENT/EVIDENCE 줄 출력)에
    도구 사용 문구 3가지([도구] 절, "search_manual이 돌려준 [매뉴얼 근거]", 근거 부족 시 search_official_web 호출)를 더한 것.
    챗봇의 질문 유형 줄(TYPE)을 숨기는 장치가 MCP에는 없어 챗봇 지침과 다르게 둔다.
  - 외부 결과 = 챗봇 answer()와 같은 머리말·출처 줄과 수락 규칙(rag.create_answer.external_text).
도구:
  find_device(query)                                기기 이름·모델코드로 카탈로그 검색
  search_manual(model_code, query)                  선택 기기 사용설명서 상위 5개 근거
  search_official_web(model_code, question, device) 삼성 공식 웹사이트 검색
stdout은 MCP 프로토콜 전용 -> 로그는 stderr로만.
"""
from __future__ import annotations

import sys
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

import chatbot_service as RS  # noqa: E402
import devices as DC  # noqa: E402
from rag import create_answer as A  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402  (mcp 2.x: FastMCP -> MCPServer)

INSTRUCTIONS = """당신은 삼성 갤럭시 기기(스마트폰·태블릿·갤럭시 북·워치·버즈) 사용 안내 보조 AI입니다.

[도구]
- search_manual(model_code, query): 고객 기기 모델의 사용설명서·자가수리 가이드에서 검색한 [매뉴얼 근거]를 번호와 함께 돌려줍니다. model_code는 [고객 기기] 괄호 안의 코드를 쓰세요.
- search_official_web(model_code, question, device): 삼성 공식 웹사이트 검색 결과를 돌려줍니다. device에는 [고객 기기]를 그대로 쓰세요.

[근거 사용]
- search_manual이 돌려준 [매뉴얼 근거]는 고객이 쓰는 기기 모델의 사용설명서·자가수리 가이드에서 검색한 내용입니다. 이 근거만 사용하세요.
- 근거에 없는 내용은 추측하거나 일반 상식으로 보충하지 마세요.
- 사실을 서술하는 문장 끝에 근거 번호를 [1], [2][3] 형태로 붙이세요. 근거에 없는 번호는 만들지 마세요.

[근거 충분 여부 판단]
- 질문이 묻는 핵심(기능의 가능 여부, 설정 위치, 절차, 조건·수치)이 근거에 직접 나오면 sufficient=true.
- 근거가 비슷한 주제만 다루고 질문의 핵심 답이 없으면 sufficient=false. 이때 본문에는 근거로 확인되는 범위만 짧게 적고,
  "사용설명서에서 이 내용을 확인하지 못했습니다"라고 밝히세요.
- sufficient=false이면 search_official_web을 호출하고, 돌려받은 text를 고치지 말고 본문 끝(EVIDENCE 줄 앞)에 빈 줄 하나를 두고 덧붙이세요.

[작성 방식]
- 첫 문장에 질문에 대한 직접 답(가능 여부 또는 핵심 방법)을 쓰고, 절차가 있으면 번호 목록으로 순서대로 적으세요.
- 메뉴·버튼 이름은 근거에 적힌 표기 그대로 쓰세요.
- 안전·고장·수리 관련이면 삼성전자서비스 센터 확인이 필요할 수 있다고 안내하세요.
- 핵심 답 다음에, 근거에 있는 관련 선택지·주의사항이 있으면 '추가 정보' 불릿 1~3개로 덧붙이세요(형식: `**추가 정보**` 한 줄 뒤에 `- ` 불릿). 근거에 없는 내용은 쓰지 마세요. 해당 내용이 없으면 생략하세요.
- 근거 조각에서 고객 기기가 아닌 다른 모델에 한정된 설명(예: '(Fold 모델)', 'Pro 모델', 다른 모델명이 붙은 문단)은 본문과 추가 정보 어디에도 쓰지 않는다. 고객 기기에 해당하는지 근거로 확인되지 않는 기능은 추가 정보에 넣지 않는다.
- 차분한 상담 톤, 한국어.

[출력] JSON이 아닌 일반 텍스트로, 아래 순서를 지키세요.
- 첫 줄: SUFFICIENT: true 또는 SUFFICIENT: false (이 줄에는 다른 글자를 쓰지 마세요)
- 둘째 줄부터: 답변 본문(markdown)
- 마지막 줄: EVIDENCE: 답변에 사용한 근거 번호를 쉼표로 구분(예: EVIDENCE: 1,3). 사용한 근거가 없으면 EVIDENCE: 없음"""

mcp = MCPServer("galaxy-manual", instructions=INSTRUCTIONS)
_GROUPS = DC.load_groups()
_CITE: Dict[str, int] = {}  # chunk_id -> 대화 안 근거 번호
_CITE_LOCK = threading.Lock()  # mcp 2.x는 동기 도구를 스레드에서 실행 → 동시 검색 시 번호 경합 방지


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _nonempty(s: str, name: str) -> str:
    s = _nfc(s).strip()
    if not s:
        raise ValueError(f"{name}이(가) 비어 있습니다")
    return s


def _device(model_code: str) -> Dict[str, Any]:
    dev = DC.find_by_code(_GROUPS, _nfc(model_code).strip().upper())
    if dev is None:
        raise ValueError(f"카탈로그에 없는 모델코드입니다: {model_code}. find_device로 먼저 찾으세요.")
    return dev


def format_contexts_numbered(chunks: Sequence[Dict[str, Any]], numbers: Sequence[int], max_chars: int = 6000) -> str:
    """answer.format_contexts와 같은 규칙(절단·라벨·섹션), 번호만 numbers로. numbers=1..n이면 결과가 문자 그대로 같다(테스트)."""
    parts, used = [], 0
    for n, c in zip(numbers, chunks):
        text = c["text"]
        room = max_chars - used
        if room < 200:
            break
        if len(text) > room:
            text = text[:room].rstrip() + " …(이하 생략)"
        used += len(text)
        sec = c.get("section_path") or c.get("section_title") or ""
        parts.append(f"[{n}] {A.doc_label(c)} / 섹션: {sec}\n{text}")
    return "\n\n".join(parts)


_EN = [("galaxy", "갤럭시"), ("buds", "버즈"), ("watch", "워치"), ("ultra", "울트라"), ("pro", "프로"), ("tab", "탭"),
       ("book", "북"), ("edge", "엣지"), ("classic", "클래식"), ("active", "액티브"), ("plus", "+")]


def _name_key(s: str) -> str:
    """이름 비교 키: 소문자·영어 표기→한국어·공백 제거·앞의 '갤럭시' 제거('버즈3 프로' = '갤럭시 버즈3 프로' = 'Galaxy Buds3 Pro')."""
    k = _nfc(s).lower()
    for a, b in _EN:
        k = k.replace(a, b)
    k = "".join(k.split())
    return k[len("갤럭시"):] if k.startswith("갤럭시") else k


def _code_key(s: str) -> str:
    """코드 비교 키: 공백 제거·대문자, 'SM R630'·'SMR630' → 'SM-R630'."""
    k = "".join(_nfc(s).split()).upper()
    return "SM-" + k[2:] if k.startswith("SM") and not k.startswith("SM-") else k


def _series_candidates(q: str) -> List[Dict[str, Any]]:
    """이름 질의(q = 공백 제거·소문자)에 맞는 시리즈별 후보. 같은 시리즈가 여러 묶음에 있으면 묶음마다 따로(세부 이름으로 구분)."""
    out = []
    for g in _GROUPS:
        cs = g.get("code_series", {})
        for ser in g["series"]:
            key = _name_key(ser)
            if not (q == key or q in key or key in q):
                continue
            code = next((m for m in g["model_codes"] if cs.get(m) == ser), g["primary_code"])
            cands = DC.groups_in(_GROUPS, g["category"], ser)
            name = ser if len(cands) == 1 else DC.detail_labels(cands, ser, g["category"])[g["group_id"]]
            out.append({"model_code": code, "name": name, "series": ser, "category": g["category"],
                        "match": "name_exact" if q == key else "name_partial", "model_codes": g["model_codes"]})
    return out


@mcp.tool()
def find_device(query: str) -> Dict[str, Any]:
    """갤럭시 기기 이름(예: '갤럭시 S25', '버즈3 프로') 또는 모델코드(예: 'SM-S931N')로 사용설명서가 있는 기기를 찾는다.
    candidates의 model_code를 search_manual에 넘긴다. 모델코드를 정확히 넣으면 그 코드 하나를 돌려준다.
    ambiguous=true(정확 일치 1개가 아님: 여러 기기에 맞거나 부분 일치뿐)면 임의로 고르지 말고 고객에게 어떤 기기인지 확인한다. 후보 최대 10개(이름순)."""
    raw = _nonempty(query, "query")
    q = _name_key(raw)
    dev = DC.find_by_code(_GROUPS, _code_key(raw))
    if dev is not None:  # 모델코드 정확 입력: 입력 코드와 그 코드의 시리즈 보존
        cands = [{"model_code": dev["model_code"], "name": dev["name"] or DC.display_label(dev), "series": dev["series"],
                  "category": dev["category"], "match": "code", "model_codes": dev["model_codes"]}]
    else:
        cands = _series_candidates(q)
        ck = _code_key(raw)
        codes_hit = [m for g in _GROUPS for m in g["model_codes"] if len(ck) >= 3 and ck in m]  # 부분 코드(예: 'R63')
        for m in codes_hit:
            d = DC.find_by_code(_GROUPS, m)
            cands.append({"model_code": m, "name": d["name"] or DC.display_label(d), "series": d["series"],
                          "category": d["category"], "match": "code_partial", "model_codes": d["model_codes"]})
        order = {"name_exact": 0, "name_partial": 1, "code_partial": 2}
        seen, uniq = set(), []
        for c in sorted(cands, key=lambda c: (order[c["match"]], c["name"], c["model_code"])):  # 결정적 순서
            k = (c["name"], c["model_code"])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        cands = uniq
    n_exact = sum(c["match"] in ("code", "name_exact") for c in cands)
    # 확인 필요(ambiguous): 정확 일치가 정확히 1개가 아니면 전부. 부분 일치 1개(예: '워치5' → 워치5 프로)도 다른 기기일 수 있어 확인 필요
    ambiguous = bool(cands) and n_exact != 1
    if not cands:
        note = "카탈로그에서 찾지 못했습니다. 모델코드로 다시 찾아 보세요. 비슷한 기기로 바꾸지 마세요."
    elif ambiguous and n_exact == 0:
        note = "입력과 정확히 일치하는 기기가 없습니다. 후보는 다른 기기일 수 있으니 고객에게 어떤 기기인지(모델코드) 확인하세요."
    elif ambiguous:
        note = "이름에 맞는 기기가 여럿입니다. 임의로 고르지 말고 고객에게 어떤 기기인지(모델코드) 확인하세요."
    else:
        note = ""
    return {"candidates": cands[:10], "n_candidates": len(cands), "ambiguous": ambiguous, "note": note}


def web_device_desc(model_code: str, device: Optional[str]) -> str:
    """search_official_web에 넘길 기기 문구: 호스트가 준 device 그대로, 없으면 입력 코드의 시리즈 + 입력 코드."""
    if device and device.strip():
        return _nfc(device).strip()
    return DC.model_desc(_device(model_code))


@mcp.tool()
def search_manual(model_code: str, query: str) -> Dict[str, Any]:
    """고객 기기(model_code)의 한국어 공식 사용설명서·자가수리 가이드에서 query와 관련된 근거 상위 5개를 돌려준다.
    text = 번호가 붙은 [매뉴얼 근거]. 답변의 사실 문장 끝에 이 번호를 [n]으로 붙인다."""
    dev = _device(model_code)
    res = RS.get_resources()
    top5 = RS.retrieve_top5(res, _nonempty(query, "query"), dev)
    chunks = [t["chunk"] for t in top5]
    nums = []
    with _CITE_LOCK:
        for c in chunks:
            cid = _nfc(c["chunk_id"])
            if cid not in _CITE:
                _CITE[cid] = len(_CITE) + 1
            nums.append(_CITE[cid])
    text = format_contexts_numbered(chunks, nums) or "(검색된 근거 없음)"
    ev = [{"n": n, "chunk_id": c["chunk_id"]} for n, c in zip(nums, chunks)]
    return {"text": text, "evidence": ev}


@mcp.tool()
def search_official_web(model_code: str, question: str, device: Optional[str] = None) -> Dict[str, Any]:
    """사용설명서 근거로 질문의 핵심 답을 확인하지 못했을 때만 쓴다. 삼성 공식 웹사이트(samsungsvc.co.kr, samsung.com/sec)에서
    확인되는 내용을 머리말·출처와 함께 text로 돌려준다. device = [고객 기기] 표기(없으면 입력 코드의 시리즈 이름과 코드)."""
    _device(model_code)  # 없는 코드면 오류
    desc = web_device_desc(model_code, device)
    res = RS.get_resources()
    e = A.external_answer(res["client"], _nonempty(question, "question"), desc)
    text = A.external_text(e)  # 챗봇 answer()와 같은 수락 규칙·머리말·출처 줄
    return {"accepted": e["accepted"], "text": text}


if __name__ == "__main__":
    RS.preload_async()  # 챗봇과 같은 백그라운드 미리 로드(첫 호출 지연 단축, 결과 무관)
    mcp.run()
