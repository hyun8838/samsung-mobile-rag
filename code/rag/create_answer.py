"""답변 생성 — 설명서 근거 답변 + 근거 부족 시 삼성 공식 웹사이트 검색.

흐름:
  질문 + 고객 기기 + 검색 상위 5개 조각 → 생성 모델이 일반 텍스트로 답한다.
    첫 줄 `SUFFICIENT: true|false`, 둘째 줄 `TYPE: 방법|가능 여부|문제 해결|범위 밖`, 본문(markdown), 마지막 줄 `EVIDENCE: n,…`
  - sufficient=true  → 설명서 답변 + 본문 인용 [n] + 코드가 붙이는 "참고 근거"(모델·문서 종류·쪽)
  - sufficient=false → (allow_external=True일 때) OpenAI Responses web_search(삼성 공식 도메인만) 결과를
                        "매뉴얼 밖 정보" 머리말·출처와 함께 붙인다. 출처 도메인은 코드에서 다시 검사한다.
  - 범위 밖 질문은 SUFFICIENT: true·근거 번호 없이 짧게 안내하므로 웹 검색을 타지 않는다.
SUFFICIENT·TYPE·EVIDENCE 줄은 내부 처리용이라 본문에서 떼어 낸다(화면 표시는 app/stream_text.py).
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# 생성·외부 검색 모델. gpt-6-luna는 temperature 0을 지원하지 않아 기본값을 쓰고, 응답의 모델명·fingerprint를 기록한다.
GEN_MODEL = "gpt-6-luna"
EXTERNAL_MODEL = "gpt-6-luna"
EXTERNAL_DOMAINS = ["samsungsvc.co.kr", "samsung.com"]
SERIES_NOTICE = "※ 정확한 모델에 따라 설명·근거가 다를 수 있습니다."
EXTERNAL_HEADER_NONE = "※ 사용설명서와 삼성 공식 웹사이트에서 이 내용을 확인하지 못했습니다. 삼성전자서비스 고객센터에 문의해 주세요."
EXTERNAL_HEADER = "※ 매뉴얼 밖 정보(삼성 공식 웹사이트 검색 결과)입니다. 사용 중인 모델의 사용설명서와 다를 수 있습니다."
TYPES = ("방법", "가능여부", "문제해결", "범위밖")

SYSTEM_PROMPT = """당신은 삼성 갤럭시 기기(스마트폰·태블릿·갤럭시 북·워치·버즈) 사용 안내 보조 AI입니다.

[근거 사용]
- 아래 [매뉴얼 근거]는 고객이 쓰는 기기 모델의 사용설명서·자가수리 가이드에서 검색한 내용입니다. 이 근거만 사용하세요.
- 근거에 없는 내용은 추측하거나 일반 상식으로 보충하지 마세요.
- 사실을 서술하는 문장 끝에 근거 번호를 [1], [2][3] 형태로 붙이세요. 근거에 없는 번호는 만들지 마세요.

[근거 충분 여부 판단]
- 질문이 묻는 핵심(기능의 가능 여부, 설정 위치, 절차, 조건·수치)이 근거에 직접 나오면 sufficient=true.
- 근거가 비슷한 주제만 다루고 질문의 핵심 답이 없으면 sufficient=false. 이때 본문에는 근거로 확인되는 범위만 짧게 적고,
  "사용설명서에서 이 내용을 확인하지 못했습니다"라고 밝히세요.
- 단, 질문 유형이 범위 밖이면 이 판단을 하지 않고 아래 [질문 유형별 작성]의 범위 밖 규칙(SUFFICIENT: true)을 따르세요.

[질문 유형별 작성]
먼저 질문을 아래 네 유형 중 하나로 정하고, 그 유형의 규칙을 따르세요.
- 방법: 기능을 켜거나 설정·사용하는 방법을 묻는 질문. 단계가 하나면 첫 문장에 메뉴 경로를 쓰세요. 단계가 둘 이상이면 첫 문장에는 무엇을 하면 되는지만 짧게 쓰고, 메뉴 경로와 순서는 번호 목록에만 적으세요.
- 가능 여부: 기능·연결·사양이 되는지 묻는 질문. 첫 문장에 근거로 확인되는 가능 여부와 조건을 쓰세요.
- 문제 해결: 증상·오류·고장을 묻는 질문. 첫 문장에 가장 먼저 확인할 것을 쓰고, 근거에 나온 점검·조치 순서를 번호 목록으로 적은 뒤, 마지막 줄에 해결되지 않으면 삼성전자서비스 센터 확인이 필요할 수 있다고 안내하세요.
- 범위 밖: 인사·잡담이나 갤럭시 기기 사용과 무관한 질문. 질문 내용 자체에는 답하거나 설명하지 말고, 근거와 근거 번호 없이 1~2문장으로 응대한 뒤 기기 사용에 관한 질문을 하도록 안내하세요. 이때 SUFFICIENT: true, EVIDENCE: 없음. 범위 밖 답에는 [근거 사용]의 근거 번호 규칙, [작성 방식]의 첫 문장 직접 답·서비스센터 안내 규칙, [출력 구성]의 순서를 적용하지 않습니다.
- 유형이 애매하면 범위 밖이 아닌 다른 유형 중 가까운 것을 고르세요.

[작성 방식]
- 첫 문장에 질문에 대한 직접 답(가능 여부 또는 핵심 방법)을 쓰고, 절차가 있으면 번호 목록으로 순서대로 적으세요.
- 번호 목록을 쓸 때는 목록에 적을 메뉴 경로·단계를 첫 문장에 미리 쓰지 마세요(같은 내용을 두 번 쓰지 않기).
- 메뉴·버튼 이름은 근거에 적힌 표기 그대로 쓰세요.
- 안전·고장·수리 관련이면 삼성전자서비스 센터 확인이 필요할 수 있다고 안내하세요.
- 핵심 답 다음에, 근거에 있는 관련 선택지·주의사항이 있으면 '추가 정보' 불릿 1~3개로 덧붙이세요(형식: `**추가 정보**` 한 줄 뒤에 `- ` 불릿). 근거에 없는 내용은 쓰지 마세요. 해당 내용이 없으면 생략하세요.
- 근거 조각에서 고객 기기가 아닌 다른 모델에 한정된 설명(예: '(Fold 모델)', 'Pro 모델', 다른 모델명이 붙은 문단)은 본문과 추가 정보 어디에도 쓰지 않는다. 고객 기기에 해당하는지 근거로 확인되지 않는 기능은 추가 정보에 넣지 않는다.
- 차분한 상담 톤, 한국어.

[출력 구성]
- 순서: 핵심 답(1~2문장) → 절차 번호 목록(있을 때만) → 추가 정보(있을 때만) → 서비스 안내 1줄(문제 해결 유형이거나 안전·고장·수리 관련일 때만).
- SUFFICIENT: false이면 핵심 답 자리에 근거로 확인되는 범위와 "사용설명서에서 이 내용을 확인하지 못했습니다"를 쓰세요.
- 번호 목록, 추가 정보 불릿, 안전 주의 문장을 뺀 설명 문장은 3문장 이내로 쓰고, 근거 문장을 길게 그대로 옮기지 마세요.

[출력] JSON이 아닌 일반 텍스트로, 아래 순서를 지키세요.
- 첫 줄: SUFFICIENT: true 또는 SUFFICIENT: false (이 줄에는 다른 글자를 쓰지 마세요)
- 둘째 줄: TYPE: 방법, TYPE: 가능 여부, TYPE: 문제 해결, TYPE: 범위 밖 중 하나 (이 줄에는 다른 글자를 쓰지 마세요)
- 셋째 줄부터: 답변 본문(markdown)
- 마지막 줄: EVIDENCE: 답변에 사용한 근거 번호를 쉼표로 구분(예: EVIDENCE: 1,3). 사용한 근거가 없으면 EVIDENCE: 없음"""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def doc_label(c: Dict[str, Any]) -> str:
    kind = "자가수리 가이드" if c.get("doc_type") == "repair_guide" else "사용설명서"
    return f"{','.join(c.get('models') or []) or '모델 미상'} {kind} p.{c.get('page_number')}"


def display_label(c: Dict[str, Any], model_desc: str = "") -> str:
    """사용자에게 보이는 '참고 근거' 표기. 공용 설명서는 모델코드가 18~28개라, 3개 초과면
    고객 기기 코드(없으면 첫 코드) + '등 N개 모델 공용'으로 줄인다. 생성 모델 입력(format_contexts)은 doc_label 그대로."""
    models = c.get("models") or []
    if len(models) <= 3:
        return doc_label(c)
    kind = "자가수리 가이드" if c.get("doc_type") == "repair_guide" else "사용설명서"
    m = re.search(r"\(([^()]+)\)\s*$", model_desc)
    head = m.group(1).strip() if m and m.group(1).strip() in models else models[0]
    return f"{head} 등 {len(models)}개 모델 공용 {kind} p.{c.get('page_number')}"


_TRACKING = re.compile(r"^(utm_[a-z]+|msockid|fbclid|gclid)$", re.I)


def strip_tracking(url: str) -> str:
    """URL의 추적 파라미터(utm_*, msockid 등) 제거(웹 검색 도구가 ?utm_source=openai를 붙인다)."""
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)]
    return urlunparse(p._replace(query=urlencode(q)))


def _strip_tracking_text(text: str) -> str:
    return re.sub(r"https?://[^\s)\]]+", lambda m: strip_tracking(m.group(0)), text)


def format_contexts(chunks: Sequence[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for n, c in enumerate(chunks, 1):
        text = c["text"]
        room = max_chars - used
        if room < 200:
            break
        if len(text) > room:
            text = text[:room].rstrip() + " …(이하 생략)"
        used += len(text)
        sec = c.get("section_path") or c.get("section_title") or ""
        parts.append(f"[{n}] {doc_label(c)} / 섹션: {sec}\n{text}")
    return "\n\n".join(parts)


def _retry(fn, tries: int = 4):
    for a in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 네트워크/레이트리밋
            if a == tries or "insufficient_quota" in str(e):  # 크레딧 소진은 재시도해도 같음
                raise
            time.sleep(2 ** a)


# 파싱 규칙: 첫 비어 있지 않은 줄 = SUFFICIENT 줄(대소문자 무시, 앞뒤 공백·** 허용). 실패 = parse_ok False·sufficient False(전환).
_SUF = re.compile(r"^\**\s*SUFFICIENT\s*\**\s*:\s*\**\s*(true|false)\s*\**$", re.I)
_EVI = re.compile(r"^\**\s*EVIDENCE\s*\**\s*:\s*(.*)$", re.I)
_EXTRA_HEAD = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*추가\s*정보\s*(?:\*\*)?\s*:?\s*(?:\*\*)?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


def normalize_type(value: str) -> str:
    v = value.strip().replace("**", "").strip()
    v = re.sub(r"\.+$", "", v)
    v = re.sub(r"\s+", "", v)
    return v if v in TYPES else "무효"


def parse_output(raw: str) -> Dict[str, Any]:
    """SUFFICIENT 줄 → (끝의) EVIDENCE 줄 → (본문 첫 줄의) TYPE 줄 순서로 떼어 낸다.
    TYPE 줄이 없으면 type "없음"(본문 보존), 값이 네 유형이 아니면 "무효". `**TYPE:`처럼 다른 글자로 시작하면 떼지 않는다."""
    lines = raw.strip().splitlines()
    if not lines or not _SUF.match(lines[0].strip()):
        return {"parse_ok": False, "sufficient": False, "body": raw, "evidence": [], "evidence_line_ok": False,
                "type": "없음", "type_raw": None}
    sufficient = _SUF.match(lines[0].strip()).group(1).lower() == "true"
    rest = lines[1:]
    ev, ev_ok = [], False
    while rest and not rest[-1].strip():
        rest.pop()
    if rest and _EVI.match(rest[-1].strip()):
        ev = [int(x) for x in re.findall(r"\d+", _EVI.match(rest[-1].strip()).group(1))]
        ev_ok = True
        rest = rest[:-1]
    p = {"parse_ok": True, "sufficient": sufficient, "body": "\n".join(rest).strip(), "evidence": ev,
         "evidence_line_ok": ev_ok, "type": "없음", "type_raw": None}
    body_lines = p["body"].splitlines()
    i = next((k for k, ln in enumerate(body_lines) if ln.strip()), None)
    if i is not None and body_lines[i].strip().startswith("TYPE:"):
        p["type_raw"] = body_lines[i].strip()
        p["type"] = normalize_type(body_lines[i].strip()[len("TYPE:"):])
        p["body"] = "\n".join(body_lines[:i] + body_lines[i + 1:]).strip()
    return p


def extract_extra_info(body: str) -> List[str]:
    """본문에서 '추가 정보' 머리줄 뒤 불릿만 뽑는다(머리줄 뒤 첫 불릿 묶음, 빈 줄은 건너뜀, 불릿 아닌 줄에서 끝)."""
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if _EXTRA_HEAD.match(ln):
            out = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    continue
                m = _BULLET.match(nxt)
                if not m:
                    break
                out.append(m.group(1).strip())
            return out
    return []


def manual_answer(client, question: str, model_desc: str, chunks: Sequence[Dict[str, Any]],
                  model: str = GEN_MODEL) -> Dict[str, Any]:
    user = (f"[고객 기기] {model_desc}\n\n[매뉴얼 근거]\n{format_contexts(chunks) or '(검색된 근거 없음)'}"
            f"\n\n[고객 질문]\n{question}")

    def call():
        t0 = time.time()
        stream = client.chat.completions.create(
            model=model, max_completion_tokens=4000, stream=True, stream_options={"include_usage": True},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        parts, ttft, usage, rmodel, fp = [], None, None, None, None
        for ch in stream:
            rmodel = rmodel or getattr(ch, "model", None)
            fp = fp or getattr(ch, "system_fingerprint", None)
            if ch.usage is not None:
                usage = ch.usage
            for c in ch.choices or []:
                d = c.delta.content if c.delta else None
                if d:
                    if ttft is None:
                        ttft = int((time.time() - t0) * 1000)
                    parts.append(d)
        return "".join(parts), ttft, usage, rmodel, fp, int((time.time() - t0) * 1000)

    raw, ttft, usage, rmodel, fp, ms = _retry(call)
    p = parse_output(raw)
    body = p["body"]
    # 인용 = EVIDENCE 줄 ∪ 본문 [n] 표기
    cited = sorted(set(p["evidence"]) | {int(x) for x in re.findall(r"\[(\d{1,2})\]", body)})
    valid = set(range(1, len(chunks) + 1))
    extra = extract_extra_info(body) if p["parse_ok"] else []
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "sufficient": p["sufficient"], "type": p["type"], "type_raw": p["type_raw"], "body": body,
        "parse_ok": p["parse_ok"], "evidence_line_ok": p["evidence_line_ok"],
        "cited": [n for n in cited if n in valid], "hallucinated_citations": [n for n in cited if n not in valid],
        "extra_info": extra, "has_extra_info": bool(extra),
        "extra_head_unparsed": (not extra) and ("추가 정보" in body), "raw": raw,
        "usage": {"resolved_model": rmodel, "system_fingerprint": fp,
                  "prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None),
                  "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                  "latency_ms": ms, "ttft_ms": ttft, "stream": True},
    }


def allowed_url(url: str) -> bool:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host == "samsungsvc.co.kr" or host.endswith(".samsungsvc.co.kr"):
        return True
    return (host == "samsung.com" or host.endswith(".samsung.com")) and p.path.startswith("/sec")


def external_answer(client, question: str, model_desc: str, model: str = EXTERNAL_MODEL) -> Dict[str, Any]:
    prompt = (f"고객 기기: {model_desc}\n질문: {question}\n\n삼성 공식 웹사이트(samsungsvc.co.kr, samsung.com/sec)에서 "
              "확인되는 내용만으로 한국어로 답하세요. 확인되지 않으면 확인되지 않는다고 답하세요.")
    t0 = time.time()
    resp = _retry(lambda: client.responses.create(
        model=model, input=prompt,
        tools=[{"type": "web_search", "filters": {"allowed_domains": EXTERNAL_DOMAINS}}]))
    urls = []
    for item in resp.output or []:
        for part in getattr(item, "content", None) or []:
            for ann in getattr(part, "annotations", None) or []:
                if getattr(ann, "type", "") == "url_citation":
                    urls.append(ann.url)
    urls = list(dict.fromkeys(strip_tracking(u) for u in urls))
    return {"body": _strip_tracking_text(resp.output_text), "urls_ok": [u for u in urls if allowed_url(u)],
            "urls_rejected": [u for u in urls if not allowed_url(u)],
            "usage": {"resolved_model": resp.model, "latency_ms": int((time.time() - t0) * 1000)}}


def external_text(e: Dict[str, Any]) -> str:
    """웹 검색 결과 → 화면 문자열. 공식 도메인 밖 출처가 하나라도 있거나 공식 출처가 없으면 본문을 버린다."""
    e["accepted"] = bool(e["urls_ok"]) and not e["urls_rejected"]
    if e["accepted"]:
        return EXTERNAL_HEADER + "\n" + e["body"] + "\n\n출처\n" + "\n".join(f"- {u}" for u in e["urls_ok"])
    return EXTERNAL_HEADER_NONE


def answer(client, question: str, model_desc: str, chunks: Sequence[Dict[str, Any]], *,
           series_level: bool = False, allow_external: bool = False) -> Dict[str, Any]:
    m = manual_answer(client, question, model_desc, chunks)
    out: Dict[str, Any] = {"manual": m, "route": "manual" if m["sufficient"] else "insufficient"}
    text = m["body"]
    if m["cited"]:
        text += "\n\n참고 근거\n" + "\n".join(f"[{n}] {display_label(chunks[n - 1], model_desc)}" for n in m["cited"])
    if not m["sufficient"] and allow_external:
        e = external_answer(client, question, model_desc)
        out["external"], out["route"] = e, "external"
        ext = external_text(e)
        # 설명서에서 확인된 부분 답(인용 있을 때)은 버리지 않고 앞에 둔다.
        text = (f"[사용설명서에서 확인된 내용]\n{text}\n\n{ext}" if m["cited"] else ext)
    if series_level:
        text += "\n\n" + SERIES_NOTICE
    out["final_text"] = text
    return out
