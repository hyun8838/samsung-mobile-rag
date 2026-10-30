"""답변 모듈·점진 표시 단위 테스트(API·모델 없음). 실행: .venv/bin/python3 tests/test_stream_text.py
 C1 표시 파서: 모든 분할의 모든 중간 상태에서 TYPE·SUFFICIENT·EVIDENCE 줄 비표시, 표시값은 항상 최종 body(answer.parse_output)의 접두,
    false·첫 줄 깨짐 = 표시 없음, 범위 밖 안내 답은 TYPE 없이 표시
 C2 등가성: 같은 raw 조각을 래퍼 없이 / StreamingClient로 받아 answer.answer() 결과 dict 전체 비교, final_text에 TYPE 없음
 C3 지침 고정: 지침 sha256 고정값"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from rag import create_answer as A  # noqa: E402
from stream_text import DisplayParser, StreamingClient  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if not cond and extra else ""))
    if not cond:
        FAILS.append(name)


A.time = NS(time=lambda: 100.0, sleep=lambda s: None)

CHUNKS = [{"chunk_id": f"c{i}", "text": f"본문{i}", "doc_type": "user_manual", "models": ["SM-S931N"], "page_number": 20 + i,
           "section_path": "x", "section_title": "x"} for i in range(1, 4)]
MAIN = "빠른 설정창에서 화면 녹화를 누르세요.[1]\n\n1. 확장 아이콘을 누르세요.[2]\n2. 녹화 시작을 누르세요.[1]"
EXTRA = "**추가 정보**\n- 저장 위치는 갤러리입니다.[3]"
RAWS = {
    "방법": "SUFFICIENT: true\nTYPE: 방법\n" + MAIN + "\n\nEVIDENCE: 1,2",
    "가능 여부(공백)": "SUFFICIENT: true\nTYPE: 가능 여부\n네, 가능합니다.[1]\n\n" + EXTRA + "\n\nEVIDENCE: 1,3",
    "TYPE 앞 빈 줄": "SUFFICIENT: true\n\nTYPE: 문제 해결\n" + MAIN + "\nEVIDENCE: 1",
    "범위 밖": "SUFFICIENT: true\nTYPE: 범위 밖\n갤럭시 기기 사용과 관련된 질문을 도와드릴게요.\nEVIDENCE: 없음",
    "TYPE 없음": "SUFFICIENT: true\n" + MAIN + "\n\nEVIDENCE: 1,2",
    "TYPE 값 무효": "SUFFICIENT: true\nTYPE: 기타\n" + MAIN + "\nEVIDENCE: 1",
    "굵은 TYPE(파서도 안 뗌)": "SUFFICIENT: true\n**TYPE:** 방법\n" + MAIN + "\nEVIDENCE: 1",
    "false": "SUFFICIENT: false\nTYPE: 방법\n사용설명서에서 이 내용을 확인하지 못했습니다.[1]\n\nEVIDENCE: 1",
    "첫 줄 깨짐": "TYPE: 방법\n" + MAIN + "\nEVIDENCE: 1",
    "TYPE 줄에서 잘림": "SUFFICIENT: true\nTYPE: 방",
    "TYPE 뒤 바로 잘림": "SUFFICIENT: true\nTYPE: 방법\n",
    "빈 출력": "",
}
NO_SHOW = {"false", "첫 줄 깨짐", "TYPE 줄에서 잘림", "TYPE 뒤 바로 잘림", "빈 출력"}
rng = random.Random(7)


def splits(s, n=40):
    out = [[s], list(s)] if s else [[]]
    for _ in range(n if len(s) > 1 else 0):
        k = rng.randint(1, min(12, len(s) - 1))
        cuts = sorted(rng.sample(range(1, len(s)), k))
        out.append([s[a:b] for a, b in zip([0] + cuts, cuts + [len(s)])])
    return out


# ---------------- C1 표시 파서
for name, raw in RAWS.items():
    po = A.parse_output(raw)
    ok_hide, ok_prefix, ok_show, any_seen = True, True, True, False
    for pcs in splits(raw):
        p, last = DisplayParser(), ""
        for x in pcs:
            r = p.feed(x)
            if r is None:
                continue
            any_seen = True
            if not r.startswith(last):
                ok_prefix = False
            last = r
            lines = [ln.strip() for ln in r.split("\n")]
            if any(ln.startswith("TYPE:") or ln.upper().startswith("SUFFICIENT") or ln.lstrip("*").strip().upper().startswith("EVIDENCE") for ln in lines):
                ok_hide = False
            if not po["body"].startswith(r) or not (po["parse_ok"] and po["sufficient"]):
                ok_prefix = False
        if name in NO_SHOW and last:
            ok_show = False
    if name not in NO_SHOW and not any_seen:
        ok_show = False
    check(f"C1 {name}: TYPE·SUFFICIENT·EVIDENCE 줄 비표시", ok_hide)
    check(f"C1 {name}: 표시값이 항상 최종 body의 접두(단조 증가)", ok_prefix)
    check(f"C1 {name}: 표시 여부 기대와 일치", ok_show)
p = DisplayParser()
for x in list(RAWS["범위 밖"]):
    p.feed(x)
check("C1 범위 밖: 안내 문장만 표시", p.shown == "갤럭시 기기 사용과 관련된 질문을 도와드릴게요.", repr(p.shown))


# ---------------- C2 등가성
def chunks_of(pieces):
    cs = [NS(model="gpt-6-luna", system_fingerprint=None, usage=None, choices=[NS(delta=NS(content=x), finish_reason=None)]) for x in pieces]
    cs.append(NS(model="gpt-6-luna", system_fingerprint="fp", usage=NS(prompt_tokens=1, completion_tokens=2), choices=[]))
    return cs


class Fake:
    def __init__(self, pieces):
        self.pieces, self.kw_log = pieces, []
        self.chat = NS(completions=NS(create=self._c))
        self.responses = NS(create=lambda **kw: (_ for _ in ()).throw(AssertionError("외부 검색 예상 안 함")))

    def _c(self, **kw):
        self.kw_log.append(kw)
        return iter(chunks_of(self.pieces))


for name in ("방법", "가능 여부(공백)", "범위 밖", "TYPE 없음"):
    raw = RAWS[name]
    for pcs in splits(raw, 5):
        base = A.answer(Fake(pcs), "q", "d", CHUNKS, allow_external=False)
        shown = []
        f2 = Fake(pcs)
        sc = StreamingClient(f2, on_text=shown.append, on_reset=lambda: None)
        got = A.answer(sc, "q", "d", CHUNKS, allow_external=False)
        same = got == base
        no_type = "TYPE:" not in got["final_text"] and all("TYPE:" not in s for s in shown)
        if not (same and no_type):
            break
    check(f"C2 {name}: 래퍼 유무와 결과 dict 동일, 화면·final_text에 TYPE 없음", same and no_type)

# ---------------- C3 지침 고정
PROMPT_SHA256 = "cb5bf0a36b79816a8aa64e78b2befe0d20761f8aba7979dc90f2614f6a986fc4"
check("C3 지침 sha256 고정값", A.PROMPT_SHA256 == PROMPT_SHA256, A.PROMPT_SHA256)

print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
sys.exit(1 if FAILS else 0)
