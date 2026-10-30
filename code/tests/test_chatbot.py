"""Streamlit AppTest + 실제 chatbot_service.ask / rag.create_answer / StreamingClient / 앱 표시 경로(가짜 OpenAI 클라이언트·가짜 검색만).
(1) 점진 표시 콜백이 실제로 불리고 (2) 최종 화면에 임시 텍스트(▌)·SUFFICIENT/EVIDENCE/TYPE 줄이 남지 않으며
(3) '추가 정보' 블록이 분리 표시되고 (4) false/깨짐은 외부 경로로 가며 (5) 범위 밖 안내 답에는 원문 상자가 없는지 확인한다.
실행: .venv/bin/python3 tests/test_chatbot.py   (API·모델 로드 없음)"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402
import devices as DC  # noqa: E402
import chatbot_service as RS  # noqa: E402

import streamlit.testing.v1.element_tree as ET  # noqa: E402


def _indices(self):   # AppTest 1.50 ButtonGroup(pills) None 값 변환 오류 보정(test_chatbot.py와 동일, 테스트 한정)
    v = self.value
    v = [] if v is None else (v if isinstance(v, list) else [v])
    return [i for i, o in enumerate(self.options) if o.content in v]


ET.ButtonGroup.indices = property(_indices)
CHUNKS = [{"chunk_id": f"c{i}", "text": f"본문{i}", "doc_type": "user_manual", "models": ["SM-S931N"], "page_number": 20 + i,
           "section_path": "x", "section_title": "x"} for i in range(1, 4)]
MAIN = "빠른 설정창에서 화면 녹화를 누르세요.[1]\n\n1. 확장 아이콘을 누르세요.[2]"
RAWS = {
    "추가": "SUFFICIENT: true\nTYPE: 방법\n" + MAIN + "\n\n**추가 정보**\n- 녹화 중 소리는 확장 아이콘에서 바꿀 수 있습니다.[2]\n- 저장 위치는 갤러리입니다.[3]\n\nEVIDENCE: 1,2,3",
    "무추가": "SUFFICIENT: true\nTYPE: 방법\n" + MAIN + "\n\nEVIDENCE: 1,2",
    "거짓": "SUFFICIENT: false\nTYPE: 방법\n사용설명서에서 이 내용을 확인하지 못했습니다.[1]\n\nEVIDENCE: 1",
    "길이": "SUFFICIENT: true\nTYPE: 방법\n" + MAIN[:30],
    "범위": "SUFFICIENT: true\nTYPE: 범위 밖\n갤럭시 기기 사용과 관련된 질문을 도와드릴게요.\nEVIDENCE: 없음",
    "깨짐": "네 가능합니다 SUFFICIENT: true\n" + MAIN,
}
EVENTS = {}


class Closable:
    def __init__(self, items):
        self.items, self.closed = items, False

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self):
        outer = self
        self.chat = NS(completions=NS(create=lambda **kw: outer._c(**kw)))
        self.responses = NS(create=lambda **kw: NS(output=[NS(content=[NS(annotations=[NS(type="url_citation", url="https://www.samsungsvc.co.kr/info/fee")])])],
                                                   output_text="외부 본문입니다.", model="gpt-6-luna"))

    def _c(self, **kw):
        user = kw["messages"][1]["content"]
        raw = next((v for k, v in RAWS.items() if f"질문-{k}" in user), None)
        items = [NS(model="gpt-6-luna", system_fingerprint="f", usage=None, choices=[NS(delta=NS(content=raw[i:i + 6]), finish_reason=None)]) for i in range(0, len(raw), 6)]
        if "길이" in user:   # 토큰 상한으로 끝난 스트림: 마지막 내용 조각에 finish_reason="length"
            items[-1].choices[0].finish_reason = "length"
        else:
            items[-1].choices[0].finish_reason = "stop"
        items.append(NS(model="gpt-6-luna", system_fingerprint="f", usage=NS(prompt_tokens=1, completion_tokens=2), choices=[]))
        return Closable(items)


CLIENT = FakeClient()
RS.preload_async = lambda: None
RS.get_resources = lambda: {"load_seconds": 0, "client": CLIENT}
RS.is_ready = lambda: True
def fake_retrieve(res, q, dev, on_stage=None):
    for st_ in ("search", "rerank"):
        on_stage and on_stage(st_)
    return [{"chunk": c, "rerank_score": 1.0 - i * 0.1, "dense_score": 0.5, "dense_rank": i + 1} for i, c in enumerate(CHUNKS)]


RS.retrieve_top5 = fake_retrieve
_real_ask = RS.ask


def spy_ask(res, question, device, on_stage=None, on_text=None, on_reset=None):
    ev = EVENTS.setdefault(question, [])

    def t(x):
        ev.append("t")
        on_text(x)

    def r():
        ev.append("r")
        on_reset()
    out = _real_ask(res, question, device, on_stage=on_stage, on_text=t, on_reset=r)
    return out


RS.ask = spy_ask
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name)


# 화면 흐름 검사는 실제 제조사 자료 대신 시험용 기기 목록을 사용한다.
_TEST_GROUPS = [{"group_id": "test-buds", "category": "buds", "series": {"갤럭시 버즈4": []},
                 "primary_code": "SM-R540", "model_codes": ["SM-R540"],
                 "code_series": {"SM-R540": "갤럭시 버즈4"}, "n_chunks": len(CHUNKS)}]
DC.load_groups = lambda *args, **kwargs: _TEST_GROUPS
_G = DC.load_groups()
APP = str(ROOT / "app/main.py")


def screen(at):
    """대화(chat_message) 영역만. 사이드바 '사용 방법' 문구에는 '추가 정보' 등이 들어 있어 범위에서 뺀다."""
    parts = []
    for m in at.chat_message:
        parts += [str(getattr(x, "value", "")) for x in m.markdown] + [c.value for c in m.caption] + [i.value for i in m.info] + [e.value for e in m.error] + [w.value for w in m.warning]
    return " ".join(parts)


def start():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="sel_cat").select("buds").run()
    at.selectbox(key="sel_ser|buds").select("갤럭시 버즈4").run()
    return at


at = start()
at.chat_input[0].set_value("질문-추가").run()
sc = screen(at)
check("F1 추가 정보 답: 예외 없음, 점진 표시 콜백 실제 호출(t 이벤트 다수), 시작 reset 1회뿐", not at.exception and EVENTS["질문-추가"].count("t") >= 3 and EVENTS["질문-추가"].count("r") == 1 and EVENTS["질문-추가"][0] == "r")
check("F1 최종 화면에 ▌·SUFFICIENT·EVIDENCE·TYPE 없음", "▌" not in sc and "SUFFICIENT" not in sc and "EVIDENCE" not in sc and "TYPE" not in sc)
check("F1 본문·추가 정보 불릿·참고 근거 표시", "빠른 설정창에서 화면 녹화를 누르세요" in sc and "저장 위치는 갤러리입니다" in sc and "📖 참고" in sc)
check("F1 '추가 정보' 블록이 별도 영역으로 표시('💡 추가 정보' 라벨, 원문 '**추가 정보**' 줄은 본문에 중복 없음)", "💡 추가 정보" in sc and sc.count("추가 정보") == 1)

at = start()
at.chat_input[0].set_value("질문-무추가").run()
sc = screen(at)
check("F2 추가 정보 없는 답: 표시 정상, '추가 정보' 영역 없음, 임시 텍스트 없음", not at.exception and "💡" not in sc and "▌" not in sc and "EVIDENCE" not in sc and "확장 아이콘을 누르세요" in sc)

at = start()
at.chat_input[0].set_value("질문-거짓").run()
sc = screen(at)
check("F3 false: 표시 이벤트 없음(t 없음), 종료 reset 후 외부 경로 결과 표시, 임시 텍스트 없음",
      not at.exception and "t" not in EVENTS["질문-거짓"] and EVENTS["질문-거짓"].count("r") == 2 and "외부 본문입니다" in sc and "▌" not in sc and "SUFFICIENT" not in sc)

at = start()
at.chat_input[0].set_value("질문-깨짐").run()
sc = screen(at)
check("F4 첫 줄 깨짐: 표시 이벤트 없음, 외부 경로(전환), 임시 텍스트 없음", not at.exception and "t" not in EVENTS["질문-깨짐"] and "외부 본문입니다" in sc and "▌" not in sc)

at = start()
at.chat_input[0].set_value("질문-길이").run()
sc = screen(at)
check("M2 finish_reason=length: 잘림 캡션 표시 + 답 본문은 그대로 + 임시 텍스트 없음", not at.exception and "※ 답이 길이 제한으로 잘렸을 수 있습니다" in sc and "빠른 설정창에서 화면" in sc and "▌" not in sc)
at = start()
at.chat_input[0].set_value("질문-무추가").run()
sc = screen(at)
check("M2 finish_reason=stop: 잘림 캡션 없음", "길이 제한" not in sc)
check("M3 응답 시간 캡션: 형식 표시 없음, 근거 원문 상자에 점수·순위·섹션 없음", "⏱" in sc and "· 사용설명서 근거" in sc and "· 답에 인용" in sc and "형식 v" not in sc and "리랭크" not in sc and "dense" not in sc and "섹션:" not in sc)
at = start()
at.chat_input[0].set_value("질문-범위").run()
sc = screen(at)
check("F5 범위 밖: 안내 문장 표시, TYPE 없음, 외부 경로 아님, 임시 텍스트 없음", not at.exception and "질문을 도와드릴게요" in sc and "TYPE" not in sc and "외부 본문입니다" not in sc and "▌" not in sc)
check("F5 범위 밖: 원문 상자(답에 인용·원문 보기) 없음, 시간 줄에 '사용설명서 근거' 없음", "· 답에 인용" not in sc and "원문" not in " ".join(e.label for e in at.expander) and "· 사용설명서 근거" not in sc and "⏱" in sc)
check("전체: 어떤 질문에서도 화면에 TYPE 줄 없음", all("TYPE" not in screen(x) for x in [at]))

print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
