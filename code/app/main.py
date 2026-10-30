"""RAG를 활용한 갤럭시 기기별 사용설명서 상담 챗봇 (로컬 Streamlit).

실행: .venv/bin/python3 -m streamlit run app/main.py
화면: 사이드바 = 기기 선택(제품 종류 -> 시리즈 -> 필요 시 세부 모델)·대화 초기화·사용 방법, 본문 = 채팅.
검색·생성 경로는 chatbot_service(평가와 동일)를 그대로 호출하고, 여기서는 표시만 바꾼다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_display as AV  # noqa: E402
import devices as DC  # noqa: E402
import chatbot_service as RS  # noqa: E402
import sample_questions as SQ  # noqa: E402
from screen_text import MODEL_HINTS, USAGE_GUIDE, external_notice, md  # noqa: E402

TITLE = "RAG를 활용한 갤럭시 기기별 사용설명서 상담 챗봇"


@st.cache_resource(show_spinner=False)
def get_resources():
    return RS.get_resources()   # 앱 시작 때 시작한 백그라운드 미리 로드(chatbot_service.preload_async)와 같은 싱글턴


@st.cache_data(show_spinner=False)
def get_groups():
    return DC.load_groups()


# 최소 CSS: config.toml [theme]로 안 되는 부분만. (1) 사이드바 선택 상자 = 흰 배경 + 진한 글자 (2) 본문 상단 얇은 파랑 띠 (3) 제목 글자 파랑
STYLE = """<style>
[data-testid="stSidebar"][aria-expanded="true"] { min-width:350px; width:350px; }   /* 파랑 비율 ~30%(1200px 창 기준) */
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#FFFFFF; color:#1B1D21; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#1B1D21 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { opacity:1 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:#1B1D21; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.35); }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { background:#EEF2FB; }
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 { color:#1428A0; }
[data-testid="stHeader"] { border-top:6px solid #1428A0; }
</style>"""

# 진행 표시(기다리는 동안 안내). 생성 호출 중에는 화면을 갱신할 기회가 없어(첫 글자 전까지 콜백 없음)
# '몇 초 뒤 바뀌는' 문구 대신, 가장 오래 기다리는 두 단계(답 작성·공식 사이트 검색)의 문구 자체를 기다림 안내로 둔다.
STAGE_LABELS = {"search": "🔎 설명서에서 찾고 있어요…", "rerank": "📑 관련 내용을 고르고 있어요…",
                "generate": "✍️ 답변을 작성 중이에요. 조금만 기다려 주세요…",
                "external": "🌐 설명서에 없는 내용이라 삼성 공식 사이트에서 찾고 있어요. 조금 더 걸려요…"}
WARMUP_LABEL = "⏳ 처음이라 준비하고 있어요. 30초쯤 걸려요…"


def render_result(r: dict) -> None:
    """답변 표시: 본문(markdown) / 참고 근거(caption) / 매뉴얼 밖 정보(테두리 container). 파싱 실패 시 원문 그대로."""
    note = external_notice(r["route"], r.get("external"), r["manual"]["cited"])
    if note and note[0] != "caption":
        getattr(st, note[0])(note[1])
    p = AV.parse_final_text(r["final_text"])
    if p is None:
        st.markdown(md(r["final_text"]))
    else:
        if p["manual_mark"]:
            st.caption("사용설명서에서 확인된 내용")
        if p["body"]:
            x = AV.split_extra(p["body"])   # '추가 정보' 블록: 내용은 그대로, 본문과 시각적으로만 분리(블록 없는 답은 x=None)
            if x is None:
                st.markdown(md(p["body"]))
            else:
                if x["before"].strip():
                    st.markdown(md(x["before"].strip("\n")))
                with st.container(border=True):
                    st.markdown("**💡 추가 정보**")
                    st.markdown(md(x["bullets"]))
                if x["after"].strip():
                    st.markdown(md(x["after"].strip("\n")))
        if p["refs"]:
            st.divider()
            st.caption("📖 참고\n\n" + "\n".join(f"- {md(x)}" for x in AV.group_refs(p["refs"])))
        if p["ext_kind"] == "accepted":
            with st.container(border=True):
                st.markdown("**🌐 매뉴얼 밖 정보**")
                st.caption(AV.EXTERNAL_HEADER.lstrip("※ "))
                st.markdown(md(p["ext_body"]))
                st.caption("출처")
                for u in p["sources"]:
                    st.markdown(f"- [{u}]({u})")
        elif p["ext_kind"] == "discarded":
            msg = AV.EXTERNAL_HEADER_NONE.lstrip("※ ")
            # 부분 답(위쪽에 매뉴얼 근거 표시) 뒤에 오면 "이 내용을 확인하지 못했습니다"가 위 답과 모순되어 보이므로 앱 쪽에서 범위를 한정한다(원문 final_text는 그대로 보관).
            st.info(("위 답에 포함되지 않은 나머지 내용: " + msg) if p["manual_mark"] else msg)
    # 관련 없는 질문(TYPE=범위 밖)은 설명서를 쓰지 않은 안내 답이라 원문 상자·'사용설명서 근거' 꼬리말을 보이지 않는다.
    oos = r["manual"].get("type") == "범위밖"
    # 근거 원문 상자: 사용자용 이름, 점수·순위 숫자와 섹션 이름(위치 표시가 틀린 청크가 있음)은 빼고 문서 종류·쪽·원문만.
    cited = set(r["manual"]["cited"])
    title = (f"챗봇이 찾아본 설명서 원문 보기 ({len(r['top5'])}개)" if r["top5"] else "찾은 설명서 내용 없음")
    if not oos:
        with st.expander(title):
            if not r["top5"]:
                st.write("선택한 기기의 설명서에서 관련 내용을 찾지 못했습니다.")
            for n, t in enumerate(r["top5"], 1):
                c = t["chunk"]
                kind = "자가수리 가이드" if c.get("doc_type") == "repair_guide" else "사용설명서"
                mark = " · 답에 인용" if n in cited else ""
                st.markdown(f"**[{n}] {kind} p.{c.get('page_number')}**{mark}")
                st.text(c["text"][:500] + (" ..." if len(c["text"]) > 500 else ""))
    if (r.get("stream") or {}).get("finish_reason") == "length":   # 생성 스트림이 토큰 상한으로 끝남(stream_text 기록)
        st.caption("※ 답이 길이 제한으로 잘렸을 수 있습니다")
    tail = "" if oos else f" · {AV.route_label(r['route'])}"
    st.caption(f"⏱ {r['timing_s']['total']:.1f}초{tail}")


def select_device(groups):
    """사이드바 선택 위젯. 하위 위젯 key에 상위 선택값을 넣어, 상위가 바뀌면 하위 선택이 자동으로 초기화된다."""
    st.markdown("**내 기기 선택**")
    cat = st.selectbox("제품 종류", list(DC.CATEGORY_LABELS), index=None, key="sel_cat", format_func=DC.CATEGORY_LABELS.get,
                       placeholder="제품 종류 선택")
    ser = st.selectbox("시리즈", DC.series_in(groups, cat) if cat else [], index=None, key=f"sel_ser|{cat}",
                       placeholder="제품 종류를 먼저 고르세요" if not cat else "시리즈 선택", disabled=not cat)
    if not (cat and ser):
        return None
    cands = DC.groups_in(groups, cat, ser)
    labels = DC.detail_labels(cands, ser, cat)
    if len(cands) == 1:
        g = cands[0]
        labels = {g["group_id"]: ser}  # 변형 목록(블루투스·Wi-Fi 등)은 이름·model_desc에 넣지 않는다
    else:  # 같은 시리즈 안에 문서 집합이 다른 모델이 있으면 세부 모델 선택 필수
        gid = st.selectbox("세부 모델", list(labels), index=None, key=f"sel_det|{cat}|{ser}", format_func=labels.get,
                           placeholder="세부 모델 선택(필수)")
        if not gid:
            st.caption("이 시리즈는 모델마다 설명서가 달라 세부 모델까지 골라야 합니다.")
            return None
        g = next(x for x in cands if x["group_id"] == gid)
    return DC.make_device(g, ser, labels[g["group_id"]])


def code_caption(dev: dict) -> str:
    codes = dev["model_codes"]
    shown = ", ".join(codes[:6]) + (f" 외 {len(codes) - 6}개" if len(codes) > 6 else "")
    same = "이 설명서를 함께 쓰는 모델코드: " if len(codes) > 1 else "모델코드: "
    return f"{same}{shown}"


st.set_page_config(page_title=TITLE, page_icon=None, layout="centered", initial_sidebar_state="expanded")
RS.preload_async()  # 모델 미리 로드(백그라운드): 첫 질문 전에 임베더·리랭커를 올려 첫 질문 지연을 줄인다. 검색·생성 조건은 불변.
st.markdown(STYLE, unsafe_allow_html=True)
groups = get_groups()
st.session_state.setdefault("histories", {})
st.session_state.setdefault("pill_nonce", {})

with st.sidebar:
    st.markdown(f"### {TITLE}")
    dev = select_device(groups)
    if dev:
        st.markdown(f"선택: **{dev['label']}**")
        st.caption(code_caption(dev))
        if dev["category"] in MODEL_HINTS:
            st.caption(MODEL_HINTS[dev["category"]])
    reset_slot = st.empty()
    with st.expander("사용 방법", expanded=not dev):   # 기기를 고르기 전(첫 화면)에는 펼쳐 둔다
        st.markdown(md(USAGE_GUIDE))

if not dev:
    with st.chat_message("assistant", avatar="🙂"):
        st.markdown("안녕하세요! 먼저 왼쪽 사이드바에서 **제품 종류 → 시리즈**를 골라 내 기기를 알려 주세요. 선택한 기기의 공식 사용설명서·자가수리 가이드에서만 답을 찾아드려요.")
    st.chat_input("먼저 사이드바에서 기기를 선택하세요", disabled=True)
    st.stop()

key = DC.device_key(dev)
history = st.session_state.histories.setdefault(key, [])
avatar = DC.AVATARS.get(dev["category"], "🤖")
with reset_slot:
    if st.button("대화 초기화", key="reset", use_container_width=True):
        st.session_state.histories[key] = []
        st.session_state.pill_nonce[key] = st.session_state.pill_nonce.get(key, 0) + 1
        st.rerun()

with st.chat_message("assistant", avatar=avatar):
    st.markdown(f"안녕하세요! **{dev['label']}** 사용설명서에서 답을 찾아드릴게요. 궁금한 점을 물어보세요.")
for h in history:
    with st.chat_message("user", avatar=DC.USER_AVATAR):
        st.write(h["question"])
    with st.chat_message("assistant", avatar=avatar):
        if h["result"] is not None:
            render_result(h["result"])
        elif h.get("error"):
            st.error(h["error"])
        else:  # 생성 중 화면이 다시 그려져(기기 변경·재입력 등) 스크립트가 중단된 질문
            st.warning("이 질문은 답변 생성 중에 중단되어 답이 없습니다(생성 중 화면 조작 시 발생). 다시 질문해 주세요.")


def unasked_pool(extra=()) -> list:
    """이 기기에서 표시 가능한 추천 풀 중 아직 묻지 않은 질문(이 기기 대화의 모든 질문 = 클릭·직접 입력 모두 '물은 질문'). 승인된 풀 그대로."""
    return SQ.unasked(dev["category"], dev["group_id"], [h["question"] for h in history] + list(extra), n=10 ** 6)


def show_pills(slot):
    """남은 추천 질문 최대 3개를 pills로. 선택 즉시 질문으로 쓰고, 질문마다 위젯 key(nonce)를 바꿔 선택값이 남아 반복되지 않게 한다. 다 썼으면 숨김."""
    pool = unasked_pool()[:3]
    if not pool:
        slot.empty()
        return None
    nonce = st.session_state.pill_nonce.get(key, 0)
    with slot.container():
        return st.pills("추천 질문", pool, selection_mode="single", key=f"pill|{key}|{nonce}", label_visibility="collapsed")


pill_slot = st.empty()
picked = show_pills(pill_slot)
pool0 = unasked_pool([picked] if picked else [])   # 방금 클릭한 질문은 이번 run에서 곧 물어보므로 placeholder 후보에서 미리 뺀다
# placeholder: 아직 묻지 않은 질문만. 화면 pills(앞 3개) 밖의 후보를 우선, 없으면 남은 첫 질문. 풀 소진 시 일반 문구.
ph_q = (pool0[3:] or pool0 or [None])[0]
placeholder = ("예: " + ph_q) if ph_q else "선택한 기기에 대해 궁금한 점을 물어보세요"
typed = st.chat_input(placeholder)
q = typed or picked
if q:
    pill_slot.empty()
    st.session_state.pill_nonce[key] = st.session_state.pill_nonce.get(key, 0) + 1
    with st.chat_message("user", avatar=DC.USER_AVATAR):
        st.write(q)
    entry = {"question": q, "result": None, "error": None}
    history.append(entry)  # 받는 즉시 기록: 생성 중 rerun으로 스크립트가 끊겨도 질문이 사라지지 않음
    with st.chat_message("assistant", avatar=avatar):
        status = st.status(STAGE_LABELS["search"], expanded=True)
        stream_slot = st.empty()   # 스트리밍 도중 임시 표시(완료되면 비우고 slot에 기존 최종 표시)
        slot = st.empty()
        _last = [0.0]

        def on_text(t):   # 해제된 answer 전체를 최대 ~20회/초로 갱신(표시 전용)
            now = time.perf_counter()
            if now - _last[0] >= 0.05:
                _last[0] = now
                stream_slot.markdown(md(t) + " ▌")
            _latest[0] = t

        _latest = [""]

        def on_reset():
            _latest[0] = ""
            stream_slot.empty()
        try:
            if not RS.is_ready():
                status.update(label=WARMUP_LABEL)
            res = get_resources()
            status.update(label=STAGE_LABELS["search"])
            r = RS.ask(res, q, dev, on_stage=lambda name: status.update(label=STAGE_LABELS[name]), on_text=on_text, on_reset=on_reset)
            stream_slot.empty()   # 스트리밍 임시 텍스트 제거 -> 아래 기존 최종 표시로 교체
            status.update(label="완료", state="complete", expanded=False)
            with slot.container():
                render_result(r)
            entry["result"] = r
        except Exception as e:  # noqa: BLE001
            stream_slot.empty()
            entry["error"] = f"오류: {type(e).__name__}: {str(e)[:300]}"
            status.update(label="오류", state="error", expanded=False)
            slot.error(entry["error"])
    # 답 아래에 아직 묻지 않은 추천 질문을 다시 보여준다(새 nonce로 만든 위젯이라 이번 run에서는 선택 이벤트가 없다).
    show_pills(st.empty())
    # 직접 입력한 질문이 현재 placeholder 질문과 같았다면 입력창을 다시 그려 placeholder를 갱신한다(클릭한 질문은 위에서 미리 제외했으므로 재실행 없음).
    if typed and ph_q and SQ.same_question(ph_q, typed):
        st.rerun()
