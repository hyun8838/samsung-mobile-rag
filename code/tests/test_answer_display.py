"""answer_display 단위 테스트(모의 final_text 4종 + 파싱 실패 + 무손실). 실행: .venv/bin/python3 tests/test_answer_display.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from answer_display import EXTERNAL_HEADER, EXTERNAL_HEADER_NONE, parse_final_text  # noqa: E402

MANUAL = "빠른 설정창에서 화면 녹화를 누르세요.[2]\n\n1. 확장 아이콘을 누르세요.[2]\n2. 녹화 시작을 누르세요.[1]"
REFS = "[1] SM-S711N 사용설명서 p.26\n[2] SM-S711N 사용설명서 p.27"
EXT_BODY = "공식 안내에 따르면 수리비는 서비스센터에서 확인할 수 있습니다."
URLS = ["https://www.samsungsvc.co.kr/info/fee?tab=service", "https://www.samsung.com/sec/support/model/SM-R420NZAAKOO/"]
EXT = EXTERNAL_HEADER + "\n" + EXT_BODY + "\n\n출처\n" + "\n".join(f"- {u}" for u in URLS)
CASES = {
    "1_manual": MANUAL + "\n\n참고 근거\n" + REFS,
    "2_partial_plus_external": "[사용설명서에서 확인된 내용]\n" + MANUAL + "\n\n참고 근거\n" + REFS + "\n\n" + EXT,
    "3_external_only": EXT,
    "4_discarded": EXTERNAL_HEADER_NONE,
    "5_discarded_after_partial": "[사용설명서에서 확인된 내용]\n" + MANUAL + "\n\n참고 근거\n" + REFS + "\n\n" + EXTERNAL_HEADER_NONE,
}
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name)


p = parse_final_text(CASES["1_manual"])
check("1 manual: body/refs", p and p["body"] == MANUAL and len(p["refs"]) == 2 and p["ext_kind"] is None and not p["manual_mark"])
p = parse_final_text(CASES["2_partial_plus_external"])
check("2 partial+external", p and p["manual_mark"] and p["body"] == MANUAL and len(p["refs"]) == 2 and p["ext_kind"] == "accepted"
      and p["ext_body"] == EXT_BODY and p["sources"] == URLS)
p = parse_final_text(CASES["3_external_only"])
check("3 external only", p and p["body"] == "" and not p["refs"] and p["ext_kind"] == "accepted" and p["sources"] == URLS)
p = parse_final_text(CASES["4_discarded"])
check("4 discarded", p and p["ext_kind"] == "discarded" and p["body"] == "")
p = parse_final_text(CASES["5_discarded_after_partial"])
check("5 discarded after partial", p and p["ext_kind"] == "discarded" and p["manual_mark"] and len(p["refs"]) == 2)
# 파싱 실패 -> None(원문 표시), 정보 손실 없음
check("fail: 출처 형식 깨짐", parse_final_text(EXTERNAL_HEADER + "\n본문\n\n출처\n- not a url") is None)
check("fail: 참고 근거 줄 형식 깨짐", parse_final_text("본문\n\n참고 근거\n그냥 문장") is None)
check("fail: 머리말 앞 구분 없음", parse_final_text("앞글" + EXT) is None)
check("plain text (구분자 없음) -> body만", (lambda q: q and q["body"] == "그냥 답" and not q["refs"])(parse_final_text("그냥 답")))
# 무손실: 성공한 파싱은 반드시 원문으로 복원
from answer_display import _rebuild  # noqa: E402
check("lossless rebuild (5종)", all(_rebuild(parse_final_text(t)) == t for t in CASES.values()))
# ---- final_text(추가 정보 불릿 포함): 파싱·무손실 재조립·추가 정보 블록 분리
from answer_display import split_extra  # noqa: E402

EXTRA_BODY = ("빠른 설정창에서 화면 녹화를 누르세요.[2]\n\n1. 확장 아이콘을 누르세요.[2]\n2. 녹화 시작을 누르세요.[1]\n\n"
           "**추가 정보**\n- 녹화 중 소리 설정은 확장 아이콘에서 바꿀 수 있습니다.[1]\n- 저장 위치는 갤러리입니다.[2]")
EXTRA_CASES = {
    "extra_1_manual_extra": EXTRA_BODY + "\n\n참고 근거\n" + REFS,
    "extra_2_partial_extra_plus_external": "[사용설명서에서 확인된 내용]\n" + EXTRA_BODY + "\n\n참고 근거\n" + REFS + "\n\n" + EXT,
    "extra_3_extra_discarded": "[사용설명서에서 확인된 내용]\n" + EXTRA_BODY + "\n\n참고 근거\n" + REFS + "\n\n" + EXTERNAL_HEADER_NONE,
    "extra_4_no_extra": MANUAL + "\n\n참고 근거\n" + REFS,
    "extra_5_extra_without_refs": EXTRA_BODY,
}
for k, t in EXTRA_CASES.items():
    q = parse_final_text(t)
    check(f"{k}: 파싱 성공 + 무손실", q is not None and _rebuild(q) == t)
q = parse_final_text(EXTRA_CASES["extra_1_manual_extra"])
check("manual: 본문에 추가 정보 불릿 보존, 참고 근거 2개 분리", q["body"] == EXTRA_BODY and len(q["refs"]) == 2 and "**추가 정보**" in q["body"])
q = parse_final_text(EXTRA_CASES["extra_2_partial_extra_plus_external"])
check("부분+외부: manual_mark·본문(불릿 포함)·외부 분리", q["manual_mark"] and q["body"] == EXTRA_BODY and q["ext_kind"] == "accepted" and q["sources"] == URLS)
x = split_extra(EXTRA_BODY)
check("split_extra: before/bullets 분리, 재조립 동일", x and x["before"].endswith("\n\n") and x["bullets"].count("\n- ") == 1
      and x["bullets"].startswith("- 녹화 중") and x["after"] == "" and x["before"] + x["head"] + x["bullets"] + x["after"] == EXTRA_BODY)
check("split_extra: 추가 정보 없는 본문 -> None", split_extra(MANUAL) is None)
check("split_extra: 머리줄만 있고 불릿 없음 -> None(원문 그대로 표시)", split_extra("답.\n\n**추가 정보**\n문장만 있음") is None)
for body in ["**추가 정보**\n- 가\n- 나", "답\n\n추가 정보\n\n- 가\n\n- 나\n\n끝 문장", "답\n\n## 추가 정보:\n1. 가\n2. 나", "답\n\n**추가 정보**\n* 가\n• 나\n\n다음 문단"]:
    x = split_extra(body)
    check(f"split_extra 무손실 {body[:14]!r}", x and x["before"] + x["head"] + x["bullets"] + x["after"] == body and x["bullets"].strip())
x = split_extra("답\n\n추가 정보\n\n- 가\n\n- 나\n\n끝 문장")
check("split_extra: 불릿 뒤 문장은 after로", x["after"].strip() == "끝 문장" and "끝 문장" not in x["bullets"])
print("ALL PASS" if ok else "SOME FAIL")
def test_group_refs():
    refs = ["[1] SM-R540 등 4개 모델 공용 사용설명서 p.21", "[2] SM-R540 등 4개 모델 공용 사용설명서 p.21",
            "[3] SM-R540 등 4개 모델 공용 사용설명서 p.19", "[4] SM-R540 자가수리 가이드 p.5", "기타 형식 줄"]
    from answer_display import group_refs
    got = group_refs(refs)
    assert got == ["SM-R540 등 4개 모델 공용 사용설명서 p.21 [1][2] · p.19 [3]", "[4] SM-R540 자가수리 가이드 p.5", "기타 형식 줄"], got
    print("PASS group_refs")
    return True


ok = test_group_refs() and ok
sys.exit(0 if ok else 1)
