"""사이드바 '사용 방법' 확인(API·모델 로드 없음). 사이드바에는 사용법만 두고 평가 수치·한계는 넣지 않는다.
실행: .venv/bin/python3 tests/test_help_text.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import screen_text as U  # noqa: E402


def main() -> int:
    g = U.USAGE_GUIDE
    for k in ("기기 고르기", "질문하기", "답 읽기", "📖 참고", "💡 추가 정보", "🌐 매뉴얼 밖 정보", "대화 초기화", "삼성전자서비스 센터", "규제 정보", "새로 고치면", "챗봇이 찾아본 설명서 원문 보기", "답에 인용"):
        assert k in g, f"사용 방법에 '{k}' 없음"
    for bad in ("%", "평가", "판정", "기준", "신뢰구간", "형식 v", "v3", "v4", "사전", "리랭커", "R@5", "한계"):
        assert bad not in g, f"사용 방법에 분석·평가 표현 '{bad}'"
    assert not re.search(r"\d+/\d+", g), "사용 방법에 비율 수치"
    assert not hasattr(U, "limits_text") and not hasattr(U, "LIMITS"), "한계 요약 문구가 남음"
    src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
    assert "md(USAGE_GUIDE)" in src and "NOTICE" not in src and "한계 요약" not in src and "limits_text" not in src
    assert not hasattr(U, "NOTICE"), "사이드바에 별도 고지 문구를 두지 않음"
    # 모델 번호 안내: 버즈는 설명서 근거가 없어 없음, 나머지는 제품별 경로
    assert set(U.MODEL_HINTS) == {"phone", "tablet", "watch", "book"} and "휴대전화 정보" in U.MODEL_HINTS["phone"] and "뒷면" in U.MODEL_HINTS["book"]
    assert "MODEL_HINTS[dev[\"category\"]]" in src and "MODEL_HINT =" not in src
    # 화면 라벨과 안내 문구가 같은지(안내에 적은 아이콘·이름을 실제 화면 코드가 쓰는지)
    for label in ("📖 참고", "💡 추가 정보", "🌐 매뉴얼 밖 정보", "대화 초기화", "챗봇이 찾아본 설명서 원문 보기", "답에 인용"):
        assert label in src, f"화면 코드에 '{label}' 없음"
    print("ALL PASS test_usage_guide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
