"""추천 질문 풀·표시 규칙·평가셋 질문 겹침 검사. 화면에 뜨는 질문은 (1) 평가셋 259문항과 문자열·bge-m3 근접중복(코사인) 검사 통과,
(2) 실제 API 1회 검증(매뉴얼 경로·인용 있음), (3) 선택 기기 묶음 청크에 필요 키워드가 있을 때만 표시.
검사 범위는 이 세 가지뿐이며 '평가셋과 무관한 질문'이라는 보장은 아니다."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
nfc = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731
_norm = lambda s: re.sub(r"[\s?？.!,]+", "", nfc(s))  # noqa: E731

COSINE_REPLACE_AT = 0.85   # 사전 고정(결과 보기 전): 평가 질문과 bge-m3 코사인 >= 0.85면 교체
AVAIL_PATH = ROOT / "data/app/suggest_availability.json"


def eval_question_texts() -> List[str]:
    """data/eval 아래 모든 jsonl의 question 필드(동결 154·양성 97 포함 상위집합) 고유 질문 원문."""
    seen = {}
    for f in sorted((ROOT / "data/eval").rglob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            try:
                q = json.loads(l).get("question")
            except Exception:  # noqa: BLE001
                continue
            if isinstance(q, str):
                seen.setdefault(_norm(q), nfc(q))
    return list(seen.values())


def eval_questions() -> set:
    return {_norm(q) for q in eval_question_texts()}


# 풀: (질문, 필요 키워드: 키워드 중 하나라도 든 묶음 청크가 min(기본 1)개 이상이면 표시; 빈 목록 = 공통). 앞에서부터 표시 가능한 것 3개를 쓴다.
POOL: Dict[str, List[dict]] = {
    "phone": [
        {"q": "화면 잠금 방법을 바꾸려면 어떻게 해요?", "kw": []},
        {"q": "배터리를 오래 쓰는 방법이 있나요?", "kw": []},
        {"q": "화면 글자 크기를 키우려면 어떻게 해요?", "kw": []},
    ],
    "tablet": [
        {"q": "S펜으로 메모하려면 어떻게 해요?", "kw": ["S펜 버튼"], "min": 3},
        {"q": "삼성 덱스는 어떻게 사용하나요?", "kw": ["DeX"], "min": 3},
        {"q": "태블릿 화면을 TV에 연결해서 보여주려면 어떻게 해요?", "kw": ["Smart View"]},
        {"q": "화면 녹화는 어떻게 하나요?", "kw": []},
        {"q": "태블릿 소프트웨어는 어떻게 업데이트하나요?", "kw": []},
    ],
    "book": [
        {"q": "터치패드에서 쓸 수 있는 제스처가 있나요?", "kw": ["터치패드"]},
        {"q": "노트북 배터리 충전 한도를 정할 수 있나요?", "kw": ["배터리 보호"]},
        {"q": "무선 인터넷(Wi-Fi)에 연결하려면 어떻게 해요?", "kw": ["Wi-Fi"]},
        {"q": "블루투스 기기는 어떻게 연결하나요?", "kw": ["블루투스"]},
    ],
    "watch": [
        {"q": "워치 페이스는 어떻게 바꾸나요?", "kw": ["시계 화면"]},
        {"q": "수면 측정은 어떻게 하나요?", "kw": ["수면"]},
        {"q": "워치로 심박수를 재려면 어떻게 해요?", "kw": ["심박수"]},
    ],
    "buds": [
        {"q": "노이즈 캔슬링은 어떻게 켜요?", "kw": ["노이즈 캔슬링"]},
        {"q": "버즈 터치로 음악을 일시정지할 수 있나요?", "kw": ["일시 정지", "일시정지"]},
        {"q": "버즈 배터리 잔량은 어디서 확인하나요?", "kw": ["배터리 잔량", "배터리 상태"]},
    ],
}


# 폐기한 후보(기록용: 근접중복·주제 겹침·답 범위 문제로 뺌). 근접중복 검사에는 함께 넣어 값을 남긴다.
RETIRED = [
    ("buds", "버즈 착용 감지 기능은 어떻게 켜고 끄나요?"),   # API 검증: 일부 기기에서 외부 경로로 감
    ("buds", "버즈 볼륨을 조절하려면 어떻게 해요?"),   # API 검증: 외부 경로로 감
    ("buds", "충전 케이스로 이어버드를 충전하려면 어떻게 해요?"),   # 다른 질문으로 대체
    ("tablet", "S펜으로 화면에 바로 메모하려면 어떻게 해요?"),     # 문구를 고쳐 다시 넣음
    ("buds", "버즈를 폰에 블루투스로 연결하려면 어떻게 해요?"),   # 평가 질문과 같은 주제
    ("book", "갤럭시 북에서 화면을 캡처하려면 어떻게 해요?"),     # 답이 다른 기능(스크린샷 아이콘)으로 감
    ("phone", "알림이 올 때 진동만 오게 하려면 어떻게 해요?"),    # 답 범위가 어긋날 수 있음
    # 아래 3개는 풀에서 제외(실제 API 검증을 하지 않은 후보; 근접중복 값만 기록)
    ("tablet", "화면 밝기를 조절하려면 어떻게 해요?"),
    ("watch", "워치를 충전하려면 어떻게 해요?"),
    ("buds", "버즈 소프트웨어는 어떻게 업데이트하나요?"),
]


def pick(category: str, group_id: Optional[str], avail: Optional[dict] = None, n: int = 3) -> List[str]:
    """묶음에서 키워드 조건을 만족하는 풀 질문을 앞에서부터 n개. 부족하면(공통 질문 포함) 가능한 만큼만. 가용성 파일이 없으면 공통 질문만."""
    avail = avail if avail is not None else load_availability()
    row = (avail or {}).get(group_id or "", {})
    out = []
    for it in POOL.get(category, []):
        if not it["kw"] or row.get(it["q"]):
            out.append(it["q"])
        if len(out) >= n:
            break
    return out


def load_availability() -> dict:
    if AVAIL_PATH.exists():
        return json.loads(AVAIL_PATH.read_text(encoding="utf-8")).get("groups", {})
    return {}


def unasked(category: str, group_id: Optional[str], asked: List[str], n: int = 3, avail: Optional[dict] = None) -> List[str]:
    """그 기기에서 표시 가능한 풀(pick과 같은 키워드 조건·승인 순서) 중 이미 물은 질문을 뺀 앞 n개.
    asked = 이 기기 대화의 모든 질문(클릭·직접 입력 구분 없음). 비교는 NFC + 공백·문장부호 정규화(_norm)."""
    done = {_norm(q) for q in asked}
    return [q for q in pick(category, group_id, avail=avail, n=10 ** 6) if _norm(q) not in done][:n]


def same_question(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)
