"""매뉴얼 청크의 `section_label`(구 `manual_section_type`) 규칙 기반 라벨링.

라벨 값은 schema.MANUAL_SECTION_TYPES 8종(usage·troubleshooting·maintenance·safety·spec·
error_code·install·other)이다. 삼성 모바일 사용설명서·자가수리 가이드의 목차 용어에 맞춰
키워드를 정했다. 키워드 목록은 실제 목차로 보정했다.

라벨링 방식: **법적 고지 → 앱 장 주의 사항 → 섹션 경로(H1>H2>H3) → 본문 보조(규격 우선).** 순서가 곧 우선순위다
(예: "안전을 위해 주의하기 > 제품을 사용할 때"는 usage가 아니라 safety로 가야 한다).

한계(정직하게): 이건 사람이 만든 규칙표지 학습된 분류기가 아니다.
정확도는 표본 수동 검수로만 확인한다.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# (label, 우선순위 순서대로, 키워드)
RULES: List[Tuple[str, List[str]]] = [
    ("safety", ["안전", "주의 사항", "주의사항", "경고", "금지 사항", "금지사항",
                 "준수 사항", "준수사항", "위험", "감전", "화재", "폭발",
                 "safety", "warning", "caution"]),
    ("error_code", ["오류 코드", "에러 코드", "오류코드", "에러코드", "오류 메시지", "오류 알림",
                     "error code", "error codes"]),
    ("troubleshooting", ["문제 해결", "문제해결", "자가 진단", "자가진단", "고장", "증상",
                          "서비스 센터에 문의하기", "문의하기", "확인 사항", "조치",
                          "해결", "이상", "troubleshoot", "faq", "자주 하는 질문", "자주하는 질문"]),
    ("maintenance", ["분리하기", "재조립하기", "재조립", "부품 교체", "배터리 교체", "교체",
                      "수리", "탈거", "분해", "조립 해제", "손질", "세척", "관리", "보관",
                      "유지", "소모품", "repair", "replace", "disassembl", "reassembl",
                      "maintenance", "care"]),
    ("spec", ["사양", "제품 정보", "제품정보", "규격", "정격", "소비 전력", "소비전력",
               "인증 정보", "규제 정보", "제품 보증", "보증", "specification", "spec",
               "warranty"]),
    ("install", ["설치", "조립", "구성품", "각 부분의 명칭", "각부의 명칭", "부품", "제품 구성",
                  "포장", "시작하기", "처음 사용", "유심", "SIM", "충전기", "액세서리", "부속품",
                  "installation", "setup", "what's in the box"]),
    # usage는 사실상 폴백에 가깝지만 명시 키워드도 함께 둔다(가장 마지막에 평가됨).
    ("usage", ["사용", "사용법", "기능", "설정", "앱", "연결", "충전", "스마트싱스", "빅스비",
                "카메라", "삼성 계정", "모드", "조작",
                "how to use", "using", "operation", "app"]),
]


# 상표·저작권·오픈 소스 고지는 섹션 제목과 무관하게 other (예: "제품 사용 시 알아두기" 아래 상표권 고지)
LEGAL_MARKERS = ["등록 상표", "등록상표", "상표권", "저작권"]
# 본문 보조 판정: 본문 앞부분에 "규격"이 있으면 spec을 먼저 준다 — 규격 표의 "보관 시 온도"가 maintenance로
# 가는 것을 막는다. "보증" 등 넓은 spec 키워드까지 앞당기면 수리 가이드 절차가 spec으로 가서 쓰지 않는다.
APP_CHAPTERS = ["앱과 기능", "앱 및 기능"]
SPEC_BODY_FIRST = ["규격"]


def label_section(section_path: Optional[str], section_title: Optional[str], text: str) -> str:
    body_head = text[:150].lower()        # 고지문이 청크 맨 앞에 올 때만(본문 중간의 "저작권" 언급은 제외)
    if any(m in body_head for m in LEGAL_MARKERS):
        return "other"
    hay_head = " ".join(filter(None, [section_path or "", section_title or ""])).lower()
    # 앱·기능 설명 장 안의 "측정 시 주의 사항"(헬스 앱 등)은 기능 사용 안내 → usage.
    # 다른 장의 주의 사항("제품 수리 시 주의 사항", "알레르기 주의 사항")은 safety 유지.
    if any(k in hay_head for k in APP_CHAPTERS) and not any(k in hay_head for k in ("안전", "경고")):
        if any(k in hay_head for k in ("주의 사항", "주의사항")):
            return "usage"
    for label, kws in RULES:
        for kw in kws:
            if kw.lower() in hay_head:
                return label
    # 본문 보조 판정 — 에러코드 표기가 실제로 들어 있으면 error_code로 승격
    if re.search(r"(오류|에러|error)\s*(코드|code)?\s*[:\-]?\s*[A-Za-z]?\d{1,3}", text):
        return "error_code"
    body = text[:400].lower()
    if any(k in body for k in SPEC_BODY_FIRST):
        return "spec"
    for label, kws in RULES:
        for kw in kws:
            if kw.lower() in body:
                return label
    return "other"
