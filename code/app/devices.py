"""기기 목록 로더(Streamlit 무관). data/app/device_catalog.json을 읽는다.

구조: groups = 문서 집합(doc_id)이 같은 모델코드 묶음. 선택 흐름 = 제품 종류 -> 시리즈 -> (시리즈 안에 묶음이 둘 이상일 때만) 세부 모델.
검색 필터는 선택한 묶음의 모델코드 전체(models=[...])를 쓴다(묶음 안 코드는 청크 집합이 동일: 카탈로그 생성 시 assert).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/app/device_catalog.json"

# 인덱스 category 값 -> 화면 표시 품목명(순서 = 선택 목록 순서)
CATEGORY_LABELS = {"phone": "스마트폰", "tablet": "태블릿", "book": "갤럭시 북", "watch": "워치", "buds": "버즈"}
AVATARS = {"phone": "📱", "tablet": "📋", "book": "💻", "watch": "⌚", "buds": "🎧"}
USER_AVATAR = "🙂"

nfc = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_groups(path: Path = CATALOG_PATH) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 - 로컬 data/ 폴더가 필요합니다")
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    for g in groups:
        g["model_codes"] = [nfc(m) for m in g["model_codes"]]
        g["primary_code"] = nfc(g["primary_code"])
    return groups


def series_in(groups: List[Dict], category: str) -> List[str]:
    return sorted({s for g in groups if g["category"] == category for s in g["series"]}, key=_natural)


def groups_in(groups: List[Dict], category: str, series: str) -> List[Dict]:
    return [g for g in groups if g["category"] == category and series in g["series"]]


def variant_text(group: Dict, series: str) -> str:
    return " / ".join(group["series"][series])


def detail_labels(cands: List[Dict], series: str, category: str) -> Dict[str, str]:
    """세부 모델 선택 항목: group_id -> 표시 이름. 시리즈명+변형(폰: S24+ / S24 울트라, 그 외: 시리즈 (블루투스, 40mm / 44mm)).
    같은 이름이 겹치면 모델코드를 덧붙인다."""
    raw = {}
    for g in cands:
        vs = g["series"][series]
        if not vs:
            raw[g["group_id"]] = series
        elif category == "phone":
            raw[g["group_id"]] = " / ".join(series + ("" if v == "+" else " ") + v for v in vs)
        else:
            raw[g["group_id"]] = f"{series} ({' / '.join(vs)})"
    cnt: Dict[str, int] = {}
    for v in raw.values():
        cnt[v] = cnt.get(v, 0) + 1
    by_id = {g["group_id"]: g for g in cands}
    return {k: (v if cnt[v] == 1 else f"{v} - {', '.join(by_id[k]['model_codes'][:2])}") for k, v in raw.items()}


def make_device(group: Dict, series: str, label: str) -> Dict:
    """선택 결과를 앱·chatbot_service가 쓰는 device 딕셔너리로. model_codes 전체가 검색 필터가 된다."""
    unnamed = series.endswith("(이름 미확인)")
    return {"group_id": group["group_id"], "category": group["category"], "series": series, "label": label,
            "name": None if unnamed else label, "model_code": next((m for m in group["model_codes"] if group.get("code_series", {}).get(m) == series),
                               group["primary_code"]),
            "model_codes": list(group["model_codes"]), "n_chunks": group["n_chunks"]}


def device_key(d: Dict) -> str:
    return f"{d['group_id']}|{d['series']}|{d['label']}"


def display_label(d: Dict) -> str:
    cat = CATEGORY_LABELS.get(d["category"], d["category"])
    return d["name"] if d.get("name") else f"{cat} {d['model_code']}"


def model_desc(d: Dict) -> str:
    """답변 생성에 넘기는 '<표시 이름> (<모델코드>)' 형식. 코드 = 묶음 대표 코드(묶음 안에서 가장 앞)."""
    cat = CATEGORY_LABELS.get(d["category"], d["category"])
    return f"{d.get('name') or cat} ({d['model_code']})"


def find_group(groups: List[Dict], code: str) -> Optional[Dict]:
    code = nfc(code)
    return next((g for g in groups if code in g["model_codes"]), None)


def find(groups: List[Dict], code: str, series: Optional[str] = None) -> Optional[Dict]:
    """모델코드 하나로 device 딕셔너리 만들기(검증 스크립트·chatbot_service 검사용)."""
    g = find_group(groups, code)
    if g is None:
        return None
    ser = series or next(iter(g["series"]))
    cands = groups_in(groups, g["category"], ser)
    # 세부 모델 선택이 필요 없는 시리즈(묶음 1개)는 시리즈명만(변형 목록 제외), 세부를 골라야 하는 시리즈는 세부 이름 유지
    label = ser if len(cands) == 1 else detail_labels(cands, ser, g["category"])[g["group_id"]]
    return make_device(g, ser, label)


def find_by_code(groups: List[Dict], code: str) -> Optional[Dict]:
    """입력 모델코드를 그대로 보존하는 device(MCP 데모용). find()는 시리즈 생략 시 묶음 첫 시리즈·대표 코드를 쓰므로
    공용 묶음의 다른 시리즈 코드(예: 버즈3 프로 SM-R630 → 버즈3 SM-R530)가 바뀐다. find()는 기존 스크립트 재현을 위해 그대로 둔다.
    시리즈 = 입력 코드의 code_series. code_series에 없고 묶음 시리즈가 여럿이면 이름을 정하지 않는다(품목명 + 입력 코드)."""
    code = nfc(code)
    g = find_group(groups, code)
    if g is None:
        return None
    ser = g.get("code_series", {}).get(code)
    if ser is None and len(g["series"]) == 1:
        ser = next(iter(g["series"]))
    if ser is None:
        dev = find(groups, code)
        cat = CATEGORY_LABELS.get(g["category"], g["category"])
        dev.update(series=None, label=f"{cat} {code}", name=None)
    else:
        dev = find(groups, code, ser)
    dev["model_code"] = code
    return dev


def catalog_input_sha(path: Path = CATALOG_PATH) -> Dict[str, str]:
    """카탈로그 manifest에 기록된 input_sha256(생성에 쓴 chunks.jsonl 등)."""
    man = path.with_name(path.name.replace(".json", ".manifest.json"))
    return json.loads(man.read_text(encoding="utf-8"))["input_sha256"]
