"""이중 전처리 경로.

    dense 인덱스 입력  : 원문 보존. 유니코드/공백/개행 정규화 + 명백한 인코딩 깨짐 복구 +
                         boilerplate(페이지번호·반복 머리말·캠페인 해시태그) 제거까지만.
                         불용어·조사 제거를 하지 않는다.
    sparse(BM25) 입력  : 위 결과에서 형태소 분석(kiwipiepy) → 내용 형태소만 남기고
                         조사/어미/문장부호/불용어 제거 + 소문자화.

두 표현은 같은 chunk_id를 공유한다(Chunk.text / Chunk.text_normalized).
이 분리가 없으면 H5(BM25+Dense 하이브리드) 비교가 "검색 방식 차이"인지
"전처리 차이"인지 구분되지 않는다.

HTML 엔티티 복구·해시태그 꼬리 제거는 전신 프로젝트(리뷰 데이터)에서 온 규칙이다. 매뉴얼 색인을 만들 때도
그대로 적용했으므로 색인 재현을 위해 남겨 둔다(아래 주석의 '리뷰' 사례는 규칙을 만든 당시 근거).
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- raw 프로파일

# 전신 프로젝트 실측: 다나와/에누리 리뷰 본문에 HTML 엔티티가 두 형태로 남아 있다.
#   (a) 정상형: &quot; &#39; &gt; &lt; &#38;
#   (b) '&'가 유실된 형태: ldquo; rdquo; lsquo; rsquo; hellip; middot; amp; #39; #38; #34;
# (b)는 html.unescape로 복구되지 않으므로 별도 치환이 필요하다.
_BARE_ENTITIES = {
    "ldquo;": "“",
    "rdquo;": "”",
    "lsquo;": "‘",
    "rsquo;": "’",
    "hellip;": "…",
    "middot;": "·",
    "nbsp;": " ",
    "quot;": '"',
    "amp;": "&",
    "lt;": "<",
    "gt;": ">",
    "#39;": "'",
    "#38;": "&",
    "#34;": '"',
}
_BARE_ENTITY_RE = re.compile("|".join(re.escape(k) for k in _BARE_ENTITIES))

# 리뷰 말미의 마케팅 캠페인 해시태그 블록.
# 주의: 본문 중간에 '이벤트'라는 단어가 들어간 리뷰(예: "삼페 이벤트를 활용해 저렴하게 샀다")는
# 실제 사용 경험이므로 지우지 않는다. 지우는 것은 '#으로 시작하는 태그가 연달아 붙은 꼬리'뿐이다.
_HASHTAG_TAIL_RE = re.compile(
    r"(?:^|\n)\s*(?:#[^\s#,\n]+[ ,]*){1,}\s*$",
)
_HASHTAG_INLINE_TAIL_RE = re.compile(r"(?:\s*#[^\s#,\n]{1,30}[ ,]*){2,}\s*$")

_WS_RE = re.compile(r"[ \t 　]+")
_NL_RE = re.compile(r"\n{3,}")

# PDF 반복 boilerplate: 페이지 번호만 있는 줄, "12 고장 신고 전 확인하기" 같은 러닝헤더
_PAGE_ONLY_RE = re.compile(r"^\s*\d{1,3}\s*$")


def repair_entities(text: str) -> str:
    # 순서 중요: 정상형(&quot;)을 먼저 풀지 않으면 bare 치환이 '&'만 남겨 &"가 된다.
    text = html.unescape(text)
    text = _BARE_ENTITY_RE.sub(lambda m: _BARE_ENTITIES[m.group(0)], text)
    return text


def strip_campaign_hashtags(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _HASHTAG_TAIL_RE.sub("\n", text)
        text = _HASHTAG_INLINE_TAIL_RE.sub("", text)
    return text.strip()


def normalize_raw(text: Optional[str], *, drop_hashtag_tail: bool = True) -> str:
    """dense 임베딩 입력용. 의미를 바꾸는 처리를 하지 않는다."""
    if not text:
        return ""
    text = repair_entities(text)
    # NFC를 쓴다(NFKC 아님). NFKC는 한글 호환 자모(ㅎㅎ, ㅋㅋ, U+3131~U+318E)를
    # 조합용 자모(U+1100대)로 바꿔 버리는데, 리뷰에서 'ㅋㅋ'·'ㅎㅎ'는 감성 신호라 원형 보존이 낫다.
    # (실측: NFKC 적용 시 'ㅎㅎ' → 'ᄒᄒ'로 깨짐)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("​", "").replace("﻿", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text)
    if drop_hashtag_tail:
        text = strip_campaign_hashtags(text)
    return text.strip()


# ---------------------------------------------------------------- PDF boilerplate

def detect_repeated_lines(pages: Sequence[str], *, min_ratio: float = 0.5, max_len: int = 60) -> set:
    """전 페이지에 반복되는 짧은 줄(러닝헤더/푸터/저작권 고지)을 문서 단위로 탐지."""
    from collections import Counter

    cnt: Counter = Counter()
    for p in pages:
        seen = set()
        for line in p.split("\n"):
            line = line.strip()
            if not line or len(line) > max_len:
                continue
            # 페이지번호가 섞인 헤더는 번호를 제거한 형태로 집계
            key = re.sub(r"\d+", "#", line)
            if key in seen:
                continue
            seen.add(key)
            cnt[key] += 1
    n = max(1, len(pages))
    return {k for k, v in cnt.items() if v / n >= min_ratio and len(k) >= 2}


def strip_page_boilerplate(page_text: str, repeated: set) -> str:
    out = []
    for line in page_text.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        if _PAGE_ONLY_RE.match(s):
            continue                     # 페이지번호는 텍스트에서 제거하고 메타데이터로 보존
        if re.sub(r"\d+", "#", s) in repeated:
            continue
        out.append(s)
    return _NL_RE.sub("\n\n", "\n".join(out)).strip()


# ---------------------------------------------------------------- normalized 프로파일

# 유지할 형태소 태그: 체언·용언 어간·부사·외국어·숫자·어근
_KEEP_TAGS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM",
    "XR", "XPN", "XSN", "XSV", "XSA",
    "SL", "SH", "SN",
    "W_SERIAL",
}
# 불용어는 최소한만 둔다. 조사·어미는 태그 필터로 이미 제거되므로,
# 여기 남는 것은 "형태소로는 명사인데 정보량이 사실상 0"인 의존명사류다.
# (불용어 제거의 근거가 약해 공격적으로 지우지 않는다.)
_STOPWORDS = {"것", "수", "등", "때", "듯", "점", "제", "저", "년", "월", "일", "개", "번"}

# 에러코드·모델명은 형태소 분석으로 쪼개면 BM25 이점이 사라진다.
# 분석 전에 뽑아 두었다가 토큰 목록에 원형 그대로 다시 넣는다.
_CODE_RE = re.compile(
    r"\b(?:"
    r"SM-[A-Za-z0-9]{4,8}"                         # 삼성 모델코드 SM-S938N, SM-X846N (한 토큰으로 유지)
    r"|NT\d{3}[A-Za-z]{1,4}"                       # 갤럭시북 NT940XJG
    r"|[A-Za-z]{1,4}[-_ ]?\d{1,4}[A-Za-z0-9\-]*"   # SM-S938N, SM-X846N, NT940XJG, C05, E-12
    r"|\d{1,3}[A-Za-z]{1,3}\b"                     # 5C
    r"|오류\s?\d{1,3}"                              # 오류 3
    r"|에러\s?\d{1,3}"
    r")\b"
)


# 도메인 사용자 사전.
# 근거(전신 프로젝트 실측): kiwi 기본 사전은 문맥에 따라 복합 제품명을 조사 등으로 쪼개
# 제품 라인명이 통째로 사라진다. 제품·기능명은 BM25에서 식별의 핵심 키워드이므로 사용자 사전으로 고정한다.
# (형태소 분석기의 도메인 미적응은 dense가 아니라 sparse 쪽 문제다)
# 실제 매뉴얼 어휘로 확인·보정한 목록.
_DOMAIN_WORDS = [
    # 제품군/브랜드
    "갤럭시", "갤럭시AI", "갤럭시버즈", "갤럭시워치", "갤럭시북", "갤럭시탭",
    # 소프트웨어/기능
    "원UI", "One UI", "S펜", "삼성덱스", "덱스", "퀵쉐어", "스마트싱스", "빅스비",
    "엣지패널", "보안폴더", "스마트스위치", "굿락", "삼성페이", "삼성월렛",
    "플렉스모드", "커버화면", "자가진단",
    # 분리 단위 보정: 복합어를 통째로 묶지 않고, 잘못 쪼개지는 하위 단위만 고정한다.
    # 기본 kiwi: 최종당화산물 → 최종/당/화/산물, 게임런처 → 게임/런/처  →  보정 후 최종/당화/산물, 게임/런처
    # ("산물"은 "당화"만 넣으면 사전 가중치 때문에 산/물로 쪼개져 함께 넣음)
    # (붙여 쓰든 띄어 쓰든 같은 토큰이 되어 BM25 매칭이 맞는다)
    "당화", "산물", "런처",
]


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    k = Kiwi()
    for w in _DOMAIN_WORDS:
        try:
            k.add_user_word(w, "NNP", 5.0)
        except Exception:  # noqa: BLE001
            pass
    return k


def extract_codes(text: str) -> List[str]:
    return [m.group(0).lower().replace(" ", "") for m in _CODE_RE.finditer(text)]


def normalize_sparse(text: str) -> str:
    """BM25 입력용: 형태소 분석 → 내용 형태소 + 코드 토큰."""
    if not text:
        return ""
    codes = extract_codes(text)
    # 코드 토큰은 형태소 분석 전에 마스킹해 파편(vr/30/t/85514/w)이 생기지 않게 한다.
    masked = _CODE_RE.sub(" ", text)
    toks: List[str] = []
    try:
        for tok in _kiwi().tokenize(masked):
            if tok.tag in _KEEP_TAGS:
                form = tok.form.lower()
                if len(form) < 1 or form in _STOPWORDS:
                    continue
                toks.append(form)
    except Exception:  # noqa: BLE001 — 형태소 분석 실패 시 공백 토큰화로 폴백
        toks = [w.lower() for w in re.findall(r"[\w가-힣]+", masked)]
    toks.extend(codes)
    return " ".join(toks)


def sparse_tokens(text_normalized: str) -> List[str]:
    return text_normalized.split()


# ---------------------------------------------------------------- 날짜 정규화

_DATE_RE = re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})")


def normalize_date(s: Optional[str]) -> Optional[str]:
    """실측: 수집 파일에 'YYYY-MM-DD' / 'YYYY.MM.DD' / 'YYYY.MM.DD.' 3종이 섞여 있다."""
    if not s:
        return None
    m = _DATE_RE.search(s)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
