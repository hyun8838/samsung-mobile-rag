"""답변 점진 표시(표시 전용). 생성 호출의 요청 인자와 결과는 바꾸지 않는다.

rag.create_answer.manual_answer는 생성 호출을 stream=True로 보내고 조각을 직접 모아 파싱한다.
StreamingClient는 OpenAI 클라이언트를 감싸 돌려주는 스트림만 바꾼다: 조각을 원래 순서·내용 그대로 흘려보내면서
옆에서 화면 표시용 텍스트를 만든다(결과 dict는 래퍼가 없을 때와 같다).

표시 규칙
  - 첫 줄(`SUFFICIENT: true|false`)이 줄바꿈까지 도착해 `true`로 확정된 뒤에만 본문을 표시한다.
    false / 첫 줄 깨짐 / 줄바꿈 없이 너무 긴 첫 줄 = 표시 없음(웹 검색 전환 또는 파싱 실패 경로).
  - 본문 첫 줄의 `TYPE:` 줄은 내부 분류용이라 숨긴다. 아직 줄바꿈 전이고 `TYPE:`가 될 수 있으면 표시를 보류한다.
    `**TYPE:`처럼 파서가 떼지 않는 줄은 파서와 똑같이 본문으로 표시한다(최종 body와 접두 일치 유지).
  - `EVIDENCE:` 줄과, EVIDENCE로 시작할 수 있는 미완성 줄은 표시하지 않는다. EVIDENCE 줄이 나오면 그 뒤는 더 표시하지 않는다.
  - 스트림이 끝나면 answer.parse_output으로 최종 판정해 parse_ok and sufficient가 아니면 on_reset()으로 임시 표시를 지운다.
  - 새 호출(재시도 포함) 시작과 예외 종료 때도 on_reset(). 스트림은 try/finally로 close().
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import create_answer as A  # noqa: E402

FIRST_LINE_MAX = 80   # 이 길이까지 줄바꿈이 안 오면 첫 줄 깨짐으로 보고 표시하지 않는다
_TYPE = "TYPE:"


def _may_be_evidence(partial: str) -> bool:
    """줄바꿈이 아직 안 온 마지막 줄이 EVIDENCE 줄이 될 수 있는가(= 표시 보류)."""
    t = partial.lstrip().lstrip("*").lstrip().upper()
    if not t:
        return True
    return "EVIDENCE".startswith(t) if len(t) <= 8 else t.startswith("EVIDENCE")


class DisplayParser:
    """feed(조각) -> 지금까지 표시 가능한 본문 전체(변화가 있을 때만 str, 없으면 None).
    state: head(첫 줄 판정 중) / stream(true 확정, 본문 표시) / tail(EVIDENCE 줄 발견: 더 표시 안 함) / hidden(false 또는 깨짐)"""

    def __init__(self):
        self.raw, self.state, self.shown = "", "head", ""
        self.sufficient_line: Optional[bool] = None

    def feed(self, piece: str) -> Optional[str]:
        if not piece:
            return None
        self.raw += piece
        if self.state in ("hidden", "tail"):
            return None
        s = self.raw.lstrip()   # parse_output은 raw.strip()으로 시작한다
        if self.state == "head":
            nl = s.find("\n")
            if nl < 0:
                if len(s) > FIRST_LINE_MAX:
                    self.state = "hidden"
                return None
            m = A._SUF.match(s[:nl].strip())
            if not m:
                self.state = "hidden"
                return None
            self.sufficient_line = m.group(1).lower() == "true"
            self.state = "stream" if self.sufficient_line else "hidden"
            if self.state == "hidden":
                return None
        s = self.raw.lstrip()
        body = s[s.find("\n") + 1:]
        lines = body.split("\n")
        keep: List[str] = []
        for ln in lines[:-1]:
            if A._EVI.match(ln.strip()):
                self.state = "tail"
                break
            keep.append(ln)
        else:
            if not _may_be_evidence(lines[-1]):
                keep.append(lines[-1])
        vis = self._visible("\n".join(keep).strip())
        if vis is None:
            return None
        if vis != self.shown and vis.startswith(self.shown):
            self.shown = vis
            return vis
        return None

    @staticmethod
    def _visible(body: str) -> Optional[str]:
        """본문 첫 줄의 TYPE 줄을 숨긴다. None = 이번에는 표시 보류."""
        if not body:
            return body
        first, nl, rest = body.partition("\n")
        head = first.strip()
        if not nl:   # 첫 줄이 아직 줄바꿈 전이거나 본문이 한 줄뿐
            if head.startswith(_TYPE) or _TYPE.startswith(head):
                return None
            return body
        return rest.strip() if head.startswith(_TYPE) else body


class _WrappedStream:
    """원래 스트림을 감싸 조각을 그대로 흘려보내며 표시용 파서에 먹인다. 소비가 끝나면(정상·예외 모두) 원 스트림을 close()."""

    def __init__(self, stream, owner, rec, t0):
        self._s, self._o, self._rec, self._t0 = stream, owner, rec, t0

    def __getattr__(self, name):
        return getattr(self._s, name)

    def close(self):
        close = getattr(self._s, "close", None)
        if callable(close):
            close()

    def __iter__(self):
        o, rec = self._o, self._rec
        parser = DisplayParser()
        parts: List[str] = []
        done = False
        try:
            for ch in self._s:
                yield ch   # 조각은 answer 모듈에 원래 그대로 전달
                for c in (getattr(ch, "choices", None) or []):
                    fr = getattr(c, "finish_reason", None)
                    if fr:
                        rec["finish_reason"] = fr   # 기록만. "length" = 토큰 상한으로 잘림
                    piece = getattr(getattr(c, "delta", None), "content", None)
                    if not piece:
                        continue
                    parts.append(piece)
                    shown = parser.feed(piece)
                    if shown is not None:
                        if rec["t_first_visible"] is None:
                            rec["t_first_visible"] = time.perf_counter() - self._t0
                        if o._on_text:
                            o._on_text(shown)
            done = True
        finally:
            raw = "".join(parts)
            rec.update(raw=raw, state=parser.state, t_total=time.perf_counter() - self._t0)
            try:
                self.close()   # rerun 중단 등(BaseException)으로 끊겨도 연결을 닫는다
            except Exception:  # noqa: BLE001
                pass
            final = A.parse_output(raw)
            ok = done and final["parse_ok"] and final["sufficient"]
            rec["sufficient"] = bool(final["parse_ok"] and final["sufficient"])
            if not ok and o._on_reset:
                rec["reset_after_stream"] = parser.shown != ""
                o._on_reset()


class StreamingClient:
    """OpenAI 클라이언트 위임 래퍼(chat.completions.create만 가로챔). 요청 인자는 그대로 전달한다.
    on_text(visible_text): 표시 가능한 본문 전체(변화 시). on_reset(): 새 호출 시작·최종이 sufficient true가 아닐 때·예외.
    last: 마지막 호출 기록 {raw, state, sufficient, finish_reason, t_first_visible(호출 시작 기준 초), t_total}."""

    def __init__(self, client, on_text: Optional[Callable[[str], None]] = None, on_reset: Optional[Callable[[], None]] = None):
        self._c, self._on_text, self._on_reset = client, on_text, on_reset
        self.calls: List[dict] = []
        self.last: dict = {}
        outer = self

        class _Completions:
            def create(_, **kw):
                return outer._create(**kw)

            def __getattr__(_, n):
                return getattr(outer._c.chat.completions, n)

        class _Chat:
            completions = _Completions()

            def __getattr__(_, n):
                return getattr(outer._c.chat, n)

        self.chat = _Chat()

    def __getattr__(self, name):   # responses 등은 그대로 위임
        return getattr(self._c, name)

    def _create(self, **kw):
        if self._on_reset:
            self._on_reset()   # 재시도·새 호출: 화면 버퍼 비움
        t0 = time.perf_counter()
        try:
            stream = self._c.chat.completions.create(**kw)   # 인자 불변
        except BaseException:
            if self._on_reset:
                self._on_reset()
            raise
        if kw.get("stream") is not True:   # 스트림이 아니면 가로챌 것이 없다
            return stream
        rec = {"raw": "", "state": "head", "sufficient": None, "finish_reason": None, "t_first_visible": None, "t_total": None}
        self.calls.append(rec)
        self.last = rec
        return _WrappedStream(stream, self, rec, t0)
