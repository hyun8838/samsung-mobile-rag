"""검색 결과의 R@5를 RAGAS IDBasedContextRecall로 계산한다(API 호출 없음).

R@5 = 질문별 (2점 정답 청크 중 top5에 든 수 / 2점 정답 청크 수)의 평균. 2점 정답이 없는 질문은 제외하고 그 수를 출력한다.
입력:
  --runs    jsonl, 줄마다 {"qid", "hits": [chunk_id, ...]} (선택 "row": 검색 구성 이름)
  --labels  jsonl, 줄마다 {"qid", "chunk_id", "label": 0|1|2}
실행(code/ 에서, requirements-ragas.txt 환경):
  python scripts/ragas_recall_at5.py --runs runs.jsonl --labels labels.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextRecall

K = 5


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def read_jsonl(p: str):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


async def score_all(runs, ref):
    metric = IDBasedContextRecall()
    out = defaultdict(dict)
    for row, by_q in runs.items():
        for q, hits in by_q.items():
            if ref.get(q):
                s = SingleTurnSample(retrieved_context_ids=hits[:K], reference_context_ids=ref[q])
                out[row][q] = await metric.single_turn_ascore(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", help="질문별 점수를 쓸 jsonl(선택)")
    a = ap.parse_args()

    ref = defaultdict(list)
    for r in read_jsonl(a.labels):
        if int(r["label"]) >= 2:
            ref[nfc(r["qid"])].append(nfc(r["chunk_id"]))
    runs = defaultdict(dict)
    for r in read_jsonl(a.runs):
        runs[r.get("row", "run")][nfc(r["qid"])] = [nfc(h) for h in r["hits"]]

    scores = asyncio.run(score_all(runs, ref))
    for row, by_q in runs.items():
        v = scores[row]
        mean = sum(v.values()) / len(v) if v else float("nan")
        print(f"{row:28s} R@5={mean:.3f}  채점 {len(v)}문항, 2점 정답 없어 제외 {len(by_q) - len(v)}문항")
    if a.out:
        with open(a.out, "x", encoding="utf-8") as fh:
            for row, v in scores.items():
                for q, s in v.items():
                    fh.write(json.dumps({"row": row, "qid": q, "recall@5": s}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
