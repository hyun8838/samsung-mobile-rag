"""검색 지표·paired bootstrap·선택 규칙.

전신 프로젝트(robot-vacuum-rag)의 지표 계산 코드를 바탕으로 새로 작성.

- 주 지표: pooled R@5 strict = |2점 정답 ∩ top5| / |2점 정답|. 보조: MRR@10. 기술: R@1·R@3·R@5.
  lenient(1·2점)는 참고. 다중 정답 문항용으로 hit@5(정답 1개 이상 top5)도 함께 낸다.
- 미판정 청크 = 0점. 행별 미판정 수를 순위대(1~5위, 6~10위)로 함께 센다.
- 정답이 0개인 문항(해당 기준)은 지표가 정의되지 않으므로 평균에서 빼고 그 수를 보고한다.
- paired bootstrap: 문항 단위 재표집 BOOTSTRAP_N회, seed STAT_SEED. 재표집 인덱스는 한 번 만들어
  모든 비교·지표에 공통으로 쓴다(비교 순서에 따라 결과가 바뀌지 않게).
- 선택 규칙: 최고 행과의 차이(최고 − 행)의 95% CI가 0을 포함하는 행 중 단순성 순서상 가장 앞의 행.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from rag import settings as C

LEVELS: Dict[str, int] = {"strict": 2, "lenient": 1}
RECALL_KS = (1, 3, 5)
MRR_K = 10
METRICS = ("recall@1", "recall@3", "recall@5", "hit@5", "mrr@10")
PRIMARY = ("strict", "recall@5")


def nfc(s: Optional[str]) -> str:
    return unicodedata.normalize("NFC", s or "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def score_hits(hit_ids: Sequence[str], labels: Mapping[str, int]) -> Dict[str, Any]:
    """한 문항·한 행의 검색 결과(순위순 chunk_id)와 판정(chunk_id → 0/1/2)으로 지표를 계산한다.

    labels에 없는 chunk_id = 미판정(0점 처리). 반환값의 지표는 정답이 0개면 None.
    """
    hits = [nfc(h) for h in hit_ids][:MRR_K]
    lab = {nfc(k): int(v) for k, v in labels.items()}
    out: Dict[str, Any] = {
        "unjudged@1-5": sum(1 for h in hits[:5] if h not in lab),
        "unjudged@6-10": sum(1 for h in hits[5:10] if h not in lab),
    }
    for name, thr in LEVELS.items():
        rel = {cid for cid, v in lab.items() if v >= thr}
        m: Dict[str, Any] = {"n_gt": len(rel)}
        if not rel:
            m.update({k: None for k in METRICS})
        else:
            for k in RECALL_KS:
                m[f"recall@{k}"] = sum(h in rel for h in hits[:k]) / len(rel)
            m["hit@5"] = 1.0 if any(h in rel for h in hits[:5]) else 0.0
            rank = next((i for i, h in enumerate(hits, 1) if h in rel), None)
            m["mrr@10"] = 1.0 / rank if rank else 0.0
        out[name] = m
    return out


def mean_metrics(per_q: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """score_hits 결과 목록 → 평균. 정답 0개 문항은 해당 기준에서 빼고 수를 센다."""
    out: Dict[str, Any] = {
        "n_questions": len(per_q),
        "unjudged@1-5": int(sum(r["unjudged@1-5"] for r in per_q)),
        "unjudged@6-10": int(sum(r["unjudged@6-10"] for r in per_q)),
    }
    for name in LEVELS:
        rows = [r[name] for r in per_q if r[name]["n_gt"] > 0]
        out[name] = {"n_scored": len(rows), "n_no_gt": len(per_q) - len(rows)}
        for k in METRICS:
            out[name][k] = float(np.mean([r[k] for r in rows])) if rows else None
    return out


def bootstrap_indices(n: int, n_boot: int = C.BOOTSTRAP_N, seed: int = C.STAT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_pair(a: Sequence[float], b: Sequence[float], idx: np.ndarray) -> Dict[str, Any]:
    """문항별로 짝지은 값 a, b의 차이(a − b) 평균과 paired bootstrap 95% CI."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if idx.shape[1] != len(d):
        raise ValueError("bootstrap 인덱스 길이와 문항 수가 다름")
    draws = d[idx].mean(axis=1)
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {"n": len(d), "diff": float(d.mean()), "ci95": [lo, hi], "ci_includes_0": lo <= 0.0 <= hi}


def paired_values(per_row: Mapping[str, Mapping[str, Mapping[str, Any]]], level: str, metric: str):
    """{row: {qid: score_hits결과}} → 모든 행에서 정답이 있는 공통 qid 목록과 행별 값 벡터."""
    rows = list(per_row)
    qids = sorted(set.intersection(*[set(per_row[r]) for r in rows]))
    qids = [q for q in qids if all(per_row[r][q][level]["n_gt"] > 0 for r in rows)]
    return qids, {r: [per_row[r][q][level][metric] for q in qids] for r in rows}


def select_simplest(per_row: Mapping[str, Mapping[str, Mapping[str, Any]]], simplicity_order: Sequence[str],
                    level: str = PRIMARY[0], metric: str = PRIMARY[1],
                    n_boot: int = C.BOOTSTRAP_N, seed: int = C.STAT_SEED) -> Dict[str, Any]:
    """선택 규칙: 최고 행과 유의차 없는(차이 95% CI가 0 포함) 행 중 단순성 순서상 가장 앞의 행."""
    missing = [r for r in simplicity_order if r not in per_row]
    if missing or set(per_row) - set(simplicity_order):
        raise ValueError(f"단순성 순서와 행 목록이 다름: 없음={missing}, 순서 밖={sorted(set(per_row) - set(simplicity_order))}")
    qids, vals = paired_values(per_row, level, metric)
    if not qids:
        raise ValueError("공통으로 채점 가능한 문항이 없음")
    idx = bootstrap_indices(len(qids), n_boot, seed)
    means = {r: float(np.mean(vals[r])) for r in simplicity_order}
    best = max(simplicity_order, key=lambda r: (means[r], -simplicity_order.index(r)))
    comps = {r: bootstrap_pair(vals[best], vals[r], idx) for r in simplicity_order if r != best}
    eligible = [r for r in simplicity_order if r == best or comps[r]["ci_includes_0"]]
    return {"level": level, "metric": metric, "n_questions": len(qids), "means": means, "best": best,
            "best_minus_row": comps, "eligible": eligible, "selected": eligible[0],
            "n_boot": n_boot, "seed": seed}


def pairwise_table(per_row: Mapping[str, Mapping[str, Mapping[str, Any]]], pairs: Sequence[Sequence[str]],
                   n_boot: int = C.BOOTSTRAP_N, seed: int = C.STAT_SEED) -> List[Dict[str, Any]]:
    """지정한 (a, b) 쌍마다 모든 기준·지표의 a − b paired bootstrap."""
    out = []
    for a, b in pairs:
        sub = {a: per_row[a], b: per_row[b]}
        for level in LEVELS:
            for metric in METRICS:
                qids, vals = paired_values(sub, level, metric)
                if not qids:
                    continue
                res = bootstrap_pair(vals[a], vals[b], bootstrap_indices(len(qids), n_boot, seed))
                out.append({"a": a, "b": b, "level": level, "metric": metric, **res})
    return out
