"""Re-pick top-K cases from an existing score_diff.csv without re-scoring.

Reads ``{out_root}/scores/score_diff.csv`` (already sorted by delta_O desc)
and writes ``{out_root}/scores/selected_topk.json`` with the requested K.

Default criterion mirrors score_edit_compare.py: ``delta_O > 0 AND
edited_object != ''`` (so stage-3 capture has a keyword to map). Use
``--no_filter`` to keep ALL rows regardless of edited_object / delta sign.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("base_SC", "dtr_SC", "base_PQ", "dtr_PQ",
                      "base_O", "dtr_O", "delta_SC", "delta_PQ", "delta_O"):
                if r.get(k) not in (None, ""):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", required=True)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--csv", default="",
                   help="Path to score_diff.csv. Default: {out_root}/scores/score_diff.csv")
    p.add_argument("--out", default="",
                   help="Output JSON path. Default: {out_root}/scores/selected_topk.json")
    p.add_argument("--no_filter", action="store_true",
                   help="Keep ALL rows, ignore edited_object / delta_O sign filters.")
    p.add_argument("--require_keyword", action="store_true",
                   help="Only keep rows whose edited_object is non-empty (default in --no_filter mode).")
    p.add_argument("--min_delta_o", type=float, default=None,
                   help="Override the default delta_O > 0 floor with this threshold.")
    p.add_argument("--task_types", default="",
                   help="Comma-separated GEdit task_type whitelist, e.g. "
                        "'subject-add,subject-remove,subject-replace'. Empty = no filter.")
    args = p.parse_args()

    csv_path = args.csv or os.path.join(args.out_root, "scores", "score_diff.csv")
    out_path = args.out or os.path.join(args.out_root, "scores", "selected_topk.json")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = _load_csv(csv_path)
    rows.sort(key=lambda r: r.get("delta_O", float("-inf")), reverse=True)

    # Task-type whitelist (applied before any other filter so the totals are
    # reported relative to the whitelisted subset).
    if args.task_types:
        wanted = {t.strip() for t in args.task_types.split(",") if t.strip()}
        rows = [r for r in rows if r.get("task_type") in wanted]

    if args.no_filter:
        kept = rows
        criterion = "no filter"
        if args.require_keyword:
            kept = [r for r in kept if (r.get("edited_object") or "").strip()]
            criterion = "edited_object != ''"
    else:
        threshold = args.min_delta_o if args.min_delta_o is not None else 0.0
        kept = [r for r in rows
                if (r.get("edited_object") or "").strip()
                and r.get("delta_O", float("-inf")) > threshold]
        criterion = f"delta_O > {threshold} AND edited_object != ''"

    topk = kept[: args.top_k]
    payload = {
        "keys": [r["key"] for r in topk],
        "rows": topk,
        "k": args.top_k,
        "criterion": criterion,
        "task_types": args.task_types or "ALL",
        "n_total": len(rows),
        "n_after_filter": len(kept),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f">>> Wrote {out_path}\n"
        f"    n_total={len(rows)}  n_after_filter={len(kept)}  "
        f"requested_k={args.top_k}  written={len(topk)}\n"
        f"    criterion: {criterion}"
    )


if __name__ == "__main__":
    main()
