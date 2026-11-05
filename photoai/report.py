"""Summarise a run: what was built, the label ranges, QC and timings. Every number comes from
the records; none is typed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photoai.shards import _records, summarise


def _col(recs, fn):
    vals = []
    for r in recs:
        try:
            v = fn(r)
        except (KeyError, TypeError):
            v = None
        if v is not None:
            vals.append(float(v))
    return np.array(vals)


def _stats(a: np.ndarray) -> dict | None:
    if len(a) == 0:
        return None
    return {"n": int(len(a)), "min": round(float(a.min()), 2), "p50": round(float(np.median(a)), 2),
            "max": round(float(a.max()), 2)}


def report(run: Path, md_path: Path | None = None) -> dict:
    recs = _records(run)
    ok = [r for r in recs if r["status"] == "ok"]
    s = summarise(recs)
    s["timings_s"] = {k: _stats(_col(ok, lambda r, k=k: r["timings_s"].get(k)))
                      for k in ("draft", "drape", "qc_labels", "render", "total")}
    s["labels"] = {
        "ease_designed_bust_cm": _stats(_col(ok, lambda r: r["labels"]["ease_designed_cm"]["bust"])),
        "ease_pattern_bust_cm": _stats(_col(ok, lambda r: r["labels"]["ease_pattern_cm"]["bust"])),
        "ease_worn_bust_cm": _stats(_col(ok, lambda r: r["labels"]["ease_worn_cm"]["bust"])),
        "ease_worn_waist_cm": _stats(_col(ok, lambda r: r["labels"]["ease_worn_cm"]["waist"])),
        "strain_pct_p95": _stats(_col(ok, lambda r: r["labels"]["strain_pct_p95"])),
        "air_gap_mm_p50": _stats(_col(ok, lambda r: r["labels"]["air_gap_mm_p50"])),
    }
    # does the worn ease follow the designed ease? (the dataset's first sanity check)
    d = _col(ok, lambda r: r["labels"]["ease_designed_cm"]["bust"])
    w = _col(ok, lambda r: r["labels"]["ease_worn_cm"]["bust"])
    if len(d) == len(w) and len(d) > 2:
        s["worn_vs_designed_bust_corr"] = round(float(np.corrcoef(d, w)[0, 1]), 3)
    # strain by fabric: the elasticity axis should show here before any model is trained
    by_fab = {}
    for r in ok:
        by_fab.setdefault(r["job"]["fabric"], []).append(r["labels"]["strain_pct_p95"])
    s["strain_p95_by_fabric"] = {k: round(float(np.median(v)), 2) for k, v in by_fab.items()}
    s["qc_fail_reasons"] = {}
    for r in ok:
        for reason in r["qc"]["fail_reasons"]:
            key = reason.split(" ")[0] + " " + reason.split(" ")[1] if " " in reason else reason
            s["qc_fail_reasons"][key] = s["qc_fail_reasons"].get(key, 0) + 1
    s["failures"] = [{"sample_id": r["sample_id"], "error": r.get("error")}
                     for r in recs if r["status"] != "ok"]
    s["samples"] = [{"sample_id": r["sample_id"], "design": r["job"]["design"],
                     "ease": r["job"]["ease_label"], "fabric": r["job"]["fabric"],
                     "body": r["body"]["body_id"],
                     "ease_worn_bust": r["labels"]["ease_worn_cm"].get("bust"),
                     "strain_p95": r["labels"]["strain_pct_p95"],
                     "qc": r["qc"]["passed"], "converged": r["drape"]["converged"],
                     "garment_ply": str((run / r["files"]["garment_ply"]).resolve()),
                     "body_ply": str((run / r["files"]["body_ply"]).resolve())}
                    for r in ok]
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown(s, run), encoding="utf-8")
    return s


def markdown(s: dict, run: Path) -> str:
    lines = [f"# Run report: `{run.resolve()}`", "",
             f"{s['n_ok']} samples built, {s['n_failed']} failed, {s['n_qc_passed']} passed QC, "
             f"{s['n_converged']} converged. {s['n_bodies']} bodies; designs {s['designs']}; "
             f"ease steps {s['ease_steps']}; fabrics {s['fabrics']}.", "",
             "## Timings (seconds per sample, one CPU core)", "",
             "| step | n | min | median | max |", "|---|---:|---:|---:|---:|"]
    for k, v in s["timings_s"].items():
        if v:
            lines.append(f"| {k} | {v['n']} | {v['min']} | {v['p50']} | {v['max']} |")
    lines += ["", "## Labels", "", "| label | n | min | median | max |", "|---|---:|---:|---:|---:|"]
    for k, v in s["labels"].items():
        if v:
            lines.append(f"| {k} | {v['n']} | {v['min']} | {v['p50']} | {v['max']} |")
    lines += ["", f"Worn bust ease against designed bust ease, correlation: "
                  f"{s.get('worn_vs_designed_bust_corr', 'n/a')}.",
              f"Median 95th-percentile strain by fabric: {s['strain_p95_by_fabric']}.",
              f"QC failure reasons: {s['qc_fail_reasons'] or 'none'}.", "",
              "## Every sample (full mesh paths, for MeshLab or Blender)", "",
              "| sample | body | design | ease | fabric | worn bust ease cm | strain p95 % | QC | converged | garment mesh | body mesh |",
              "|---|---|---|---|---|---:|---:|---|---|---|---|"]
    for r in s["samples"]:
        lines.append(f"| {r['sample_id']} | {r['body']} | {r['design']} | {r['ease']} | {r['fabric']} | "
                     f"{r['ease_worn_bust']} | {r['strain_p95']} | {r['qc']} | {r['converged']} | "
                     f"`{r['garment_ply']}` | `{r['body_ply']}` |")
    if s["failures"]:
        lines += ["", "## Failed", ""] + [f"- {f['sample_id']}: {f['error']}" for f in s["failures"]]
    return "\n".join(lines) + "\n"
