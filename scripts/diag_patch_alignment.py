#!/usr/bin/env python3
"""Task 1 — HE/PR feature row-level correspondence diagnosis (diagnostic-only, no model changes).

Purpose
-------
Determine whether HE feature row i and PR feature row i are the *same* original
patch.  This is the precondition for the shared `patch_indices` logic in
`data/c16_multimodal_dataset.py` (which only checks COUNT, not ORDER).

What we can and cannot do
-------------------------
- The source JPEGs live on /media/kemove/data_hdd0/... which is NOT mounted, and
  the .pt files are plain tensors (no filename/coordinate metadata), so a direct
  filename or coordinate comparison is IMPOSSIBLE.  → Case C for that route.
- We therefore fall back to the *feature-correlation* test: PR is a virtual-stain
  transform of HE, so if HE[i] and PR[i] are the same patch, HE[i] should be far
  more similar to PR[i] (the diagonal) than to PR[j!=i] (off-diagonal / shuffled).

Tests
-----
  A. count equality            (shape check)
  B. diagonal vs shuffled      (L2-normalized cosine, mean over rows)
  C. diagonal vs offset d      (d = 1,2,5,10,50,100,500 circular offsets)
  D. argmax-match              (best PR match for each HE row; identity ⇒ aligned)

Conclusion rule (printed + written to JSON):
  Case A  — empirically aligned:  diag >> shuffled AND argmax≈identity (frac>0.9)
  Case B  — set matches, order does NOT: diag ≈ shuffled (permuted rows)
  Case C  — cannot determine (weak / ambiguous signal)

Usage:
  python scripts/diag_patch_alignment.py --slides normal_001 normal_002 normal_003 \
      normal_004 normal_005 tumor_001 tumor_002 tumor_003 tumor_004 tumor_005
"""
import argparse, json, sys
from pathlib import Path

import torch
import numpy as np

FEATURE_BASE = Path("/home/Public/lillan/features_result/C16_features")
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
OFFSETS = [1, 2, 5, 10, 50, 100, 500]
SUBSAMPLE = 3000          # rows used for diagonal/offset/shuffled estimates
ARGMAX_SUBSET = 400       # rows used for the argmax-match test
SEED = 0


def load(mod: str, cat: str, idx: int):
    p = FEATURE_BASE / DIR_MAPPING[mod] / cat / f"{cat}_{idx:03d}.pt"
    return torch.load(str(p), map_location="cpu", weights_only=True)


def l2norm(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def analyze(he, pr):
    n = min(he.shape[0], pr.shape[0])
    he, pr = he[:n].float(), pr[:n].float()
    he_n, pr_n = l2norm(he), l2norm(pr)

    # deterministic subsample for cheap estimates
    rng = np.random.RandomState(SEED)
    idx = rng.choice(n, min(n, SUBSAMPLE), replace=False)
    idx.sort()
    he_s, pr_s = he_n[idx], pr_n[idx]
    m = he_s.shape[0]

    # B. diagonal vs shuffled
    diag = (he_s * pr_s).sum(1).mean().item()
    perm = rng.permutation(m)
    shuffled = (he_s * pr_s[perm]).sum(1).mean().item()

    # C. diagonal vs circular offsets
    offset_sims = {}
    for d in OFFSETS:
        if d >= m:
            continue
        pr_shift = torch.cat([pr_s[d:], pr_s[:d]], dim=0)
        offset_sims[d] = (he_s * pr_shift).sum(1).mean().item()

    # D. argmax-match on a smaller subset
    am_idx = rng.choice(n, min(n, ARGMAX_SUBSET), replace=False)
    am_idx.sort()
    he_a, pr_a = he_n[am_idx], pr_n[am_idx]
    S = he_a @ pr_a.T                      # [K, K]
    best = S.argmax(dim=1).cpu().numpy()   # best PR row for each HE row
    k = best.shape[0]
    exact = (best == np.arange(k)).mean()
    within1 = (np.abs(best - np.arange(k)) <= 1).mean()
    mean_disp = np.abs(best - np.arange(k)).mean()

    return {
        "n_rows": n,
        "diag_cos": diag,
        "shuffled_cos": shuffled,
        "offset_cos": offset_sims,
        "argmax_exact": float(exact),
        "argmax_within1": float(within1),
        "argmax_mean_disp": float(mean_disp),
    }


def classify(r):
    diag = r["diag_cos"]
    shuff = r["shuffled_cos"]
    exact = r["argmax_exact"]
    # gap to shuffled is the decisive signal for "aligned vs permuted"
    gap = diag - shuff
    if exact > 0.9 and gap > 0.05:
        return "A"
    if gap < 0.02 and exact < 0.5:
        return "B"
    return "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", default=[
        "normal_001", "normal_002", "normal_003", "normal_004", "normal_005",
        "tumor_001", "tumor_002", "tumor_003", "tumor_004", "tumor_005",
    ])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    print("=" * 84)
    print("Task 1 — HE/PR row-level correspondence (feature-correlation test)")
    print("=" * 84)
    print(f"{'slide':14s} {'N':>6s} {'diag':>7s} {'shuff':>7s} {'gap':>7s} "
          f"{'off1':>7s} {'argmax=id':>10s} {'meanDisp':>9s}  case")
    print("-" * 84)

    for name in args.slides:
        parts = name.rsplit("_", 1)
        cat, idx = parts[0], int(parts[1])
        try:
            he = load("HE", cat, idx)
            pr = load("PR", cat, idx)
        except FileNotFoundError as e:
            print(f"{name:14s}  MISSING {e}")
            continue
        r = analyze(he, pr)
        case = classify(r)
        results[name] = {**r, "case": case}
        off1 = r["offset_cos"].get(1, float("nan"))
        print(f"{name:14s} {r['n_rows']:6d} {r['diag_cos']:7.4f} {r['shuffled_cos']:7.4f} "
              f"{r['diag_cos']-r['shuffled_cos']:7.4f} {off1:7.4f} "
              f"{r['argmax_exact']:10.3f} {r['argmax_mean_disp']:9.1f}  {case}")

    print("-" * 84)
    print("offset similarity profile (mean over slides):")
    all_offsets = {}
    for d in OFFSETS:
        vals = [r["offset_cos"][d] for r in results.values() if d in r["offset_cos"]]
        if vals:
            all_offsets[d] = float(np.mean(vals))
            print(f"  d={d:4d}  cos={all_offsets[d]:.4f}")
    diag_mean = float(np.mean([r["diag_cos"] for r in results.values()]))
    shuff_mean = float(np.mean([r["shuffled_cos"] for r in results.values()]))
    print(f"  d=0 (diag)   cos={diag_mean:.4f}   <-- compare to offsets above")
    print(f"  shuffled      cos={shuff_mean:.4f}")

    nA = sum(1 for r in results.values() if r["case"] == "A")
    nB = sum(1 for r in results.values() if r["case"] == "B")
    nC = sum(1 for r in results.values() if r["case"] == "C")
    print("-" * 84)
    print(f"Case tally: A={nA}  B={nB}  C={nC}  (of {len(results)} slides)")
    print("=" * 84)

    summary = {
        "feature_base": str(FEATURE_BASE),
        "source_unmounted": True,
        "note": ("filename/coordinate comparison impossible (source JPEGs on "
                 "/media/kemove/data_hdd0 are unmounted; .pt is a plain tensor). "
                 "Conclusion rests on the feature-correlation diagonal test."),
        "per_slide": results,
        "mean_diag": diag_mean,
        "mean_shuffled": shuff_mean,
        "mean_offsets": all_offsets,
        "case_tally": {"A": nA, "B": nB, "C": nC},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
