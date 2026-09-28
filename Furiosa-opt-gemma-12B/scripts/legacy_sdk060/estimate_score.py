#!/usr/bin/env python3
"""Estimate the Stage 1 score from real RNGD cycle counts.

    python3 scripts/estimate_score.py <qkv_cycles> <attn_out_cycles> <ffn_cycles>

The score is the geometric mean of the per-kernel speedup over the (unpublished) baseline.
The product of the three real baseline cycle counts follows from any leaderboard row as
`score^3 * k1 * k2 * k3`; the four rows visible on 2026-09-13 agree on 3.754e17 to four digits,
so that constant is used here. Arena logs print the three cycle counts this script needs.
"""
import sys

BASELINE_PRODUCT = 3.754e17

if len(sys.argv) != 4:
    sys.exit(__doc__)
k1, k2, k3 = (float(v.replace(",", "")) for v in sys.argv[1:])
score = (BASELINE_PRODUCT / (k1 * k2 * k3)) ** (1 / 3)
print(f"estimated score: {score:.3f}x")
