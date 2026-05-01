"""Daily Feed evaluation plotting entrypoint.

These plots are for the Daily Feed evaluation section. Variant comparison
defaults to the presentation setting ``delta = 0.5`` so algorithm variants are
compared at the same diversity-index value. Use ``--pool-deltas`` only for
exploratory plots, because pooling mixes variant effects with the diversity
sweep.

Run from the project root:

    python evaluation/plot_daily_feed_metrics.py \\
      --input evaluation/data/daily_feed_eval_raw_ev1_ev90.json \\
      --outdir evaluation/figures \\
      --delta 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_evaluation_results import main_daily_feed


if __name__ == "__main__":
    raise SystemExit(main_daily_feed())
