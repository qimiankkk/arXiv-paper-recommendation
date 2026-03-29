"""CLI script to run the offline data pipeline.

Usage:
    python scripts/run_offline_pipeline.py               # Process all papers (~2M)
    python scripts/run_offline_pipeline.py --limit 50000  # Dev: ~5 min on CPU
    python scripts/run_offline_pipeline.py --limit 10000  # Quick test

Produces all artifacts in data/ that the app needs to serve recommendations.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.offline import run

parser = argparse.ArgumentParser(
    description="Run the ArXiv offline pipeline: download → embed → cluster → save."
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Max papers to process (None = all). Use 10000-50000 for dev.",
)
parser.add_argument(
    "--k",
    type=int,
    default=500,
    help="Number of k-means clusters. Default 500.",
)
args = parser.parse_args()

run(limit=args.limit, k=args.k)
