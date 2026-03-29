"""Reset the local SQLite user database used by the Streamlit app.

Usage:
    python scripts/reset_db.py
    python scripts/reset_db.py --db-path data/arxiv_rec.db

Behavior:
    - Deletes the DB file if it exists.
    - Recreates an empty DB with the required tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from user.db import init_db


def reset_db(db_path: str) -> None:
    """Delete and recreate the SQLite DB with empty tables."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    if db_file.exists():
        db_file.unlink()
        print(f"Deleted existing DB: {db_file}")
    else:
        print(f"No existing DB found at: {db_file}")

    init_db(str(db_file))
    print(f"Created fresh DB schema: {db_file}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Reset local user SQLite DB (delete file + recreate schema)."
    )
    parser.add_argument(
        "--db-path",
        default="data/arxiv_rec.db",
        help="Path to SQLite database file. Default: data/arxiv_rec.db",
    )
    args = parser.parse_args()
    reset_db(args.db_path)


if __name__ == "__main__":
    main()
