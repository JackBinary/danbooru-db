"""End-to-end: fetch tags + wikis, then embed. Each phase is resumable."""

from __future__ import annotations

import argparse
import sys

from .db import connect
from .embed import embed_pending
from .fetch import fetch_tags, fetch_wikis


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Danbooru wiki vector DB end-to-end")
    p.add_argument("--db", default="danbooru.db")
    p.add_argument("--min-posts", type=int, default=1000)
    p.add_argument("--skip", choices=["tags", "wikis", "embed"], action="append", default=[])
    args = p.parse_args()

    conn = connect(args.db)

    if "tags" not in args.skip:
        print(">>> phase 1: fetching tag list", file=sys.stderr)
        n = fetch_tags(conn, min_post_count=args.min_posts)
        print(f"   upserted {n} tags", file=sys.stderr)

    if "wikis" not in args.skip:
        print(">>> phase 2: fetching wiki bodies", file=sys.stderr)
        n = fetch_wikis(conn)
        print(f"   fetched {n} wiki pages", file=sys.stderr)

    if "embed" not in args.skip:
        print(">>> phase 3: embedding bodies", file=sys.stderr)
        n = embed_pending(conn)
        print(f"   embedded {n} tags", file=sys.stderr)


if __name__ == "__main__":
    main()
