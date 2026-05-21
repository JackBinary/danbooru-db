"""CLI: query the vector DB with a natural-language string."""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

from .db import EMBED_DIM, connect
from .embed import load_query_model

# The Harrier model expects an instruction prefix for queries (but not docs).
DEFAULT_INSTRUCTION = (
    "Given a natural-language description of an image, retrieve Danbooru "
    "tags whose wiki pages describe matching visual content."
)


def encode_query(model, text: str, instruction: str = DEFAULT_INSTRUCTION) -> bytes:
    prompt = f"Instruct: {instruction}\nQuery: {text}"
    out = model.create_embedding(prompt)
    vec = np.asarray(out["data"][0]["embedding"], dtype=np.float32)
    if vec.ndim == 2:
        vec = vec[-1]
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return struct.pack(f"{EMBED_DIM}f", *vec.tolist())


def search(db: str, query: str, k: int = 10, instruction: str | None = None):
    conn = connect(db)
    model = load_query_model()
    qblob = encode_query(model, query, instruction or DEFAULT_INSTRUCTION)

    rows = conn.execute(
        """
        SELECT t.name, t.post_count, v.distance, t.body_clean
        FROM vec_tags v
        JOIN tags t ON t.rowid = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        (qblob, k),
    ).fetchall()
    return rows


def _snippet(body: str | None, width: int = 120) -> str:
    if not body:
        return ""
    s = " ".join(body.split())  # collapse all whitespace incl. newlines
    return s[:width] + ("…" if len(s) > width else "")


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic search over Danbooru wiki tags")
    p.add_argument("query", help="Search text")
    p.add_argument("--db", default="danbooru.db")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--instruction", help="Override task instruction")
    args = p.parse_args()

    rows = search(args.db, args.query, k=args.k, instruction=args.instruction)
    if not rows:
        print("(no results)", file=sys.stderr)
        return
    for name, post_count, dist, body in rows:
        print(f"{dist:6.4f}  {name:<35}  ({post_count:>8} posts)  {_snippet(body)}")


if __name__ == "__main__":
    main()
