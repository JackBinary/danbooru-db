"""Embed wiki bodies using the Harrier OSS GGUF via llama-cpp-python.

The model uses last-token pooling with L2 normalization. llama.cpp's
embedding mode handles pooling internally; we set `pooling_type="last"` and
normalize ourselves to be explicit.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from datetime import datetime, timezone

import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from tqdm import tqdm

from .db import EMBED_DIM, connect

MODEL_REPO = "mykor/harrier-oss-v1-270m-GGUF"
# BF16 is used at indexing time: Q8_0 quantization noise compounds across the
# corpus and collapses the embedding space (cat_ears ends up near gold_chain
# instead of dog_ears). BF16 takes one slow run but the index is built once.
INDEX_MODEL_FILE = "harrier-oss-v1-270M-BF16.gguf"
# Q8_0 from the same repo for queries: cosine(BF16, Q8) = 0.9997 on the same
# input, so target ranks against the BF16 index are unchanged. Faster to load
# and run at query time.
QUERY_MODEL_FILE = "harrier-oss-v1-270M-Q8_0.gguf"
# Harrier-OSS supports 32k tokens, but llama-cpp-python's internal n_seq_max
# hard-caps per-sequence context at 256 tokens. Inputs above this fail rather
# than silently truncate, so we truncate via the tokenizer below.
N_CTX = 4096
MAX_INPUT_TOKENS = 248  # leave headroom under the 256 cap


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(filename: str) -> Llama:
    path = hf_hub_download(MODEL_REPO, filename)
    # llama_pooling_type: 0=NONE, 1=MEAN, 2=CLS, 3=LAST.
    # Harrier is a decoder-only embedding model that uses LAST pooling.
    # n_gpu_layers=-1 offloads the whole model to GPU; no-op for CPU builds.
    return Llama(
        model_path=path,
        embedding=True,
        n_ctx=N_CTX,
        pooling_type=3,
        n_gpu_layers=-1,
        verbose=False,
    )


def load_index_model() -> Llama:
    return _load(INDEX_MODEL_FILE)


def load_query_model() -> Llama:
    return _load(QUERY_MODEL_FILE)


def _truncate_to_tokens(model: Llama, text: str, max_tokens: int) -> str:
    """Truncate `text` so it tokenizes to at most `max_tokens` tokens.

    Cheap path first (try the full string), only re-tokenize and detokenize
    the prefix if it's actually too long.
    """
    tokens = model.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = model.detokenize(tokens[:max_tokens]).decode("utf-8", errors="ignore")
    return truncated


def _embed(model: Llama, text: str) -> np.ndarray:
    """Return a single L2-normalized embedding vector."""
    text = _truncate_to_tokens(model, text, MAX_INPUT_TOKENS)
    out = model.create_embedding(text)
    vec = np.asarray(out["data"][0]["embedding"], dtype=np.float32)
    if vec.ndim == 2:
        # Defensive: some builds yield per-token rows even with pooling set.
        vec = vec[-1]
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def embed_pending(conn: sqlite3.Connection, batch_commit: int = 16) -> int:
    rows = conn.execute(
        """
        SELECT rowid, name, body_clean
        FROM tags
        WHERE body_clean IS NOT NULL
          AND body_clean != ''
          AND embedded_at IS NULL
        ORDER BY post_count DESC
        """
    ).fetchall()
    if not rows:
        return 0

    model = load_index_model()
    actual_dim = model.n_embd()
    if actual_dim != EMBED_DIM:
        raise RuntimeError(
            f"Model embedding dim {actual_dim} != schema dim {EMBED_DIM}"
        )

    done = 0
    pending: list[tuple[int, bytes]] = []
    pending_marks: list[int] = []
    for rowid, name, body_clean in tqdm(rows, desc="embed", unit="tag"):
        vec = _embed(model, body_clean)
        blob = struct.pack(f"{EMBED_DIM}f", *vec.tolist())
        pending.append((rowid, blob))
        pending_marks.append(rowid)
        done += 1

        if len(pending) >= batch_commit:
            _flush(conn, pending, pending_marks)
            pending.clear()
            pending_marks.clear()

    if pending:
        _flush(conn, pending, pending_marks)
    return done


def _flush(
    conn: sqlite3.Connection,
    rows: list[tuple[int, bytes]],
    marks: list[int],
) -> None:
    now = _now()
    with conn:
        for rowid, blob in rows:
            # Replace any existing vector for this rowid (vec0 keys by rowid).
            conn.execute("DELETE FROM vec_tags WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO vec_tags(rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
        placeholders = ",".join("?" * len(marks))
        conn.execute(
            f"UPDATE tags SET embedded_at = ? WHERE rowid IN ({placeholders})",
            (now, *marks),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Embed wiki bodies into sqlite-vec")
    p.add_argument("--db", default="danbooru.db")
    args = p.parse_args()

    conn = connect(args.db)
    n = embed_pending(conn)
    print(f"Embedded {n} tags", file=sys.stderr)


if __name__ == "__main__":
    main()
