"""SQLite + sqlite-vec storage layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

EMBED_DIM = 640

SCHEMA = """
CREATE TABLE IF NOT EXISTS tags (
    rowid       INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    post_count  INTEGER NOT NULL,
    tag_id      INTEGER,
    wiki_id     INTEGER,
    body_raw    TEXT,
    body_clean  TEXT,
    see_also    TEXT,
    other_names TEXT,
    wiki_updated_at TEXT,
    fetched_at  TEXT,
    embedded_at TEXT
);

CREATE INDEX IF NOT EXISTS tags_post_count_idx ON tags(post_count DESC);
CREATE INDEX IF NOT EXISTS tags_needs_wiki_idx ON tags(name) WHERE body_raw IS NULL;
CREATE INDEX IF NOT EXISTS tags_needs_embed_idx ON tags(name) WHERE body_clean IS NOT NULL AND embedded_at IS NULL;
"""

VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_tags USING vec0(
    embedding float[{EMBED_DIM}]
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(SCHEMA)
    conn.executescript(VEC_SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
