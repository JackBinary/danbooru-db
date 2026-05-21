"""Fetch tags and wiki pages from Danbooru into SQLite.

Two phases, both resumable:
    1. fetch_tags(): paginate /tags.json with category=general,
       post_count>min, has_wiki_page=true. Upserts into tags(name, post_count).
    2. fetch_wikis(): for every tag missing body_raw, GET /wiki_pages/<name>.json
       and write body_raw / other_names / wiki metadata.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone

import httpx
from tqdm import tqdm

from .db import connect
from .dtext import parse as parse_dtext

BASE_URL = "https://danbooru.donmai.us"
USER_AGENT = "danbooru-db/0.1 (vector-db builder; +https://github.com/)"
PAGE_LIMIT = 1000  # Danbooru max
REQUEST_INTERVAL = 1.0  # seconds between requests (polite)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=httpx.Timeout(30.0, connect=10.0),
    )


def _get_with_retry(client: httpx.Client, path: str, params: dict) -> httpx.Response:
    backoff = 2.0
    for attempt in range(6):
        try:
            resp = client.get(path, params=params)
        except httpx.RequestError as exc:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == 5:
                resp.raise_for_status()
            retry_after = float(resp.headers.get("retry-after", backoff))
            time.sleep(retry_after)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


def fetch_tags(
    conn: sqlite3.Connection,
    min_post_count: int = 1000,
    category: int = 0,
) -> int:
    """Populate the `tags` table. Cursor pagination by tag id descending."""
    inserted = 0
    last_id: int | None = None

    with _client() as client:
        pbar = tqdm(desc="tags", unit="tag")
        while True:
            params = {
                "search[category]": category,
                "search[post_count]": f">={min_post_count}",
                "search[has_wiki_page]": "true",
                "search[is_deprecated]": "false",
                "search[order]": "id_desc",
                "limit": PAGE_LIMIT,
            }
            if last_id is not None:
                # Cursor: fetch ids strictly less than last_id
                params["page"] = f"b{last_id}"

            resp = _get_with_retry(client, "/tags.json", params)
            page = resp.json()
            if not page:
                break

            with conn:
                for row in page:
                    conn.execute(
                        """
                        INSERT INTO tags (name, post_count, tag_id)
                        VALUES (?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            post_count = excluded.post_count,
                            tag_id = excluded.tag_id
                        """,
                        (row["name"], row["post_count"], row["id"]),
                    )
                    inserted += 1

            pbar.update(len(page))
            last_id = page[-1]["id"]
            if len(page) < PAGE_LIMIT:
                break
            time.sleep(REQUEST_INTERVAL)
        pbar.close()

    return inserted


def _fetch_one_wiki(client: httpx.Client, name: str) -> dict | None:
    """Return the wiki page dict for `name`, or None if no match.

    `/wiki_pages/<name>.json` usually returns a dict, but for some names it
    routes to the index action and returns a list (e.g. names that look like
    numeric IDs, or names with characters Rails treats specially). Fall back
    to an explicit title search in that case.
    """
    try:
        resp = _get_with_retry(client, f"/wiki_pages/{name}.json", params={})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    data = resp.json()
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("title") == name:
                return entry
        # No exact-title match in the list. Try an explicit title search.
        resp = _get_with_retry(
            client,
            "/wiki_pages.json",
            params={"search[title]": name, "limit": 1},
        )
        results = resp.json()
        if isinstance(results, list) and results and isinstance(results[0], dict):
            if results[0].get("title") == name:
                return results[0]
    return None


def fetch_wikis(conn: sqlite3.Connection) -> int:
    """Fetch wiki body for every tag where body_raw IS NULL."""
    rows = conn.execute(
        "SELECT name FROM tags WHERE body_raw IS NULL ORDER BY post_count DESC"
    ).fetchall()
    if not rows:
        return 0

    fetched = 0
    with _client() as client:
        for (name,) in tqdm(rows, desc="wiki", unit="page"):
            page = _fetch_one_wiki(client, name)
            if page is None:
                # Wiki page can't be found - mark as empty so we don't retry.
                with conn:
                    conn.execute(
                        "UPDATE tags SET body_raw = '', body_clean = '', "
                        "see_also = '[]', other_names = '[]', fetched_at = ? "
                        "WHERE name = ?",
                        (_now(), name),
                    )
                time.sleep(REQUEST_INTERVAL)
                continue

            body = page.get("body") or ""
            parsed = parse_dtext(body)
            other_names = page.get("other_names") or []

            with conn:
                conn.execute(
                    """
                    UPDATE tags SET
                        wiki_id = ?,
                        body_raw = ?,
                        body_clean = ?,
                        see_also = ?,
                        other_names = ?,
                        wiki_updated_at = ?,
                        fetched_at = ?
                    WHERE name = ?
                    """,
                    (
                        page.get("id"),
                        body,
                        parsed.body_clean,
                        json.dumps(parsed.see_also),
                        json.dumps(other_names),
                        page.get("updated_at"),
                        _now(),
                        name,
                    ),
                )
            fetched += 1
            time.sleep(REQUEST_INTERVAL)

    return fetched


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Danbooru tags + wikis into SQLite")
    p.add_argument("--db", default="danbooru.db", help="SQLite path")
    p.add_argument("--min-posts", type=int, default=1000)
    p.add_argument("--phase", choices=["tags", "wikis", "all"], default="all")
    args = p.parse_args()

    conn = connect(args.db)
    if args.phase in ("tags", "all"):
        n = fetch_tags(conn, min_post_count=args.min_posts)
        print(f"Upserted {n} tags", file=sys.stderr)
    if args.phase in ("wikis", "all"):
        n = fetch_wikis(conn)
        print(f"Fetched {n} new wiki pages", file=sys.stderr)


if __name__ == "__main__":
    main()
