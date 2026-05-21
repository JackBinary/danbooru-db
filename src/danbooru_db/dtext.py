"""Strip Danbooru dtext markup and extract the See Also section.

dtext reference: https://danbooru.donmai.us/wiki_pages/help:dtext

This is a pragmatic parser — not a full reimplementation. It handles the
constructs that appear in general-category wiki pages and ignores edge cases
that don't affect embedding quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Headers: h1. through h6., optionally with #anchor before the period
# (dtext syntax: "h4#anchor. Title").
_HEADER_RE = re.compile(r"^h([1-6])(?:#\S+)?\.\s*(.+?)\s*$", re.MULTILINE)

# Block tags we treat as paragraph breaks (open/close)
_BLOCK_TAGS = {
    "quote", "code", "expand", "spoiler", "spoilers", "table", "tr", "td", "th",
    "thead", "tbody", "colgroup", "col", "tn",
}

_INLINE_TAGS = {"b", "i", "u", "s", "sub", "sup", "color", "small", "big"}

# [[wiki link]] or [[wiki link|display]]. Display may be empty: [[target|]].
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]*?))?\]\]")
# !post #12345 / !asset #12345 - dtext shorthand for embedded media.
_POST_EMBED_RE = re.compile(r"!\w+\s*#\d+:?")
# {{search}} -> drop search syntax, keep display
_SEARCH_LINK_RE = re.compile(r"\{\{([^}|]+?)(?:\|([^}]+?))?\}\}")
# "text":[url] or "text":url
_NAMED_URL_RE = re.compile(r'"([^"]+)":\[?([^\]\s]+)\]?')
# Bare brackets [tag] / [/tag] / [tag=foo]
_BRACKET_TAG_RE = re.compile(r"\[/?[a-zA-Z][a-zA-Z0-9]*(?:[ =][^\]]*)?\]")
# Reference like post #123 or topic #123 -- keep as-is, but strip the marker links
_HR_RE = re.compile(r"^\s*\*{3,}\s*$", re.MULTILINE)
# Multiple blank lines -> single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# List bullets: lines starting with "* " or "** "
_LIST_RE = re.compile(r"^\*+\s+", re.MULTILINE)
# Empty bullet lines (left over after !post #N removal): "- ", "- - ", etc.
_EMPTY_BULLET_RE = re.compile(r"^\s*-(?:\s*[-*])*\s*$", re.MULTILINE)


@dataclass
class ParsedWiki:
    body_clean: str          # dtext-stripped, no See Also, no Posts
    see_also: list[str]      # tag names extracted from See Also section


def _is_posts_header(title: str) -> bool:
    t = title.strip().lower().rstrip(":")
    return t in {"posts", "post"}


def _is_see_also_header(title: str) -> bool:
    t = title.strip().lower().rstrip(":")
    return t in {"see also", "related tags", "related", "see"}


def split_sections(body: str) -> tuple[str, str | None, str | None]:
    """Split body into (preamble, see_also_section, _) cutting everything from
    a Posts header onward.

    Walks headers in order. Anything from the first Posts/Examples header
    onward is dropped. The See Also section is everything between its own
    header and the next header (or the Posts header, whichever comes first).
    """
    headers = list(_HEADER_RE.finditer(body))
    if not headers:
        return body, None, None

    cut_idx = len(body)            # body[:cut_idx] is kept (before Posts)
    see_also_header_start: int | None = None  # start of "hN. See Also" line
    see_also_content_start: int | None = None  # first char after the header line
    see_also_end: int | None = None  # exclusive end of see-also span

    for i, m in enumerate(headers):
        title = m.group(2)
        # Start of the line after this header (skip its trailing \n if present).
        nl = body.find("\n", m.start())
        line_after = nl + 1 if nl != -1 else m.end()
        next_start = headers[i + 1].start() if i + 1 < len(headers) else len(body)

        if _is_posts_header(title):
            cut_idx = min(cut_idx, m.start())
            if see_also_header_start is not None and see_also_end is None:
                see_also_end = m.start()
            break

        if _is_see_also_header(title):
            see_also_header_start = m.start()
            see_also_content_start = line_after
            see_also_end = next_start  # may be overwritten if Posts comes next

    if see_also_header_start is not None:
        end = see_also_end if see_also_end is not None else cut_idx
        see_also_text = body[see_also_content_start:end]
        body_kept = body[:see_also_header_start] + body[end:cut_idx]
    else:
        see_also_text = None
        body_kept = body[:cut_idx]

    return body_kept, see_also_text, None


def extract_see_also_tags(see_also_text: str) -> list[str]:
    """Pull tag names from [[...]] links inside a See Also section."""
    tags: list[str] = []
    seen: set[str] = set()
    for m in _WIKI_LINK_RE.finditer(see_also_text):
        target = m.group(1).strip()
        # Wiki targets use underscores; titles may use spaces interchangeably.
        # Normalize to underscore form, lowercase, drop anchors.
        target = target.split("#", 1)[0]
        target = target.replace(" ", "_").lower()
        if not target or target in seen:
            continue
        seen.add(target)
        tags.append(target)
    return tags


def _wiki_link_replace(m: re.Match[str]) -> str:
    display = (m.group(2) or "").strip()
    if display:
        return display
    target = m.group(1).strip()
    # "tag_name (qualifier)" -> "tag_name" for cleaner reading
    return target.split(" (", 1)[0].replace("_", " ")


def _drop_empty_sections(text: str) -> str:
    """Drop section headers that are immediately followed by no content.

    A "header" here is any line whose original form started with hN. — at this
    point in the pipeline the hN. prefix is gone, so we recognise headers
    structurally: a short line followed by a blank line that's then either
    another short line, end-of-text, or more blank lines.

    Simpler heuristic: split by blank-line paragraphs and drop one-line
    paragraphs whose text is a known section-header word and that have no
    following content paragraph before the next header.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    keep: list[str] = []
    section_word = re.compile(r"^[A-Z][A-Za-z][\w \-]{0,40}$")
    for i, p in enumerate(paragraphs):
        stripped = p.strip()
        # Drop empty paragraphs.
        if not stripped:
            continue
        # If a one-line paragraph looks like a stranded header and the next
        # paragraph is also a header (or there is no next), drop it.
        if "\n" not in stripped and section_word.match(stripped):
            next_p = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ""
            if not next_p or (
                "\n" not in next_p and section_word.match(next_p)
            ):
                continue
        keep.append(stripped)
    return "\n\n".join(keep)


def strip_dtext(text: str) -> str:
    """Convert dtext to plain text suitable for embedding."""
    # Normalize line endings up front so all multiline regexes behave.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Wiki links: keep display text (group 2) or target (group 1)
    text = _WIKI_LINK_RE.sub(_wiki_link_replace, text)
    # Drop !post #N / !asset #N references
    text = _POST_EMBED_RE.sub("", text)
    # Search-style links {{...}} - keep display or target
    text = _SEARCH_LINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    # Named URLs - keep the visible text only
    text = _NAMED_URL_RE.sub(lambda m: m.group(1), text)
    # Headers: keep title text, drop the "hN." prefix
    text = _HEADER_RE.sub(lambda m: m.group(2), text)
    # List bullets - turn into "- "
    text = _LIST_RE.sub("- ", text)
    # Drop horizontal rules
    text = _HR_RE.sub("", text)
    # Strip any remaining bracket tags like [b] [/quote] [color=red]
    text = _BRACKET_TAG_RE.sub("", text)
    # Remove empty bullet lines left by stripped !post #N markers.
    text = _EMPTY_BULLET_RE.sub("", text)
    # Drop now-orphan section headers (e.g. "Examples" with no content under it).
    text = _drop_empty_sections(text)
    # Collapse blank lines.
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def parse(body: str) -> ParsedWiki:
    """Full pipeline: split, extract see-also, strip the rest."""
    kept, see_also_text, _ = split_sections(body)
    see_also = extract_see_also_tags(see_also_text) if see_also_text else []
    return ParsedWiki(body_clean=strip_dtext(kept), see_also=see_also)
