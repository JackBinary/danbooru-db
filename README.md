# danbooru-db

A vector database of [Danbooru](https://danbooru.donmai.us) general-category
tag wiki pages, embedded with
[`mykor/harrier-oss-v1-270m-GGUF`](https://huggingface.co/mykor/harrier-oss-v1-270m-GGUF)
— a GGUF of `microsoft/harrier-oss-v1-270m`, a 270M-param Gemma-embedding
model with 640-dim output and last-token pooling.

> **Indexing uses BF16, queries use Q8\_0** (both from `mykor`'s repo). BF16
> at index time keeps the cosine geometry clean across 9k embeddings; Q8\_0
> at query time is ~5× faster and produces query vectors that are
> numerically identical (cosine ≈ 0.9997) to BF16 — target ranks against
> the BF16 corpus are unchanged.
>
> Don't use [`keisuke-miyako/harrier-oss-v1-270m-gguf-q8_0`](https://huggingface.co/keisuke-miyako/harrier-oss-v1-270m-gguf-q8_0)
> for either role — its quantization is degraded enough that `long_hair`
> ranks below the top 500 for the query `"long hair"`.

## What's in the database

A single SQLite file (default `danbooru.db`) with:

| table | purpose |
|---|---|
| `tags` | one row per general tag with `post_count >= --min-posts` (default 1000) and a valid wiki page. Columns: `name`, `post_count`, `body_raw` (dtext source), `body_clean` (stripped), `see_also` (JSON array of tag names from the wiki's See Also section), `other_names`, timestamps. |
| `vec_tags` | `sqlite-vec` virtual table holding the 640-dim embedding for each tag, keyed by `tags.rowid`. |

The cleaned body has dtext markup removed, the entire `See Also` section
extracted to its own column, and everything from the first `Posts` header
onward dropped.

## Install

Requires Python 3.12 and a working C toolchain (for `llama-cpp-python`).

```sh
uv sync
```

For GPU acceleration, rebuild `llama-cpp-python` against your platform's
backend. On AMD/Intel/Apple iGPUs the easiest path is Vulkan (needs `glslc`,
Vulkan headers, and a Vulkan ICD installed):

```sh
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --reinstall --no-cache-dir \
    --no-binary llama-cpp-python llama-cpp-python
```

Other backends use the same pattern with a different `CMAKE_ARGS`:
`-DGGML_CUDA=on`, `-DGGML_HIP=on`, `-DGGML_METAL=on`. Once a GPU build is
installed, `load_index_model()` / `load_query_model()` pass `n_gpu_layers=-1`
automatically and the model offloads transparently — no API changes.

On a Strix Halo iGPU via Vulkan: ~37 emb/s for BF16, ~52 emb/s for Q8\_0
(against ~5.5 emb/s for the CPU BF16 build). Full re-embed of the 9k-tag
corpus drops from ~28 min to under 5 min.

## Build the DB

End-to-end (resumable — each phase skips rows that are already complete):

```sh
uv run danbooru-db-build --db danbooru.db --min-posts 1000
```

Or run phases individually:

```sh
uv run danbooru-db-fetch --db danbooru.db --phase tags    # ~1 min
uv run danbooru-db-fetch --db danbooru.db --phase wikis   # ~1 req/sec, hours
uv run danbooru-db-embed --db danbooru.db                 # ~10 tag/s on CPU
```

The wiki fetch is rate-limited to 1 request/second to be polite to Danbooru.
For ~5,000 tags expect ~90 minutes of wiki fetching.

## Query

```sh
uv run danbooru-db-query --db danbooru.db "a girl wearing a sailor uniform"
```

Output: `<distance>  <tag_name>  (<post_count>)  <snippet>`.

The query is wrapped in the Harrier instruction prefix (`Instruct: <task>\nQuery: <q>`),
which the model expects for query-side encoding. Override the task instruction with
`--instruction "..."` if your retrieval task differs.

## Notes / caveats

- `llama-cpp-python` hard-caps per-sequence context at 256 tokens (a baked-in
  `n_seq_max` constant, not configurable). Inputs are explicitly truncated to
  248 tokens (≈ 1000 chars) before embedding; the definition at the top of each
  wiki survives, the trailing related-tag lists do not.
- Tag fetch uses cursor pagination (`page=b<id>`) so it remains efficient past
  Danbooru's 1000-row soft limit.
- The wiki endpoint occasionally returns a list rather than a dict for slugs
  Danbooru can't uniquely resolve (e.g. tag names like `?`). `fetch.py` handles
  both shapes and falls back to an explicit title search.
