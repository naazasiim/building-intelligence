"""Create/reuse the Atlas Search keyword index on chunks.text (Story 4.1).

Run: uv run python -m building_with_rag.ingestion.keyword_index
No Voyage calls, no data writes; a differing index is reported, never replaced.
"""

import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

WAIT_SECONDS = 120


def _find(coll):
    return next((i for i in coll.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME), None)


def _diffs(have: dict, want: dict, path: str = "") -> list[str]:
    out = []
    for key, value in want.items():
        here = f"{path}.{key}" if path else key
        if key not in have:
            out.append(f"{here} missing")
        elif isinstance(value, dict):
            out += _diffs(have[key], value, here)
        elif have[key] != value:
            out.append(f"{here}={have[key]!r} (expected {value!r})")
    return out


def main() -> None:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise SystemExit("MONGODB_URI is not set.")
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10000)
    coll = client[settings.mongodb_db_name][schema.CHUNKS_COLLECTION]
    idx = _find(coll)
    if idx is None:
        coll.create_search_index(
            SearchIndexModel(
                name=schema.KEYWORD_INDEX_NAME, type="search", definition=schema.KEYWORD_INDEX_DEFINITION
            )
        )
    else:
        diffs = _diffs(idx.get("latestDefinition") or idx.get("definition") or {}, schema.KEYWORD_INDEX_DEFINITION)
        if diffs:
            raise SystemExit(
                f"Index '{schema.KEYWORD_INDEX_NAME}' differs: {'; '.join(diffs)}. "
                "Drop it manually in Atlas, then re-run."
            )
    status = "UNKNOWN"
    for _ in range(WAIT_SECONDS // 5 + 1):
        idx = _find(coll)
        status = idx.get("status") if idx else "UNKNOWN"
        if idx and (idx.get("queryable") or status == "READY"):
            print(f"{schema.KEYWORD_INDEX_NAME}: {status} (queryable)")
            return
        time.sleep(5)
    raise SystemExit(f"{schema.KEYWORD_INDEX_NAME}: not ready after {WAIT_SECONDS}s (status: {status}).")


if __name__ == "__main__":
    main()
