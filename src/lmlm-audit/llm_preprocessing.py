"""LLM-based atomic-filter preprocessing for custom-database JSONs.

Given a database JSON like data/custom_databases/<domain>/<variant>.json with
top-level keys {entities, relationships, return_values, triplets}, this module:

  1. Loads the DB.
  2. Sends the full triplet list to an LLM judge (e.g. Llama 3.3 70B via vLLM)
     with a single-prompt-per-DB request asking it to group triplets that
     express the SAME FACT (aliases, paraphrased relations, inverse relations).
  3. Parses and validates the judge's JSON response.
  4. Builds a filtered DB dict keeping only the canonical triplet from each
     group plus all unmerged triplets, while preserving the other top-level
     keys (entities, relationships, return_values) unchanged.
  5. Optionally caches the filtered DB and the raw LLM judgment for
     reproducibility and auditability.

The atomic-filter pass is designed to be run BEFORE `run_audit.py` so that the
auditor sees a minimal, de-duplicated fact set rather than the noisy-variant
DBs (e.g. countries/alias.json, countries/collision.json).
"""

from __future__ import annotations

import copy
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


Triplet = tuple[str, str, str]


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


def load_database(db_path: Path) -> dict[str, Any]:
    """Load a custom database JSON into a dict.

    The DB is expected to have the top-level keys {entities, relationships,
    return_values, triplets}. This function does not enforce that schema
    strictly; it just returns whatever the JSON decodes to, as long as it's a
    dict.
    """
    with Path(db_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected database JSON to decode to a dict, got {type(payload).__name__}"
        )
    return payload


def load_triplets(db_path: Path) -> list[Triplet]:
    """Load just the triplet list from a database JSON.

    Accepts either a dict with a "triplets" field or a raw list for robustness.
    Each triplet is coerced to a 3-tuple of strings.
    """
    with Path(db_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload["triplets"] if isinstance(payload, dict) else payload
    triplets: list[Triplet] = []
    for row in raw:
        if not (isinstance(row, (list, tuple)) and len(row) == 3):
            raise ValueError(f"Expected 3-tuple triplet, got: {row!r}")
        subject, relation, obj = row
        triplets.append((str(subject), str(relation), str(obj)))
    return triplets


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

ATOMIC_FILTER_SYSTEM_PROMPT = """You judge whether a list of knowledge-base triplets can be grouped because some of them express the SAME FACT.

A triplet is (subject, relation, object). Two or more triplets express the same fact if and only if they assert the same thing about the world, possibly using different surface forms. Group them together when any of the following apply:

- Aliases or paraphrases of the subject, relation, or object. For example, ("USA", "Capital City", "Washington, D.C.") and ("United States", "Capital", "Washington, D.C.") are aliases + paraphrased relation, same fact, so MERGE them.
- Inverse relations that together describe the same fact. For example, ("Paris", "capital_of", "France") and ("France", "has_capital", "Paris") are inverse relations expressing the same fact, so MERGE them.

Be conservative. Do NOT merge triplets that share only the subject or only the object but describe different relations. For example, ("USA", "Capital", "Washington, D.C.") vs ("USA", "Currency", "US Dollar") have the same subject but different relations and therefore describe different facts, so DO NOT MERGE them.

You will receive a numbered list of triplets. Your job is to partition the indices 0..N-1 such that:
- indices of triplets that express the same fact are grouped together, and
- one member of each group is designated the "canonical" triplet (prefer the shortest / most canonical surface form),
- any triplet that doesn't merge with anything goes into "unmerged".

Reply with a single JSON object of the exact shape:
{"groups": [{"group_id": <int>, "canonical_triplet_id": <int>, "member_indices": [<int>, ...], "reason": "<short>"}, ...], "unmerged": [<int>, ...]}

Every triplet index from 0..N-1 must appear exactly once across groups.member_indices and unmerged. Each group's canonical_triplet_id must be an element of its own member_indices. No prose outside the JSON."""


ATOMIC_FILTER_USER_TEMPLATE = """Triplets:
{enumerated_triplets}

Group indices that express the same fact. Return JSON matching this schema:
{{"groups": [{{"group_id": <int>, "canonical_triplet_id": <int>, "member_indices": [<int>, ...], "reason": "<short>"}}, ...], "unmerged": [<int>, ...]}}
Every triplet index from 0..N-1 must appear exactly once across groups.member_indices and unmerged. canonical_triplet_id must be in its own member_indices. No prose outside the JSON."""


def _enumerate_triplets(triplets: list[Triplet]) -> str:
    return "\n".join(f"{i}. ({t[0]}, {t[1]}, {t[2]})" for i, t in enumerate(triplets))


def build_atomic_filter_messages(
    triplets: list[Triplet],
) -> list[dict[str, str]]:
    """Return the chat messages used for the atomic-filter request."""
    user_prompt = ATOMIC_FILTER_USER_TEMPLATE.format(
        enumerated_triplets=_enumerate_triplets(triplets)
    )
    return [
        {"role": "system", "content": ATOMIC_FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Judge clients
# ---------------------------------------------------------------------------


class JudgeClient(ABC):
    """Abstract LLM client that answers atomic-filter grouping requests."""

    @abstractmethod
    def filter_atomic(self, triplets: list[Triplet]) -> dict[str, Any]:
        """Return the parsed, validated grouping payload for `triplets`.

        The return value must match:
            {"groups": [{"group_id": int, "canonical_triplet_id": int,
                         "member_indices": [int, ...], "reason": str}, ...],
             "unmerged": [int, ...]}
        """


class VLLMJudgeClient(JudgeClient):
    """OpenAI-compatible (vLLM serve) judge that does atomic filtering."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        import requests

        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "not-needed"

    def filter_atomic(self, triplets: list[Triplet]) -> dict[str, Any]:
        messages = build_atomic_filter_messages(triplets)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_atomic_filter_response(content, expected_n=len(triplets))


class StubJudgeClient(JudgeClient):
    """Returns a canned response. For tests."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def filter_atomic(self, triplets: list[Triplet]) -> dict[str, Any]:
        # Return a deep copy so callers can mutate the result without
        # corrupting the canned response.
        return copy.deepcopy(self._response)


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(content: str) -> dict[str, Any]:
    """Find the first top-level JSON object in `content` and json.loads it.

    Uses a greedy `{...}` match and then falls back to a brace-balanced scan
    if that fails to decode (handles stray text around the JSON).
    """
    if not content or not content.strip():
        raise ValueError("empty LLM response")

    stripped = content.strip()
    # Strip a leading code fence if present (```json ... ```).
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    # Try the whole string first.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Greedy {...} capture.
    match = _JSON_BLOCK_RE.search(stripped)
    if match is not None:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Brace-balanced scan as a last resort.
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    raise ValueError(
        f"could not parse JSON object from LLM response: {content.strip()[:200]!r}"
    )


def parse_atomic_filter_response(
    content: str,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Extract and validate a grouping JSON payload from raw LLM output.

    Returns a dict of the form:
        {"groups": [{"group_id": int, "canonical_triplet_id": int,
                     "member_indices": [int, ...], "reason": str}, ...],
         "unmerged": [int, ...]}

    Raises `ValueError` with a helpful message if:
      - the content is not valid JSON,
      - the top-level shape is missing `groups` or `unmerged`,
      - indices are negative, duplicated, or (when `expected_n` is given)
        out of range,
      - a group's canonical_triplet_id is not in its member_indices,
      - not every index in [0, expected_n) appears exactly once across
        groups and unmerged.
    """
    obj = _extract_json_object(content)

    if "groups" not in obj or "unmerged" not in obj:
        raise ValueError(
            "atomic-filter JSON must have both 'groups' and 'unmerged' keys; "
            f"got keys={sorted(obj.keys())!r}"
        )

    groups_raw = obj["groups"]
    unmerged_raw = obj["unmerged"]

    if not isinstance(groups_raw, list):
        raise ValueError(f"'groups' must be a list, got {type(groups_raw).__name__}")
    if not isinstance(unmerged_raw, list):
        raise ValueError(
            f"'unmerged' must be a list, got {type(unmerged_raw).__name__}"
        )

    # Normalize unmerged.
    unmerged: list[int] = []
    for item in unmerged_raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"'unmerged' must contain ints, got {item!r} ({type(item).__name__})"
            )
        if item < 0:
            raise ValueError(f"'unmerged' contains a negative index: {item}")
        unmerged.append(int(item))

    # Normalize groups.
    groups: list[dict[str, Any]] = []
    for idx, g in enumerate(groups_raw):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{idx}] must be a dict, got {type(g).__name__}")
        for required in ("group_id", "canonical_triplet_id", "member_indices"):
            if required not in g:
                raise ValueError(f"groups[{idx}] is missing required key {required!r}")

        gid = g["group_id"]
        cid = g["canonical_triplet_id"]
        members_raw = g["member_indices"]
        reason = g.get("reason", "")

        if isinstance(gid, bool) or not isinstance(gid, int):
            raise ValueError(f"groups[{idx}].group_id must be an int, got {gid!r}")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise ValueError(
                f"groups[{idx}].canonical_triplet_id must be an int, got {cid!r}"
            )
        if not isinstance(members_raw, list):
            raise ValueError(
                f"groups[{idx}].member_indices must be a list, got "
                f"{type(members_raw).__name__}"
            )
        if not isinstance(reason, str):
            reason = str(reason)

        members: list[int] = []
        for m in members_raw:
            if isinstance(m, bool) or not isinstance(m, int):
                raise ValueError(
                    f"groups[{idx}].member_indices must contain ints, got {m!r}"
                )
            if m < 0:
                raise ValueError(
                    f"groups[{idx}].member_indices contains a negative index: {m}"
                )
            members.append(int(m))

        if not members:
            raise ValueError(f"groups[{idx}] has empty member_indices")
        if cid not in members:
            raise ValueError(
                f"groups[{idx}].canonical_triplet_id={cid} is not in its "
                f"member_indices={members!r}"
            )
        if len(set(members)) != len(members):
            raise ValueError(
                f"groups[{idx}].member_indices has duplicates: {members!r}"
            )

        groups.append(
            {
                "group_id": int(gid),
                "canonical_triplet_id": int(cid),
                "member_indices": members,
                "reason": reason,
            }
        )

    # Check disjointness + coverage.
    all_indices: list[int] = []
    for g in groups:
        all_indices.extend(g["member_indices"])
    all_indices.extend(unmerged)

    if len(set(all_indices)) != len(all_indices):
        seen: set[int] = set()
        dups: list[int] = []
        for i in all_indices:
            if i in seen:
                dups.append(i)
            seen.add(i)
        raise ValueError(
            f"indices are not disjoint across groups/unmerged; duplicates={sorted(set(dups))!r}"
        )

    if expected_n is not None:
        expected = set(range(expected_n))
        got = set(all_indices)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                "indices do not cover 0..N-1 exactly once "
                f"(expected_n={expected_n}, missing={missing!r}, extra={extra!r})"
            )

    return {"groups": groups, "unmerged": unmerged}


# ---------------------------------------------------------------------------
# Filtered DB construction
# ---------------------------------------------------------------------------


def build_filtered_database(
    original: dict[str, Any],
    groups_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a new DB dict keeping only canonical triplets + unmerged.

    Preserves the other top-level keys (entities, relationships,
    return_values) unchanged. Does not mutate `original`.

    The resulting `triplets` list is:
      1. Canonical triplets (one per group), ordered by group_id ascending.
      2. Unmerged triplets, ordered by index ascending.
    """
    if "triplets" not in original:
        raise ValueError("original database is missing a 'triplets' key")

    original_triplets = original["triplets"]
    n = len(original_triplets)

    groups = groups_payload.get("groups", [])
    unmerged = groups_payload.get("unmerged", [])

    def _check_idx(i: int, source: str) -> None:
        if not isinstance(i, int) or isinstance(i, bool):
            raise ValueError(f"{source} contains non-int index: {i!r}")
        if not (0 <= i < n):
            raise ValueError(f"{source} contains out-of-range index {i} (N={n})")

    sorted_groups = sorted(groups, key=lambda g: g["group_id"])
    sorted_unmerged = sorted(unmerged)

    new_triplets: list[Any] = []
    for g in sorted_groups:
        cid = g["canonical_triplet_id"]
        _check_idx(cid, "groups.canonical_triplet_id")
        new_triplets.append(copy.deepcopy(original_triplets[cid]))
    for i in sorted_unmerged:
        _check_idx(i, "unmerged")
        new_triplets.append(copy.deepcopy(original_triplets[i]))

    filtered: dict[str, Any] = {}
    for key, value in original.items():
        if key == "triplets":
            continue
        filtered[key] = copy.deepcopy(value)
    filtered["triplets"] = new_triplets
    return filtered


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, db_path: Path) -> tuple[Path, Path]:
    stem = db_path.stem
    filtered_cache = cache_dir / f"{stem}.filtered.json"
    groups_cache = cache_dir / f"{stem}.groups.json"
    return filtered_cache, groups_cache


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Top-level preprocessing entry points
# ---------------------------------------------------------------------------


def preprocess_database(
    db_path: Path,
    judge: JudgeClient,
    *,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Load a DB, run the LLM atomic filter, and return the filtered DB dict.

    If `cache_dir` is given, the filtered DB is cached as
    `<cache_dir>/<db_stem>.filtered.json` and the raw LLM judgment as
    `<cache_dir>/<db_stem>.groups.json`. A cached filtered DB is reused
    unless `force_recompute=True`.
    """
    db_path = Path(db_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    if cache_dir is not None and not force_recompute:
        filtered_cache, _ = _cache_paths(cache_dir, db_path)
        if filtered_cache.exists():
            cached = _read_json(filtered_cache)
            if isinstance(cached, dict):
                return cached

    original = load_database(db_path)
    triplets = load_triplets(db_path)

    groups_payload = judge.filter_atomic(triplets)
    # Defensive re-validation: parse_atomic_filter_response already checks
    # shape for VLLMJudgeClient, but a StubJudgeClient may return an
    # unvalidated payload.
    groups_payload = _ensure_valid_payload(groups_payload, expected_n=len(triplets))

    filtered = build_filtered_database(original, groups_payload)

    if cache_dir is not None:
        filtered_cache, groups_cache = _cache_paths(cache_dir, db_path)
        _write_json(filtered_cache, filtered)
        _write_json(groups_cache, groups_payload)

    return filtered


def preprocess_database_to_file(
    db_path: Path,
    judge: JudgeClient,
    output_path: Path,
    *,
    force_recompute: bool = False,
) -> Path:
    """Preprocess `db_path` and write the filtered DB to `output_path`.

    Uses `output_path.parent` as the groups cache directory, so the LLM
    judgment lands alongside the filtered DB for auditability. Returns
    `output_path`.
    """
    db_path = Path(db_path)
    output_path = Path(output_path)
    cache_dir = output_path.parent

    if output_path.exists() and not force_recompute:
        # Trust the cached filtered DB at output_path itself. Still load it
        # to validate JSON-ness before returning.
        try:
            _read_json(output_path)
            return output_path
        except (json.JSONDecodeError, OSError):
            pass  # fall through and recompute

    filtered = preprocess_database(
        db_path,
        judge,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    _write_json(output_path, filtered)
    return output_path


def _ensure_valid_payload(
    payload: dict[str, Any],
    expected_n: int,
) -> dict[str, Any]:
    """Validate a groups payload dict (as if it had been parsed from LLM text)."""
    if not isinstance(payload, dict):
        raise ValueError(f"judge returned non-dict payload: {type(payload).__name__}")
    # Re-serialize and re-parse to exercise the same validation path.
    return parse_atomic_filter_response(json.dumps(payload), expected_n=expected_n)
