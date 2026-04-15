"""LLM-assisted DB canonicalization.

Reads a custom-database JSON (list of (subject, relation, object) triplets),
groups triplets that express the same fact via:
  1. literal normalized-string dedup,
  2. FAISS top-k candidate generation over sentence-transformer embeddings,
  3. LLM adjudication of ambiguous candidate pairs, and
  4. union-find over "yes" edges to produce equivalence classes.

Writes an `equivalence_classes.json` that downstream code (e.g. a
DEL-ON-CANON state) consumes to expand what "delete fact f" covers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from equivalence import normalize_text


Triplet = tuple[str, str, str]
TripletId = int


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


def load_triplets(database_path: Path) -> list[Triplet]:
    with database_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload["triplets"] if isinstance(payload, dict) else payload
    triplets: list[Triplet] = []
    for row in raw:
        if not (isinstance(row, (list, tuple)) and len(row) == 3):
            raise ValueError(f"Expected 3-tuple triplet, got: {row!r}")
        subject, relation, obj = row
        triplets.append((str(subject), str(relation), str(obj)))
    return triplets


def triplet_text(triplet: Triplet) -> str:
    return f"{triplet[0]} | {triplet[1]} | {triplet[2]}"


# ---------------------------------------------------------------------------
# Step 1: literal dedup
# ---------------------------------------------------------------------------


def literal_key(triplet: Triplet) -> tuple[str, str, str]:
    return (
        normalize_text(triplet[0]),
        normalize_text(triplet[1]),
        normalize_text(triplet[2]),
    )


def literal_dedup_groups(
    triplets: list[Triplet],
) -> dict[tuple[str, str, str], list[TripletId]]:
    groups: dict[tuple[str, str, str], list[TripletId]] = {}
    for idx, triplet in enumerate(triplets):
        groups.setdefault(literal_key(triplet), []).append(idx)
    return groups


# ---------------------------------------------------------------------------
# Step 2: FAISS candidate generation
# ---------------------------------------------------------------------------


@dataclass
class CandidatePair:
    left: TripletId
    right: TripletId
    score: float


def build_candidate_pairs(
    triplets: list[Triplet],
    top_k: int,
    embedding_model_name: str,
    min_score: float,
) -> list[CandidatePair]:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    model = SentenceTransformer(embedding_model_name)
    texts = [triplet_text(t) for t in triplets]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    effective_k = min(top_k + 1, len(triplets))
    scores, neighbors = index.search(embeddings, effective_k)

    seen: set[tuple[TripletId, TripletId]] = set()
    pairs: list[CandidatePair] = []
    for i, (row_scores, row_neighbors) in enumerate(zip(scores, neighbors)):
        for score, j in zip(row_scores, row_neighbors):
            if j == -1 or j == i:
                continue
            if score < min_score:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append(CandidatePair(left=a, right=b, score=float(score)))
    return pairs


# ---------------------------------------------------------------------------
# Step 3: LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You judge whether two knowledge-base triplets express the SAME FACT.

A triplet is (subject, relation, object). Two triplets express the same fact if
and only if they assert the same thing about the world, possibly using different
surface forms:

- Aliases or paraphrases of the subject, relation, or object (e.g. "USA" vs
  "United States of America"; "capital" vs "capital city").
- Inverse relations that together describe the same fact (e.g. (Paris,
  capital_of, France) and (France, has_capital, Paris)).

Do NOT merge triplets that share only the subject or only the object but
describe different relations (e.g. (Paris, capital, France) and (Paris,
located_in, France) are DIFFERENT facts).

Reply with a single JSON object of the form:
{"same_fact": true|false, "reason": "<one short sentence>"}
No prose outside the JSON."""


JUDGE_USER_TEMPLATE = """Triplet A: ({a_sub}, {a_rel}, {a_obj})
Triplet B: ({b_sub}, {b_rel}, {b_obj})

Do A and B express the same fact?"""


JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def canonical_pair_key(a: Triplet, b: Triplet) -> str:
    def tkey(t: Triplet) -> str:
        return "\x1f".join(normalize_text(field) for field in t)

    lo, hi = sorted([tkey(a), tkey(b)])
    digest = hashlib.sha1(f"{lo}\x1e{hi}".encode("utf-8")).hexdigest()
    return digest


class JudgeClient:
    def judge(self, a: Triplet, b: Triplet) -> tuple[bool, str]:
        raise NotImplementedError


class VLLMJudgeClient(JudgeClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: float = 60.0,
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

    def judge(self, a: Triplet, b: Triplet) -> tuple[bool, str]:
        user_msg = JUDGE_USER_TEMPLATE.format(
            a_sub=a[0],
            a_rel=a[1],
            a_obj=a[2],
            b_sub=b[0],
            b_rel=b[1],
            b_obj=b[2],
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
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
        return parse_judge_response(content)


def parse_judge_response(content: str) -> tuple[bool, str]:
    match = JSON_BLOCK_RE.search(content)
    if not match:
        return False, f"unparseable response: {content.strip()[:120]}"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, f"invalid json: {match.group(0)[:120]}"
    verdict = bool(parsed.get("same_fact", False))
    reason = str(parsed.get("reason", "")).strip()
    return verdict, reason


# ---------------------------------------------------------------------------
# Judgment cache
# ---------------------------------------------------------------------------


@dataclass
class JudgmentCache:
    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "JudgmentCache":
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
        else:
            entries = {}
        return cls(path=path, entries=entries)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.entries[key] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Step 4: union-find merge
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry

    def classes(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return groups


def pick_canonical(members: list[TripletId], triplets: list[Triplet]) -> TripletId:
    # Shortest normalized total length = most canonical-looking form.
    return min(
        members,
        key=lambda i: (
            sum(len(normalize_text(field)) for field in triplets[i]),
            i,
        ),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def canonicalize(
    triplets: list[Triplet],
    judge: JudgeClient,
    cache: JudgmentCache,
    embedding_model_name: str,
    top_k: int,
    min_embedding_score: float,
    progress: bool = True,
) -> dict[str, Any]:
    uf = UnionFind(len(triplets))

    literal_groups = literal_dedup_groups(triplets)
    literal_merge_count = 0
    for members in literal_groups.values():
        if len(members) > 1:
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)
                literal_merge_count += 1

    pairs = build_candidate_pairs(
        triplets=triplets,
        top_k=top_k,
        embedding_model_name=embedding_model_name,
        min_score=min_embedding_score,
    )

    if progress:
        try:
            from tqdm import tqdm

            iterator: Iterable[CandidatePair] = tqdm(pairs, desc="LLM judge")
        except ImportError:
            iterator = pairs
    else:
        iterator = pairs

    judged_yes = 0
    judged_no = 0
    cache_hits = 0
    for pair in iterator:
        a, b = triplets[pair.left], triplets[pair.right]
        if uf.find(pair.left) == uf.find(pair.right):
            continue

        key = canonical_pair_key(a, b)
        cached = cache.get(key)
        if cached is not None:
            cache_hits += 1
            verdict = bool(cached["same_fact"])
        else:
            verdict, reason = judge.judge(a, b)
            cache.put(
                key,
                {
                    "triplet_a": list(a),
                    "triplet_b": list(b),
                    "embedding_score": pair.score,
                    "same_fact": verdict,
                    "reason": reason,
                },
            )

        if verdict:
            uf.union(pair.left, pair.right)
            judged_yes += 1
        else:
            judged_no += 1

    cache.save()

    classes: list[dict[str, Any]] = []
    for class_id, (_, members) in enumerate(sorted(uf.classes().items())):
        canonical_id = pick_canonical(members, triplets)
        classes.append(
            {
                "class_id": class_id,
                "canonical_triplet_id": canonical_id,
                "canonical": list(triplets[canonical_id]),
                "member_triplet_ids": members,
                "members": [list(triplets[i]) for i in members],
                "size": len(members),
            }
        )

    return {
        "triplet_count": len(triplets),
        "class_count": len(classes),
        "literal_merges": literal_merge_count,
        "candidate_pairs_considered": len(pairs),
        "llm_judged_same_fact": judged_yes,
        "llm_judged_different": judged_no,
        "cache_hits": cache_hits,
        "classes": classes,
    }
