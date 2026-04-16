"""
Unit tests for src/lmlm-audit/llm_preprocessing.py.

All tests use StubJudgeClient; VLLMJudgeClient is NOT exercised here (it requires
a live vLLM server). Filesystem tests go through tmp_path.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from llm_preprocessing import (
    JudgeClient,
    StubJudgeClient,
    build_filtered_database,
    load_database,
    load_triplets,
    parse_atomic_filter_response,
    preprocess_database,
    preprocess_database_to_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _sample_db() -> dict:
    return {
        "entities": ["USA", "United States", "Paris", "France"],
        "relationships": ["capital", "has_capital", "capital_of"],
        "return_values": ["Washington, D.C.", "Paris"],
        "triplets": [
            ["USA", "Capital City", "Washington, D.C."],
            ["United States", "Capital", "Washington, D.C."],
            ["Paris", "capital_of", "France"],
            ["France", "has_capital", "Paris"],
            ["USA", "Currency", "US Dollar"],
        ],
    }


def _sample_groups_payload() -> dict:
    # Merge 0 with 1 (aliases/paraphrased relation), 2 with 3 (inverse relation),
    # 4 unmerged.
    return {
        "groups": [
            {
                "group_id": 0,
                "canonical_triplet_id": 1,
                "member_indices": [0, 1],
                "reason": "alias + paraphrase",
            },
            {
                "group_id": 1,
                "canonical_triplet_id": 2,
                "member_indices": [2, 3],
                "reason": "inverse relation",
            },
        ],
        "unmerged": [4],
    }


# ===========================================================================
# load_database / load_triplets
# ===========================================================================


class TestLoadDatabase:
    def test_happy_path(self, tmp_path):
        db = _sample_db()
        p = _write_json(tmp_path / "sample.json", db)
        result = load_database(p)
        assert set(result.keys()) == {
            "entities",
            "relationships",
            "return_values",
            "triplets",
        }
        assert result["triplets"][0] == ["USA", "Capital City", "Washington, D.C."]

    def test_non_dict_raises(self, tmp_path):
        p = _write_json(tmp_path / "list.json", [1, 2, 3])
        with pytest.raises(ValueError, match="dict"):
            load_database(p)


class TestLoadTriplets:
    def test_dict_with_triplets(self, tmp_path):
        p = _write_json(tmp_path / "db.json", _sample_db())
        triplets = load_triplets(p)
        assert len(triplets) == 5
        assert triplets[0] == ("USA", "Capital City", "Washington, D.C.")
        # each row is coerced to tuple of str
        for row in triplets:
            assert isinstance(row, tuple)
            assert len(row) == 3
            assert all(isinstance(x, str) for x in row)

    def test_raw_list_payload(self, tmp_path):
        raw = [["A", "R", "X"], ["B", "R", "Y"]]
        p = _write_json(tmp_path / "raw.json", raw)
        triplets = load_triplets(p)
        assert triplets == [("A", "R", "X"), ("B", "R", "Y")]

    def test_malformed_row_wrong_length(self, tmp_path):
        bad = {"triplets": [["only", "two"]]}
        p = _write_json(tmp_path / "bad.json", bad)
        with pytest.raises(ValueError, match="3-tuple"):
            load_triplets(p)

    def test_malformed_row_too_long(self, tmp_path):
        bad = {"triplets": [["a", "b", "c", "d"]]}
        p = _write_json(tmp_path / "bad.json", bad)
        with pytest.raises(ValueError, match="3-tuple"):
            load_triplets(p)

    def test_coerces_non_string_fields(self, tmp_path):
        # ints are valid JSON but should be coerced to str.
        p = _write_json(tmp_path / "ints.json", {"triplets": [[1, 2, 3]]})
        triplets = load_triplets(p)
        assert triplets == [("1", "2", "3")]


# ===========================================================================
# parse_atomic_filter_response
# ===========================================================================


class TestParseAtomicFilterResponse:
    def _valid_payload_str(self, n: int = 5) -> str:
        return json.dumps(_sample_groups_payload())

    def test_valid_json_parses(self):
        result = parse_atomic_filter_response(self._valid_payload_str(), expected_n=5)
        assert len(result["groups"]) == 2
        assert result["unmerged"] == [4]

    def test_code_fence_json(self):
        raw = "```json\n" + self._valid_payload_str() + "\n```"
        result = parse_atomic_filter_response(raw, expected_n=5)
        assert result["unmerged"] == [4]

    def test_code_fence_no_lang(self):
        raw = "```\n" + self._valid_payload_str() + "\n```"
        result = parse_atomic_filter_response(raw, expected_n=5)
        assert len(result["groups"]) == 2

    def test_surrounding_prose(self):
        raw = (
            "Here is my response.\n\n"
            + self._valid_payload_str()
            + "\n\nLet me know if you need anything else."
        )
        result = parse_atomic_filter_response(raw, expected_n=5)
        assert len(result["groups"]) == 2
        assert result["unmerged"] == [4]

    def test_missing_index_raises(self):
        # total N=5 declared but index 3 is missing from coverage.
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0, 1],
                    "reason": "",
                },
            ],
            "unmerged": [2, 4],
        }
        with pytest.raises(ValueError, match="missing"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=5)

    def test_duplicated_index_raises(self):
        # Index 1 appears in both a group and unmerged.
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0, 1],
                    "reason": "",
                },
            ],
            "unmerged": [1, 2],
        }
        with pytest.raises(ValueError, match="disjoint"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=3)

    def test_canonical_not_in_members_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 7,  # not in [0,1]
                    "member_indices": [0, 1],
                    "reason": "",
                },
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="canonical_triplet_id"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=2)

    def test_negative_index_in_unmerged_raises(self):
        payload = {"groups": [], "unmerged": [-1, 2]}
        with pytest.raises(ValueError, match="negative"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_negative_index_in_members_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0, -1],
                    "reason": "",
                },
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="negative"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_atomic_filter_response("totally not json {{{")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_atomic_filter_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_atomic_filter_response("   \n  ")

    def test_expected_n_mismatch_raises(self):
        # Payload covers 5 indices; expected_n=6 → missing=[5].
        with pytest.raises(ValueError, match="cover"):
            parse_atomic_filter_response(self._valid_payload_str(), expected_n=6)

    def test_missing_groups_key_raises(self):
        payload = {"unmerged": [0, 1]}
        with pytest.raises(ValueError, match="groups"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=2)

    def test_missing_unmerged_key_raises(self):
        payload = {"groups": []}
        with pytest.raises(ValueError, match="unmerged"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=0)


# ===========================================================================
# build_filtered_database
# ===========================================================================


class TestBuildFilteredDatabase:
    def test_triplet_order_canonicals_then_unmerged(self):
        original = _sample_db()
        payload = _sample_groups_payload()
        filtered = build_filtered_database(original, payload)
        # Canonicals: group 0 → index 1 ("United States", ...), group 1 → index 2
        # ("Paris", "capital_of", ...). Unmerged: [4].
        assert filtered["triplets"] == [
            ["United States", "Capital", "Washington, D.C."],
            ["Paris", "capital_of", "France"],
            ["USA", "Currency", "US Dollar"],
        ]

    def test_canonical_is_picked_not_other_members(self):
        original = _sample_db()
        payload = _sample_groups_payload()
        filtered = build_filtered_database(original, payload)
        # The merged group's other member (index 0, "USA") must NOT appear.
        assert ["USA", "Capital City", "Washington, D.C."] not in filtered["triplets"]

    def test_other_top_level_keys_preserved(self):
        original = _sample_db()
        payload = _sample_groups_payload()
        filtered = build_filtered_database(original, payload)
        assert filtered["entities"] == original["entities"]
        assert filtered["relationships"] == original["relationships"]
        assert filtered["return_values"] == original["return_values"]

    def test_original_not_mutated(self):
        original = _sample_db()
        reference = copy.deepcopy(original)
        payload = _sample_groups_payload()
        _ = build_filtered_database(original, payload)
        assert original == reference

    def test_returned_triplets_are_independent_copies(self):
        original = _sample_db()
        payload = _sample_groups_payload()
        filtered = build_filtered_database(original, payload)
        # Mutating the filtered result must not touch the original.
        filtered["triplets"][0][0] = "MUTATED"
        assert original["triplets"][1][0] == "United States"

    def test_all_unmerged_returns_sorted_by_index(self):
        original = _sample_db()
        # All 5 triplets unmerged, and intentionally shuffled input order.
        payload = {"groups": [], "unmerged": [4, 0, 3, 1, 2]}
        filtered = build_filtered_database(original, payload)
        assert filtered["triplets"] == original["triplets"]

    def test_missing_triplets_key_raises(self):
        bad = {"entities": [], "relationships": []}
        with pytest.raises(ValueError, match="triplets"):
            build_filtered_database(bad, _sample_groups_payload())

    def test_groups_sorted_by_group_id(self):
        original = _sample_db()
        # Reverse the group order in the payload; canonicals should still be
        # emitted in group_id-ascending order.
        payload = {
            "groups": [
                {
                    "group_id": 10,
                    "canonical_triplet_id": 4,
                    "member_indices": [4],
                    "reason": "",
                },
                {
                    "group_id": 1,
                    "canonical_triplet_id": 0,
                    "member_indices": [0],
                    "reason": "",
                },
            ],
            "unmerged": [1, 2, 3],
        }
        filtered = build_filtered_database(original, payload)
        # Group_id=1 canonical (index 0) must come before group_id=10
        # canonical (index 4).
        assert filtered["triplets"][0] == ["USA", "Capital City", "Washington, D.C."]
        assert filtered["triplets"][1] == ["USA", "Currency", "US Dollar"]
        # Then unmerged [1,2,3] sorted asc.
        assert filtered["triplets"][2] == ["United States", "Capital", "Washington, D.C."]
        assert filtered["triplets"][3] == ["Paris", "capital_of", "France"]
        assert filtered["triplets"][4] == ["France", "has_capital", "Paris"]


# ===========================================================================
# preprocess_database (StubJudgeClient)
# ===========================================================================


class CountingStubJudge(JudgeClient):
    """Stub that counts filter_atomic calls and can raise after the Nth call."""

    def __init__(self, response: dict, raise_after: int | None = None) -> None:
        self._response = response
        self.calls = 0
        self.raise_after = raise_after

    def filter_atomic(self, triplets):
        self.calls += 1
        if self.raise_after is not None and self.calls > self.raise_after:
            raise AssertionError(
                f"judge called more than {self.raise_after} time(s)"
            )
        return copy.deepcopy(self._response)


class TestPreprocessDatabase:
    def test_end_to_end_no_cache(self, tmp_path):
        db_path = _write_json(tmp_path / "countries.json", _sample_db())
        judge = StubJudgeClient(_sample_groups_payload())
        result = preprocess_database(db_path, judge)
        # Canonicals first (group_id asc), then unmerged.
        assert result["triplets"] == [
            ["United States", "Capital", "Washington, D.C."],
            ["Paris", "capital_of", "France"],
            ["USA", "Currency", "US Dollar"],
        ]
        # Other top-level keys preserved.
        assert result["entities"] == _sample_db()["entities"]

    def test_cache_written_on_first_call(self, tmp_path):
        db_path = _write_json(tmp_path / "countries.json", _sample_db())
        cache_dir = tmp_path / "cache"
        judge = StubJudgeClient(_sample_groups_payload())
        preprocess_database(db_path, judge, cache_dir=cache_dir)

        filtered_cache = cache_dir / "countries.filtered.json"
        groups_cache = cache_dir / "countries.groups.json"
        assert filtered_cache.exists()
        assert groups_cache.exists()

        with filtered_cache.open() as f:
            cached_filtered = json.load(f)
        assert cached_filtered["triplets"][0] == [
            "United States",
            "Capital",
            "Washington, D.C.",
        ]

        with groups_cache.open() as f:
            cached_groups = json.load(f)
        assert cached_groups["unmerged"] == [4]
        assert len(cached_groups["groups"]) == 2

    def test_cache_reused_without_rerunning_judge(self, tmp_path):
        db_path = _write_json(tmp_path / "countries.json", _sample_db())
        cache_dir = tmp_path / "cache"

        # First call populates the cache.
        judge = CountingStubJudge(_sample_groups_payload(), raise_after=1)
        preprocess_database(db_path, judge, cache_dir=cache_dir)
        assert judge.calls == 1

        # Second call must hit the cache and NOT invoke filter_atomic again.
        result = preprocess_database(db_path, judge, cache_dir=cache_dir)
        assert judge.calls == 1  # still 1
        assert result["triplets"][0] == [
            "United States",
            "Capital",
            "Washington, D.C.",
        ]

    def test_force_recompute_bypasses_cache(self, tmp_path):
        db_path = _write_json(tmp_path / "countries.json", _sample_db())
        cache_dir = tmp_path / "cache"

        judge = CountingStubJudge(_sample_groups_payload())
        preprocess_database(db_path, judge, cache_dir=cache_dir)
        assert judge.calls == 1

        preprocess_database(
            db_path, judge, cache_dir=cache_dir, force_recompute=True
        )
        assert judge.calls == 2

    def test_no_cache_dir_runs_judge_every_time(self, tmp_path):
        db_path = _write_json(tmp_path / "countries.json", _sample_db())
        judge = CountingStubJudge(_sample_groups_payload())
        preprocess_database(db_path, judge)
        preprocess_database(db_path, judge)
        assert judge.calls == 2


# ===========================================================================
# preprocess_database_to_file
# ===========================================================================


class TestPreprocessDatabaseToFile:
    def test_writes_filtered_db_and_returns_path(self, tmp_path):
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        output_path = tmp_path / "out" / "countries.filtered.json"
        judge = StubJudgeClient(_sample_groups_payload())

        returned = preprocess_database_to_file(db_path, judge, output_path)

        assert returned == output_path
        assert output_path.exists()

    def test_groups_cache_in_output_parent(self, tmp_path):
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        output_path = tmp_path / "out" / "countries.filtered.json"
        judge = StubJudgeClient(_sample_groups_payload())

        preprocess_database_to_file(db_path, judge, output_path)

        groups_cache = output_path.parent / "countries.groups.json"
        assert groups_cache.exists()
        with groups_cache.open() as f:
            cached = json.load(f)
        assert cached["unmerged"] == [4]

    def test_roundtrip_matches_preprocess_database(self, tmp_path):
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        output_path = tmp_path / "out" / "countries.filtered.json"

        in_memory = preprocess_database(
            db_path, StubJudgeClient(_sample_groups_payload())
        )
        preprocess_database_to_file(
            db_path, StubJudgeClient(_sample_groups_payload()), output_path
        )
        with output_path.open() as f:
            on_disk = json.load(f)

        assert in_memory == on_disk

    def test_existing_output_reused_without_judge(self, tmp_path):
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        output_path = tmp_path / "out" / "countries.filtered.json"

        # First call populates output.
        first_judge = CountingStubJudge(_sample_groups_payload())
        preprocess_database_to_file(db_path, first_judge, output_path)
        assert first_judge.calls == 1

        # A fresh judge that would blow up if invoked is never called, because
        # the output file already exists.
        second_judge = CountingStubJudge(_sample_groups_payload(), raise_after=0)
        preprocess_database_to_file(db_path, second_judge, output_path)
        assert second_judge.calls == 0

    def test_force_recompute_rewrites_output(self, tmp_path):
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        output_path = tmp_path / "out" / "countries.filtered.json"

        first_judge = CountingStubJudge(_sample_groups_payload())
        preprocess_database_to_file(db_path, first_judge, output_path)
        second_judge = CountingStubJudge(_sample_groups_payload())
        preprocess_database_to_file(
            db_path, second_judge, output_path, force_recompute=True
        )
        assert second_judge.calls == 1
