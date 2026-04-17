"""
Unit tests for src/lmlm-audit/llm_preprocessing.py.

All tests use StubJudgeClient; VLLMJudgeClient is NOT exercised here (it requires
a live vLLM server). Filesystem tests go through tmp_path.
"""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from llm_preprocessing import (
    ATOMIC_FILTER_SYSTEM_PROMPT,
    JudgeClient,
    StubJudgeClient,
    VLLMJudgeClient,
    _enumerate_triplets,
    _ensure_valid_payload,
    build_atomic_filter_messages,
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
        assert filtered["triplets"][2] == [
            "United States",
            "Capital",
            "Washington, D.C.",
        ]
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
            raise AssertionError(f"judge called more than {self.raise_after} time(s)")
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

        preprocess_database(db_path, judge, cache_dir=cache_dir, force_recompute=True)
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

    def test_existing_invalid_output_is_recomputed(self, tmp_path):
        """Corrupt cached output on disk should be recomputed, not crash."""
        db_path = _write_json(tmp_path / "db" / "countries.json", _sample_db())
        # Use a filename that does NOT collide with the internal cache path
        # (<output_parent>/<db_stem>.filtered.json).
        output_path = tmp_path / "out" / "my_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write junk at the output_path that json.loads cannot parse.
        output_path.write_text("not valid json {{{", encoding="utf-8")

        judge = StubJudgeClient(_sample_groups_payload())
        returned = preprocess_database_to_file(db_path, judge, output_path)

        # The invalid file was replaced with a real filtered DB.
        assert returned == output_path
        with output_path.open() as f:
            cached = json.load(f)
        assert cached["triplets"][0] == [
            "United States",
            "Capital",
            "Washington, D.C.",
        ]


# ===========================================================================
# _enumerate_triplets & build_atomic_filter_messages
# ===========================================================================


class TestEnumerateTriplets:
    def test_empty_list(self):
        assert _enumerate_triplets([]) == ""

    def test_single_triplet(self):
        result = _enumerate_triplets([("A", "R", "B")])
        assert result == "0. (A, R, B)"

    def test_multiple_triplets_numbered_sequentially(self):
        triplets = [("A", "R", "B"), ("C", "S", "D"), ("E", "T", "F")]
        result = _enumerate_triplets(triplets)
        lines = result.split("\n")
        assert lines[0] == "0. (A, R, B)"
        assert lines[1] == "1. (C, S, D)"
        assert lines[2] == "2. (E, T, F)"


class TestBuildAtomicFilterMessages:
    def test_returns_two_messages_with_correct_roles(self):
        messages = build_atomic_filter_messages([("A", "R", "B")])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_is_the_canonical_constant(self):
        messages = build_atomic_filter_messages([("A", "R", "B")])
        assert messages[0]["content"] == ATOMIC_FILTER_SYSTEM_PROMPT

    def test_user_prompt_contains_enumerated_triplets(self):
        messages = build_atomic_filter_messages(
            [("USA", "Capital", "DC"), ("France", "Capital", "Paris")]
        )
        user_content = messages[1]["content"]
        assert "0. (USA, Capital, DC)" in user_content
        assert "1. (France, Capital, Paris)" in user_content

    def test_empty_triplet_list(self):
        messages = build_atomic_filter_messages([])
        assert len(messages) == 2
        assert messages[1]["role"] == "user"


# ===========================================================================
# VLLMJudgeClient (mocked requests)
# ===========================================================================


@pytest.fixture
def fake_requests(monkeypatch):
    """Inject a fake `requests` module into sys.modules.

    CI does not install real requests, and VLLMJudgeClient.__init__ does
    `import requests`. A MagicMock stands in without any real HTTP dep.
    """
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


class TestVLLMJudgeClient:
    def _build_client(self, **overrides):
        # The module stores a reference to the `requests` module in
        # self._requests during __init__; swap it for a Mock right after.
        defaults = dict(
            base_url="http://localhost:8000",
            model="my-model",
            temperature=0.0,
            max_tokens=1024,
            timeout=30.0,
        )
        defaults.update(overrides)
        client = VLLMJudgeClient(**defaults)
        client._requests = MagicMock()
        return client

    def test_base_url_is_stripped_of_trailing_slash(self, fake_requests):
        client = self._build_client(base_url="http://host/")
        assert client.base_url == "http://host"

    def test_api_key_from_kwarg_wins(self, fake_requests):
        client = self._build_client(api_key="secret-key")
        assert client.api_key == "secret-key"

    def test_api_key_from_env_when_not_provided(self, fake_requests, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "env-key")
        client = VLLMJudgeClient(base_url="http://x", model="m")
        assert client.api_key == "env-key"

    def test_api_key_default_when_no_source(self, fake_requests, monkeypatch):
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        client = VLLMJudgeClient(base_url="http://x", model="m")
        assert client.api_key == "not-needed"

    def test_filter_atomic_posts_to_chat_completions_endpoint(self, fake_requests):
        client = self._build_client(base_url="http://host:9000")
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(_sample_groups_payload())}}]
        }
        client._requests.post.return_value = response

        triplets = [("A", "R", "B")] * 5
        result = client.filter_atomic(triplets)

        # Called the right URL.
        call_args = client._requests.post.call_args
        assert call_args.args[0] == "http://host:9000/v1/chat/completions"
        # Posted the right payload.
        posted = call_args.kwargs["json"]
        assert posted["model"] == "my-model"
        assert posted["temperature"] == 0.0
        assert posted["max_tokens"] == 1024
        assert len(posted["messages"]) == 2
        # Authorization header is set.
        assert "Authorization" in call_args.kwargs["headers"]
        assert call_args.kwargs["headers"]["Authorization"].startswith("Bearer ")
        # Parses the response.
        assert result["unmerged"] == [4]

    def test_filter_atomic_raises_on_http_error(self, fake_requests):
        client = self._build_client()
        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("500 Internal")
        client._requests.post.return_value = response
        with pytest.raises(RuntimeError, match="500"):
            client.filter_atomic([("A", "R", "B")])

    def test_filter_atomic_validates_response_shape(self, fake_requests):
        """Malformed LLM content should raise through parse_atomic_filter_response."""
        client = self._build_client()
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "not JSON at all"}}]
        }
        client._requests.post.return_value = response
        with pytest.raises(ValueError):
            client.filter_atomic([("A", "R", "B")])


# ===========================================================================
# Extra parse_atomic_filter_response error paths (shape validation)
# ===========================================================================


class TestParseAtomicFilterResponseExtra:
    def test_brace_balanced_fallback_recovers_json(self):
        """Greedy {.*} fails on trailing '}' but brace-balanced scan succeeds."""
        # Valid JSON followed by a stray '}' — greedy regex overshoots,
        # forcing the brace-balanced fallback to handle it.
        valid = json.dumps(_sample_groups_payload())
        raw = valid + "\nstray: }"
        result = parse_atomic_filter_response(raw, expected_n=5)
        assert result["unmerged"] == [4]
        assert len(result["groups"]) == 2

    def test_non_dict_json_raises(self):
        # JSON parses fine as list, but we want a dict.
        with pytest.raises(ValueError):
            parse_atomic_filter_response("[1, 2, 3]")

    def test_groups_not_list_raises(self):
        payload = {"groups": {}, "unmerged": []}
        with pytest.raises(ValueError, match="'groups' must be a list"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_unmerged_not_list_raises(self):
        payload = {"groups": [], "unmerged": {}}
        with pytest.raises(ValueError, match="'unmerged' must be a list"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_unmerged_contains_non_int_raises(self):
        payload = {"groups": [], "unmerged": ["0", 1]}
        with pytest.raises(ValueError, match="ints"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_unmerged_contains_bool_raises(self):
        payload = {"groups": [], "unmerged": [True]}
        with pytest.raises(ValueError, match="ints"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_group_not_dict_raises(self):
        payload = {"groups": [["not", "a", "dict"]], "unmerged": []}
        with pytest.raises(ValueError, match="must be a dict"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_group_missing_required_key_raises(self):
        payload = {
            "groups": [
                {"group_id": 0, "member_indices": [0]}
            ],  # no canonical_triplet_id
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="canonical_triplet_id"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=1)

    def test_group_id_not_int_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": "zero",
                    "canonical_triplet_id": 0,
                    "member_indices": [0],
                }
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="group_id must be an int"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_canonical_id_not_int_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": "zero",
                    "member_indices": [0],
                }
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="canonical_triplet_id must be an int"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_members_not_list_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": "not a list",
                }
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="member_indices must be a list"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_members_non_int_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0, "1"],
                }
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="member_indices must contain ints"):
            parse_atomic_filter_response(json.dumps(payload))

    def test_empty_members_raises(self):
        payload = {
            "groups": [
                {"group_id": 0, "canonical_triplet_id": 0, "member_indices": []}
            ],
            "unmerged": [0],
        }
        with pytest.raises(ValueError, match="empty member_indices"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=1)

    def test_duplicate_members_within_group_raises(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0, 0, 1],
                }
            ],
            "unmerged": [],
        }
        with pytest.raises(ValueError, match="duplicates"):
            parse_atomic_filter_response(json.dumps(payload), expected_n=2)

    def test_brace_balanced_scan_breaks_on_unparseable_candidate(self):
        """All three parse strategies fail → ValueError."""
        # A '{...}' span that is not valid JSON, plus a trailing '}' to throw
        # off the greedy regex. Brace-balanced scan extracts '{ bad }' which
        # also fails json.loads, triggering the break on line 253-254.
        raw = "{ bad }} extra"
        with pytest.raises(ValueError, match="could not parse JSON"):
            parse_atomic_filter_response(raw)

    def test_non_string_reason_is_coerced(self):
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 0,
                    "member_indices": [0],
                    "reason": 42,  # int instead of str
                }
            ],
            "unmerged": [],
        }
        result = parse_atomic_filter_response(json.dumps(payload), expected_n=1)
        assert isinstance(result["groups"][0]["reason"], str)
        assert result["groups"][0]["reason"] == "42"


# ===========================================================================
# build_filtered_database – index validation
# ===========================================================================


class TestBuildFilteredDatabaseIndexChecks:
    def test_unmerged_out_of_range_raises(self):
        original = _sample_db()  # N=5
        payload = {"groups": [], "unmerged": list(range(5)) + [99]}
        with pytest.raises(ValueError, match="out-of-range"):
            build_filtered_database(original, payload)

    def test_non_int_canonical_index_raises_via_build(self):
        """_check_idx rejects bool canonical index even if upstream missed it."""
        original = _sample_db()  # N=5
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    # bool is subclass of int, but _check_idx rejects bools.
                    "canonical_triplet_id": True,
                    "member_indices": [0],
                }
            ],
            "unmerged": [1, 2, 3, 4],
        }
        with pytest.raises(ValueError, match="non-int"):
            build_filtered_database(original, payload)

    def test_canonical_out_of_range_raises(self):
        original = _sample_db()  # N=5
        payload = {
            "groups": [
                {
                    "group_id": 0,
                    "canonical_triplet_id": 99,
                    "member_indices": [99],
                }
            ],
            "unmerged": [0, 1, 2, 3, 4],
        }
        with pytest.raises(ValueError, match="out-of-range"):
            build_filtered_database(original, payload)


# ===========================================================================
# _ensure_valid_payload
# ===========================================================================


class TestEnsureValidPayload:
    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="non-dict payload"):
            _ensure_valid_payload([], expected_n=0)  # type: ignore[arg-type]

    def test_valid_payload_roundtrip(self):
        result = _ensure_valid_payload(_sample_groups_payload(), expected_n=5)
        assert result["unmerged"] == [4]
        assert len(result["groups"]) == 2
