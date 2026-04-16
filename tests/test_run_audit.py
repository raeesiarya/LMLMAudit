"""
Comprehensive unit tests for src/lmlm-audit/run_audit.py.

All pure-Python helpers (no model weights, no GPU) are exercised with normal,
boundary, and edge-case inputs.  Functions that require a live model are only
tested through mocks.  Several tests log diagnostic plots to W&B.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from run_audit import (
    _default_retrieval_trace,
    choose_answer,
    clean_answer,
    compute_generation_budget,
    extract_lookup_values,
    load_prompts,
    parse_args,
    prepare_prompt,
    retrieve_lookup_value,
    save_results,
)
from database_states import DatabaseState


# ===========================================================================
# clean_answer – original tests
# ===========================================================================


def test_clean_answer_strips_db_markup_and_html_tags() -> None:
    answer = clean_answer(
        "&lt;/poem&gt; <|db_entity|>Madhur Jaffrey<|db_relationship|>"
        "Award<|db_return|>Madison Sharma<|db_end|> Biography"
    )
    assert answer == "Biography"


def test_clean_answer_strips_standalone_db_special_tokens() -> None:
    answer = clean_answer('"<|db_entity|> Spice Girls <|db_return|>"')
    assert answer == "Spice Girls"


# ===========================================================================
# clean_answer – extended
# ===========================================================================


class TestCleanAnswer:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_answer("  Paris  ") == "Paris"

    def test_strips_answer_prefix_single(self):
        assert clean_answer("Answer: Paris") == "Paris"

    def test_strips_answer_prefix_repeated(self):
        assert clean_answer("Answer: Answer: Paris") == "Paris"

    def test_strips_answer_prefix_case_insensitive(self):
        assert clean_answer("answer: Paris") == "Paris"

    def test_strips_the_answer_is_prefix(self):
        assert clean_answer("The answer is Paris") == "Paris"

    def test_strips_it_is_prefix(self):
        assert clean_answer("It is Paris") == "Paris"

    def test_strips_its_prefix(self):
        assert clean_answer("It's Paris") == "Paris"

    def test_stops_at_question_marker(self):
        result = clean_answer("Paris\nQuestion: What is the capital?")
        assert result == "Paris"

    def test_stops_at_context_marker(self):
        result = clean_answer("Paris\nContext: France is a country.")
        assert result == "Paris"

    def test_stops_at_double_newline(self):
        result = clean_answer("Paris\n\nSome extra text")
        assert result == "Paris"

    def test_stops_at_fact_marker(self):
        result = clean_answer("Paris\nFact: France is in Europe.")
        assert result == "Paris"

    def test_stops_at_answer_marker(self):
        result = clean_answer("Paris\nAnswer: Berlin")
        assert result == "Paris"

    def test_unescapes_html_entities(self):
        result = clean_answer("&amp; &lt; &gt;")
        assert "&" in result

    def test_removes_db_markup_span(self):
        result = clean_answer(
            "<|db_entity|>X<|db_relationship|>Y<|db_return|>Z<|db_end|>"
        )
        assert "<|db_entity|>" not in result
        assert "<|db_end|>" not in result

    def test_removes_standalone_db_token(self):
        result = clean_answer("Hello <|db_return|> World")
        assert "<|db_return|>" not in result

    def test_removes_html_tags(self):
        result = clean_answer("<b>Paris</b>")
        assert "<b>" not in result
        assert "Paris" in result

    def test_keeps_first_sentence(self):
        result = clean_answer("Paris. It is the capital of France.")
        assert result == "Paris"

    def test_strips_trailing_punctuation(self):
        result = clean_answer("Paris.")
        assert result == "Paris"

    def test_strips_trailing_quotes(self):
        result = clean_answer('"Paris"')
        assert result == "Paris"

    def test_empty_string(self):
        result = clean_answer("")
        assert result == ""

    def test_only_whitespace(self):
        result = clean_answer("   ")
        assert result == ""

    def test_only_db_markup(self):
        result = clean_answer(
            "<|db_entity|>X<|db_relationship|>Y<|db_return|>Z<|db_end|>"
        )
        # The markup is removed; only surrounding whitespace remains → empty or the value before it
        assert result.strip() == "" or result.isspace() or result == ""

    def test_collapses_internal_spaces(self):
        result = clean_answer("New    York")
        assert result == "New York"

    def test_strips_leading_comma(self):
        result = clean_answer(",Paris")
        assert result == "Paris"

    def test_strips_trailing_semicolon(self):
        result = clean_answer("Paris;")
        assert result == "Paris"

    def test_complex_combined(self):
        result = clean_answer(
            'Answer: The answer is "Paris"\nQuestion: What city?'
        )
        assert result == "Paris"

    def test_answer_prefix_mixed_case(self):
        assert clean_answer("ANSWER: Paris") == "Paris"

    def test_exclamation_sentence_split(self):
        # The regex splits AFTER the punctuation; ! is not in the final strip
        # chars, so it is preserved in the first segment.
        result = clean_answer("Paris! It is wonderful.")
        assert result == "Paris!"

    def test_question_sentence_split(self):
        result = clean_answer("Paris? Yes, Paris.")
        assert result == "Paris?"


# ===========================================================================
# extract_lookup_values
# ===========================================================================


class TestExtractLookupValues:
    TEMPLATE = (
        "<|db_entity|>{entity}<|db_relationship|>{rel}<|db_return|>{value}<|db_end|>"
    )

    def test_single_lookup(self):
        raw = self.TEMPLATE.format(entity="Hexol", rel="First Described By", value="Jorgensen")
        result = extract_lookup_values(raw)
        assert result == ["Jorgensen"]

    def test_multiple_distinct_lookups(self):
        raw = (
            self.TEMPLATE.format(entity="A", rel="R", value="X")
            + self.TEMPLATE.format(entity="B", rel="S", value="Y")
        )
        result = extract_lookup_values(raw)
        assert "X" in result
        assert "Y" in result
        assert len(result) == 2

    def test_deduplicates_repeated_value(self):
        raw = (
            self.TEMPLATE.format(entity="A", rel="R", value="X")
            + self.TEMPLATE.format(entity="A", rel="R", value="X")
        )
        result = extract_lookup_values(raw)
        assert result.count("X") == 1

    def test_no_lookup_returns_empty(self):
        assert extract_lookup_values("plain text") == []

    def test_empty_string(self):
        assert extract_lookup_values("") == []

    def test_value_cleaned_before_return(self):
        # value with answer prefix should be cleaned
        raw = self.TEMPLATE.format(entity="A", rel="R", value="Answer: Paris")
        result = extract_lookup_values(raw)
        assert result == ["Paris"]

    def test_empty_value_ignored(self):
        raw = self.TEMPLATE.format(entity="A", rel="R", value="")
        result = extract_lookup_values(raw)
        assert result == []

    def test_multiline_value(self):
        raw = self.TEMPLATE.format(entity="A", rel="R", value="Paris\nFrance")
        result = extract_lookup_values(raw)
        # clean_answer truncates at \n\n but a single \n is fine here
        assert len(result) >= 1


# ===========================================================================
# choose_answer
# ===========================================================================


class TestChooseAnswer:
    def test_lookup_value_preferred_for_fact_query(self):
        answer, source = choose_answer("What is the capital?", "Berlin", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_lookup_value_preferred_for_fill_blank(self):
        answer, source = choose_answer("The capital is ____.", "Berlin", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_processed_text_used_when_no_lookup(self):
        answer, source = choose_answer("The capital is Paris.", "Paris", [])
        assert answer == "Paris"
        assert source == "postprocessed_text"

    def test_lookup_value_used_as_fallback_for_non_fact_query(self):
        # non-question, no blank → lookup not preferred, but postprocessed_text is empty
        answer, source = choose_answer("Tell me about Paris.", "", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_empty_when_nothing_available(self):
        answer, source = choose_answer("Some prompt.", "", [])
        assert answer == ""
        assert source == "empty"

    def test_question_mark_triggers_lookup_preference(self):
        answer, source = choose_answer("Where was she born?", "France", ["Paris"])
        assert answer == "Paris"

    def test_blank_triggers_lookup_preference(self):
        answer, source = choose_answer("She was born in ____", "France", ["Paris"])
        assert answer == "Paris"

    def test_first_lookup_value_used(self):
        answer, source = choose_answer("Q?", "", ["First", "Second"])
        assert answer == "First"

    def test_non_question_uses_postprocessed_text(self):
        answer, source = choose_answer("Describe Paris.", "The City of Light", ["Paris"])
        assert answer == "The City of Light"
        assert source == "postprocessed_text"


# ===========================================================================
# compute_generation_budget
# ===========================================================================


class TestComputeGenerationBudget:
    def _make_tokenizer(self, token_count: int):
        tok = MagicMock()
        tok.encode.return_value = list(range(token_count))
        return tok

    def test_minimum_is_32(self):
        tok = self._make_tokenizer(0)
        result = compute_generation_budget(tok, "", target_answer_tokens=0)
        assert result >= 32

    def test_includes_prompt_length(self):
        tok = self._make_tokenizer(100)
        result = compute_generation_budget(tok, "x" * 100, target_answer_tokens=12)
        # 100 + 12 + 16 = 128 > 32
        assert result == 128

    def test_includes_slack(self):
        tok = self._make_tokenizer(10)
        result = compute_generation_budget(tok, "short", target_answer_tokens=5)
        # 10 + 5 + 16 = 31 → max(32, 31) = 32
        assert result == 32

    def test_large_prompt(self):
        tok = self._make_tokenizer(1000)
        result = compute_generation_budget(tok, "x" * 1000, target_answer_tokens=12)
        assert result == 1028  # 1000 + 12 + 16

    def test_returns_int(self):
        tok = self._make_tokenizer(50)
        result = compute_generation_budget(tok, "prompt", target_answer_tokens=10)
        assert isinstance(result, int)


# ===========================================================================
# prepare_prompt
# ===========================================================================


class TestPreparePrompt:
    def test_strips_whitespace(self):
        assert prepare_prompt("  hello  ") == "hello"

    def test_no_change_needed(self):
        assert prepare_prompt("hello") == "hello"

    def test_empty(self):
        assert prepare_prompt("") == ""

    def test_newline_stripped(self):
        assert prepare_prompt("hello\n") == "hello"

    def test_internal_content_preserved(self):
        text = "What is the capital of France?"
        assert prepare_prompt(text) == text


# ===========================================================================
# retrieve_lookup_value
# ===========================================================================


class TestRetrieveLookupValue:
    def test_no_db_manager_returns_unknown(self):
        model = MagicMock()
        del model.db_manager  # AttributeError → getattr returns None
        model.db_manager = None
        result = retrieve_lookup_value(model, "some query")
        assert result == "unknown"

    def test_successful_retrieval(self):
        db = MagicMock()
        db.retrieve_from_database.return_value = "Paris"
        model = MagicMock()
        model.db_manager = db
        result = retrieve_lookup_value(model, "query")
        assert result == "Paris"

    def test_exception_with_top1_fallback(self):
        db = MagicMock()
        # First call raises, second (threshold=-1) succeeds
        db.retrieve_from_database.side_effect = [
            ValueError("no result"),
            "Paris",
        ]
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "top1_anyway"
        result = retrieve_lookup_value(model, "query")
        assert result == "Paris"

    def test_exception_with_non_top1_policy_returns_unknown(self):
        db = MagicMock()
        db.retrieve_from_database.side_effect = ValueError("fail")
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "raise"
        result = retrieve_lookup_value(model, "query")
        assert result == "unknown"

    def test_both_calls_fail_returns_unknown(self):
        db = MagicMock()
        db.retrieve_from_database.side_effect = ValueError("fail")
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "top1_anyway"
        result = retrieve_lookup_value(model, "query")
        assert result == "unknown"


# ===========================================================================
# load_prompts
# ===========================================================================


class TestLoadPrompts:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "prompts.jsonl"
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        result = load_prompts(p)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["text"] == "world"

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n', encoding="utf-8"
        )
        result = load_prompts(p)
        assert len(result) == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        result = load_prompts(p)
        assert result == []

    def test_single_record(self, tmp_path):
        p = tmp_path / "single.jsonl"
        p.write_text('{"fact_id": 42}\n', encoding="utf-8")
        result = load_prompts(p)
        assert len(result) == 1
        assert result[0]["fact_id"] == 42

    def test_unicode_content(self, tmp_path):
        p = tmp_path / "unicode.jsonl"
        p.write_text('{"text": "Jørgensen"}\n', encoding="utf-8")
        result = load_prompts(p)
        assert result[0]["text"] == "Jørgensen"


# ===========================================================================
# save_results
# ===========================================================================


class TestSaveResults:
    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "a" / "b" / "results.jsonl"
        save_results([{"key": "val"}], output)
        assert output.exists()

    def test_saves_each_result_as_jsonl(self, tmp_path):
        output = tmp_path / "out.jsonl"
        results = [{"id": 1}, {"id": 2}]
        save_results(results, output)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    def test_empty_results(self, tmp_path):
        output = tmp_path / "empty.jsonl"
        save_results([], output)
        assert output.read_text() == ""

    def test_unicode_preserved(self, tmp_path):
        output = tmp_path / "unicode.jsonl"
        save_results([{"text": "Jørgensen"}], output)
        data = json.loads(output.read_text(encoding="utf-8").strip())
        assert data["text"] == "Jørgensen"

    def test_overwrites_existing_file(self, tmp_path):
        output = tmp_path / "out.jsonl"
        output.write_text("old content\n", encoding="utf-8")
        save_results([{"new": True}], output)
        data = json.loads(output.read_text(encoding="utf-8").strip())
        assert data == {"new": True}


# ===========================================================================
# _default_retrieval_trace
# ===========================================================================


class TestDefaultRetrievalTrace:
    def test_full_state(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        assert trace["state"] == "FULL"
        assert trace["retrieval_enabled"] is True
        assert trace["lookup_query"] is None
        assert trace["all_candidates"] == []
        assert trace["error"] is None

    def test_del_on_state(self):
        trace = _default_retrieval_trace(DatabaseState.DEL_ON)
        assert trace["state"] == "DEL-ON"
        assert trace["retrieval_enabled"] is True

    def test_del_off_state(self):
        trace = _default_retrieval_trace(DatabaseState.DEL_OFF)
        assert trace["state"] == "DEL-OFF"
        assert trace["retrieval_enabled"] is False

    def test_all_keys_present(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        expected_keys = {
            "state", "retrieval_enabled", "lookup_query", "threshold",
            "all_candidates", "deleted_candidates", "retained_candidates",
            "selected_candidate", "selected_value", "error",
        }
        assert expected_keys <= set(trace.keys())

    def test_selected_value_none(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        assert trace["selected_value"] is None


# ===========================================================================
# W&B visualisation tests
# ===========================================================================


def test_clean_answer_processing_logged_to_wandb(wandb_run):
    """Bar chart showing before/after clean_answer character counts."""
    import matplotlib.pyplot as plt

    test_inputs = [
        "Answer: Paris",
        "&lt;b&gt;Berlin&lt;/b&gt;",
        "<|db_entity|>X<|db_relationship|>Y<|db_return|>Value<|db_end|> Paris",
        "The answer is Rome.",
        '"Tokyo"',
        "London\nQuestion: follow-up?",
        "  Madrid  ",
        "Vienna; city of music.",
    ]
    before_lens = [len(t) for t in test_inputs]
    cleaned = [clean_answer(t) for t in test_inputs]
    after_lens = [len(c) for c in cleaned]

    if wandb_run is not None:
        try:
            import numpy as np
            import wandb

            x = np.arange(len(test_inputs))
            width = 0.4
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width / 2, before_lens, width, label="before", color="tomato")
            ax.bar(x + width / 2, after_lens, width, label="after", color="seagreen")
            ax.set_xticks(x)
            ax.set_xticklabels([t[:20] + "…" if len(t) > 20 else t for t in test_inputs], rotation=45, ha="right")
            ax.set_ylabel("Characters")
            ax.set_title("clean_answer: before vs after character count")
            ax.legend()
            plt.tight_layout()
            wandb_run.log({"run_audit/clean_answer_lengths": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    # Assertions: cleaned version should never be longer than original (after strip)
    for inp, out in zip(test_inputs, cleaned):
        assert len(out) <= len(inp.strip()) + 5  # small slack for unescape


def test_choose_answer_distribution_logged_to_wandb(wandb_run):
    """Pie chart of choose_answer source distribution over test cases."""
    import matplotlib.pyplot as plt
    from collections import Counter

    cases = [
        ("Q?", "Berlin", ["Paris"]),
        ("Q?", "", ["Paris"]),
        ("Statement.", "Paris", []),
        ("Statement.", "", []),
        ("Blank ____", "Berlin", ["Paris"]),
        ("Statement.", "", ["Paris"]),
    ]
    sources = [choose_answer(p, out, lv)[1] for p, out, lv in cases]
    counts = Counter(sources)

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            ax.pie(
                counts.values(),
                labels=counts.keys(),
                autopct="%1.0f%%",
                colors=["steelblue", "seagreen", "tomato"],
            )
            ax.set_title("choose_answer source distribution")
            plt.tight_layout()
            wandb_run.log({"run_audit/choose_answer_sources": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert set(sources) <= {"lookup_value", "postprocessed_text", "empty"}


# ===========================================================================
# parse_args – atomic-filter preprocessing CLI flags
# ===========================================================================


class TestParseArgsPreprocessing:
    def test_preprocessing_defaults_to_false(self):
        with patch.object(sys, "argv", ["run_audit.py"]):
            args = parse_args()
        assert args.preprocessing is False

    def test_preprocessing_flag_sets_true(self):
        with patch.object(sys, "argv", ["run_audit.py", "--preprocessing"]):
            args = parse_args()
        assert args.preprocessing is True

    def test_preprocess_cache_dir_defaults_under_output_dir(self, tmp_path):
        with patch.object(
            sys,
            "argv",
            ["run_audit.py", "--output-dir", str(tmp_path / "results")],
        ):
            args = parse_args()
        assert args.preprocess_cache_dir == tmp_path / "results" / "preprocessed"

    def test_preprocess_cache_dir_explicit_overrides_default(self, tmp_path):
        explicit = tmp_path / "custom_cache"
        with patch.object(
            sys,
            "argv",
            [
                "run_audit.py",
                "--output-dir",
                str(tmp_path / "results"),
                "--preprocess-cache-dir",
                str(explicit),
            ],
        ):
            args = parse_args()
        assert args.preprocess_cache_dir == explicit

    def test_judge_base_url_default(self):
        with patch.object(sys, "argv", ["run_audit.py"]):
            args = parse_args()
        assert args.judge_base_url == "http://localhost:8000"

    def test_judge_model_default(self):
        with patch.object(sys, "argv", ["run_audit.py"]):
            args = parse_args()
        assert args.judge_model == "meta-llama/Llama-3.3-70B-Instruct"

    def test_judge_base_url_override(self):
        with patch.object(
            sys,
            "argv",
            ["run_audit.py", "--judge-base-url", "http://example.com:9000"],
        ):
            args = parse_args()
        assert args.judge_base_url == "http://example.com:9000"

    def test_judge_model_override(self):
        with patch.object(
            sys,
            "argv",
            ["run_audit.py", "--judge-model", "my-org/my-model"],
        ):
            args = parse_args()
        assert args.judge_model == "my-org/my-model"
