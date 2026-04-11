import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from run_audit import clean_answer


def test_clean_answer_strips_db_markup_and_html_tags() -> None:
    answer = clean_answer(
        '&lt;/poem&gt; <|db_entity|>Madhur Jaffrey<|db_relationship|>'
        'Award<|db_return|>Madison Sharma<|db_end|> Biography'
    )
    assert answer == "Biography"


def test_clean_answer_strips_standalone_db_special_tokens() -> None:
    answer = clean_answer('"<|db_entity|> Spice Girls <|db_return|>"')
    assert answer == "Spice Girls"
