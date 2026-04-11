import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from database_states import (
    AuditDatabaseManager,
    DatabaseState,
    TargetFact,
    candidate_supports_target_fact,
    extract_lookup_query,
    is_deleted_triplet,
    retrieval_enabled,
)


class FakeModel:
    def encode(self, *_args, **_kwargs):
        return [[1.0]]


class FakeIndex:
    def __init__(self, indices: list[int], distances: list[float]) -> None:
        self.indices = indices
        self.distances = distances

    def search(self, _query_embedding, _top_k):
        return [self.distances], [self.indices]


class FakeRetriever:
    def __init__(self) -> None:
        self.top_k = 3
        self.default_threshold = 0.6
        self.model = FakeModel()
        self.index = FakeIndex(
            indices=[0, 1, 2],
            distances=[0.95, 0.90, 0.80],
        )
        self.id_to_triplet = {
            0: ("Hexol", "First Described By", "Jorgensen"),
            1: ("Hexol", "Structure Recognized By", "Werner"),
            2: ("Jocelyne Girard-Bujold", "Term End", "2004"),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().strip()


class FakeBaseManager:
    def __init__(self) -> None:
        self.database_name = "fake"
        self.database_org_file = []
        self.database = {}
        self.topk_retriever = FakeRetriever()

    def init_topk_retriever(self, *args, **kwargs) -> None:
        return None

    def retrieve_from_database(self, _prompt: str, threshold=None) -> str:
        return "Jorgensen"


def test_retrieval_enabled() -> None:
    assert retrieval_enabled(DatabaseState.FULL) is True
    assert retrieval_enabled(DatabaseState.DEL_ON) is True
    assert retrieval_enabled(DatabaseState.DEL_OFF) is False


def test_extract_lookup_query() -> None:
    entity, relation = extract_lookup_query(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_is_deleted_triplet() -> None:
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )
    assert (
        is_deleted_triplet(
            ("Hexol", "First Described By", "Jorgensen"),
            target_fact,
        )
        is True
    )
    assert (
        is_deleted_triplet(
            ("Hexol", "Structure Recognized By", "Werner"),
            target_fact,
        )
        is False
    )


def test_candidate_supports_target_fact_flags() -> None:
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )

    assert candidate_supports_target_fact(
        ("Hexol", "First Described By", "Jorgensen"),
        target_fact,
    ) == (True, True, True, True)
    assert candidate_supports_target_fact(
        ("Hexol", "Structure Recognized By", "Jorgensen"),
        target_fact,
    ) == (True, False, True, False)


def test_audit_database_manager_filters_deleted_fact() -> None:
    base_manager = FakeBaseManager()
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )
    audit_manager = AuditDatabaseManager(
        base_db_manager=base_manager,
        state=DatabaseState.DEL_ON,
        target_fact=target_fact,
    )

    value = audit_manager.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert value == "Werner"
    assert audit_manager.last_trace is not None
    assert audit_manager.last_trace["lookup_query"] == {
        "entity": "Hexol",
        "relation": "First Described By",
    }
    assert len(audit_manager.last_trace["all_candidates"]) == 3
    assert len(audit_manager.last_trace["deleted_candidates"]) == 1
    assert audit_manager.last_trace["deleted_candidates"][0]["object"] == "Jorgensen"
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_subject"] is True
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_relation"] is True
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_object"] is True
    assert audit_manager.last_trace["deleted_candidates"][0]["supports_target_fact"] is True
    assert audit_manager.last_trace["selected_candidate"]["matches_relation"] is False
    assert audit_manager.last_trace["selected_candidate"]["supports_target_fact"] is False
    assert audit_manager.last_trace["selected_value"] == "Werner"


def test_full_state_falls_back_to_base_manager_when_trace_parse_fails() -> None:
    base_manager = FakeBaseManager()
    target_fact = TargetFact(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )
    audit_manager = AuditDatabaseManager(
        base_db_manager=base_manager,
        state=DatabaseState.FULL,
        target_fact=target_fact,
    )

    value = audit_manager.retrieve_from_database("not a dblookup prompt")
    assert value == "Jorgensen"
    assert audit_manager.last_trace is not None
    assert audit_manager.last_trace["error"] is not None
    assert audit_manager.last_trace["selected_value"] == "Jorgensen"
