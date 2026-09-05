from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.retrieval.graph_retriever import Neo4jGraphBackend, graph_search
from src.data.medical_quality import DEFAULT_REJECTION_AUDIT_PATH
from src.retrieval.graph_schema import (
    CORE_RELATIONS,
    NODE_LABELS,
    RELATION_SPECS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIPLES_PATH = PROJECT_ROOT / "data" / "processed" / "triples.csv"
DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_BATCH_SIZE = 1_000
EXPECTED_COLUMNS = frozenset({"head", "relation", "tail", "source"})


class Neo4jDependencyError(RuntimeError):
    """Raised when the optional official Neo4j driver is unavailable."""


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str = DEFAULT_NEO4J_URI
    username: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("Neo4j uri must not be empty")
        if not self.username.strip():
            raise ValueError("Neo4j username must not be empty")
        if not self.database.strip():
            raise ValueError("Neo4j database must not be empty")


@dataclass(frozen=True)
class TripleValidationReport:
    input_path: str
    scanned_records: int
    core_records: int
    skipped_candidate_records: int
    relation_distribution: dict[str, int]


@dataclass(frozen=True)
class GraphCountReport:
    total_nodes: int
    disease_nodes: int
    total_relationships: int
    node_label_distribution: dict[str, int]
    relation_distribution: dict[str, int]


@dataclass(frozen=True)
class GraphIngestionReport:
    input_path: str
    scanned_records: int
    submitted_core_records: int
    skipped_candidate_records: int
    batches: int
    node_count_before: int
    node_count_after: int
    relationship_count_before: int
    relationship_count_after: int
    relation_distribution_after: dict[str, int]


@dataclass(frozen=True)
class GraphPruneReport:
    audit_path: str
    reason_prefixes: tuple[str, ...]
    rejected_values: int
    deleted_relationships: int
    deleted_orphan_nodes: int
    deleted_relation_distribution: dict[str, int]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def load_project_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def create_neo4j_driver(settings: Neo4jSettings) -> object:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise Neo4jDependencyError(
            "The official neo4j Python driver is not installed. Install project "
            "requirements, then provide NEO4J_PASSWORD for a running Neo4j server."
        ) from exc

    if not settings.password:
        raise ValueError(
            "Neo4j password is required. Set NEO4J_PASSWORD or pass --password."
        )
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    )


def _result_records(result: object) -> Sequence[Mapping[str, object]]:
    records = getattr(result, "records", None)
    if records is None and isinstance(result, tuple) and result:
        records = result[0]
    if records is None:
        return []
    return records


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def iter_triple_rows(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Triples file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                "Triples CSV must contain columns: "
                + ", ".join(sorted(EXPECTED_COLUMNS))
            )
        for line_no, row in enumerate(reader, start=2):
            cleaned = {column: _clean(row.get(column)) for column in EXPECTED_COLUMNS}
            if not cleaned["relation"]:
                raise ValueError(f"Missing relation at {path}:{line_no}")
            if cleaned["relation"] in RELATION_SPECS:
                missing = [
                    column for column in ("head", "tail", "source") if not cleaned[column]
                ]
                if missing:
                    raise ValueError(
                        f"Missing {', '.join(missing)} at {path}:{line_no}"
                    )
            yield line_no, cleaned


def validate_triples(
    path: Path = DEFAULT_TRIPLES_PATH,
    *,
    limit: int | None = None,
) -> TripleValidationReport:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    scanned = 0
    core = 0
    skipped = 0
    distribution: Counter[str] = Counter()
    for _, row in iter_triple_rows(path):
        scanned += 1
        relation = row["relation"]
        if relation not in RELATION_SPECS:
            skipped += 1
            continue
        core += 1
        distribution[relation] += 1
        if limit is not None and core >= limit:
            break
    return TripleValidationReport(
        input_path=str(path),
        scanned_records=scanned,
        core_records=core,
        skipped_candidate_records=skipped,
        relation_distribution=dict(sorted(distribution.items())),
    )


def _constraint_query(label: str) -> str:
    return (
        f"/* medical_graph:constraint:{label} */\n"
        f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS\n"
        f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
    )


def _ingest_query(relation: str) -> str:
    spec = RELATION_SPECS[relation]
    return f"""
/* medical_graph:ingest:{spec.relationship_type}:{spec.tail_label} */
UNWIND $rows AS row
MERGE (d:Disease {{name: row.head}})
  ON CREATE SET d.source = row.source
  ON MATCH SET d.source = CASE
    WHEN d.source IS NULL OR d.source = '' THEN row.source ELSE d.source END
MERGE (t:{spec.tail_label} {{name: row.tail}})
  ON CREATE SET t.source = row.source
  ON MATCH SET t.source = CASE
    WHEN t.source IS NULL OR t.source = '' THEN row.source ELSE t.source END
MERGE (d)-[r:{spec.relationship_type}]->(t)
  ON CREATE SET r.source = row.source
  ON MATCH SET r.source = CASE
    WHEN r.source IS NULL OR r.source = '' THEN row.source ELSE r.source END
RETURN count(r) AS processed
""".strip()


def _prune_relationship_query(relation: str) -> str:
    spec = RELATION_SPECS[relation]
    return f"""
/* medical_graph:prune:{spec.relationship_type}:{spec.tail_label} */
MATCH (:Disease)-[r:{spec.relationship_type}]->(t:{spec.tail_label})
WHERE r.source = $source AND t.name IN $values
WITH collect(r) AS relationships
WITH relationships, size(relationships) AS deleted_relationships
FOREACH (relationship IN relationships | DELETE relationship)
RETURN deleted_relationships
""".strip()


def _prune_orphan_query(relation: str) -> str:
    spec = RELATION_SPECS[relation]
    return f"""
/* medical_graph:prune_orphans:{spec.tail_label} */
MATCH (t:{spec.tail_label})
WHERE t.source = $source AND t.name IN $values AND NOT (t)--()
WITH collect(t) AS nodes
WITH nodes, size(nodes) AS deleted_orphan_nodes
FOREACH (node IN nodes | DELETE node)
RETURN deleted_orphan_nodes
""".strip()


NODE_COUNT_QUERY = """
/* medical_graph:count:nodes */
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY label
""".strip()

RELATION_COUNT_QUERY = """
/* medical_graph:count:relationships */
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY relationship_type
""".strip()


class MedicalNeo4jStore:
    """Own Neo4j schema, core-relation ingestion, and graph counts."""

    def __init__(self, driver: object, settings: Neo4jSettings) -> None:
        self.driver = driver
        self.settings = settings

    def _execute(self, query: str, parameters: Mapping[str, object] | None = None):
        execute_query = getattr(self.driver, "execute_query", None)
        if not callable(execute_query):
            raise TypeError("Neo4j driver must expose a callable execute_query method")
        return execute_query(
            query,
            parameters_=dict(parameters or {}),
            database_=self.settings.database,
        )

    def verify_connectivity(self) -> None:
        verify = getattr(self.driver, "verify_connectivity", None)
        if not callable(verify):
            raise TypeError("Neo4j driver must expose a callable verify_connectivity method")
        verify()

    def ensure_schema(self) -> int:
        for label in NODE_LABELS:
            self._execute(_constraint_query(label))
        return len(NODE_LABELS)

    def count_summary(self) -> GraphCountReport:
        node_records = _result_records(self._execute(NODE_COUNT_QUERY))
        relation_records = _result_records(self._execute(RELATION_COUNT_QUERY))

        node_distribution = {
            str(record.get("label", "")): int(record.get("count", 0))
            for record in node_records
            if str(record.get("label", ""))
        }
        relationship_to_relation = {
            spec.relationship_type: relation for relation, spec in RELATION_SPECS.items()
        }
        relation_distribution: dict[str, int] = {}
        for record in relation_records:
            relationship_type = str(record.get("relationship_type", ""))
            relation = relationship_to_relation.get(
                relationship_type, relationship_type.lower()
            )
            if relation:
                relation_distribution[relation] = int(record.get("count", 0))

        return GraphCountReport(
            total_nodes=sum(node_distribution.values()),
            disease_nodes=node_distribution.get("Disease", 0),
            total_relationships=sum(relation_distribution.values()),
            node_label_distribution=dict(sorted(node_distribution.items())),
            relation_distribution=dict(sorted(relation_distribution.items())),
        )

    def ingest_triples(
        self,
        path: Path = DEFAULT_TRIPLES_PATH,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        limit: int | None = None,
    ) -> GraphIngestionReport:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")

        self.ensure_schema()
        before = self.count_summary()
        buffers: dict[str, list[dict[str, str]]] = {
            relation: [] for relation in CORE_RELATIONS
        }
        scanned = 0
        submitted = 0
        skipped = 0
        batches = 0

        def flush(relation: str) -> None:
            nonlocal batches
            rows = buffers[relation]
            if not rows:
                return
            self._execute(_ingest_query(relation), {"rows": list(rows)})
            rows.clear()
            batches += 1

        for _, row in iter_triple_rows(path):
            scanned += 1
            relation = row["relation"]
            if relation not in RELATION_SPECS:
                skipped += 1
                continue
            buffers[relation].append(row)
            submitted += 1
            if len(buffers[relation]) >= batch_size:
                flush(relation)
            if limit is not None and submitted >= limit:
                break

        for relation in CORE_RELATIONS:
            flush(relation)

        after = self.count_summary()
        return GraphIngestionReport(
            input_path=str(path),
            scanned_records=scanned,
            submitted_core_records=submitted,
            skipped_candidate_records=skipped,
            batches=batches,
            node_count_before=before.total_nodes,
            node_count_after=after.total_nodes,
            relationship_count_before=before.total_relationships,
            relationship_count_after=after.total_relationships,
            relation_distribution_after=after.relation_distribution,
        )

    def prune_rejected_items(
        self,
        audit_path: Path = DEFAULT_REJECTION_AUDIT_PATH,
        *,
        reason_prefixes: Sequence[str] = ("confirmed_",),
    ) -> GraphPruneReport:
        """Delete only audited graph facts with explicitly approved reason prefixes."""

        prefixes = tuple(_clean(prefix) for prefix in reason_prefixes if _clean(prefix))
        if not prefixes:
            raise ValueError("At least one non-empty rejection reason prefix is required")
        if not audit_path.exists():
            raise FileNotFoundError(f"Rejection audit not found: {audit_path}")

        field_relations = {
            "symptom": "has_symptom",
            "check": "diagnosed_by",
            "recommand_drug": "treated_by",
            "common_drug": "treated_by",
            "cure_department": "belongs_to",
            "acompany": "has_complication",
        }
        grouped: dict[tuple[str, str], set[str]] = {}
        with audit_path.open("r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid rejection audit JSON at {audit_path}:{line_no}"
                    ) from exc
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"Expected rejection audit object at {audit_path}:{line_no}"
                    )
                reason = _clean(item.get("reason"))
                if not any(reason.startswith(prefix) for prefix in prefixes):
                    continue
                relation = field_relations.get(_clean(item.get("field")))
                value = _clean(item.get("value"))
                item_source = _clean(item.get("source"))
                if relation and value and item_source:
                    grouped.setdefault((relation, item_source), set()).add(value)

        deleted_relationships = 0
        deleted_orphans = 0
        relation_distribution: Counter[str] = Counter()
        for (relation, item_source), values in sorted(grouped.items()):
            parameters = {"source": item_source, "values": sorted(values)}
            relationship_records = _result_records(
                self._execute(_prune_relationship_query(relation), parameters)
            )
            relationship_count = (
                int(relationship_records[0].get("deleted_relationships", 0))
                if relationship_records
                else 0
            )
            orphan_records = _result_records(
                self._execute(_prune_orphan_query(relation), parameters)
            )
            orphan_count = (
                int(orphan_records[0].get("deleted_orphan_nodes", 0))
                if orphan_records
                else 0
            )
            deleted_relationships += relationship_count
            deleted_orphans += orphan_count
            relation_distribution[relation] += relationship_count

        return GraphPruneReport(
            audit_path=str(audit_path),
            reason_prefixes=prefixes,
            rejected_values=sum(len(values) for values in grouped.values()),
            deleted_relationships=deleted_relationships,
            deleted_orphan_nodes=deleted_orphans,
            deleted_relation_distribution=dict(sorted(relation_distribution.items())),
        )


def _settings_from_args(args: argparse.Namespace) -> Neo4jSettings:
    return Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the medical Neo4j core-relation graph"
    )
    parser.add_argument(
        "--uri", default=os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI)
    )
    parser.add_argument(
        "--username", default=os.getenv("NEO4J_USERNAME", "neo4j")
    )
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate triples without connecting to Neo4j"
    )
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_TRIPLES_PATH)
    validate_parser.add_argument("--limit", type=int)

    subparsers.add_parser("init", help="Verify connectivity and create constraints")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Idempotently MERGE core triples into Neo4j"
    )
    ingest_parser.add_argument("--input", type=Path, default=DEFAULT_TRIPLES_PATH)
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ingest_parser.add_argument("--limit", type=int)

    prune_parser = subparsers.add_parser(
        "prune-rejected",
        help="Delete only graph facts listed in the rejection audit",
    )
    prune_parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_REJECTION_AUDIT_PATH,
    )
    prune_parser.add_argument(
        "--reason-prefix",
        action="append",
        default=["confirmed_"],
        help="Only delete audited items whose reason starts with this value",
    )

    subparsers.add_parser("count", help="Count typed nodes and relationships")

    search_parser = subparsers.add_parser(
        "search", help="Expand one exact disease entity by core relations"
    )
    search_parser.add_argument("disease_name")
    search_parser.add_argument(
        "--relation", action="append", choices=CORE_RELATIONS
    )
    search_parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    load_project_environment()
    args = parse_args()
    if args.command == "validate":
        print(
            json.dumps(
                asdict(validate_triples(args.input, limit=args.limit)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    settings = _settings_from_args(args)
    driver = create_neo4j_driver(settings)
    try:
        store = MedicalNeo4jStore(driver, settings)
        store.verify_connectivity()
        if args.command == "init":
            constraints = store.ensure_schema()
            payload: object = {
                "uri": settings.uri,
                "database": settings.database,
                "constraints_ensured": constraints,
                "counts": asdict(store.count_summary()),
            }
        elif args.command == "ingest":
            payload = asdict(
                store.ingest_triples(
                    args.input,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
            )
        elif args.command == "prune-rejected":
            payload = asdict(
                store.prune_rejected_items(
                    args.audit,
                    reason_prefixes=args.reason_prefix,
                )
            )
        elif args.command == "count":
            payload = asdict(store.count_summary())
        else:
            backend = Neo4jGraphBackend(driver, database=settings.database)
            payload = graph_search(
                args.disease_name,
                backend=backend,
                relations=args.relation,
                limit=args.limit,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
