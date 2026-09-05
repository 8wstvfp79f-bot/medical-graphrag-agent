from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from src.retrieval.graph_schema import (
    RELATIONSHIP_TO_RELATION,
    RELATION_SPECS,
    normalize_relations,
)


GRAPH_SEARCH_QUERY = """
/* medical_graph:search */
MATCH (d:Disease {name: $disease_name})-[r]->(target)
WHERE type(r) IN $relationship_types
RETURN d.name AS disease_name,
       type(r) AS relationship_type,
       target.name AS entity_name,
       labels(target)[0] AS entity_type,
       coalesce(r.source, '') AS source
ORDER BY relationship_type, entity_name
LIMIT $limit
""".strip()


class GraphSearchBackend(Protocol):
    def search(
        self,
        disease_name: str,
        *,
        relations: tuple[str, ...],
        limit: int,
    ) -> Sequence[Mapping[str, object]]:
        """Return graph candidates for one disease entity."""


def _result_records(result: object) -> Sequence[Mapping[str, object]]:
    records = getattr(result, "records", None)
    if records is None and isinstance(result, tuple) and result:
        records = result[0]
    if records is None:
        return []
    return records


class Neo4jGraphBackend:
    """Execute a bounded one-hop expansion with an official Neo4j driver."""

    def __init__(self, driver: object, *, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    def search(
        self,
        disease_name: str,
        *,
        relations: tuple[str, ...],
        limit: int,
    ) -> Sequence[Mapping[str, object]]:
        execute_query = getattr(self.driver, "execute_query", None)
        if not callable(execute_query):
            raise TypeError("Neo4j driver must expose a callable execute_query method")

        relationship_types = [
            RELATION_SPECS[relation].relationship_type for relation in relations
        ]
        result = execute_query(
            GRAPH_SEARCH_QUERY,
            parameters_={
                "disease_name": disease_name,
                "relationship_types": relationship_types,
                "limit": limit,
            },
            database_=self.database,
        )
        return _result_records(result)


def graph_search(
    disease_name: str,
    *,
    backend: GraphSearchBackend,
    relations: list[str] | tuple[str, ...] | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Return typed, auditable graph evidence for an exact disease entity.

    The disease name and relationship allowlist are structured parameters. This keeps
    graph expansion deterministic and prevents unrelated or low-value relationship
    types from entering the generation context.
    """
    disease_name = " ".join(disease_name.split())
    if not disease_name:
        raise ValueError("disease_name must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    normalized_relations = normalize_relations(relations)
    candidates = backend.search(
        disease_name,
        relations=normalized_relations,
        limit=limit,
    )

    evidence: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    allowed = set(normalized_relations)
    for candidate in candidates:
        head = str(candidate.get("disease_name", "")).strip()
        relationship_type = str(candidate.get("relationship_type", "")).strip()
        relation = RELATIONSHIP_TO_RELATION.get(relationship_type)
        tail = str(candidate.get("entity_name", "")).strip()
        source = str(candidate.get("source", "")).strip()
        if head != disease_name or relation not in allowed or not tail:
            continue
        key = (head, relation, tail, source)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "evidence_type": "graph",
                "disease_name": head,
                "relation": relation,
                "entity_name": tail,
                "entity_type": str(candidate.get("entity_type", "")).strip(),
                "source": source,
                "path_text": f"{head} -[{relation}]-> {tail}",
            }
        )
        if len(evidence) >= limit:
            break
    return evidence
