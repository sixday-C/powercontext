"""Durable source dependency state for Memory and Topic Memory."""

# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.runtime.models import CaptureSource, GetMemoryEntryRequest
from powercontext.builtin.sources import ContentSource
from powercontext.sources import SourceRef


class SourceLifecycleError(ValueError):
    """A lifecycle binding or event violates the dependency contract."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be nonblank")
        return value


_MARKER = "powercontext.source-lifecycle.v1"


def _record_id(kind: str, node_id: str, sequence: int | None = None) -> str:
    identity = json.dumps((kind, node_id, sequence), separators=(",", ":")).encode()
    return "lifecycle-" + hashlib.sha256(identity).hexdigest()


def _active_nodes(bindings: dict[str, dict[str, Any]], events: dict[str, SourceLifecycleEvent]) -> set[str]:
    active: set[str] = set()
    for node_id, binding in bindings.items():
        if binding["kind"] == "source":
            if events.get(node_id) is None or events[node_id].available:
                active.add(node_id)
            continue
        support = [parent in active for parent in binding["parents"]]
        if all(support) if binding["policy"] == "all" else any(support):
            active.add(node_id)
    return active


class ScopedSourceLifecycle:
    def __init__(self, runtime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    async def _load(self) -> tuple[dict[str, dict[str, Any]], dict[str, SourceLifecycleEvent]]:
        async with self._runtime._context(self.scope_id) as context:
            entries = await context.sources.journal.entries()
        bindings: dict[str, dict[str, Any]] = {}
        events: dict[str, SourceLifecycleEvent] = {}
        for entry in entries:
            source = entry.source
            if not isinstance(source, ContentSource) or source.metadata.get(_MARKER) != "1":
                continue
            record = json.loads(source.content)
            if record["record_type"] == "binding":
                bindings[record["node_id"]] = record
            else:
                event = SourceLifecycleEvent.model_validate(record["event"])
                events[event.node_id] = event
        return bindings, events

    async def _store(self, kind: str, node_id: str, record: dict[str, Any], sequence: int | None = None) -> None:
        await self._runtime.sources.for_scope(self.scope_id).capture(
            CaptureSource(
                source_id=_record_id(kind, node_id, sequence),
                content=json.dumps(record, sort_keys=True, separators=(",", ":")),
                metadata={_MARKER: "1"},
            )
        )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if not isinstance(node_id, str) or not node_id.strip() or type(revision) is not int or revision < 1:
            raise SourceLifecycleError("invalid source registration")
        async with self._runtime._context(self.scope_id) as context:
            entries = await context.sources.journal.entries()
        if not any(entry.source_ref == source_ref for entry in entries):
            raise SourceLifecycleError("source reference is not readable")
        await self._register(
            node_id, {"kind": "source", "reference": source_ref.model_dump(mode="json"), "revision": revision}
        )

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("memory citation is not readable") from error
        await self._register(
            node_id,
            {"kind": "memory", "reference": citation.model_dump(mode="json"), "parents": parents, "policy": policy},
        )

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic reference has the wrong family")
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                "topic-memory", artifact_ref.artifact_id, artifact_ref.revision
            )
        except Exception as error:
            raise SourceLifecycleError("topic reference is not readable") from error
        await self._register(
            node_id,
            {"kind": "topic", "reference": artifact_ref.model_dump(mode="json"), "parents": parents, "policy": policy},
        )

    async def _register(self, node_id: str, binding: dict[str, Any]) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must be nonblank")
        bindings, events = await self._load()
        record = {"record_type": "binding", "node_id": node_id, **binding}
        existing = bindings.get(node_id)
        if existing is not None:
            if existing != json.loads(json.dumps(record)):
                raise SourceLifecycleError("node identity is already bound")
            await self._reconcile(bindings, events)
            return
        if binding["kind"] != "source":
            parents = binding["parents"]
            if binding["policy"] not in {"all", "any"} or not parents or len(set(parents)) != len(parents):
                raise SourceLifecycleError("invalid dependency policy or parents")
            if any(parent not in bindings for parent in parents):
                raise SourceLifecycleError("parent must be registered first")
        await self._store("binding", node_id, record)
        bindings[node_id] = record
        await self._reconcile(bindings, events)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        bindings, events = await self._load()
        binding = bindings.get(event.node_id)
        if binding is None or binding["kind"] != "source" or binding["revision"] != event.revision:
            raise SourceLifecycleError("unknown source or lifecycle revision")
        previous = events.get(event.node_id)
        if previous is not None and (
            event.sequence < previous.sequence or (event.sequence == previous.sequence and event != previous)
        ):
            raise SourceLifecycleError("source event sequence conflict")
        if previous is None or event.sequence > previous.sequence:
            await self._store(
                "event", event.node_id, {"record_type": "event", "event": event.model_dump(mode="json")}, event.sequence
            )
            events[event.node_id] = event
        await self._reconcile(bindings, events)
        return sorted(_active_nodes(bindings, events))

    async def resolve(self, node_id: str, *, purpose: Literal["generation", "audit"]):
        bindings, events = await self._load()
        binding = bindings.get(node_id)
        if binding is None or purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown node or purpose")
        if purpose == "generation" and node_id not in _active_nodes(bindings, events):
            raise SourceLifecycleError("node has no available support")
        reference = binding["reference"]
        if binding["kind"] == "source":
            return SourceRef.model_validate(reference)
        if binding["kind"] == "memory":
            return MemoryCitation.model_validate(reference)
        return ArtifactRef.model_validate(reference)

    async def inactive_topics(self) -> set[tuple[str, int]]:
        bindings, events = await self._load()
        active = _active_nodes(bindings, events)
        return {
            (binding["reference"]["artifact_id"], binding["reference"]["revision"])
            for node_id, binding in bindings.items()
            if binding["kind"] == "topic" and node_id not in active
        }

    async def _reconcile(self, bindings: dict[str, dict[str, Any]], events: dict[str, SourceLifecycleEvent]) -> None:
        if not any(binding["kind"] == "memory" for binding in bindings.values()):
            return
        active = _active_nodes(bindings, events)
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            for node_id, binding in bindings.items():
                if binding["kind"] != "memory":
                    continue
                citation = MemoryCitation.model_validate(binding["reference"])
                entries = await service.entries(current)
                entry = next((entry for entry in entries if entry.entry_id == citation.entry_id), None)
                if entry is None or entry.entry_version_id != citation.entry_version_id:
                    raise SourceLifecycleError("registered memory has changed")
                if node_id in active:
                    current = await service.reactivate(current, entries=(entry,), reason="source lifecycle")
                else:
                    current = await service.forget(current, entries=(entry,), reason="source lifecycle")


class SourceLifecycleApplication:
    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]
