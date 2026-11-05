# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the visible, uncompressed history of a selected Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    """A complete Pi v3 JSONL archive and an optional selected branch leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event and the complete record from which it was rendered."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """An immutable captured event with its full original Pi record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _archive(value: PiSessionInput) -> tuple[str, dict[str, dict[str, JsonValue]], tuple[str, ...]]:  # noqa: C901
    try:
        decoded = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be valid UTF-8") from error  # noqa: TRY003
    try:
        records = [json.loads(line) for line in decoded.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError("Pi session contains invalid JSON") from error  # noqa: TRY003
    if not records or not isinstance(records[0], dict):
        raise ValueError("Pi session must start with a v3 header")  # noqa: TRY003
    header = records[0]
    session_id = header.get("id")
    if (
        header.get("type") != "session"
        or header.get("version") != 3
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise ValueError("Pi session must start with a valid v3 header")  # noqa: TRY003

    by_id: dict[str, dict[str, JsonValue]] = {}
    last_id: str | None = None
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi events must be JSON objects")  # noqa: TRY003, TRY004
        event_id = record.get("id")
        parent_id = record.get("parentId")
        if not isinstance(event_id, str) or not event_id or event_id in by_id:
            raise ValueError("Pi event IDs must be non-empty and unique")  # noqa: TRY003
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in by_id):
            raise ValueError("Pi event parent must precede its child")  # noqa: TRY003
        by_id[event_id] = record
        last_id = event_id

    leaf_id = value.leaf_id if value.leaf_id is not None else last_id
    if leaf_id is None:
        return session_id, by_id, ()
    if leaf_id not in by_id:
        raise ValueError("Pi branch leaf is missing")  # noqa: TRY003
    path: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        path.append(current)
        parent = by_id[current]["parentId"]
        current = parent if isinstance(parent, str) else None
    path.reverse()
    return session_id, by_id, tuple(path)


def _visible_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if isinstance(name, str) and isinstance(call_id, str):
                arguments = json.dumps(
                    block.get("arguments", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return "\n".join(parts)


def _event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    role: str
    body: str
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return None
        role = message["role"]
        if role == "toolResult":
            name = message.get("toolName", "")
            call_id = message.get("toolCallId", "")
            text = _visible_text(message.get("content"))
            body = f"[tool_result {name} {call_id}] {text}"
        elif role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            body = f"$ {command}\n{output}"
        else:
            body = _visible_text(message.get("content"))
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        body = record.get("summary", "") if isinstance(record.get("summary"), str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _visible_text(record.get("content"))
    else:
        return None
    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record.get("parentId"),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate the whole graph and return visible events on the selected path."""

    session_id, by_id, path = _archive(value)
    return tuple(event for event_id in path if (event := _event(session_id, by_id[event_id])) is not None)


def _source_id(session_id: str, event_id: str) -> str:
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(
        json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return f"pi:sha256:{digest}"


class _PiEventProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventSourceDefinition:
    name = "pi-event"
    version = "1"
    input_class = PiEventInput
    source_class = PiEventSource
    projections = (_PiEventProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = _source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION = _PiEventSourceDefinition()


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PiSessionConnector:
    """Import new visible events and checkpoint their exact original records."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        value = PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id)
        session_id, by_id, path = _archive(value)
        previous = session.checkpoint
        if previous is None:
            imported: dict[str, str] = {}
        elif (
            isinstance(previous, dict)
            and previous.get("session_id") == session_id
            and isinstance(previous.get("events"), dict)
        ):
            imported = previous["events"].copy()
        else:
            raise ValueError("Pi checkpoint does not match the current session")  # noqa: TRY003

        for event_id, digest in imported.items():
            if event_id not in by_id or _record_digest(by_id[event_id]) != digest:
                raise ValueError("an imported Pi event was changed or removed")  # noqa: TRY003

        updated = imported.copy()
        for event_id in path:
            event = _event(session_id, by_id[event_id])
            if event is None or event_id in imported:
                continue
            await session.submit(event_id, "pi-event", event)
            updated[event_id] = _record_digest(by_id[event_id])
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE, checkpoint={"session_id": session_id, "events": updated}
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
