"""Portable full and prefix-delta packets for prepared context text."""

# ruff: noqa: TRY003, C901

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """An invalid context packet or delta."""


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_object(data: bytes, keys: set[str], schema: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
        raise ContextPackError("invalid JSON packet") from error
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != schema:
        raise ContextPackError("invalid packet fields or schema")
    return value


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decoded(items: Any) -> tuple[bytes, ...]:
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ContextPackError("invalid item array")
    try:
        result = tuple(base64.b64decode(item, validate=True) for item in items)
    except (ValueError, binascii.Error) as error:
        raise ContextPackError("invalid Base64 item") from error
    if _encoded(result) != items:
        raise ContextPackError("noncanonical Base64 item")
    return result


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or not all(isinstance(item, bytes) for item in items):
            raise ContextPackError("items must be a bytes tuple")
        unsigned = {"schema": _PACK_SCHEMA, "items": _encoded(items)}
        digest = "sha256:" + hashlib.sha256(_json_bytes(unsigned)).hexdigest()
        packet = {**unsigned, "digest": digest, "byte_count": 0}
        count = len(_json_bytes(packet))
        while True:
            next_count = len(_json_bytes({**packet, "byte_count": count}))
            if next_count == count:
                return cls(items, digest, count)
            count = next_count

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str) or type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("invalid content or chunk limit")
        chunks: list[bytes] = []
        current = bytearray()
        # Prefer line boundaries so appended context does not shift all later chunks.
        for line in content.splitlines(keepends=True):
            encoded = line.encode("utf-8")
            if len(encoded) <= chunk_bytes:
                if current and len(current) + len(encoded) > chunk_bytes:
                    chunks.append(bytes(current))
                    current.clear()
                current.extend(encoded)
                continue
            if current:
                chunks.append(bytes(current))
                current.clear()
            for char in line:
                value = char.encode("utf-8")
                if len(value) > chunk_bytes:
                    raise ContextPackError("chunk limit cannot hold one character")
                if len(current) + len(value) > chunk_bytes:
                    chunks.append(bytes(current))
                    current.clear()
                current.extend(value)
            if current:
                chunks.append(bytes(current))
                current.clear()
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _json_bytes({
            "schema": _PACK_SCHEMA,
            "items": _encoded(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        value = _decode_object(data, {"schema", "items", "digest", "byte_count"}, _PACK_SCHEMA)
        items = _decoded(value["items"])
        if not _valid_digest(value["digest"]) or type(value["byte_count"]) is not int:
            raise ContextPackError("invalid digest or byte count")
        result = cls.build(items)
        if result.digest != value["digest"] or result.byte_count != value["byte_count"]:
            raise ContextPackError("packet checksum mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _json_bytes({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        value = _decode_object(
            data, {"schema", "base_digest", "target_digest", "prefix_items", "appended"}, _DELTA_SCHEMA
        )
        if not _valid_digest(value["base_digest"]) or not _valid_digest(value["target_digest"]):
            raise ContextPackError("invalid delta digest")
        if type(value["prefix_items"]) is not int or value["prefix_items"] < 0:
            raise ContextPackError("invalid prefix count")
        return cls(value["base_digest"], value["target_digest"], value["prefix_items"], _decoded(value["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if (
        delta.base_digest != base.digest
        or type(delta.prefix_items) is not int
        or not 0 <= delta.prefix_items <= len(base.items)
    ):
        raise ContextPackError("base or prefix mismatch")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target digest mismatch")
    return target
