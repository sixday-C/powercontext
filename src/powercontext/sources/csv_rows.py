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

"""Captured CSV records with stable, document-scoped row identities."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


def _validate_document_id(document_id: str) -> None:
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003


def _validate_columns(primary_key: str, columns: tuple[str, ...]) -> None:
    if not primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("CSV columns must be non-empty")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("CSV columns must be unique")  # noqa: TRY003
    if primary_key not in columns:
        raise ValueError("primary_key must match a CSV column exactly")  # noqa: TRY003


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the bounded identity of a row without normalizing either component."""

    _validate_document_id(document_id)
    if not primary_key_value or not primary_key_value.strip():
        raise ValueError("primary_key_value must be non-blank")  # noqa: TRY003
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("CSV row source_id exceeds the Source identity limit")  # noqa: TRY003
    return source_id


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document and its exact primary-key column."""

    document_id: str
    primary_key: str
    content: bytes

    @model_validator(mode="after")
    def validate_identity(self) -> CSVDocumentInput:
        _validate_document_id(self.document_id)
        if not self.primary_key:
            raise ValueError("primary_key must be non-empty")  # noqa: TRY003
        return self


class CSVRowInput(BaseModel):
    """One validated record, retaining the original header order and strings."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        _validate_document_id(self.document_id)
        _validate_columns(self.primary_key, self.columns)
        if len(self.values) != len(self.columns):
            raise ValueError("CSV row width must equal header width")  # noqa: TRY003
        if self.primary_key_value != self.values[self.columns.index(self.primary_key)]:
            raise ValueError("primary_key_value must match the original row value")  # noqa: TRY003
        csv_row_source_id(self.document_id, self.primary_key_value)
        return self


class CSVRowSource(Source):
    """Captured, complete original row with its stable identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate the entire document before returning any rows."""

    _validate_document_id(value.document_id)
    if not value.primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003
    try:
        decoded = value.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003

    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
    try:
        columns = tuple(next(reader))
        _validate_columns(value.primary_key, columns)
        key_index = columns.index(value.primary_key)
        rows: list[CSVRowInput] = []
        seen_keys: set[str] = set()
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV row width must equal header width")  # noqa: TRY003
            primary_key_value = record[key_index]
            if primary_key_value in seen_keys:
                raise ValueError("CSV primary-key values must be unique")  # noqa: TRY003
            row = CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
            seen_keys.add(primary_key_value)
            rows.append(row)
    except StopIteration as error:
        raise ValueError("CSV document must contain a header") from error  # noqa: TRY003
    except csv.Error as error:
        raise ValueError("CSV document is malformed") from error  # noqa: TRY003
    return tuple(rows)


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


class _CSVRowSourceDefinition:
    name = "csv-row"
    version = "1"
    input_class = CSVRowInput
    source_class = CSVRowSource
    projections = (_CSVRowTextEvidenceProjection(),)

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_SOURCE_DEFINITION = _CSVRowSourceDefinition()


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
