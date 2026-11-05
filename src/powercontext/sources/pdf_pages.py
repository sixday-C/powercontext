"""Complete physical PDF pages as captured Sources with exact page citations."""

# ruff: noqa: TRY003, TRY301

from __future__ import annotations

import hashlib
import io
import json
import re

from pydantic import BaseModel, model_validator
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    @model_validator(mode="after")
    def validate_id(self) -> PDFDocumentInput:
        if not self.document_id.strip():
            raise ValueError("document_id must be nonblank")
        return self


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Read every physical page before returning any page to the caller."""
    if not value.content.startswith(b"%PDF-") or b"%%EOF" not in value.content[-2048:]:
        raise ValueError("invalid PDF document")
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF requires a password")
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=number,
                text=page.extract_text() or "",
            )
            for number, page in enumerate(reader.pages, start=1)
        )
        if not pages:
            raise ValueError("PDF has no pages")
    except Exception as error:
        raise ValueError("PDF could not be parsed completely") from error
    return pages


class _PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageSourceDefinition:
    name = "pdf-page"
    version = "1"
    input_class = PDFPageInput
    source_class = PDFPageSource
    projections = (_PDFPageTextEvidenceProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = json.dumps(
            [value.document_id, value.document_sha256, value.page_number],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        name = "pdf-page:" + hashlib.sha256(identity).hexdigest()
        return PDFPageSource(
            name=name,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION = _PDFPageSourceDefinition()

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "as",
    "of",
    "in",
    "on",
    "for",
    "to",
    "from",
    "by",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "how",
    "does",
    "did",
    "has",
    "have",
    "any",
    "with",
    "within",
    "that",
    "this",
    "its",
    "it",
    "then",
    "can",
    "only",
    "use",
    "using",
    "give",
    "answer",
    "question",
    "reported",
    "report",
    "fy",
    "fy2022",
    "fy2021",
    "2022",
    "2021",
    "company",
    "statement",
    "statements",
}

_FINANCIAL_QUERY_EXPANSIONS = (
    (
        {"geography", "geographies", "geographic", "region", "regions", "operates"},
        ("operations segments", "geographic divisions"),
    ),
    ({"legal", "litigation", "lawsuits", "battles"}, ("legal proceedings", "legal matters")),
    ({"restructuring", "reorganization"}, ("restructuring charges", "restructuring costs")),
    ({"ebitda", "depreciation", "amortization"}, ("depreciation amortization", "operating income")),
    ({"capex", "capital"}, ("capital expenditures", "capital spending")),
    ({"cash", "flows"}, ("statement cash flows",)),
    ({"income", "profit", "loss", "p&l"}, ("consolidated statement income",)),
    ({"assets", "liabilities", "balance"}, ("consolidated balance sheets",)),
)


def _search_queries(question: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]+|\d+", question.lower())
    distinctive = list(dict.fromkeys(word for word in tokens if len(word) > 2 and word not in _STOP_WORDS))
    queries = [question]
    if distinctive:
        queries.append(" ".join(distinctive[:12]))
    if len(distinctive) > 4:
        middle = len(distinctive) // 2
        queries.extend((" ".join(distinctive[middle - 1 : middle + 1]), " ".join(distinctive[-2:])))
    vocabulary = set(tokens)
    for triggers, expansions in _FINANCIAL_QUERY_EXPANSIONS:
        if vocabulary & triggers:
            queries.extend(expansions)
    return tuple(dict.fromkeys(queries[:10]))


def _ranked_page_ids(channels: list[list[str]], scores: dict[str, float], limit: int) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(identity: str) -> None:
        if identity not in seen and len(selected) < limit:
            seen.add(identity)
            selected.append(identity)

    for identity in channels[0][: max(1, limit * 2 // 5)]:
        add(identity)
    # Reserve space for specialized queries so a generic long question cannot
    # crowd out pages using the document's terminology for the same concept.
    for rank in range(max((len(channel) for channel in channels[1:]), default=0)):
        for channel in channels[1:]:
            if rank < len(channel):
                add(channel[rank])
    for identity in sorted(scores, key=lambda item: (-scores[item], item)):
        add(identity)
    return tuple(selected)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Diversify public scoped Memory searches and return exact entry citations."""
    if limit <= 0:
        return ()
    queries = _search_queries(question)
    scores: dict[str, float] = {}
    citations: dict[str, MemoryCitation] = {}
    channels: list[list[str]] = []
    for query in queries:
        page = await memory.search(SearchMemoryRequest(query=query, limit=max(20, limit), mode="fts"))
        channel: list[str] = []
        for rank, hit in enumerate(page.hits, start=1):
            identity = hit.entry_id
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (20 + rank)
            channel.append(identity)
            citations[identity] = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        channels.append(channel)
    return tuple(citations[identity] for identity in _ranked_page_ids(channels, scores, limit))


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "parse_pdf_document",
    "search_pdf_pages",
]
