"""Share concurrent embedding work and retain successful vectors in an LRU cache."""

# ruff: noqa: TRY003, TRY301, TRY300

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _Pending:
    vector: asyncio.Future[EmbeddingVector]
    batch: asyncio.Task[InferenceUsage] | None = None


class CoalescingEmbeddingModel:
    """Deduplicate texts across callers while preserving each caller's usage."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not 0 <= coalesce_ms < float("inf")
        ):
            raise ValueError("coalesce_ms must be finite and nonnegative")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Pending] = {}
        self._queued: list[str] = []
        self._flush_handle: asyncio.Handle | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=())

        # All bookkeeping before the first await is atomic within one event loop turn.
        known: dict[str, EmbeddingVector] = {}
        waiting: dict[str, _Pending] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            if text in known or text in waiting:
                continue
            if text in self._cache:
                known[text] = self._cache[text]
                self._cache.move_to_end(text)
                continue
            pending = self._inflight.get(text)
            if pending is None:
                future: asyncio.Future[EmbeddingVector] = loop.create_future()
                # A cancelled sole waiter can leave a failed provider result unobserved.
                future.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
                pending = _Pending(future)
                self._inflight[text] = pending
                self._queued.append(text)
            waiting[text] = pending

        if self._queued and self._flush_handle is None:
            self._flush_handle = loop.call_later(self._coalesce_seconds, self._flush)

        if waiting:
            values = await asyncio.gather(*(asyncio.shield(item.vector) for item in waiting.values()))
            known.update(zip(waiting, values, strict=True))

        batches = {item.batch for item in waiting.values()}
        usage = InferenceUsage(requests=0)
        for batch in batches:
            if batch is None:
                raise RuntimeError("embedding batch was not assigned")
            actual = await asyncio.shield(batch)
            usage = InferenceUsage(
                requests=usage.requests + actual.requests,
                input_tokens=_sum_optional(usage.input_tokens, actual.input_tokens),
                output_tokens=_sum_optional(usage.output_tokens, actual.output_tokens),
            )
        return EmbeddingResult(vectors=tuple(known[text] for text in texts), usage=usage)

    def _flush(self) -> None:
        self._flush_handle = None
        queued, self._queued = self._queued, []
        for offset in range(0, len(queued), self._max_batch_size):
            group = tuple(queued[offset : offset + self._max_batch_size])
            task = asyncio.create_task(self._run_batch(group))
            for text in group:
                self._inflight[text].batch = task

    async def _run_batch(self, texts: tuple[str, ...]) -> InferenceUsage:
        try:
            result = await self._model.embed(texts)
            if len(result.vectors) != len(texts):
                raise ValueError("embedding vector count does not match input count")
            for text, vector in zip(texts, result.vectors, strict=True):
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                self._inflight[text].vector.set_result(vector)
            return result.usage
        except Exception as error:
            for text in texts:
                self._inflight[text].vector.set_exception(error)
            return InferenceUsage(requests=0)
        finally:
            for text in texts:
                del self._inflight[text]


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)
