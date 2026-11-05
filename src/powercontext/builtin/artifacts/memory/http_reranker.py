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

"""HTTP scoring adapter for identity-preserving Memory reranking."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score bounded request batches, then select from the complete candidate pool."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates or type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("rerank requires candidates and a limit within their count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self.batch_size):
            batch = candidates[offset : offset + self.batch_size]
            expected = set(range(offset + 1, offset + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(batch, start=offset + 1)
                ],
            }
            response = await self.client.post(self.endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as error:
                raise ValueError("reranker response must be valid JSON") from error  # noqa: TRY003
            scores.update(_validated_scores(body, expected))

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(selected),
            usage=InferenceUsage(requests=requests),
        )


def _validated_scores(body: object, expected: set[int]) -> dict[int, float]:
    if not isinstance(body, dict) or set(body) != {"scores"} or not isinstance(body["scores"], list):
        raise ValueError("reranker response must contain only a scores list")  # noqa: TRY003
    scores: dict[int, float] = {}
    for item in body["scores"]:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("reranker score must contain only rank and score")  # noqa: TRY003
        rank, score = item["rank"], item["score"]
        if type(rank) is not int or rank not in expected or rank in scores:
            raise ValueError("reranker ranks must cover the current batch exactly once")  # noqa: TRY003
        if type(score) not in {int, float} or not math.isfinite(score):
            raise ValueError("reranker scores must be finite numbers")  # noqa: TRY003
        scores[rank] = float(score)
    if set(scores) != expected:
        raise ValueError("reranker response is missing candidate ranks")  # noqa: TRY003
    return scores


__all__ = ["HTTPMemoryReranker"]
