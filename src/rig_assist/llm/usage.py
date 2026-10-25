"""Turning token counts into a cost estimate."""

from dataclasses import dataclass

from rig_assist.llm.models import Usage


@dataclass(frozen=True)
class Prices:
    """USD per 1 million tokens, as providers list them. Set via LLM_*_PRICE_PER_1M."""

    input_per_1m: float
    output_per_1m: float


def cost_usd(usage: Usage, prices: Prices | None) -> float | None:
    """Estimated cost of `usage`, or None when prices aren't configured (unknown, not free)."""
    if prices is None:
        return None
    return (
        usage.input_tokens * prices.input_per_1m + usage.output_tokens * prices.output_per_1m
    ) / 1_000_000
