"""SFT generation eval helpers for AccRL distill experiments."""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import generate
from miles.utils.metric_utils import dict_add_prefix
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


async def generate_with_zero_reward(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate with SGLang and skip reward-model calls for SFT eval."""
    if isinstance(input.sample.prompt, list):
        input.sample.prompt = input.state.tokenizer.apply_chat_template(
            input.sample.prompt,
            tools=input.sample.metadata.get("tools") if input.sample.metadata else None,
            tokenize=False,
            add_generation_prompt=True,
        )
    sample = await generate(input.args, input.sample, input.sampling_params)
    sample.reward = 0.0
    sample.loss_mask = [1] * sample.response_length
    return GenerateFnOutput(samples=sample)


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "mean": statistics.fmean(values),
        "median": pct(0.50),
        "p90": pct(0.90),
        "max": ordered[-1],
        "min": ordered[0],
    }


def _find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for idx in range(len(values) - len(needle) + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return None


def _generated_thinking(sample: Any, tokenizer: Any) -> tuple[int, int, bool]:
    response_tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
    end_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    end_pos = _find_subsequence(response_tokens, end_ids)
    if end_pos is None:
        return len(response_tokens), 0, False
    return end_pos, max(0, len(response_tokens) - end_pos - len(end_ids)), True


def _flatten_samples(data: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    return {name: list(payload.get("samples") or []) for name, payload in data.items()}


def log_generated_thinking_metrics(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    """Add generated thinking metrics to Miles eval logs, then let default eval logging run."""
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    if extra_metrics is None:
        extra_metrics = {}

    for dataset_name, samples in _flatten_samples(data).items():
        thinking_lengths = []
        visible_lengths = []
        has_end = []
        for sample in samples:
            thinking_tokens, visible_tokens, found_end = _generated_thinking(sample, tokenizer)
            thinking_lengths.append(thinking_tokens)
            visible_lengths.append(visible_tokens)
            has_end.append(found_end)

        metrics: dict[str, Any] = {}
        metrics |= dict_add_prefix(_summary(thinking_lengths), "generated_thinking/")
        metrics |= dict_add_prefix(_summary(visible_lengths), "generated_visible_after_think/")
        if has_end:
            metrics["thinking_end_ratio"] = statistics.fmean(1.0 if x else 0.0 for x in has_end)
            metrics["thinking_missing_end_ratio"] = 1.0 - metrics["thinking_end_ratio"]
        if thinking_lengths and visible_lengths:
            ratios = [
                thinking / (thinking + visible)
                for thinking, visible in zip(thinking_lengths, visible_lengths, strict=True)
                if thinking + visible > 0
            ]
            if ratios:
                metrics["generated_thinking_ratio/mean"] = statistics.fmean(ratios)

        metrics = {key: value for key, value in metrics.items() if not isinstance(value, float) or math.isfinite(value)}
        extra_metrics.update({f"eval/{dataset_name}/{key}": value for key, value in metrics.items()})

    # Attach a compact sample table to the normal eval log so it shares eval/step.
    if getattr(args, "use_wandb", False):
        try:
            import wandb

            table = wandb.Table(
                columns=[
                    "rollout_id",
                    "dataset",
                    "sample_index",
                    "row_id",
                    "generated_thinking_tokens",
                    "found_end_think",
                    "response",
                    "target",
                ]
            )
            for dataset_name, samples in _flatten_samples(data).items():
                for sample in samples[:16]:
                    metadata = sample.metadata or {}
                    thinking_tokens, _visible_tokens, found_end = _generated_thinking(sample, tokenizer)
                    table.add_data(
                        rollout_id,
                        dataset_name,
                        sample.index,
                        metadata.get("sft_eval_row_id"),
                        thinking_tokens,
                        found_end,
                        sample.response,
                        sample.label,
                    )
            extra_metrics["eval/generated_thinking_table"] = table
        except Exception:
            logger.warning("failed to build SFT eval W&B table", exc_info=True)

    return False
