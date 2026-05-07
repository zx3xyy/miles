import logging
import math
import os
import statistics

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_processor, load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _summarize(values):
    if not values:
        return {}
    sorted_values = sorted(values)

    def percentile(pct):
        idx = round((len(sorted_values) - 1) * pct)
        return sorted_values[idx]

    return {
        "mean": statistics.fmean(values),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": sorted_values[-1],
        "min": sorted_values[0],
    }


def _find_thinking_span(token_ids, loss_mask, response_length, tokenizer):
    """Return supervised target token counts before and after </think>.

    For SFT this measures the target trace, not sampled model output. The Qwen3
    loss mask starts after the assistant generation prefix, so counting masked
    tokens until </think> gives the length of the supervised thinking trace.
    """
    if response_length <= 0:
        return 0, 0, False

    response_tokens = token_ids[-response_length:]
    response_loss_mask = loss_mask[-response_length:]
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    special_ids = set(tokenizer.all_special_ids)

    thinking_tokens = 0
    visible_tokens = 0
    found_end = False
    after_end = False

    for token_id, mask in zip(response_tokens, response_loss_mask, strict=True):
        if mask != 1:
            continue
        if not found_end and token_id == end_think_id:
            found_end = True
            after_end = True
            continue
        if after_end:
            if token_id not in special_ids and tokenizer.decode([token_id]).strip():
                visible_tokens += 1
        else:
            thinking_tokens += 1

    return thinking_tokens, visible_tokens, found_end


def _thinking_metrics(thinking_lengths, visible_lengths, has_end):
    metrics = {}
    for prefix, values in (
        ("rollout/thinking", thinking_lengths),
        ("rollout/target_thinking", thinking_lengths),
        ("rollout/visible_after_think", visible_lengths),
        ("rollout/target_visible_after_think", visible_lengths),
    ):
        for name, value in _summarize(values).items():
            metrics[f"{prefix}/{name}"] = value

    if has_end:
        metrics["rollout/thinking_end_ratio"] = statistics.fmean(1.0 if x else 0.0 for x in has_end)
        metrics["rollout/thinking_missing_end_ratio"] = 1.0 - metrics["rollout/thinking_end_ratio"]
        metrics["rollout/target_thinking_end_ratio"] = metrics["rollout/thinking_end_ratio"]
        metrics["rollout/target_thinking_missing_end_ratio"] = metrics["rollout/thinking_missing_end_ratio"]
    if thinking_lengths and visible_lengths:
        ratios = [
            thinking / (thinking + visible)
            for thinking, visible in zip(thinking_lengths, visible_lengths, strict=True)
            if thinking + visible > 0
        ]
        if ratios:
            metrics["rollout/thinking_ratio/mean"] = statistics.fmean(ratios)

    return {key: value for key, value in metrics.items() if isinstance(value, int) or math.isfinite(value)}


def _token_rows(tokenizer, token_ids, loss_mask, *, rollout_id, sample_id, center, radius, region):
    special_ids = set(tokenizer.all_special_ids)
    start = max(0, center - radius)
    end = min(len(token_ids), center + radius + 1)
    return [
        [
            rollout_id,
            sample_id,
            region,
            idx,
            token_ids[idx],
            tokenizer.convert_ids_to_tokens([token_ids[idx]])[0],
            tokenizer.decode([token_ids[idx]]),
            loss_mask[idx],
            token_ids[idx] in special_ids,
        ]
        for idx in range(start, end)
    ]


def _maybe_build_wandb_token_table(args, rollout_id, sample_id, token_ids, loss_mask, tokenizer):
    every = int(os.environ.get("SFT_WANDB_TOKEN_TABLE_EVERY", "0"))
    if every <= 0 or rollout_id % every != 0 or not getattr(args, "use_wandb", False):
        return {}

    try:
        import wandb
    except ImportError:
        logger.warning("SFT_WANDB_TOKEN_TABLE_EVERY is set, but wandb is not importable")
        return {}

    if 1 not in loss_mask:
        return {}

    radius = int(os.environ.get("SFT_WANDB_TOKEN_TABLE_RADIUS", "24"))
    first_loss = loss_mask.index(1)
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    end_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == end_think_id]

    rows = _token_rows(
        tokenizer,
        token_ids,
        loss_mask,
        rollout_id=rollout_id,
        sample_id=sample_id,
        center=first_loss,
        radius=radius,
        region="first_loss",
    )
    if end_positions:
        rows.extend(
            _token_rows(
                tokenizer,
                token_ids,
                loss_mask,
                rollout_id=rollout_id,
                sample_id=sample_id,
                center=end_positions[0],
                radius=radius,
                region="end_think",
            )
        )

    table = wandb.Table(
        columns=[
            "rollout_id",
            "sample_id",
            "region",
            "token_index",
            "token_id",
            "token",
            "text",
            "loss_mask",
            "is_special",
        ],
        data=rows,
    )
    return {"rollout/sft_token_mask_table": table}


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    thinking_lengths = []
    visible_lengths = []
    has_end = []
    debug_metrics = {}

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        if sample.metadata is None:
            sample.metadata = {}
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        thinking_tokens, visible_tokens, found_end = _find_thinking_span(
            token_ids, loss_mask, response_length, TOKENIZER
        )
        thinking_lengths.append(thinking_tokens)
        visible_lengths.append(visible_tokens)
        has_end.append(found_end)
        sample.metadata["sft_target_thinking_tokens"] = thinking_tokens
        sample.metadata["sft_target_visible_after_think_tokens"] = visible_tokens
        sample.metadata["sft_target_has_think_end"] = found_end

        if i == 0 and not SAMPLE_PRINTED:
            preview_tokens = token_ids[-min(response_length, 64) :] if response_length > 0 else []
            target_tail = TOKENIZER.decode(preview_tokens)[-500:] if preview_tokens else ""
            prompt_chars = sum(len(message.get("content", "")) for message in messages)
            logger.info(
                "sft_rollout::generate_rollout example data: "
                "sample_id=%s num_messages=%d prompt_chars=%d total_tokens=%d "
                "response_length=%d thinking_tokens=%d visible_tokens=%d "
                "found_end=%s target_tail=%r",
                getattr(sample, "id", None),
                len(messages),
                prompt_chars,
                len(token_ids),
                response_length,
                thinking_tokens,
                visible_tokens,
                found_end,
                target_tail,
            )
            SAMPLE_PRINTED = True
        if i == 0:
            debug_metrics = _maybe_build_wandb_token_table(
                args, rollout_id, getattr(sample, "id", None), token_ids, loss_mask, TOKENIZER
            )

    metrics = _thinking_metrics(thinking_lengths, visible_lengths, has_end)
    metrics.update(debug_metrics)
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
