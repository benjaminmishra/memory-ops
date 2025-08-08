"""QR‑HEAD context compression implementation.

This module provides a `reduce` function which trims a long input
conversation down to the most relevant tokens for a given query.  It
implements the algorithm described in the paper "Query‑Focused
Retrieval Heads Improve Long‑Context Reasoning and Re‑ranking" by
Zhang et al.  The core idea is to identify a small set of attention
heads that naturally attend to the query‑relevant parts of the
context.  At runtime we run the model once with `output_attentions`
enabled, extract the attention scores of those heads and select the
top‑K tokens.

If no LoRA weights are provided or the weights do not specify
retrieval heads metadata, a naive fallback summarisation is used
instead.  This fallback simply keeps the first and last few tokens of
the context, ensuring that at least some history is retained.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, List[Tuple[int, int]]]:
    """Load the base model and optional LoRA retrieval heads.

    Returns a tuple of (model, tokenizer, retrieval_heads).  The
    retrieval_heads is a list of (layer_index, head_index) tuples
    indicating which attention heads to use for scoring.
    """
    settings = get_settings()
    model_name = settings.model_name
    lora_id = settings.lora_id
    logger.info(f"Loading base model {model_name}")
    # Use 8bit quantisation if bitsandbytes is available; otherwise use
    # half precision to reduce memory usage.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
        )
    except Exception:
        # Fallback to fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    retrieval_heads: List[Tuple[int, int]] = []
    if lora_id:
        logger.info(f"Applying LoRA weights {lora_id}")
        # Apply LoRA weights to the model.
        model = PeftModel.from_pretrained(model, lora_id)
        # Retrieve the set of retrieval heads encoded in the LoRA config
        # If the LoRA was trained using the QR‑HEAD trainer, the config
        # includes a `retrieval_heads` attribute (list of (layer, head)).
        try:
            config = model.peft_config
            rh = config.get("retrieval_heads")  # type: ignore[attr-defined]
            if isinstance(rh, (list, tuple)):
                retrieval_heads = [(int(layer), int(head)) for layer, head in rh]
        except Exception:
            pass
    return model, tokenizer, retrieval_heads


def reduce(query: str, context: str, top_k: int | None = None) -> Tuple[str, int, int]:
    """Reduce a long context to its most relevant parts.

    :param query: The question or message for which we need context.
    :param context: The full conversation history or document.
    :param top_k: Number of tokens to keep; if None uses settings.top_k.
    :returns: tuple of (condensed_text, tokens_before, tokens_after)
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k
    model, tokenizer, retrieval_heads = _load_model()
    device = next(model.parameters()).device
    # Compose input with a separator token.  Using a clear delimiter helps
    # the model distinguish between query and context segments.
    sep = "\n\n### CONTEXT ###\n\n"
    full_text = query + sep + context
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids.to(device)
    tokens_before = input_ids.shape[1]
    # If no retrieval heads available, fall back to naive summarisation
    if not retrieval_heads:
        # Keep the first K/2 and last K/2 tokens of the context
        half = max(top_k // 2, 1)
        # Extract only the context part of the ids (after sep)
        # We'll locate the sep by tokenising separately for robustness.
        sep_ids = tokenizer(sep, return_tensors="pt").input_ids.squeeze().tolist()
        sep_len = len(sep_ids)
        # Remove the query tokens and separator from the input
        query_tokens = tokenizer(query, return_tensors="pt").input_ids.squeeze().tolist()
        start_idx = len(query_tokens) + sep_len
        context_ids = input_ids.squeeze().tolist()[start_idx:]
        if len(context_ids) <= top_k:
            selected = context_ids
        else:
            selected = context_ids[:half] + context_ids[-half:]
        condensed = tokenizer.decode(selected, skip_special_tokens=True)
        tokens_after = len(selected)
        return condensed, tokens_before, tokens_after
    # Run the model once to get attention scores.  We disable caching
    # because we need full attentions across the entire context.
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=enc.attention_mask.to(device),
            output_attentions=True,
            use_cache=False,
        )
    # outputs.attentions is a tuple of length num_layers; each element is
    # (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    # Compute per-token relevance by averaging the attention scores of
    # the selected heads onto the query token (position 0).  We collect
    # relevance across all selected heads and normalise.
    seq_len = input_ids.shape[1]
    relevance = torch.zeros(seq_len, device=device)
    for layer_idx, head_idx in retrieval_heads:
        # Clamp layer index because LoRA might refer to last layers
        layer = attentions[layer_idx]
        # Attention from query token (position 0) to all tokens
        # shape: (batch, num_heads, tgt_len, src_len)
        attn_scores = layer[0, head_idx, 0, :]
        relevance += attn_scores
    # Normalise
    relevance /= max(len(retrieval_heads), 1)
    # Pick the top_k highest scoring tokens *after the query and separator*
    # We avoid selecting query tokens by zeroing their relevance.
    sep_ids = tokenizer(sep, return_tensors="pt").input_ids.squeeze().tolist()
    sep_len = len(sep_ids)
    query_len = len(tokenizer(query, return_tensors="pt").input_ids.squeeze().tolist())
    # Zero out relevance for query tokens and separator
    relevance[: query_len + sep_len] = -float("inf")
    # Identify top_k indices
    top_scores, top_indices = torch.topk(relevance, k=min(top_k, seq_len))
    # Sort indices to preserve order in input
    sorted_indices = torch.sort(top_indices).values.tolist()
    selected_ids = input_ids.squeeze()[sorted_indices].tolist()
    condensed = tokenizer.decode(selected_ids, skip_special_tokens=True)
    tokens_after = len(selected_ids)
    return condensed, tokens_before, tokens_after