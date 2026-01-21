"""
Data preparation for language modeling tasks (WikiText-103).

This module implements a standard LM pipeline:
1) Tokenize texts into a continuous token stream.
2) Concatenate into one long sequence.
3) Split into fixed-length blocks (block_size) without padding.
"""

from typing import Dict, List, Optional, Tuple

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Ensure tokenizer has a pad token for compatibility."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _filter_empty(example: Dict[str, str]) -> bool:
    """Filter out empty lines."""
    return len(example["text"].strip()) > 0


def _tokenize_texts(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    add_eos: bool,
) -> Dict[str, List[List[int]]]:
    """Tokenize raw texts into token id lists.

    Args:
        examples: Batch of examples with key "text".
        tokenizer: Tokenizer instance.
        add_eos: Whether to append EOS to each sequence.

    Returns:
        Dict with "input_ids" as list of token id lists.
    """
    outputs = tokenizer(
        examples["text"],
        add_special_tokens=False,
        return_attention_mask=False,
    )
    if add_eos and tokenizer.eos_token_id is not None:
        outputs["input_ids"] = [
            ids + [tokenizer.eos_token_id] for ids in outputs["input_ids"]
        ]
    return {"input_ids": outputs["input_ids"]}


def _group_texts(
    examples: Dict[str, List[List[int]]],
    block_size: int,
) -> Dict[str, List[List[int]]]:
    """Concatenate token lists and split into fixed-length blocks.

    Args:
        examples: Dict with "input_ids" as list of token id lists.
        block_size: Length of each block.

    Returns:
        Dict with "input_ids" and "attention_mask" lists of length block_size.
    """
    concatenated: List[int] = sum(examples["input_ids"], [])
    total_length = len(concatenated)
    if total_length < block_size:
        return {"input_ids": [], "attention_mask": []}

    total_length = (total_length // block_size) * block_size
    input_ids = [
        concatenated[i : i + block_size] for i in range(0, total_length, block_size)
    ]
    attention_mask = [[1] * block_size for _ in range(len(input_ids))]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def prepare_lm_data(
    tokenizer_path: str,
    max_length: int = 512,
    train_sample_size: Optional[int] = None,
    val_sample_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    add_eos: bool = True,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """Prepare data for language modeling on WikiText-103.

    Args:
        tokenizer_path: Path to pre-trained tokenizer.
        max_length: Maximum sequence length (used as block_size if block_size is None).
        train_sample_size: Number of training samples (None for all).
        val_sample_size: Number of validation samples (None for all).
        cache_dir: Directory to cache dataset.
        block_size: Length of each LM block. Defaults to max_length.
        add_eos: Whether to append EOS between documents.

    Returns:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        vocab_size: Vocabulary size.
    """
    block_size = block_size or max_length

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    _ensure_pad_token(tokenizer)

    if cache_dir:
        dataset: DatasetDict = load_dataset(cache_dir)
    else:
        dataset = load_dataset("wikitext", "wikitext-103-v1")

    dataset = dataset.filter(_filter_empty)
    dataset = dataset.map(
        lambda x: _tokenize_texts(x, tokenizer=tokenizer, add_eos=add_eos),
        batched=True,
        remove_columns=["text"],
    )
    dataset = dataset.map(
        lambda x: _group_texts(x, block_size=block_size),
        batched=True,
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if train_sample_size is not None:
        train_dataset = train_dataset.select(range(train_sample_size))
    if val_sample_size is not None:
        val_dataset = val_dataset.select(range(val_sample_size))

    train_dataset = train_dataset.with_format("torch", columns=["input_ids", "attention_mask"])
    val_dataset = val_dataset.with_format("torch", columns=["input_ids", "attention_mask"])

    return train_dataset, val_dataset, len(tokenizer)

