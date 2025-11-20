"""
Data preparation for language modeling tasks (WikiText-103)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer


class LanguageModelingDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def prepare_lm_data(
    tokenizer_path,
    max_length=512,
    train_sample_size=None,
    val_sample_size=None,
    cache_dir=None
):
    """
    Prepare data for language modeling on WikiText-103.
    
    Args:
        tokenizer_path (str): Path to pre-trained tokenizer
        max_length (int): Maximum sequence length
        train_sample_size (int): Number of training samples (None for all)
        val_sample_size (int): Number of validation samples (None for all)
        cache_dir (str): Directory to cache dataset
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        vocab_size: Vocabulary size
    """
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load WikiText-103
    if cache_dir:
        dataset = load_dataset(cache_dir)
    else:
        dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    # Extract and filter texts (remove empty lines)
    train_texts = [
        text for text in dataset['train']['text']
        if len(text.strip()) > 0
    ]
    val_texts = [
        text for text in dataset['validation']['text']
        if len(text.strip()) > 0
    ]
    
    # Sub-sample if requested
    if train_sample_size is not None:
        train_texts = train_texts[:train_sample_size]
    if val_sample_size is not None:
        val_texts = val_texts[:val_sample_size]
    
    # Create datasets
    train_dataset = LanguageModelingDataset(train_texts, tokenizer, max_length)
    val_dataset = LanguageModelingDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset, tokenizer.vocab_size

