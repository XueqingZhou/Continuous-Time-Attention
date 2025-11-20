"""
Data preparation for character-level IMDb (Long Range Arena)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class CharLevelDataset(Dataset):
    """Dataset for character-level text classification"""
    
    def __init__(self, texts, labels, char_to_idx, max_length=2048):
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert characters to indices
        char_indices = [
            self.char_to_idx.get(c, self.char_to_idx['<UNK>'])
            for c in text[:self.max_length]
        ]
        
        # Pad to max_length
        padding_length = self.max_length - len(char_indices)
        char_indices = char_indices + [self.char_to_idx['<PAD>']] * padding_length
        
        # Create attention mask (1 for real characters, 0 for padding)
        attention_mask = [1] * min(len(text), self.max_length) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(char_indices, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_char_vocab(texts, min_freq=1):
    """
    Build character vocabulary from texts.
    
    Args:
        texts (list): List of text strings
        min_freq (int): Minimum frequency for a character to be included
    
    Returns:
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
    """
    from collections import Counter
    
    # Count character frequencies
    char_counts = Counter()
    for text in texts:
        char_counts.update(text)
    
    # Build vocabulary
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    idx_to_char = {0: '<PAD>', 1: '<UNK>'}
    
    for char, count in char_counts.most_common():
        if count >= min_freq:
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[idx] = char
    
    return char_to_idx, idx_to_char


def prepare_char_level_data(
    max_length=2048,
    min_freq=1,
    cache_dir=None
):
    """
    Prepare character-level IMDb data for Long Range Arena.
    
    This follows the LRA setup with character-level tokenization
    and sequences up to 2048 characters.
    
    Args:
        max_length (int): Maximum sequence length (default 2048 for LRA)
        min_freq (int): Minimum character frequency
        cache_dir (str): Directory to cache dataset
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        vocab_size: Vocabulary size (number of unique characters)
    """
    # Load IMDb dataset
    if cache_dir:
        dataset = load_dataset(cache_dir)
    else:
        dataset = load_dataset('imdb')
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Build character vocabulary from training set
    char_to_idx, idx_to_char = build_char_vocab(train_texts, min_freq=min_freq)
    vocab_size = len(char_to_idx)
    
    print(f"Character vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Create datasets
    train_dataset = CharLevelDataset(
        train_texts, train_labels, char_to_idx, max_length
    )
    test_dataset = CharLevelDataset(
        test_texts, test_labels, char_to_idx, max_length
    )
    
    return train_dataset, test_dataset, vocab_size

