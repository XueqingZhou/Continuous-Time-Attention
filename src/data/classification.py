"""
Data preparation for classification tasks (IMDb, AG News, SST-2)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer


class ClassificationDataset(Dataset):
    """Dataset for text classification tasks"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def prepare_classification_data(
    dataset_name,
    tokenizer_path,
    max_length=256,
    cache_dir=None
):
    """
    Prepare data for classification tasks.
    
    Args:
        dataset_name (str): Name of dataset ('imdb', 'ag_news', 'sst2')
        tokenizer_path (str): Path to pre-trained tokenizer
        max_length (int): Maximum sequence length
        cache_dir (str): Directory to cache dataset
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test/validation dataset
        num_classes: Number of classes
        vocab_size: Vocabulary size
    """
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load dataset
    if dataset_name == 'imdb':
        if cache_dir:
            dataset = load_dataset(cache_dir)
        else:
            dataset = load_dataset('imdb')
        train_split = 'train'
        test_split = 'test'
        text_field = 'text'
        label_field = 'label'
        num_classes = 2
        
    elif dataset_name == 'ag_news':
        if cache_dir:
            dataset = load_dataset(cache_dir)
        else:
            dataset = load_dataset('ag_news')
        train_split = 'train'
        test_split = 'test'
        text_field = 'text'
        label_field = 'label'
        num_classes = 4
        
    elif dataset_name == 'sst2' or dataset_name == 'glue/sst2':
        if cache_dir:
            dataset = load_dataset(cache_dir)
        else:
            dataset = load_dataset('glue', 'sst2')
        train_split = 'train'
        test_split = 'validation'  # SST-2 uses validation as test
        text_field = 'sentence'
        label_field = 'label'
        num_classes = 2
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Extract texts and labels
    train_texts = dataset[train_split][text_field]
    train_labels = dataset[train_split][label_field]
    test_texts = dataset[test_split][text_field]
    test_labels = dataset[test_split][label_field]
    
    # Tokenize
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    # Create datasets
    train_dataset = ClassificationDataset(train_encodings, train_labels)
    test_dataset = ClassificationDataset(test_encodings, test_labels)
    
    return train_dataset, test_dataset, num_classes, tokenizer.vocab_size

