"""
PDE-Enhanced Transformer Models

This module implements Transformer models integrated with PDE layers for
classification and language modeling tasks.
"""

import torch
import torch.nn as nn
from .pde_layers import create_pde_layer


class PDETransformerClassifier(nn.Module):
    """
    PDE-Transformer for sequence classification tasks.
    
    Integrates PDE refinement layers after each Transformer layer to enable
    continuous-time modeling of token interactions.
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): FFN hidden dimension
        num_layers (int): Number of Transformer layers
        num_classes (int): Number of output classes
        pde_type (str): Type of PDE layer ('diffusion', 'wave', etc.)
        pde_steps (int): Number of PDE refinement steps per layer
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate
        **pde_kwargs: Additional arguments for PDE layers
    """
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        num_classes,
        pde_type='diffusion',
        pde_steps=1,
        max_length=512,
        dropout=0.1,
        **pde_kwargs
    ):
        super(PDETransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # PDE refinement layers (one per Transformer layer)
        self.pde_layers = nn.ModuleList([
            nn.ModuleList([
                create_pde_layer(pde_type, embed_dim, **pde_kwargs)
                for _ in range(pde_steps)
            ])
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len), 1 for valid tokens
        
        Returns:
            Logits for classification (batch_size, num_classes)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create padding mask (True for positions to ignore)
        padding_mask = (attention_mask == 0)
        
        # Apply Transformer + PDE layers
        for transformer_layer, pde_layer_list in zip(self.transformer_layers, self.pde_layers):
            # Transformer layer
            x = transformer_layer(x, src_key_padding_mask=padding_mask)
            
            # PDE refinement steps
            # Note: PDE layers expect (batch, hidden, seq_len) format
            x_transposed = x.transpose(1, 2)  # (batch, seq_len, hidden) -> (batch, hidden, seq_len)
            for pde_layer in pde_layer_list:
                x_transposed = pde_layer(x_transposed)
            x = x_transposed.transpose(1, 2)  # Back to (batch, seq_len, hidden)
        
        # Masked mean pooling
        expanded_mask = attention_mask.unsqueeze(-1).expand(x.size())
        sum_embeddings = torch.sum(x * expanded_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
        mean_pooled = sum_embeddings / sum_mask
        
        # Classification
        return self.fc(mean_pooled)


class StandardTransformerClassifier(nn.Module):
    """
    Standard Transformer for sequence classification (baseline model).
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): FFN hidden dimension
        num_layers (int): Number of Transformer layers
        num_classes (int): Number of output classes
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        num_classes,
        max_length=512,
        dropout=0.1
    ):
        super(StandardTransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits for classification (batch_size, num_classes)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create padding mask
        padding_mask = (attention_mask == 0)
        
        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Masked mean pooling
        expanded_mask = attention_mask.unsqueeze(-1).expand(x.size())
        sum_embeddings = torch.sum(x * expanded_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
        mean_pooled = sum_embeddings / sum_mask
        
        # Classification
        return self.fc(mean_pooled)


class PDETransformerLM(nn.Module):
    """
    PDE-Transformer for language modeling tasks.
    
    Predicts next tokens in a sequence using PDE-enhanced representations.
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): FFN hidden dimension
        num_layers (int): Number of Transformer layers
        pde_type (str): Type of PDE layer
        pde_steps (int): Number of PDE refinement steps
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate
        **pde_kwargs: Additional PDE layer arguments
    """
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        pde_type='diffusion',
        pde_steps=1,
        max_length=512,
        dropout=0.1,
        **pde_kwargs
    ):
        super(PDETransformerLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # PDE layers
        self.pde_layers = nn.ModuleList([
            nn.ModuleList([
                create_pde_layer(pde_type, embed_dim, **pde_kwargs)
                for _ in range(pde_steps)
            ])
            for _ in range(num_layers)
        ])
        
        # Language modeling head
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create padding mask
        padding_mask = (attention_mask == 0)
        
        # Apply Transformer + PDE layers
        for transformer_layer, pde_layer_list in zip(self.transformer_layers, self.pde_layers):
            x = transformer_layer(x, src_key_padding_mask=padding_mask)
            
            # Apply PDE refinement
            x_transposed = x.transpose(1, 2)
            for pde_layer in pde_layer_list:
                x_transposed = pde_layer(x_transposed)
            x = x_transposed.transpose(1, 2)
        
        # Project to vocabulary
        return self.fc(x)


class StandardTransformerLM(nn.Module):
    """
    Standard Transformer for language modeling (baseline).
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): FFN hidden dimension
        num_layers (int): Number of Transformer layers
        max_length (int): Maximum sequence length
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_length=512,
        dropout=0.1
    ):
        super(StandardTransformerLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Language modeling head
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create padding mask
        padding_mask = (attention_mask == 0)
        
        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Project to vocabulary
        return self.fc(x)

