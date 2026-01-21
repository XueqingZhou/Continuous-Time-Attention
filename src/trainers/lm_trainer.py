"""
Trainer for language modeling tasks
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class LanguageModelingTrainer:
    """Trainer for language modeling tasks"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Use ignore_index=-100 to ignore padding tokens in loss calculation
        # reduction='sum' enables strict token-weighted aggregation.
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, int]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            perplexity: Training perplexity
        """
        self.model.train()
        total_nll = 0.0
        total_tokens = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            try:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Targets are input_ids shifted by one position
                targets = input_ids[:, 1:].contiguous()
                
                # Mask out padding positions in targets (align with outputs[:, :-1])
                # attention_mask[:, 1:] corresponds to positions in targets
                labels_mask = attention_mask[:, 1:].contiguous()
                # Set padding positions to ignore_index
                targets_masked = targets.clone()
                targets_masked[labels_mask == 0] = -100
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                
                # Compute loss (predict next token for each position)
                # outputs[:, :-1] predicts tokens at positions 1:
                # Only valid tokens (where targets_masked != -100) will contribute to loss
                nll_sum = self.criterion(
                    outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                    targets_masked.view(-1)
                )
                batch_tokens = int(labels_mask.sum().item())
                if batch_tokens == 0:
                    continue
                loss = nll_sum / batch_tokens
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                total_nll += float(nll_sum.item())
                total_tokens += batch_tokens
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM error. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_nll = total_nll / max(total_tokens, 1)
        perplexity = np.exp(avg_nll) if avg_nll < 100 else float("inf")
        
        return avg_nll, perplexity, total_tokens
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, int]:
        """
        Evaluate the model.
        
        Returns:
            avg_loss: Average validation loss
            perplexity: Validation perplexity
        """
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Targets
                    targets = input_ids[:, 1:].contiguous()
                    
                    # Mask out padding positions in targets (align with outputs[:, :-1])
                    labels_mask = attention_mask[:, 1:].contiguous()
                    targets_masked = targets.clone()
                    targets_masked[labels_mask == 0] = -100
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Compute loss (only on valid tokens)
                    nll_sum = self.criterion(
                        outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                        targets_masked.view(-1)
                    )
                    batch_tokens = int(labels_mask.sum().item())
                    if batch_tokens == 0:
                        continue
                    total_nll += float(nll_sum.item())
                    total_tokens += batch_tokens
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("CUDA OOM error. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_nll = total_nll / max(total_tokens, 1)
        perplexity = np.exp(avg_nll) if avg_nll < 100 else float("inf")
        
        return avg_nll, perplexity, total_tokens
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Returns:
            history: Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': []
        }
        
        best_val_ppl = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_ppl, train_tokens = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            
            # Validate
            val_loss, val_ppl, val_tokens = self.evaluate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            history.setdefault("train_tokens", []).append(train_tokens)
            history.setdefault("val_tokens", []).append(val_tokens)
            
            # Track best model
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                print(f"New best validation perplexity: {best_val_ppl:.2f}")
        
        return history

