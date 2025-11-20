"""
Trainer for language modeling tasks
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class LanguageModelingTrainer:
    """Trainer for language modeling tasks"""
    
    def __init__(self, model, device, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            perplexity: Training perplexity
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            try:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Targets are input_ids shifted by one position
                targets = input_ids[:, 1:].contiguous()
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                
                # Compute loss (predict next token for each position)
                # outputs[:, :-1] predicts tokens at positions 1:
                loss = self.criterion(
                    outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                    targets.view(-1)
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                total_loss += loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM error. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, perplexity
    
    def evaluate(self, dataloader):
        """
        Evaluate the model.
        
        Returns:
            avg_loss: Average validation loss
            perplexity: Validation perplexity
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Targets
                    targets = input_ids[:, 1:].contiguous()
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Compute loss
                    loss = self.criterion(
                        outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
                    
                    total_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("CUDA OOM error. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, perplexity
    
    def train(self, train_loader, val_loader, num_epochs):
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
            train_loss, train_ppl = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            
            # Validate
            val_loss, val_ppl = self.evaluate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            
            # Track best model
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                print(f"New best validation perplexity: {best_val_ppl:.2f}")
        
        return history

