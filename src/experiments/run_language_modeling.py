"""
Run language modeling experiment (Tables 3-5: WikiText-103)

This script reproduces the WikiText-103 results from Tables 3, 4, and 5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json

from models import PDETransformerLM, StandardTransformerLM
from data import prepare_lm_data
from trainers import LanguageModelingTrainer


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Prepare data
    print("\nPreparing WikiText-103 dataset...")
    train_dataset, val_dataset, vocab_size = prepare_lm_data(
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        train_sample_size=args.train_sample_size,
        val_sample_size=args.val_sample_size,
        cache_dir=args.cache_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {args.max_length}")
    
    # Model configurations
    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'max_length': args.max_length,
        'dropout': args.dropout,
    }
    
    # Run experiments for both models
    results = {}
    
    for model_type in ['standard', 'pde']:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Transformer (Language Modeling)")
        print(f"{'='*60}")
        
        # Create model
        if model_type == 'pde':
            model = PDETransformerLM(
                **model_config,
                pde_type=args.pde_type,
                pde_steps=args.pde_steps,
            ).to(device)
            print(f"PDE Type: {args.pde_type}, PDE Steps: {args.pde_steps}")
        else:
            model = StandardTransformerLM(**model_config).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        
        # Create trainer
        trainer = LanguageModelingTrainer(model, device, optimizer, scheduler)
        
        # Train
        history = trainer.train(train_loader, val_loader, args.num_epochs)
        
        # Save results
        best_ppl = min(history['val_ppl'])
        results[model_type] = {
            'best_perplexity': best_ppl,
            'final_perplexity': history['val_ppl'][-1],
            'history': history,
            'num_params': num_params
        }
        
        print(f"\n{model_type.upper()} - Best Perplexity: {best_ppl:.2f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY (WikiText-103)")
    print(f"{'='*60}")
    print(f"Standard Transformer PPL: {results['standard']['best_perplexity']:.2f}")
    print(f"PDE-Transformer PPL:      {results['pde']['best_perplexity']:.2f}")
    
    improvement = ((results['standard']['best_perplexity'] - results['pde']['best_perplexity']) 
                   / results['standard']['best_perplexity'] * 100)
    print(f"Relative Improvement:     {improvement:.2f}%")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'wikitext103_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run WikiText-103 language modeling experiment')
    
    # Dataset arguments
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache dataset')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--train_sample_size', type=int, default=None,
                        help='Number of training samples (None for all)')
    parser.add_argument('--val_sample_size', type=int, default=None,
                        help='Number of validation samples (None for all)')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='FFN hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # PDE arguments
    parser.add_argument('--pde_type', type=str, default='diffusion',
                        choices=['diffusion', 'wave', 'reaction-diffusion', 'advection-diffusion'],
                        help='Type of PDE layer')
    parser.add_argument('--pde_steps', type=int, default=4,
                        help='Number of PDE refinement steps (Table 4 shows 4 is optimal)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args)

