"""
Run data size ablation study on WikiText-103

This script studies the effect of training data size on model performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from models import PDETransformerLM, StandardTransformerLM
from data import prepare_lm_data
from trainers import LanguageModelingTrainer
from utils.config import load_yaml_config, set_defaults_from_config
from utils.metadata import collect_metadata, write_metadata


def plot_datasize_results(
    results: dict,
    data_sizes: list,
    output_dir: str,
) -> None:
    """Plot comprehensive data size ablation results."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    colors_std = plt.cm.Blues(np.linspace(0.4, 0.9, len(data_sizes)))
    colors_pde = plt.cm.Oranges(np.linspace(0.4, 0.9, len(data_sizes)))
    
    for i, size in enumerate(data_sizes):
        std_key = f"Standard-{size:.1%}"
        pde_key = f"PDE-{size:.1%}"
        
        if std_key in results and results[std_key]['train_loss']:
            epochs = range(1, len(results[std_key]['train_loss']) + 1)
            ax1.plot(epochs, results[std_key]['train_loss'], 
                    '--', alpha=0.7, color=colors_std[i], label=f'Std {size:.1%}')
        
        if pde_key in results and results[pde_key]['train_loss']:
            epochs = range(1, len(results[pde_key]['train_loss']) + 1)
            ax1.plot(epochs, results[pde_key]['train_loss'], 
                    '-', linewidth=2, color=colors_pde[i], label=f'PDE {size:.1%}')
    
    ax1.set_title('Training Loss vs Data Size', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation perplexity curves
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, size in enumerate(data_sizes):
        std_key = f"Standard-{size:.1%}"
        pde_key = f"PDE-{size:.1%}"
        
        if std_key in results and results[std_key]['val_ppl']:
            epochs = range(1, len(results[std_key]['val_ppl']) + 1)
            ax2.plot(epochs, results[std_key]['val_ppl'], 
                    '--', alpha=0.7, color=colors_std[i], label=f'Std {size:.1%}')
        
        if pde_key in results and results[pde_key]['val_ppl']:
            epochs = range(1, len(results[pde_key]['val_ppl']) + 1)
            ax2.plot(epochs, results[pde_key]['val_ppl'], 
                    '-', linewidth=2, color=colors_pde[i], label=f'PDE {size:.1%}')
    
    ax2.set_title('Validation Perplexity vs Data Size', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final perplexity bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    
    std_final = []
    pde_final = []
    
    for size in data_sizes:
        std_key = f"Standard-{size:.1%}"
        pde_key = f"PDE-{size:.1%}"
        
        if std_key in results and results[std_key]['val_ppl']:
            std_final.append(results[std_key]['val_ppl'][-1])
        else:
            std_final.append(np.nan)
            
        if pde_key in results and results[pde_key]['val_ppl']:
            pde_final.append(results[pde_key]['val_ppl'][-1])
        else:
            pde_final.append(np.nan)
    
    x = np.arange(len(data_sizes))
    width = 0.35
    
    ax3.bar(x - width/2, std_final, width, label='Standard', 
            color=colors_std, edgecolor='black', linewidth=1.5)
    ax3.bar(x + width/2, pde_final, width, label='PDE', 
            color=colors_pde, edgecolor='black', linewidth=1.5)
    
    ax3.set_title('Final Perplexity Comparison', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Dataset Size')
    ax3.set_ylabel('Perplexity')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{s:.1%}' for s in data_sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (s, p) in enumerate(zip(std_final, pde_final)):
        if not np.isnan(s):
            ax3.text(i - width/2, s, f'{s:.1f}', ha='center', va='bottom', fontsize=9)
        if not np.isnan(p):
            ax3.text(i + width/2, p, f'{p:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Relative improvement heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    
    improvements = []
    for i, size in enumerate(data_sizes):
        std_key = f"Standard-{size:.1%}"
        pde_key = f"PDE-{size:.1%}"
        
        if (std_key in results and pde_key in results and 
            results[std_key]['val_ppl'] and results[pde_key]['val_ppl']):
            std_ppl = results[std_key]['val_ppl'][-1]
            pde_ppl = results[pde_key]['val_ppl'][-1]
            improvement = (std_ppl - pde_ppl) / std_ppl * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    heatmap_data = [improvements]
    
    sns.heatmap(heatmap_data, ax=ax4, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, cbar_kws={'label': 'Improvement (%)'},
                xticklabels=[f'{s:.1%}' for s in data_sizes],
                yticklabels=['PDE vs Std'])
    
    ax4.set_title('Relative Performance Improvement', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Dataset Size')
    
    plt.suptitle('Data Size Ablation Study on WikiText-103', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'datasize_ablation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'datasize_ablation.pdf'), bbox_inches='tight')
    plt.close()


def main(args: argparse.Namespace) -> None:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Prepare full dataset
    print("\nPreparing WikiText-103 dataset...")
    train_dataset, val_dataset, vocab_size = prepare_lm_data(
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
        add_eos=args.add_eos,
    )
    
    print(f"Full train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Fixed validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'max_length': args.max_length,
        'dropout': args.dropout,
    }
    
    # Storage for results
    results = {}
    
    # Run experiments for each data size
    for data_ratio in args.data_sizes:
        print(f"\n{'='*60}")
        print(f"Experimenting with {data_ratio:.1%} of training data")
        print(f"{'='*60}")
        
        # Sample training data
        num_samples = int(len(train_dataset) * data_ratio)
        print(f"Using {num_samples} training samples")
        
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_subset = Subset(train_dataset, indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        # Train both models
        for model_type in ['standard', 'pde']:
            print(f"\nTraining {model_type.upper()} Transformer")
            
            # Create model
            if model_type == 'pde':
                model = PDETransformerLM(
                    **model_config,
                    pde_type=args.pde_type,
                    pde_steps=args.pde_steps
                ).to(device)
            else:
                model = StandardTransformerLM(**model_config).to(device)
            
            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            
            # Scheduler
            total_steps = len(train_loader) * args.num_epochs
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_steps
            ) if warmup_steps > 0 else None
            
            # Trainer
            trainer = LanguageModelingTrainer(model, device, optimizer, scheduler)
            
            # Train
            history = trainer.train(train_loader, val_loader, args.num_epochs)
            
            # Store results (match plotting/summary keys)
            key_prefix = "Standard" if model_type == "standard" else "PDE"
            key = f"{key_prefix}-{data_ratio:.1%}"
            results[key] = {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_ppl': history['val_ppl'],
                'best_ppl': min(history['val_ppl'])
            }
            
            print(f"{model_type.upper()} - Best PPL: {results[key]['best_ppl']:.2f}")
            
            # Clean up
            del model, optimizer
            torch.cuda.empty_cache()
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_datasize_results(results, args.data_sizes, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'datasize_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATA SIZE ABLATION SUMMARY")
    print(f"{'='*60}")
    for data_ratio in args.data_sizes:
        std_key = f"Standard-{data_ratio:.1%}"
        pde_key = f"PDE-{data_ratio:.1%}"
        
        if std_key in results and pde_key in results:
            std_ppl = results[std_key]['best_ppl']
            pde_ppl = results[pde_key]['best_ppl']
            improvement = (std_ppl - pde_ppl) / std_ppl * 100
            
            print(f"\nData Size: {data_ratio:.1%}")
            print(f"  Standard: {std_ppl:.2f}")
            print(f"  PDE:      {pde_ppl:.2f}")
            print(f"  Improvement: {improvement:+.2f}%")
    
    print(f"\nResults saved to: {output_file}")

    meta = collect_metadata(
        command=" ".join(sys.argv),
        config_path=args.config,
        extra={"args": vars(args)},
    )
    meta_file = os.path.join(args.output_dir, "datasize_ablation_meta.json")
    write_metadata(meta_file, meta)
    print(f"Metadata saved to: {meta_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data size ablation study')

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./local_models/tinyllama",
    )
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for LM token stream (no padding)",
    )
    parser.add_argument('--data_sizes', type=float, nargs='+',
                        default=[0.001, 0.01, 0.05, 0.1],
                        help='Data size ratios to test')
    parser.add_argument(
        "--add_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS between documents in LM token stream",
    )
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # PDE arguments
    parser.add_argument('--pde_type', type=str, default='diffusion')
    parser.add_argument('--pde_steps', type=int, default=4)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    
    config_mapping = {
        "tokenizer_path": ("tokenizer", "path"),
        "max_length": ("dataset", "max_length"),
        "block_size": ("dataset", "block_size"),
        "cache_dir": ("dataset", "cache_dir"),
        "add_eos": ("dataset", "add_eos"),
        "embed_dim": ("model", "embed_dim"),
        "num_heads": ("model", "num_heads"),
        "hidden_dim": ("model", "hidden_dim"),
        "num_layers": ("model", "num_layers"),
        "dropout": ("model", "dropout"),
        "pde_type": ("pde", "type"),
        "pde_steps": ("pde", "steps"),
        "batch_size": ("training", "batch_size"),
        "num_epochs": ("training", "num_epochs"),
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "warmup_ratio": ("training", "warmup_ratio"),
        "num_workers": ("training", "num_workers"),
        "seed": ("training", "seed"),
        "output_dir": ("output", "dir"),
    }

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        cfg = load_yaml_config(pre_args.config)
        set_defaults_from_config(parser, cfg, config_mapping)

    args = parser.parse_args()
    main(args)

