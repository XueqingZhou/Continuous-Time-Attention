"""
Run PDE type ablation study on WikiText-103 (Table 5)

This script compares different PDE variants: diffusion, wave, 
reaction-diffusion, and advection-diffusion.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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


def plot_pdetype_results(
    results: dict,
    pde_types: list,
    output_dir: str,
) -> None:
    """Plot PDE type comparison results."""
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(pde_types) + 1))
    
    # 1. Training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Standard first
    if 'Standard' in results and results['Standard']['train_loss']:
        epochs = range(1, len(results['Standard']['train_loss']) + 1)
        ax1.plot(epochs, results['Standard']['train_loss'], 
                marker='o', linewidth=2.5, color=colors[0],
                label='Standard (No PDE)', linestyle='--', alpha=0.8)
    
    # Then PDE types
    for i, pde_type in enumerate(pde_types, 1):
        key = f"PDE-{pde_type}"
        if key in results and results[key]['train_loss']:
            epochs = range(1, len(results[key]['train_loss']) + 1)
            ax1.plot(epochs, results[key]['train_loss'], 
                    marker='s', linewidth=2, color=colors[i],
                    label=pde_type.replace('-', ' ').title())
    
    ax1.set_title('Training Loss: PDE Type Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation perplexity curves
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'Standard' in results and results['Standard']['val_ppl']:
        epochs = range(1, len(results['Standard']['val_ppl']) + 1)
        ax2.plot(epochs, results['Standard']['val_ppl'], 
                marker='o', linewidth=2.5, color=colors[0],
                label='Standard (No PDE)', linestyle='--', alpha=0.8)
    
    for i, pde_type in enumerate(pde_types, 1):
        key = f"PDE-{pde_type}"
        if key in results and results[key]['val_ppl']:
            epochs = range(1, len(results[key]['val_ppl']) + 1)
            ax2.plot(epochs, results[key]['val_ppl'], 
                    marker='s', linewidth=2, color=colors[i],
                    label=pde_type.replace('-', ' ').title())
    
    ax2.set_title('Validation Perplexity: PDE Type Comparison', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final perplexity bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    
    model_names = ['Standard'] + [f"PDE-{t}" for t in pde_types]
    final_ppls = []
    
    for name in model_names:
        if name in results and results[name]['val_ppl']:
            final_ppls.append(results[name]['val_ppl'][-1])
        else:
            final_ppls.append(np.nan)
    
    bars = ax3.bar(range(len(model_names)), final_ppls, 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, ppl in zip(bars, final_ppls):
        if not np.isnan(ppl):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{ppl:.2f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    ax3.set_title('Final Perplexity Comparison', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Perplexity')
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([n.replace('PDE-', '').replace('-', ' ').title() 
                         for n in model_names], rotation=30, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Relative improvement table (Table 5)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create table data matching Table 5 format
    table_data = [['Model', 'PDE Params', 'PPL ↓', 'Δ(%)']]
    
    if 'Standard' in results and results['Standard']['val_ppl']:
        baseline_ppl = results['Standard']['val_ppl'][-1]
        table_data.append(['Standard Transformer', '—', f'{baseline_ppl:.2f}', '—'])
        
        # PDE type configurations from paper
        pde_configs = {
            'diffusion': 'α=0.10',
            'wave': 'α=0.15',
            'reaction-diffusion': 'α=0.10, β=0.02',
            'advection-diffusion': 'α=0.10, β=0.03'
        }
        
        for pde_type in pde_types:
            key = f"PDE-{pde_type}"
            if key in results and results[key]['val_ppl']:
                ppl = results[key]['val_ppl'][-1]
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
                params = pde_configs.get(pde_type, 'α=0.10')
                
                table_data.append([
                    pde_type.replace('-', ' ').title(),
                    params,
                    f'{ppl:.2f}',
                    f'{delta:.2f}'
                ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best PDE
    if len(table_data) > 1:
        ppls = [float(row[2]) for row in table_data[2:]]  # Skip header and standard
        if ppls:
            best_idx = np.argmin(ppls) + 2  # +2 for header and standard
            for i in range(4):
                table[(best_idx, i)].set_facecolor('#FFD700')
    
    ax4.set_title('Summary Table (Table 5 from Paper)', fontweight='bold', fontsize=14, pad=20)
    
    plt.suptitle('PDE Type Ablation Study on WikiText-103', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pdetype_ablation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pdetype_ablation.pdf'), bbox_inches='tight')
    plt.close()


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Prepare data
    print("\nPreparing WikiText-103 dataset...")
    train_dataset, val_dataset, vocab_size = prepare_lm_data(
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        block_size=args.block_size,
        train_sample_size=args.train_sample_size,
        val_sample_size=args.val_sample_size,
        cache_dir=args.cache_dir,
        add_eos=args.add_eos,
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
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
    
    # Model config
    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'max_length': args.max_length,
        'dropout': args.dropout,
    }
    
    results = {}
    
    # Train standard baseline first
    print(f"\n{'='*60}")
    print("Training Standard Transformer (Baseline)")
    print(f"{'='*60}")
    
    model = StandardTransformerLM(**model_config).to(device)
    
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
    
    trainer = LanguageModelingTrainer(model, device, optimizer, scheduler)
    history = trainer.train(train_loader, val_loader, args.num_epochs)
    
    results['Standard'] = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_ppl': history['val_ppl'],
        'best_ppl': min(history['val_ppl'])
    }
    
    print(f"Standard - Best PPL: {results['Standard']['best_ppl']:.2f}")
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    # Train each PDE type
    for pde_type in args.pde_types:
        print(f"\n{'='*60}")
        print(f"Training PDE-Transformer: {pde_type}")
        print(f"{'='*60}")
        
        # PDE-specific hyperparameters (from paper)
        pde_kwargs = {}
        if pde_type == 'diffusion':
            pde_kwargs = {'alpha_init': 0.10}
        elif pde_type == 'wave':
            pde_kwargs = {'alpha_init': 0.15}
        elif pde_type == 'reaction-diffusion':
            pde_kwargs = {'alpha_init': 0.10, 'beta_init': 0.02}
        elif pde_type == 'advection-diffusion':
            pde_kwargs = {'alpha_init': 0.10, 'beta_init': 0.03}
        
        model = PDETransformerLM(
            **model_config,
            pde_type=pde_type,
            pde_steps=args.pde_steps,
            **pde_kwargs
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        
        trainer = LanguageModelingTrainer(model, device, optimizer, scheduler)
        
        try:
            history = trainer.train(train_loader, val_loader, args.num_epochs)
            
            results[f"PDE-{pde_type}"] = {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_ppl': history['val_ppl'],
                'best_ppl': min(history['val_ppl'])
            }
            
            print(f"PDE-{pde_type} - Best PPL: {results[f'PDE-{pde_type}']['best_ppl']:.2f}")
            
        except Exception as e:
            print(f"Training failed for {pde_type}: {e}")
            results[f"PDE-{pde_type}"] = {
                'train_loss': [],
                'val_loss': [],
                'val_ppl': [float('nan')],
                'best_ppl': float('nan')
            }
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_pdetype_results(results, args.pde_types, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'pdetype_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary (Table 5 format)
    print(f"\n{'='*60}")
    print("PDE TYPE ABLATION SUMMARY (Table 5)")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'PDE Params':<20} {'PPL ↓':<10} {'Δ(%)'}")
    print(f"{'-'*75}")
    
    if 'Standard' in results:
        baseline_ppl = results['Standard']['best_ppl']
        print(f"{'Standard Transformer':<30} {'—':<20} {baseline_ppl:<10.2f} {'—'}")
        
        for pde_type in args.pde_types:
            key = f"PDE-{pde_type}"
            if key in results:
                ppl = results[key]['best_ppl']
                if not np.isnan(ppl):
                    delta = (ppl - baseline_ppl) / baseline_ppl * 100
                    
                    # Parameter strings from paper
                    param_str = {
                        'diffusion': 'α=0.10',
                        'wave': 'α=0.15',
                        'reaction-diffusion': 'α=0.10, β=0.02',
                        'advection-diffusion': 'α=0.10, β=0.03'
                    }.get(pde_type, 'α=0.10')
                    
                    name = pde_type.replace('-', ' ').title()
                    print(f"{name:<30} {param_str:<20} {ppl:<10.2f} {delta:.2f}")
    
    print(f"\nResults saved to: {output_file}")

    meta = collect_metadata(
        command=" ".join(sys.argv),
        config_path=args.config,
        extra={"args": vars(args)},
    )
    meta_file = os.path.join(args.output_dir, "pdetype_ablation_meta.json")
    write_metadata(meta_file, meta)
    print(f"Metadata saved to: {meta_file}")
    print(f"\nNote: Δ(%) is relative to baseline: (PPL_MODEL - PPL_STD) / PPL_STD × 100")
    print(f"      All PDE types show ~99.98% improvement in the paper (Table 5)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDE type ablation study')

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    # Dataset
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
    parser.add_argument('--train_sample_size', type=int, default=None,
                        help='Use subset for faster experiments')
    parser.add_argument("--val_sample_size", type=int, default=None)
    parser.add_argument(
        "--add_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS between documents in LM token stream",
    )
    parser.add_argument('--pde_types', type=str, nargs='+',
                        default=['diffusion', 'wave', 'reaction-diffusion', 'advection-diffusion'],
                        help='PDE types to compare')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # PDE
    parser.add_argument('--pde_steps', type=int, default=4,
                        help='Number of PDE steps (use 4 based on Table 4)')
    
    # Training
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
        "train_sample_size": ("dataset", "train_sample_size"),
        "val_sample_size": ("dataset", "val_sample_size"),
        "cache_dir": ("dataset", "cache_dir"),
        "add_eos": ("dataset", "add_eos"),
        "embed_dim": ("model", "embed_dim"),
        "num_heads": ("model", "num_heads"),
        "hidden_dim": ("model", "hidden_dim"),
        "num_layers": ("model", "num_layers"),
        "dropout": ("model", "dropout"),
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

