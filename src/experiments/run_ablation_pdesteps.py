"""
Run PDE steps ablation study on WikiText-103 (Table 4)

This script studies the optimal number of PDE refinement steps.
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


def plot_pdesteps_results(results, pde_steps_list, output_dir):
    """Plot PDE steps ablation results"""
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(pde_steps_list)))
    
    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    for i, steps in enumerate(pde_steps_list):
        key = f"PDE-{steps}steps" if steps > 0 else "Standard"
        if key in results and results[key]['train_loss']:
            epochs = range(1, len(results[key]['train_loss']) + 1)
            ax1.plot(epochs, results[key]['train_loss'], 
                    marker='o', linewidth=2, color=colors[i],
                    label=f'{steps} steps' if steps > 0 else 'Standard (0 steps)')
    
    ax1.set_title('Training Loss vs PDE Steps', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    for i, steps in enumerate(pde_steps_list):
        key = f"PDE-{steps}steps" if steps > 0 else "Standard"
        if key in results and results[key]['val_ppl']:
            epochs = range(1, len(results[key]['val_ppl']) + 1)
            ax2.plot(epochs, results[key]['val_ppl'], 
                    marker='s', linewidth=2, color=colors[i],
                    label=f'{steps} steps' if steps > 0 else 'Standard (0 steps)')
    
    ax2.set_title('Validation Perplexity vs PDE Steps', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final perplexity vs steps
    ax3 = fig.add_subplot(gs[1, 0])
    
    steps_list = []
    final_ppls = []
    stable_flags = []
    
    for steps in pde_steps_list:
        key = f"PDE-{steps}steps" if steps > 0 else "Standard"
        if key in results and results[key]['val_ppl']:
            steps_list.append(steps)
            ppl = results[key]['val_ppl'][-1]
            final_ppls.append(ppl)
            
            # Check stability (no NaN/Inf in training)
            stable = not (np.isnan(ppl) or np.isinf(ppl))
            stable_flags.append(stable)
    
    # Plot line
    valid_steps = [s for s, stable in zip(steps_list, stable_flags) if stable]
    valid_ppls = [p for p, stable in zip(final_ppls, stable_flags) if stable]
    
    if valid_steps:
        ax3.plot(valid_steps, valid_ppls, 'o-', linewidth=2, markersize=10, color='#2a9d8f')
        
        # Mark best
        best_idx = np.argmin(valid_ppls)
        best_steps = valid_steps[best_idx]
        best_ppl = valid_ppls[best_idx]
        
        ax3.axvline(x=best_steps, color='red', linestyle='--', alpha=0.7)
        ax3.annotate(f'Best: {best_steps} steps\nPPL: {best_ppl:.2f}',
                    xy=(best_steps, best_ppl),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=12, fontweight='bold')
    
    # Mark unstable points
    unstable_steps = [s for s, stable in zip(steps_list, stable_flags) if not stable]
    if unstable_steps:
        ax3.scatter(unstable_steps, [max(valid_ppls)*1.1]*len(unstable_steps),
                   marker='x', s=200, c='red', label='Unstable')
    
    ax3.set_title('Final Perplexity vs Number of PDE Steps', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Number of PDE Steps')
    ax3.set_ylabel('Perplexity')
    ax3.set_xticks(pde_steps_list)
    ax3.grid(True, alpha=0.3)
    if unstable_steps:
        ax3.legend()
    
    # 4. Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create table data
    table_data = [['Steps', 'PPL', 'Δ%', 'Stable', 'Rank']]
    
    if valid_steps:
        baseline_ppl = valid_ppls[0]  # 0 steps (standard)
        
        sorted_indices = np.argsort(valid_ppls)
        ranks = {valid_steps[i]: rank+1 for rank, i in enumerate(sorted_indices)}
        
        for steps, ppl, stable in zip(steps_list, final_ppls, stable_flags):
            if stable:
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
                rank = ranks[steps]
                status = 'YES'
            else:
                delta = float('nan')
                rank = '-'
                status = 'NO'
            
            table_data.append([
                str(steps),
                f'{ppl:.2f}' if not np.isnan(ppl) else 'NaN',
                f'{delta:.2f}%' if not np.isnan(delta) else '-',
                status,
                str(rank)
            ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row
    if valid_steps:
        best_row = valid_steps.index(best_steps) + 1
        for i in range(5):
            table[(best_row, i)].set_facecolor('#FFD700')
    
    ax4.set_title('Summary Table (Table 4 from Paper)', fontweight='bold', fontsize=14, pad=20)
    
    plt.suptitle('PDE Steps Ablation Study on WikiText-103', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pdesteps_ablation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pdesteps_ablation.pdf'), bbox_inches='tight')
    plt.close()


def main(args):
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
        train_sample_size=args.train_sample_size,
        val_sample_size=args.val_sample_size,
        cache_dir=args.cache_dir
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
    
    # Run for each PDE steps configuration
    for pde_steps in args.pde_steps_list:
        print(f"\n{'='*60}")
        print(f"Training with {pde_steps} PDE steps")
        print(f"{'='*60}")
        
        # Create model
        if pde_steps == 0:
            model = StandardTransformerLM(**model_config).to(device)
            key = "Standard"
        else:
            model = PDETransformerLM(
                **model_config,
                pde_type=args.pde_type,
                pde_steps=pde_steps
            ).to(device)
            key = f"PDE-{pde_steps}steps"
        
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
        try:
            history = trainer.train(train_loader, val_loader, args.num_epochs)
            
            results[key] = {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_ppl': history['val_ppl'],
                'best_ppl': min(history['val_ppl']),
                'stable': True
            }
            
            print(f"{key} - Best PPL: {results[key]['best_ppl']:.2f}")
            
        except Exception as e:
            print(f"Training failed for {key}: {e}")
            results[key] = {
                'train_loss': [],
                'val_loss': [],
                'val_ppl': [float('nan')],
                'best_ppl': float('nan'),
                'stable': False
            }
        
        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_pdesteps_results(results, args.pde_steps_list, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'pdesteps_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PDE STEPS ABLATION SUMMARY (Table 4)")
    print(f"{'='*60}")
    print(f"{'Steps':<10} {'PPL':<10} {'Δ%':<10} {'Stable':<10} {'Rank'}")
    print(f"{'-'*60}")
    
    # Sort by PPL
    valid_results = [(k, v) for k, v in results.items() if v['stable']]
    sorted_results = sorted(valid_results, key=lambda x: x[1]['best_ppl'])
    
    if sorted_results:
        baseline_ppl = results.get('Standard', {}).get('best_ppl', sorted_results[0][1]['best_ppl'])
        
        for rank, (key, data) in enumerate(sorted_results, 1):
            steps = key.replace('PDE-', '').replace('steps', '') if 'PDE' in key else '0'
            ppl = data['best_ppl']
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            stable = 'YES' if data['stable'] else 'NO'
            
            print(f"{steps:<10} {ppl:<10.2f} {delta:+.2f}%    {stable:<10} {rank}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nNote: Table 4 in the paper shows that 4 PDE steps is optimal")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDE steps ablation study')
    
    # Dataset
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--train_sample_size', type=int, default=None,
                        help='Use subset of training data for faster experiments')
    parser.add_argument('--val_sample_size', type=int, default=1024)
    parser.add_argument('--pde_steps_list', type=int, nargs='+',
                        default=[0, 1, 2, 4, 8],
                        help='List of PDE steps to test (0 = standard)')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # PDE
    parser.add_argument('--pde_type', type=str, default='diffusion')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    
    args = parser.parse_args()
    main(args)

