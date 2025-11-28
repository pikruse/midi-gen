import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import MidiDataset, collate_fn, get_tokenizer
from model.model import MidiTransformer


def train(
    data_path: str = "../data/input/",
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 1024,
    max_seq_len: int = 1024,
    dropout: float = 0.1,
    device: str = "auto",
    wandb_project: str = "midi-gen",
    wandb_entity: str = "pikruse-personal",
    wandb_run_name: str = None,
    use_wandb: bool = True,
    early_stop_patience: int = 10,
):
    """
    Train the MIDI transformer model.
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Initialize wandb
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
        "dropout": dropout,
        "device": str(device),
    }
    
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config,
        )
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = MidiDataset(data_path=data_path, max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"Dataset size: {len(dataset)} MIDI files")
    print(f"Vocab size: {dataset.vocab_size}")
    
    if use_wandb:
        wandb.config.update({"dataset_size": len(dataset), "vocab_size": dataset.vocab_size})
    
    # Create model
    model = MidiTransformer(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_id=dataset.pad_id,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    if use_wandb:
        wandb.config.update({"n_params": n_params})
        wandb.watch(model, log="all", log_freq=100)  # Log gradients and parameters
    
    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    global_step = 0
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log batch metrics to wandb
            if use_wandb:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                })
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log epoch metrics to wandb
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch,
                "train/best_loss": best_loss if best_loss != float('inf') else avg_loss,
            })
        
        # Save checkpoint only if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            
            checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "model_best.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"New best loss: {avg_loss:.4f} - Saved checkpoint: {checkpoint_path}")
            
            # Log model checkpoint to wandb
            if use_wandb:
                wandb.save(str(checkpoint_path))
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s) (best: {best_loss:.4f})")
        
        # Early stopping check
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stop_patience} epochs)")
            break
    
    # Save final model
    final_path = Path(__file__).parent.parent / "checkpoints" / "model_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    print(f"Training complete! Saved final model: {final_path}")
    
    if use_wandb:
        wandb.save(str(final_path))
        wandb.finish()
    
    return model


if __name__ == "__main__":
    import argparse
    
    # Get the project root directory (parent of scripts/)
    project_root = Path(__file__).parent.parent
    default_data_path = str(project_root / "data" / "input")
    
    parser = argparse.ArgumentParser(description="Train MIDI Transformer")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="midi-gen")
    parser.add_argument("--wandb_entity", type=str, default="pikruse-personal", help="Wandb entity (your username for personal projects)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stop after N epochs without improvement (0 to disable)")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        early_stop_patience=args.early_stop_patience,
    )
