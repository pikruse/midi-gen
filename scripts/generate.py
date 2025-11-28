import torch
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import get_tokenizer
from model.model import MidiTransformer


def generate_midi(
    checkpoint_path: str,
    output_path: str = "data/output/generated.mid",
    num_samples: int = 1,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "auto",
) -> List[Tuple[List[int], object]]:
    """
    Generate MIDI file(s) using a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save generated MIDI file(s). For multiple samples,
                     files are named with suffix _0, _1, etc.
        num_samples: Number of MIDI files to generate in parallel
        max_len: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling
        device: Device to run on
        
    Returns:
        List of (token_ids, midi_object) tuples
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer.vocab)
    bos_id = tokenizer.vocab["BOS_None"]
    eos_id = tokenizer.vocab["EOS_None"]
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MidiTransformer(
        vocab_size=vocab_size,
        d_model=256,  # Should match training config
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.0,  # No dropout during inference
        pad_id=0,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Generating {num_samples} sample(s) (max_len={max_len}, temp={temperature}, top_k={top_k}, top_p={top_p})...")
    
    # Generate tokens (batch)
    generated_batch = model.generate(
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        batch_size=num_samples,
    )
    
    results = []
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        # Get this sample's tokens
        generated_ids = generated_batch[i].tolist()
        
        # Remove BOS and EOS tokens for decoding
        if generated_ids[0] == bos_id:
            generated_ids = generated_ids[1:]
        # Remove EOS and everything after
        if eos_id in generated_ids:
            eos_idx = generated_ids.index(eos_id)
            generated_ids = generated_ids[:eos_idx]
        
        print(f"Sample {i+1}: Generated {len(generated_ids)} tokens")
        
        # Filter out any invalid token IDs (outside vocab range)
        valid_ids = [id for id in generated_ids if 0 <= id < vocab_size]
        if len(valid_ids) != len(generated_ids):
            print(f"  Warning: Filtered out {len(generated_ids) - len(valid_ids)} invalid tokens")
            generated_ids = valid_ids
        
        # Decode tokens to MIDI (wrap in list for batch dimension)
        midi = tokenizer.decode([generated_ids])
        
        # Determine output filename
        if num_samples == 1:
            sample_path = output_path
        else:
            sample_path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
        
        # Save MIDI file
        midi.dump_midi(sample_path)
        print(f"  Saved to: {sample_path}")
        
        results.append((generated_ids, midi))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MIDI with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="data/output/generated.mid")
    parser.add_argument("--num_samples", "-n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    generate_midi(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )