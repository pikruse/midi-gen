import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MidiTransformer(nn.Module):
    """
    Autoregressive Transformer for MIDI generation.
    Uses decoder-only architecture (like GPT).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (autoregressive) attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def _generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate padding mask from input tensor."""
        return (x == self.pad_id).bool()
    
    def forward(
        self, 
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            memory: Optional encoder memory (not used for decoder-only)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create masks
        causal_mask = self._generate_causal_mask(seq_len, device)
        padding_mask = self._generate_padding_mask(x)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # For decoder-only, we use x as both input and memory
        # Create a dummy memory of zeros if not provided
        if memory is None:
            memory = torch.zeros(batch_size, 1, self.d_model, device=device)
        
        # Pass through transformer decoder
        x = self.transformer(
            x, 
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        bos_id: int = 1,
        eos_id: int = 2,
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Autoregressive generation starting from BOS token.
        
        Args:
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            max_len: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling with this probability mass
            device: Device to generate on
            
        Returns:
            generated: Generated token IDs (1, seq_len)
        """
        self.eval()
        
        # Start with BOS token
        generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Forward pass
            logits = self.forward(generated)
            
            # Get logits for next token (last position)
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS generated
            if next_token.item() == eos_id:
                break
        
        return generated
