import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from miditok import REMI, TokenizerConfig
from pathlib import Path
from typing import List, Tuple

### TOKENIZER SETUP ###
def get_tokenizer() -> REMI:
    """Create and return the REMI tokenizer with consistent config."""
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=False,
    )
    return REMI(config)

### MIDI DATASET CLASS ###
class MidiDataset(Dataset):
    def __init__(self, 
                 data_path: str = "../data/input/",
                 max_seq_len: int = 1024):
        """
        Dataset for tokenized MIDI files.
        
        Args:
            data_path: Path to directory containing .mid files
            max_seq_len: Maximum sequence length (truncate if longer)
        """
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.file_list = list(self.data_path.glob("*.mid"))
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        
        # Get special token IDs
        self.bos_id = self.tokenizer.vocab["BOS_None"]  # 1
        self.eos_id = self.tokenizer.vocab["EOS_None"]  # 2
        self.pad_id = self.tokenizer.vocab["PAD_None"]  # 0
        
        # Pre-tokenize all files for faster training
        self.tokenized_data = self._tokenize_all()
    
    def _tokenize_all(self) -> List[List[int]]:
        """Tokenize all MIDI files and add BOS/EOS tokens."""
        tokenized = []
        for file_path in self.file_list:
            try:
                # Encode MIDI file
                tokens = self.tokenizer.encode(file_path)
                token_ids = tokens[0].ids  # Get IDs from first track
                
                # Add BOS and EOS tokens
                token_ids = [self.bos_id] + token_ids + [self.eos_id]
                
                # Truncate if too long
                if len(token_ids) > self.max_seq_len:
                    token_ids = token_ids[:self.max_seq_len - 1] + [self.eos_id]
                
                tokenized.append(token_ids)
            except Exception as e:
                print(f"Error tokenizing {file_path}: {e}")
                continue
        
        return tokenized

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input and target tensors for autoregressive training.
        
        Input:  [BOS, tok1, tok2, ..., tokN]
        Target: [tok1, tok2, ..., tokN, EOS]
        """
        token_ids = self.tokenized_data[idx]
        
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.vocab)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    Pads sequences to the longest in the batch.
    """
    inputs, targets = zip(*batch)
    
    # Pad sequences (pad_id = 0)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded