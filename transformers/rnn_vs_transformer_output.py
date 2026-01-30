import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

# set random seeds for reproducability
torch.manual_seed(42)
np.random.seed(42)

# ----------------------- data preperation --------------------------

class TextDataset:
    def __init__(self, text, seq_length = 5):
        self.text = text.lower()
        self.seq_length = seq_length
        self.vocab = sorted(set(self.text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.text) - self.seq_length):
            seq = self.text[i:i + self.seq_length]
            target = self.text[i + self.seq_length]
            
            seq_encoded = [self.char_to_idx[c] for c in seq]
            target_encoded = self.char_to_idx[target]
            
            self.sequences.append(seq_encoded)
            self.targets.append(target_encoded)
    
    # pytorch dataset methods
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), \
               torch.tensor(self.targets[idx], dtype=torch.long)

# -------------------------- Models ---------------------------

# defining a sequential model
class TextRNN(nn.Module):
    def __init__(self,):
        super().__init__()