import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
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

# defining a sequential modelclass TextRNN(nn.Module):
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, return_hidden=False):
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        logits = self.fc(lstm_out[:, -1, :])  # Use last timestep output
        
        if return_hidden:
            return logits, lstm_out
        return logits

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, embedding_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=128,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x, return_attention=False):
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded + self.positional_encoding[:, :embedded.size(1), :]
        transformer_out = self.transformer(embedded)  # (batch, seq_len, embedding_dim)
        logits = self.fc(transformer_out[:, -1, :])  # Use last position output
        
        if return_attention:
                return logits, transformer_out
        return logits

# --------------------------- Training --------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for sequences, targets in dataloader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            logits = model(sequences)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total


# -------------------------- Generaliztion and Visualization -----------------------

def generate_text(model, dataset, initial_text, length=20, device='cpu'):
    model.eval()
    generated = initial_text
    
    with torch.no_grad():
        for _ in range(length):
            # Get last seq_length characters
            input_seq = generated[-dataset.seq_length:]
            
            # Encode
            encoded = [dataset.char_to_idx[c] for c in input_seq]
            x = torch.tensor([encoded], dtype=torch.long).to(device)
            
            # Predict
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            next_char_idx = torch.argmax(probs, dim=1).item()
            next_char = dataset.idx_to_char[next_char_idx]
            
            generated += next_char
    
    return generated

def get_model_attention(model, dataset, input_text, device='cpu'):
    """Get what the model 'sees' at each timestep"""
    model.eval()
    
    encoded = [dataset.char_to_idx[c] for c in input_text]
    x = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        if isinstance(model, TextRNN):
            _, hidden_states = model(x, return_hidden=True)
            # hidden_states: (batch, seq_len, hidden_dim)
            return hidden_states[0].cpu().numpy()
        else:  # Transformer
            _, attention_outputs = model(x, return_attention=True)
            # attention_outputs: (batch, seq_len, embedding_dim)
            return attention_outputs[0].cpu().numpy()

# ------------------- EXECUTION --------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    text = """the quick brown fox jumps over the lazy dog. the dog was sleeping under the tree. 
    the fox ran quickly through the forest. the brown dog barked at the fox. """
    
    dataset = TextDataset(text, seq_length=5)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True
    )
    
    # Initialize models
    rnn_model = TextRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=32,
        hidden_dim=64
    ).to(device)
    
    transformer_model = TextTransformer(
        vocab_size=dataset.vocab_size,
        embedding_dim=32,
        n_heads=4,
        n_layers=2
    ).to(device)
    
    # Training setup
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50
    rnn_losses = []
    transformer_losses = []
    
    print("=" * 60)
    print("TRAINING RNN vs TRANSFORMER ON TEXT GENERATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)} sequences")
    print()
    
    # Train both models
    for epoch in range(epochs):
        rnn_loss = train_epoch(rnn_model, dataloader, rnn_optimizer, criterion, device)
        transformer_loss = train_epoch(transformer_model, dataloader, transformer_optimizer, criterion, device)
        
        rnn_losses.append(rnn_loss)
        transformer_losses.append(transformer_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  RNN Loss: {rnn_loss:.4f}")
            print(f"  Transformer Loss: {transformer_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("GENERATION RESULTS")
    print("=" * 60)
    
    # Generate text
    initial_text = "the quick brown fox"
    rnn_generated = generate_text(rnn_model, dataset, initial_text, length=20, device=device)
    transformer_generated = generate_text(transformer_model, dataset, initial_text, length=20, device=device)
    
    print(f"\nInitial text: '{initial_text}'")
    print(f"\nRNN output: '{rnn_generated}'")
    print(f"Transformer output: '{transformer_generated}'")
    
    # ----------------------- VISUALIZATION -------------------------
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rnn_losses, label='RNN (LSTM)', linewidth=2, color='#FF6B6B')
    ax1.plot(transformer_losses, label='Transformer', linewidth=2, color='#4ECDC4')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Comparison: RNN vs Transformer', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. RNN Hidden States (what RNN "sees")
    ax2 = fig.add_subplot(gs[1, 0])
    rnn_states = get_model_attention(rnn_model, dataset, initial_text, device)
    # Take mean across hidden dimension for visualization
    rnn_heatmap = rnn_states.mean(axis=1, keepdims=True)
    sns.heatmap(
        rnn_heatmap.T,
        cmap='RdYlGn',
        ax=ax2,
        cbar_kws={'label': 'Activation Magnitude'},
        xticklabels=[c for c in initial_text],
        yticklabels=['Hidden State']
    )
    ax2.set_title('RNN: What It "Sees" at Each Timestep\n(Hidden State Activations)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Character Position', fontsize=11)
    
    # 3. Transformer Embeddings (what Transformer "sees")
    ax3 = fig.add_subplot(gs[1, 1])
    transformer_states = get_model_attention(transformer_model, dataset, initial_text, device)
    # Take mean across embedding dimension for visualization
    transformer_heatmap = transformer_states.mean(axis=1, keepdims=True)
    sns.heatmap(
        transformer_heatmap.T,
        cmap='RdYlGn',
        ax=ax3,
        cbar_kws={'label': 'Activation Magnitude'},
        xticklabels=[c for c in initial_text],
        yticklabels=['Embedding']
    )
    ax3.set_title('Transformer: What It "Sees" at Each Timestep\n(Embedding Activations)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Character Position', fontsize=11)
    
    # 4. Generation comparison
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    comparison_text = f"""
    INPUT TEXT:  '{initial_text}'
    
    RNN (LSTM) OUTPUT:
    {rnn_generated}
    
    TRANSFORMER OUTPUT:
    {transformer_generated}
    
    KEY OBSERVATIONS:
    • RNN processes sequentially (left to right), building memory as it goes
    • Transformer sees all tokens simultaneously, enabling parallel processing
    • RNN's hidden state can fade with distance (vanishing gradient problem)
    • Transformer has direct connections between all tokens (no fading)
    • Notice how Transformer typically converges faster (fewer epochs to low loss)
    """
    
    ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('RNN vs Transformer: Complete Comparison on Text Generation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('rnn_vs_transformer_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved as 'rnn_vs_transformer_comparison.png'")
    plt.show()
    
    # Print model statistics
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    
    print(f"\nRNN Parameters: {rnn_params:,}")
    print(f"Transformer Parameters: {transformer_params:,}")
    print(f"Parameter Ratio (T/R): {transformer_params/rnn_params:.2f}x")
    
    print(f"\nFinal RNN Loss: {rnn_losses[-1]:.4f}")
    print(f"Final Transformer Loss: {transformer_losses[-1]:.4f}")
    print(f"RNN Loss Reduction: {(rnn_losses[0] - rnn_losses[-1]) / rnn_losses[0] * 100:.1f}%")
    print(f"Transformer Loss Reduction: {(transformer_losses[0] - transformer_losses[-1]) / transformer_losses[0] * 100:.1f}%")