import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
from itertools import cycle

BYTE_VOCAB_SIZE = 256

def load_data(filename="data/enwik9", size=100_000): # 100k bytes
    with open(filename, "rb") as f:
        data = f.read(size)
    return torch.tensor(list(data), dtype=torch.long)

def compute_empirical_entropy(data):
    counts = torch.bincount(data, minlength=BYTE_VOCAB_SIZE)
    probabilities = counts.float() / data.numel()
    probabilities = probabilities[probabilities > 0]  # avoid log(0)
    return -(probabilities * torch.log(probabilities)).sum().item()  # nats

def print_compression_stats(nats, model_size_bytes=0, data_len=0):
    bits_per_byte = nats / math.log(2) # nats to bits
    compression_ratio = 8 / bits_per_byte

    print(f"\nCompression Statistics:\n{'─' * 40}")
    print(f"{'Loss (nats):':<25} {nats:.4f}")
    print(f"{'Bits (per byte):':<25} {bits_per_byte:.2f}")
    print(f"{'Compression ratio:':<25} {compression_ratio:.2f}x")

    if model_size_bytes > 0 and data_len > 0:
        dataset_bytes = data_len
        compressed_bytes = data_len / compression_ratio + model_size_bytes
        effective_ratio = dataset_bytes / compressed_bytes

        print(f"\nEffective Compression Ratio: {effective_ratio:.2f}x")

    print("─" * 40)

def get_model_bytes(model):
    return sum(param.numel() * param.element_size() for param in model.parameters())

class ChunkByteDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=128):
        self.data = data
        self.seq_len = seq_len
        self.num_chunks = (len(self.data) - 1) // self.seq_len

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        return x, y

def eval(model, dataloader, data, device='auto', verbose=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_loss = 0
    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, BYTE_VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if verbose:
        print_compression_stats(avg_loss, get_model_bytes(model), len(data))
    return avg_loss

def train(model, dataloader, data, total_steps, eval_interval=None, device='auto', lr=1e-4, verbose=True):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if verbose:
        print("Initial evaluation:")
    eval_loss = eval(model, dataloader, data, device=device, verbose=verbose)
    
    data_iter = cycle(dataloader)
    iterator = tqdm(range(total_steps), desc="Training") if verbose else range(total_steps)
    
    for step in iterator:
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, BYTE_VOCAB_SIZE), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if eval_interval is not None and (step + 1) % eval_interval == 0:
            if verbose:
                print(f"\nStep {step + 1} complete:")
            eval_loss = eval(model, dataloader, data, device=device, verbose=verbose)
            if verbose and isinstance(iterator, tqdm):
                iterator.set_description(f"Training (loss: {eval_loss:.4f})")
    
    # final evaluation if we haven't just done one
    if eval_interval is None or total_steps % eval_interval != 0:
        if verbose:
            print(f"\nFinal evaluation:")
        eval_loss = eval(model, dataloader, data, device=device, verbose=verbose)
    
    return eval_loss

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=BYTE_VOCAB_SIZE, hidden_size=64, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        _x, _ = self.rnn(x)
        x = (x + _x) / np.sqrt(2) # residual
        x = self.out_proj(x)

        return x