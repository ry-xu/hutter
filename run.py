import torch

from utils import (
    load_data,
    compute_empirical_entropy,
    print_compression_stats,
    ChunkByteDataset,
    LSTMModel,
    train
)

## load data
data = load_data("data/enwik9", size=1000000)

print("Computing empirical entropy...")
empirical_entropy_nats = compute_empirical_entropy(data)
print_compression_stats(empirical_entropy_nats)

## hparams
seq_len = 16

lr = 1e-3
batch_size = 32
total_steps = 5000
eval_interval = total_steps // 4
verbose = True

lstm_hidden_size = 16
lstm_num_layers = 2

## train
dataset = ChunkByteDataset(data, seq_len=seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = LSTMModel(hidden_size=lstm_hidden_size, num_layers=lstm_num_layers)

train(
    model,
    dataloader,
    data,
    total_steps=total_steps,
    eval_interval=eval_interval,
    lr=lr,
    verbose=verbose
)