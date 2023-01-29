

# Taking our script from examine_dataset.ipynb and make it a bit more clear

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1337)

batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
lr = 1e-2
device = torch.device("mps") # for macos 
eval_iters = 200

# open the dataset
with open('/Users/swarajnanda/Karpathy_lectures/nanoGPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all the characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a dictionary to map characters to indices and vice versa and make a function to encode and decode
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Split the samples into training and testing
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y, = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # context manager to disable gradient calculation
def estimate_loss(): # takes a batch average of the loss for both train and val
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range (eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Write the Bigram model now

class Bigram(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_state = nn.Embedding(vocab_size, vocab_size) # our C matrix

    def forward(self, idx, targets = None):
        logits = self.token_embedding_state(idx)
        if targets is None:
            loss = []
        else:
            B,T,C = logits.shape
            # change the dimensioning
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_ = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# Train the model
model = Bigram(vocab_size)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
for i in range(max_iters):
    # print training and testing loss
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
    # sample from the batch
    xb,yb = get_batch('train')
    # forward pass
    logits,loss = model(xb,yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate some text
context = torch.zeros((1,1),dtype=torch.long).to(device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))
