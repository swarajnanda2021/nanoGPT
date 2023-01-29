

# Taking our script from examine_dataset.ipynb and make it a bit more clear

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1337)

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = torch.device("mps") # for macos 
eval_iters = 200
n_embed = 384
n_transformerblocks = 6 # means every head has 384/6 = 64 dimensions
dropout = 0.2 # every forward pass, 20% of the activations are set to zero to prevent overfitting

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
            _,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module): # Transformer head

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # makes it a decoder block
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # randomly prevents some nodes from being updated so that the network does not overfit
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module): # Transformer head

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1) # concatenated over the channel dimension
        out = self.dropout(self.proj(out))
        return out


class LayerNorm1d:

    def __init__(self, dim, eps=1e-5):
        self.eps = eps # set the epsilon to 1e-8 to prevent division by zero
        # set the gain and bias of the batchnorm layer to 1 and 0 respectively on initialization.
        self.gamma = torch.ones(dim) # set the gamma to 1
        self.beta = torch.zeros(dim) # set the beta to 0
            
    def __call__(self,x): # what happens when an input is given to the BatchNorm1d layer.
        # forward pass
        xmean = x.mean(1,keepdim=True) # calculate the mean of the input x
        xvar = x.var(1,keepdim=True) # calculate the variance of the input x
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps) # calculate the normalized input x
        self.out = self.gamma * xhat + self.beta # calculate the output of the batchnorm layer using its gain and bias, i.e., gamma and beta
        # when calling without training on, update the running meand and standard deviation of the batchnorm layer.
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]




class FeedForward(nn.Module):
    # Single layer feed forward network
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed), # the 4* comes from the paper, which says that the hidden layer should be 4 times the size of the input
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # projection layer going back to the residual connection
            nn.Dropout(dropout), # can be added before the reconnection into the residual connection
        )
    def forward(self, x):
        return self.net(x)
        
# Let us now combine the multi headed self attention and the feed forward network into a single block
# so that we can stack blocks together.
class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head,head_size) # this achieves communication
        self.ffwd = FeedForward(n_embed) # this achieves the computation
        self.lnl1 = nn.LayerNorm(n_embed)
        self.lnl2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x+self.sa(self.lnl1(x)) # adding residual connection and layer norm to both self attention and feed forward layers
        x = x+self.ffwd(self.lnl2(x))
        return x

# Bigram_2

class Bigram_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.blocks = nn.Sequential(
        #    Block(n_embed, n_head = 4),
        #    Block(n_embed, n_head = 4),
        #    Block(n_embed, n_head = 4),
        #    nn.LayerNorm(n_embed), # one more layer norm added at the end before the vocabulary projection
        #)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = 4) for _ in range(n_transformerblocks)])
        self.lnl_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
    
    def forward(self, idx, targets = None):
        B,T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T,device = device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.lnl_f(x)
        logits = self.lm_head(x)
        

        #logits = self.token_embedding_state(idx)
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
            # crop idx to the last block size as you cannot exceed block size
            idx_cond = idx[:,-block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# Write the Bigram model now

class Bigram(nn.Module):

    def __init__(self): # vocab_size removed because globally declared
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # n_embed is the embedding size
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # block_size is the sequence length
        self.sa_head = MultiHeadAttention(4,n_embed//4) # i.e., 4 heads of 8-dimensional(32//4=8) self-attention
        self.lm_head = nn.Linear(n_embed,vocab_size) # vocab_size is the output size, lm-> language model
        self.ffwd = FeedForward(n_embed)
        

    def forward(self, idx, targets = None):
        B,T = idx.shape

        token_embed = self.token_embedding_table(idx) # (B,T,n_embed)
        position_embedding = self.position_embedding_table(torch.arange(T,device = device)) # basically integers from 0 to T-1
        # They get embedded into a table with size (T,C)
        x = token_embed + position_embedding # (B,T,n_embed) contains token identity and position
        x = self.sa_head(x) # (B,T,n_embed)
        x = self.ffwd(x) # (B,T,n_embed)
        logits = self.lm_head(x) # (B,T,vocab_size) 

        #logits = self.token_embedding_state(idx)
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
            # crop idx to the last block size as you cannot exceed block size
            idx_cond = idx[:,-block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx



# Train the model
#model = Bigram()
model = Bigram_2()
print(sum(p.nelement() for p in model.parameters() if p.requires_grad))
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
loss_fin = estimate_loss()
print(f"step {i}: train loss = {loss_fin['train']:.4f}, val loss = {loss_fin['val']:.4f}")
# Generate some text
context = torch.zeros((1,1),dtype=torch.long).to(device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))
