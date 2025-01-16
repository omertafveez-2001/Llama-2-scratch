import torch
import torch.nn as nn
import sentencepiece as spm
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int= 4096 # Dimension of the model
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the K and V
    vocab_size: int = -1 # set when we load the tokenizer
    multiple_of: int = 256 # parameters of feedforward network
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV-Cache
    max_batch_int: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_cos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Head dim must be divisible by 2" 
    # RoPE is only applied to head_dim divisible by 2.
    # Build the theta parameters that encode the angle of the positional encodings (m) for the tokens.

    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim)) for i = [1,2...d/2]
    # Shape: (head_dim/2)
    thera_numerator = torch.arange(0, head_dim, 2).float()

    # Shape: (head_dim/2)
    theta = 1.0 /(theta ** (thera_numerator / head_dim)).to(device)

    # construct the positions (the 'm' parameter) 
    # shape (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product

    # Shape: (seq_len) outerproduct * (head_dim/2) -> (seq_len, head_dim/2)
    # get all possible combinations of m and theta.
    # theta = [1, 2, .. head_dim/2]
    # m = [0, 1, 2, ... seq_len]
    freqs = torch.outer(m, theta).float()

    # write the numbers in the complex form. (polar form) c = R * exp(i * m * theta) where R = 1 as follows:
    # Shape: (Seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex



class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs)-> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # Rotary Embeddings
        self.freqs_complex = precompute_theta_cos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # compared to transformers, here only the latest tokens are passed and we do not need the previously computed tokens due to KV caching.
        # (B, seq_len) 
        batch_size, seq_len = tokens.shape 
        assert seq_len == 1, "Only one token should be passed at a time"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, freqs_complex)
        h = self.norm(h)
        output = self.output(h)
        return output
