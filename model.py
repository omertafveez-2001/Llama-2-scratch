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

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seqlen, H, head_dim) -> (B, seqlen, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (1, seqlen, 1, head_dim/2) -> adding dimensions to freqs_complex
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)
    # (B, Seq_len, H, Head_dim/2) * (1, Seq_len, 1, Head_dim/2) -> (B, Seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, Seq_len, H, Head_dim/2) -> (B, Seq_len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    #(B, Seq_len, H, Head_dim/2, 2) -> (B, Seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def norm(self, x:torch.Tensor):
        # (B, Seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, dim) -> (B, Seq_len, dim)
        return self.weight * self.norm(x.float()).type_as(x)

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            # (B, Seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )
    
class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # indicates the number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # indicates how many times the Keys and Values should be repeated to match the head of the Queries.
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, n_heads * head_dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, n_heads * head_dim) -> (B, 1, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the vectors.
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry int he cache for this token. 
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve all the caches keys and values so far
        # (B, Seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # repear the heads of the K and V to reach the number of heads of the queries.
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, Seq_len, n_heads, head_dim) -> (B, n_heads, Seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len) --> (B, H_Q, 1, Seq_Len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q * Head_Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) --> (B, 1, Dim)
    

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        # increase the size of the model (increase parameters) since we decrease the number of heads using GQA.
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim /3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        
        # Round the hidden dim to the nearest multiple of the multiple_of parameter to ensure that it is the multiple of the parameter.
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of -1) // args.multiple_of)
        # hidden_size = 7, multiple of = 5
        # (7 + 4) // 5 = 2 
        # 2 * 5 = 10 (multiple of 5)

        # SwiGLU 
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x



        


