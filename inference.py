from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

#https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYXR0czAzbWdjMzJxYjNuanJ6OW14NHN6IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTI4NzczM319fV19&Signature=r54AAsw%7E9bQXztsJOC0bJnSxL3VgMf%7E8gMFeJnmLK3rztprNE64vLoz9CojGXw3m93gv7%7EXWHEMsH%7E3K6yH6mqMOrk-Nc4zgbLQRE67WGPjMuALWmy0RNrpT8h793-nkYO30JC5kZB-TLVIBOK-cD2Br02pdk3AdLYp0NehEYKvDW4xu2x%7ErDR%7Ec-Nns0znwXK38c8aAEYDYU2%7EDFWFWQESJRRI2tCUZOOMDq3YoOY1wgUokcGIYYOF5ii62XyE9qlqp71h5LVFGWwZU4v3V7ZFJHShoYaBxAyL6wlCSmGYgdgDlRTqUTSJVQOWvQ%7EeR35uIFmHTdHHTfRHDeP8%7EaA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=9442375782474052

class LLaMa:

    def __init__(self, model:Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pt')) # get all the checkpoints
            assert len(checkpoints) > 0, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f} seconds")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # removing frequencies from the checkpoint 
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {(time.time() - prev_time):.2f} seconds")
        
        return LLaMa(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len

        # convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos = True, add_eos=False) for prompt in prompts]
        # make sure the batch size is not too large 
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, f"Batch size {batch_size} is too large"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # make sure the prompt length is not larger than maximum seq length
        assert max_prompt_len <= self.model_args.max_seq_len, f"Prompt length {max_prompt_len} is too large"
        total_length = min(self.model_args.max_seq_len, max_prompt_len + max_gen_len)

        # create the list that will contain the generated token, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # populate the initial tokens into prompt tokens
            # example ==> k = 0, t = [1, 2, 3, 4, 5], tokens[k, :len(t)] = [1, 2, 3, 4, 5]
            # k = 1, t = [1, 2, 3, 4], tokens[k, :len(t)] = [1, 2, 3, 4]
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        
        eos_reached = torch.Tensor([False] * batch_size).to(device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, false otherwise. 
        for cur_pos in tqdm(range(1, total_length), desc="Generating tokens"):
            with torch.no_grad():
                # logits shape = (batch_size, seq_len, vocab_size)
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1]/ temperature, dim = -1) # only use the last token.
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the highest probability 
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # next token shape: (batch_size) => (batch_size, 1)
            next_token = next_token.unsqueeze(-1) 
            # Only replace the token if it is a padding token since the rest are replaced with beginning tokens.
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position. 
            # eos is reached if its not a prompt token or the next token reaches the eos token. 
            eos_reached != (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break 
            
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx + 1]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p # shifting the number to the right. 
        probs_sort[mask] = 0
        probs_sort.div_(probs_sort.sum(div=-1, keepdim=True))
        next_tokens = torch.multinomial(probs_sort, num_samples=1) # we need one token. 
        next_token = torch.gather(probs_idx, -1, next_tokens) # get the token index.
        return next_token


    
if __name__ == "main":
    torch.manual_seed(0)

    allow_cuda = True
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"

    prompts = [
        ""
    ]

    model = LLaMa.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path= "tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    # inference the model 
    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_tokens) == len(prompts)
    for i in range(len(out_text)):
        print(f"{out_text}")
        print('-' * 50)
        
    