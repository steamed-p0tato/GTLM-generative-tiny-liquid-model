#!/usr/bin/env python3
import os, math, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress Hugging Face warnings
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0" 
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

import tiktoken
from transformers import GPT2LMHeadModel
from datasets import load_dataset

# ========================== LIQ-LM CONFIG & ARCHITECTURE ==========================

@dataclass
class Config:
    hidden_size: int = 768     
    num_layers: int = 12       
    num_heads: int = 12        
    max_seq_length: int = 1024 
    vocab_size: int = 50257    
    dropout: float = 0.0       # Must be 0.0 for evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "liq_150M_instruct_final.pt"

def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor):
    B, L, D = a.shape
    next_pow2 = 2 ** math.ceil(math.log2(max(1, L)))
    if next_pow2 != L:
        pad_len = next_pow2 - L
        a = F.pad(a, (0, 0, 0, pad_len), value=1.0)
        b = F.pad(b, (0, 0, 0, pad_len), value=0.0)
    num_steps = int(math.log2(next_pow2))
    for i in range(num_steps):
        step = 2**i
        a_curr, b_curr = a[:, step:, :], b[:, step:, :]
        a_prev, b_prev = a[:, :-step, :], b[:, :-step, :]
        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        a = torch.cat([a[:, :step, :], new_a], dim=1)
        b = torch.cat([b[:, :step, :], new_b], dim=1)
    return b[:, :L, :]

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=1024, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(L)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        return self.resid_drop(self.out_proj(out.transpose(1, 2).reshape(B, L, D)))

class ParallelLiquidCell(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.tau_proj = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(dim, dim)
        self.log_dt = nn.Parameter(torch.zeros(dim)) 
    def forward(self, x, h0):
        tau = F.softplus(self.tau_proj(x)) + 1e-3
        dt = F.softplus(self.log_dt)
        alpha = torch.clamp(dt / tau, 0.0, 1.0) 
        candidate = torch.tanh(self.input_proj(x))
        a, b = 1.0 - alpha, alpha * candidate
        b_init = a[:, :1, :] * h0.unsqueeze(1) + b[:, :1, :]
        b = torch.cat([b_init, b[:, 1:, :]], dim=1)
        return parallel_associative_scan(a, b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=1024, dropout=0.0):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, max_seq_len, dropout)
        self.ln_liq = RMSNorm(dim)
        self.liquid = ParallelLiquidCell(dim)
        self.liq_gate = nn.Linear(dim, dim, bias=False) 
        self.ln_ffn = RMSNorm(dim)
        # FIXED: Dummy dropouts included so weights load perfectly
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
    def forward(self, x, h0):
        x = x + self.attn(self.ln_attn(x))
        h_seq = self.liquid(self.ln_liq(x), h0)
        x = x + torch.sigmoid(self.liq_gate(h_seq)) * h_seq
        x = x + self.ffn(self.ln_ffn(x))
        return x, h_seq[:, -1]

class LiquidLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.max_seq_length, cfg.dropout) 
            for _ in range(cfg.num_layers)
        ])
        self.h0 = nn.ParameterList([nn.Parameter(torch.zeros(cfg.hidden_size)) for _ in range(cfg.num_layers)])
        self.ln_out = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.emb_drop(self.token_emb(input_ids))
        for i, block in enumerate(self.blocks):
            h0_val = self.h0[i].expand(B, -1)
            x, _ = block(x, h0_val)
        return self.lm_head(self.ln_out(x))

# ========================== BENCHMARKING ENGINE ==========================

@torch.inference_mode()
def calculate_perplexity(model, enc, dataset_text, device, max_length=1024, stride=512, is_hf=False):
    """Calculates Perplexity using a sliding window approach."""
    model.eval()
    tokens = enc.encode_ordinary(dataset_text)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    seq_len = input_ids.size(1)
    nlls = []
    
    print(f"  -> Evaluating {seq_len} tokens with sliding window...")
    for i in tqdm(range(0, seq_len, stride), desc="Perplexity"):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_ids[:, begin_loc:end_loc].clone()
        target_chunk[:, :-trg_len] = -100 # Ignore context for loss calculation
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            if is_hf:
                outputs = model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss
            else:
                logits = model(input_chunk)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, 50257)
                shift_labels = target_chunk[:, 1:].contiguous().view(-1)
                neg_log_likelihood = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                
        nlls.append(neg_log_likelihood * trg_len)
        
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

@torch.inference_mode()
def benchmark_speed(model, enc, device, is_hf=False, prompt="The fundamental difference between machine learning and", max_new_tokens=100):
    """Measures raw token generation speed."""
    model.eval()
    ids = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Warmup
    for _ in range(5):
        if is_hf: model(ids)
        else: model(ids)
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Generate
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=torch.float16):
            if is_hf:
                outputs = model(ids)
                next_token_logits = outputs.logits[:, -1, :]
            else:
                logits = model(ids)
                next_token_logits = logits[:, -1, :]
                
            next_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=-1)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_sec = max_new_tokens / total_time
    
    # Profile memory
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) 
    torch.cuda.reset_peak_memory_stats()
    
    return tokens_per_sec, peak_mem

def generate_benchmark_plot(results):
    print("\n[!] Generating publication-ready benchmark plot...")
    
    models = list(results.keys())
    
    # Extract data
    ppl_vals = [results[m]["PPL"] for m in models]
    speed_vals = [results[m]["Speed"] for m in models]
    mem_vals = [results[m]["Mem"] for m in models]
    
    # Set up a 1x3 subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2ca02c', '#1f77b4'] # Green for your LiQ-LM, Blue for GPT-2
    
    # Panel 1: Perplexity
    axes[0].bar(models, ppl_vals, color=colors)
    axes[0].set_title('WikiText-2 Perplexity\n(Lower is Better)', fontweight='bold')
    axes[0].set_ylabel('Perplexity Score')
    for i, v in enumerate(ppl_vals):
        axes[0].text(i, v + (max(ppl_vals)*0.02), f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
        
    # Panel 2: Inference Speed
    axes[1].bar(models, speed_vals, color=colors)
    axes[1].set_title('Inference Speed\n(Higher is Better)', fontweight='bold')
    axes[1].set_ylabel('Tokens / Second')
    for i, v in enumerate(speed_vals):
        axes[1].text(i, v + (max(speed_vals)*0.02), f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
        
    # Panel 3: VRAM Usage
    axes[2].bar(models, mem_vals, color=colors)
    axes[2].set_title('Peak VRAM Footprint\n(Lower is Better)', fontweight='bold')
    axes[2].set_ylabel('Megabytes (MB)')
    for i, v in enumerate(mem_vals):
        axes[2].text(i, v + (max(mem_vals)*0.02), f"{v:.0f}", ha='center', va='bottom', fontweight='bold')
        
    # Global formatting
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    
    # Save at 300 DPI for journal submission
    plot_filename = "journal_benchmark_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[!] Plot saved successfully to {plot_filename}")

def run_benchmarks():
    print(f"\n{'='*60}\nJOURNAL BENCHMARK: LiQ-LM 150M vs GPT-2 124M\n{'='*60}")
    cfg = Config()
    enc = tiktoken.get_encoding("gpt2")
    
    # Load WikiText-2 Test Set (Combine into single string for fast eval)
    print("\n[1/3] Loading WikiText-2 Test Dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(dataset["text"])
    
    # Restrict to ~200k tokens to keep eval time reasonable on 8GB GPU
    test_text = test_text[:200000] 
    
    results = {}

    # ================= EVALUATE LIQ-LM =================
    print("\n[2/3] Initializing LiQ-LM 150M...")
    if not os.path.exists(cfg.model_path):
        print(f"[FATAL ERROR] Cannot find {cfg.model_path}. Exiting.")
        return
        
    model_liq = LiquidLM(cfg).to(cfg.device)
    model_liq.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device, weights_only=True))
    
    # FIXED: torch.compile REMOVED to prevent dynamic shape crash!
    # model_liq = torch.compile(model_liq) 
    
    print("  -> Benchmarking LiQ-LM Speed...")
    liq_tok_sec, liq_mem = benchmark_speed(model_liq, enc, cfg.device, is_hf=False)
    
    print("  -> Calculating LiQ-LM Perplexity...")
    liq_ppl = calculate_perplexity(model_liq, enc, test_text, cfg.device, is_hf=False)
    
    results["LiQ-LM 150M"] = {"PPL": liq_ppl, "Speed": liq_tok_sec, "Mem": liq_mem}
    
    # Free memory
    del model_liq
    gc.collect()
    torch.cuda.empty_cache()

    # ================= EVALUATE GPT-2 =================
    print("\n[3/3] Initializing OpenAI GPT-2 Base (124M)...")
    model_gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(cfg.device)
    
    # FIXED: torch.compile removed here as well to ensure a perfectly fair test
    # model_gpt = torch.compile(model_gpt)
    
    print("  -> Benchmarking GPT-2 Speed...")
    gpt_tok_sec, gpt_mem = benchmark_speed(model_gpt, enc, cfg.device, is_hf=True)
    
    print("  -> Calculating GPT-2 Perplexity...")
    gpt_ppl = calculate_perplexity(model_gpt, enc, test_text, cfg.device, is_hf=True)
    
    results["GPT-2 124M"] = {"PPL": gpt_ppl, "Speed": gpt_tok_sec, "Mem": gpt_mem}

    # ================= PRINT FINAL REPORT =================
    print(f"\n{'='*60}\nFINAL BENCHMARK RESULTS (Hardware: AMD RX7600)\n{'='*60}")
    print(f"{'Model':<15} | {'WikiText-2 PPL (↓)':<20} | {'Tokens/Sec (↑)':<15} | {'Peak VRAM (MB) (↓)'}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<15} | {metrics['PPL']:<20.2f} | {metrics['Speed']:<15.2f} | {metrics['Mem']:.1f} MB")
    print("=" * 60)
    
    generate_benchmark_plot(results)
    
    print("\nHow to read this for your paper:")
    print("1. Perplexity (PPL): If LiQ-LM is close to or lower than GPT-2, your Liquid Time-Constant (LTC) architecture proves superior parameter efficiency.")
    print("2. Tokens/Sec: This proves the parallel_associative_scan allows recurrent networks to generate tokens at comparable speeds to pure Transformers.")

if __name__ == "__main__":
    run_benchmarks()
