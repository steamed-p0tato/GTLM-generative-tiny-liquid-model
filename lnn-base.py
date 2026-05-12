#!/usr/bin/env python3
import os, math, random, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from tqdm import tqdm

# ================= PERFORMANCE & STABILITY =================
torch.set_float32_matmul_precision('high') 

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0" 
import tiktoken

gc.collect()
torch.cuda.empty_cache()

@dataclass
class Config:
    """LiQ-LM 150M - PHASE 2: WIKIPEDIA PUSH (8GB AMD Profile)"""
    hidden_size: int = 768     
    num_layers: int = 12       
    num_heads: int = 12        
    max_seq_length: int = 1024 
    vocab_size: int = 50257    
    dropout: float = 0.1       
    
    # --- DATA SCALING ---
    batch_size: int = 2        
    grad_accum: int = 64       
    num_epochs: int = 1        
    steps_per_epoch: int = 25000 
    
    # --- OPTIMIZER ---
    # Shorter warmup since the model is already pre-trained
    warmup_steps: int = 500   
    max_lr: float = 3e-4       # Lower peak LR for Phase 2 refinement
    min_lr: float = 3e-5       
    weight_decay: float = 0.1  
    label_smoothing: float = 0.0 
    max_grad_norm: float = 1.0 
    
    seed: int = 42             
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True       
    num_workers: int = 2       
    log_interval: int = 1      
    eval_interval: int = 1000  
    
    # New output targets so we don't overwrite Phase 1
    base_model_to_load: str = "liq_150M_final.pt" 
    output_model: str = "liq_150M_wiki_final.pt"
    checkpoint_path: str = "liq_150M_wiki_latest.pt"

    @property
    def total_steps(self) -> int: return self.num_epochs * self.steps_per_epoch

# ========================== UTILITIES ==========================

def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor):
    """Parallel Prefix Sum for Liquid Recurrence with dynamic padding."""
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

# ========================== MODEL COMPONENTS ==========================

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
    def __init__(self, dim, num_heads, max_seq_len=1024, dropout=0.1):
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

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True 
        )
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
    def __init__(self, dim, num_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, max_seq_len, dropout)
        
        self.ln_liq = RMSNorm(dim)
        self.liquid = ParallelLiquidCell(dim)
        self.liq_gate = nn.Linear(dim, dim, bias=False) 
        
        self.ln_ffn = RMSNorm(dim)
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
        self.use_checkpointing = True 
        
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        x = self.emb_drop(self.token_emb(input_ids))
        
        for i, block in enumerate(self.blocks):
            h0_val = self.h0[i].expand(B, -1)
            if self.training and self.use_checkpointing:
                x, _ = checkpoint(block, x, h0_val, use_reentrant=False)
            else:
                x, _ = block(x, h0_val)
                
        logits = self.lm_head(self.ln_out(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss, logits

# ========================== DATA LOADING (WIKIPEDIA) ==========================

class BPEStream(IterableDataset):
    def __init__(self, cfg: Config, epoch: int = 0, resume_step: int = 0):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.epoch = epoch
        self.resume_step = resume_step 

    def __iter__(self):
        from datasets import load_dataset
        import datasets
        datasets.config.STREAMING_READ_MAX_RETRIES = 100
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        num_w = worker_info.num_workers if worker_info else 1
        
        samples_processed, max_retries = 0, 50
        while max_retries > 0:
            try:
                # [FIXED] Using the modern, official Wikimedia dataset
                ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
                
                dynamic_seed = self.cfg.seed + (self.epoch * 1000) + self.resume_step + wid
                ds = ds.shuffle(buffer_size=1000, seed=dynamic_seed)
                
                if num_w > 1: ds = ds.shard(num_shards=num_w, index=wid)
                
                for sample in ds:
                    tokens = self.enc.encode_ordinary(sample["text"])
                    if not tokens: continue
                    sl = self.cfg.max_seq_length
                    
                    if len(tokens) > sl:
                        s = random.randint(0, len(tokens) - sl)
                        ids = tokens[s : s+sl]; lbl = ids
                    else:
                        pad = sl - len(tokens)
                        ids = tokens + [self.eot] * pad; lbl = tokens + [-100] * pad
                    yield {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(lbl, dtype=torch.long)}
                    samples_processed += 1
                break
            except Exception as e:
                # [FIXED] Loudly print the error so we aren't blind!
                print(f"\n[DATASET STREAM ERROR] {e}") 
                max_retries -= 1
                time.sleep(5)
                
        if max_retries <= 0:
            print("\n[FATAL] Could not connect to Hugging Face dataset after 50 retries.")

# ========================== TRAINING HELPERS ==========================

def get_lr(cfg, step):
    if step < cfg.warmup_steps: return cfg.max_lr * step / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps), 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
@torch.compiler.disable 
def evaluate_prompts(model, prompts, max_new=50):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"\n{'='*40}\nPROMPT EVALUATION\n{'='*40}")
    for p in prompts:
        ids = torch.tensor(enc.encode(p), dtype=torch.long, device=model.cfg.device).unsqueeze(0)
        for _ in range(max_new):
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            next_id = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, -1), 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == 50256: break
        print(f"Prompt: {p}\nOutput: {enc.decode(ids[0].tolist())}\n{'-'*40}")
    model.train()

def train():
    cfg = Config()
    model = LiquidLM(cfg).to(cfg.device)
    compiled_model = torch.compile(model)
    
    # --- PHASE 2 PRE-LOAD LOGIC ---
    if os.path.exists(cfg.base_model_to_load) and not os.path.exists(cfg.checkpoint_path):
        print(f"\n[!] Phase 1 Weights Found: '{cfg.base_model_to_load}'. Injecting into model...")
        model.load_state_dict(torch.load(cfg.base_model_to_load, map_location=cfg.device))
        print("[!] Phase 1 Weights successfully loaded. Ready for Phase 2 (Wikipedia).")
    elif not os.path.exists(cfg.checkpoint_path):
        print(f"\n[WARNING] Base model '{cfg.base_model_to_load}' not found! Starting completely from scratch.\n")
    
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim < 2 or "ln" in name or "bias" in name or "log_dt" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.max_lr, betas=(0.9, 0.95))
    
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- PHASE 2 RESUME LOGIC ---
    global_step = 0
    micro_step = 0
    start_epoch = 1

    if os.path.exists(cfg.checkpoint_path):
        print(f"\n[!] Found Phase 2 checkpoint '{cfg.checkpoint_path}'. Resuming training...")
        checkpoint_data = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
        
        model.load_state_dict(checkpoint_data['model_state_dict'])
        opt.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        global_step = checkpoint_data['global_step']
        start_epoch = checkpoint_data['epoch']
        print(f"[!] Resumed Wikipedia Push at Epoch {start_epoch}, Global Step {global_step}\n")
    else:
        total_p = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*50}\nSTARTING PHASE 2: WIKIPEDIA (AMD RX7600)\nTotal Params: {total_p:,}\n{'='*50}\n")
        
    test_prompts = [
        "The history of the Roman Empire is", 
        "Albert Einstein was a theoretical physicist who",
        "The primary function of mitochondria in a cell is"
    ]

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        loader = DataLoader(BPEStream(cfg, epoch-1, resume_step=global_step), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        
        initial_pbar_step = global_step * cfg.grad_accum
        total_micro_steps = cfg.steps_per_epoch * cfg.grad_accum
        pbar = tqdm(loader, total=total_micro_steps, initial=initial_pbar_step, desc=f"Epoch {epoch} (Wiki)")
        
        for batch in pbar:
            input_ids, labels = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
            
            if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_begin()

            device_type_str = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type_str, dtype=torch.float16):
                loss, _ = compiled_model(input_ids, labels)
            
            scaler.scale(loss / cfg.grad_accum).backward()
            micro_step += 1
            
            if micro_step % cfg.grad_accum == 0:
                global_step += 1
                
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                
                for pg in opt.param_groups: pg["lr"] = get_lr(cfg, global_step)
                
                if global_step > 0 and global_step % cfg.eval_interval == 0:
                    evaluate_prompts(model, test_prompts)
                    
                    checkpoint_dict = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }
                    torch.save(checkpoint_dict, cfg.checkpoint_path)
                    torch.save(checkpoint_dict, f"liq_150M_wiki_step_{global_step}.pt")
                
                if global_step % cfg.log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                    
                if global_step >= cfg.steps_per_epoch: break

    torch.save(model.state_dict(), cfg.output_model)

if __name__ == "__main__":
    train()
