# GTRM: Generative Tiny Recursive Model

**GTRM** is a 150-million parameter hybrid Small Language Model (SLM) designed to bridge the gap between global context-awareness and recurrent efficiency. By merging **FlashAttention** with **Parallel Liquid Neural Networks (LNNs)**, GTRM achieves high-performance generative modeling on consumer-grade hardware.

---

## 🚀 Key Features

* **Hybrid Liquid-Transformer Architecture:** Combines the targeted retrieval of self-attention with the continuous-time dynamics of Liquid Cells.
* **Parallel Associative Scan:** Overcomes the $O(N)$ sequential bottleneck of traditional recurrent models, allowing for $O(\log N)$ parallel training.
* **Memory Optimized:** Engineered to train on **8GB VRAM** (e.g., AMD RX7600) using Gradient Accumulation and Automatic Mixed Precision (AMP).
* **Advanced Embeddings:** Utilizes **Rotary Positional Embeddings (RoPE)** for superior relative positioning and **RMSNorm** for training stability.
* **Wikipedia Refined:** Pre-configured for a **3.2 Billion token** refinement phase using the modern Wikimedia stream.

---

## 🏗️ Architecture Overview

GTRM leverages a unique "Recurrence-First" approach. Unlike standard Transformers that suffer from quadratic complexity, GTRM gates a Liquid recurrent state within each block to compress historical context while retaining the ability to recall specific facts via attention.

| Component | Implementation | Benefit |
| --- | --- | --- |
| **Attention** | FlashAttention-2 | Memory-efficient global context |
| **Recurrence** | Parallel Liquid Cell | Continuous-time state dynamics |
| **Normalization** | RMSNorm | Lower compute overhead vs LayerNorm |
| **Positioning** | RoPE | Robust handling of long-range dependencies |
| **Optimization** | AdamW + Cosine Decay | Stable convergence on small datasets |

---

## 🛠️ Installation

Ensure you have a modern Python environment and the necessary dependencies installed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken tqdm datasets

```

---

## 📈 Training Configuration

The model is currently optimized for **Phase 2: Wikipedia Push**.

```python
# Hardware Profile: 8GB VRAM
Config:
    hidden_size: 768
    num_layers: 12
    num_heads: 12
    batch_size: 2
    grad_accum: 64  # Effective batch size: 128
    max_lr: 3e-4

```

To start training:

```bash
python train.py

```

---

## 🧪 Evaluation

The training script includes an automated evaluation loop. Every `1000` steps, the model is prompted to generate completions for historical and scientific queries to benchmark its factual recall evolution.

**Sample Prompt:**

> *Input:* "The primary function of mitochondria in a cell is"
> *Output:* "...to generate most of the chemical energy needed to power the cell's biochemical reactions."

---

## 🎓 Research & Recognition

This project was developed as part of the **ACM Kolkata Chapter UG Project Award 2026** (India East & North-East).

* **Lead Researcher:** Ariyan Basu
* **Collaborator:** Pallav Anand Singh
* **Institution:** Adamas University
* **Supervisor:** Dr. Tamal Ghosh

---

## 📄 License

This project is open-source and available under the **MIT License**.

> **Note:** This model is part of an ongoing research initiative into Small Language Model (SLM) optimizers and Liquid Neural Network scalability. For inquiries regarding the GTRM architecture or research paper, please open an issue or contact the authors.

---

*Inspired by minimalist, developer-centric design—built for the next generation of edge-deployed AI.*
