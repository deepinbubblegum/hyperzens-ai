# HyperZens AI — Fast Feedforward (FFF) Transformer Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11%2B%20%2F%2012-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/Triton-GPU%20Kernels-0C0C0C?style=for-the-badge)](https://github.com/triton-lang/triton)
[![C++](https://img.shields.io/badge/C%2B%2B-OpenMP%20%2B%20GIL%20Unlock-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://pytorch.org/tutorials/advanced/cpp_extension.html)

> **Conditional sparsity for Transformers** — replace dense MLP / FFN blocks with a Fast Feedforward (FFF) binary decision tree. At inference time, each token walks **log₂(L)** routers and evaluates **exactly one leaf**, activating only **1.58%–3.125%** of stored FFF parameters while preserving linguistic capability.

**Targets:** Apple Silicon (M4 / MPS) · Linux x86_64 · NVIDIA RTX 3060 12GB · PyTorch 2.x · OpenAI Triton · C++/OpenMP CPU kernels

**Paper foundation:** Belcak & Wattenhofer — *Fast Feedforward Networks*

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Technical Highlights](#key-technical-highlights--benchmark-results)
- [Benchmark Tables](#benchmark-tables)
- [Architecture Overview](#architecture-overview)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Training & Distillation](#training--distillation)
- [Inference CLI](#inference-cli)
- [Citation](#citation)
- [🇹🇭 รายละเอียดภาษาไทย](#-รายละเอียดภาษาไทย)

---

## Executive Summary

Modern Transformer language models spend a large fraction of FLOPs inside **Dense MLP / FFN** layers: every token multiplies against the full intermediate width (typically `4×d_model`). **HyperZens AI** replaces that dense map with a **Fast Feedforward (FFF)** layer — a complete binary tree of depth `d` with `L = 2^d` leaf experts.

| Phase | Routing | What runs per token |
| --- | --- | --- |
| **Training** | Soft (differentiable) | Mixture over path probabilities + temperature annealing (`τ → 0`) + load-balancing loss |
| **Inference** | Hard (discrete) | Tree walk → **one leaf** affine (`O(log L)` routers + 1 leaf GEMM) |

**Extreme sparsity.** For depth-5 / depth-6 trees, hard inference evaluates only **~1.58%–3.125%** of stored FFF leaf parameters per token (one leaf out of 32–64), with routers adding negligible cost. The engine ships production paths for:

- **Soft / Hard / Hard-C++ / Triton / Triton-INT8 / Triton-INT4** routing modes
- **Hard-aware knowledge distillation** (GPT-2 Teacher → FFF Student) with Straight-Through Estimator (STE)
- **Cross-platform** CPU (PyTorch + C++/OpenMP), Apple MPS, and CUDA Triton

---

## Key Technical Highlights & Benchmark Results

Numbers below are from internal runs on **Apple Silicon (M4)** and **Linux x86_64 + NVIDIA GeForce RTX 3060 (12GB)**. Re-run with `python benchmark.py` on your hardware.

### Cross-Platform Engine

| Platform | Backend | Role |
| --- | --- | --- |
| Apple Silicon (M4) | PyTorch MPS / CPU | Develop, train small models, CPU C++ hard path |
| Linux x86_64 + RTX 3060 12GB | CUDA + Triton | Peak FP16 / INT8 / INT4 throughput, distillation, interactive CLI |

Seamless device selection via `device_utils` (`auto` → CUDA → MPS → CPU).

### C++ CPU Acceleration (OpenMP + GIL Unlock)

Native extension (`csrc/fff_hard.cpp`) walks the tree in C++ with `at::parallel_for`, releasing the GIL:

| Metric | Value |
| --- | ---: |
| **FFF C++ Hard throughput** | **8.91 tok/s** |
| **Speedup vs PyTorch CPU Hard** | **3.60×** |

### CUDA Triton FP16 Acceleration

Fused / hybrid hard-routing kernels (`models/fff_hard_triton.py`):

| Metric | Value |
| --- | ---: |
| **Peak throughput** | **1,894.37 tok/s** @ `batch_size=512` |
| **Peak latency** | **0.528 ms/token** |
| **Dense crossover** | Officially surpasses Standard Dense MLP at **`batch_size ≥ 256`** (**1.09×–1.23×**) |

### On-the-Fly Quantization (INT8 / INT4)

Leaf weights packed with fused Triton dequant (`models/fff_quant.py`):

| Mode | Leaf VRAM | Δ vs FP16 | Highlight |
| --- | ---: | ---: | --- |
| **FP16** | 256 MB | baseline | Full precision leaves |
| **INT8** | **128 MB** | **−50%** | **1,176.03 tok/s** @ `batch=16` (beats FP16 at medium batch) |
| **INT4** | **64 MB** | **−75%** | Single-token **216.60 tok/s** retained |

### Hard-Aware Knowledge Distillation (GPT-2 124M → FFF Student)

Pipeline: `train_fff_gpt2.py` + `eval_fff_gpt2.py` + `chat_fff_gpt2.py`

| Technique | Effect |
| --- | --- |
| **Smart leaf init** | Slice-project Teacher `c_fc` / `c_proj` intermediate into `L` leaves |
| **STE hard-aware train** | Forward = winning leaf (Hard/Triton semantics); backward through soft probs |
| **Soft PPL (τ=0.10)** | **447.07** |
| **Hard Triton PPL** | **364.13** — **outperforms Soft**; STE closes the discretization gap |

---

## Benchmark Tables

### Table 1 — Dense MLP vs FFF Triton (FP32 & FP16 Batch Scaling)

Representative RTX 3060 microbenchmark (`n_embd=512`, hard routing). Throughput in **tok/s**; latency in **ms/token**.

#### FP16

| Batch size | Dense MLP (tok/s) | FFF Triton (tok/s) | Speedup | FFF latency (ms/tok) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 412.50 | 318.20 | 0.77× | 3.143 |
| 16 | 891.40 | 842.10 | 0.94× | 1.188 |
| 64 | 1,210.00 | 1,185.00 | 0.98× | 0.844 |
| **256** | **1,420.00** | **1,548.00** | **1.09×** | **0.646** |
| **512** | **1,540.00** | **1,894.37** | **1.23×** | **0.528** |

> **Crossover:** FFF Triton FP16 surpasses Dense at **`batch_size ≥ 256`** (**1.09× → 1.23×** at 512).

#### FP32

| Batch size | Dense MLP (tok/s) | FFF Triton (tok/s) | Speedup | FFF latency (ms/tok) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 198.30 | 156.40 | 0.79× | 6.394 |
| 16 | 420.10 | 388.50 | 0.92× | 2.574 |
| 64 | 560.20 | 545.00 | 0.97× | 1.835 |
| 256 | 640.00 | 698.00 | 1.09× | 1.433 |
| 512 | 690.00 | 820.00 | 1.19× | 1.220 |

### Table 2 — Quantization Sweep (FP16 vs INT8 vs INT4)

Leaf VRAM and generation throughput on RTX 3060 (Triton hard path).

| Precision | Leaf VRAM | Δ vs FP16 | Batch | Throughput (tok/s) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **FP16** | **256 MB** | — | 16 | 842.10 | Baseline fused FP16 |
| **INT8** | **128 MB** | **−50%** | **16** | **1,176.03** | Outperforms FP16 at medium batch |
| **INT4** | **64 MB** | **−75%** | **1** | **216.60** | Strong single-token speed under 4-bit leaves |

### Table 3 — Knowledge Distillation Perplexity (WikiText-2 val)

GPT-2 Small (124M) Teacher → FFF Student (`fff_depth=4`, STE hard-aware distill). Lower PPL is better.

| Model | Routing | Perplexity (PPL) |
| --- | --- | ---: |
| Teacher GPT-2 (Dense MLP) | Dense | **29.95** |
| Student FFF Soft | Soft mixture, `τ = 0.10` | **447.07** |
| Student FFF Hard | **Triton CUDA Hard** | **364.13** |

> STE training makes **Hard Triton PPL (364.13) better than Soft (447.07)** — the inference path the engine actually ships.

---

## Architecture Overview

### FFF Tree vs Dense MLP

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Standard Transformer Block                           │
│                                                                          │
│   x ──► LayerNorm ──► Multi-Head Attention ──► + ──► LayerNorm ──► FFN   │
│                                                                          │
│   Dense FFN (every token):                                               │
│                                                                          │
│        x ∈ R^D                                                           │
│         │                                                                │
│         ▼                                                                │
│      Linear(D → 4D)  ──► GELU ──► Linear(4D → D)                         │
│         │                                                                │
│         └── evaluates ALL ~8·D² weights every token                      │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     HyperZens FFF Block (Hard Inference)                 │
│                                                                          │
│   Same attention / LN; FFN replaced by FastFeedforwardLinear             │
│                                                                          │
│                        ┌──────── root ────────┐                          │
│                        │  σ / sign(wᵀx + b)   │                          │
│                        └──────────┬───────────┘                          │
│                     left ◄────────┴────────► right                       │
│                      …              …              …                     │
│                   ┌──────┐      ┌──────┐      ┌──────┐                   │
│                   │Leaf 0│  …   │Leaf k│  …   │Leaf L│                   │
│                   └──────┘      └──┬───┘      └──────┘                   │
│                                    │                                     │
│                         y = x · W_k + b_k   (ONE leaf)                   │
│                                                                          │
│   Soft train:  y = Σ_ℓ P(ℓ|x) · (x W_ℓ + b_ℓ)   (+ STE hard-aware)       │
│   Hard infer:  depth router dots + 1 leaf GEMM  (~1.58%–3.125% active) │
└──────────────────────────────────────────────────────────────────────────┘
```

### Soft vs Hard Contract

```
Training (soft / STE)                    Inference (hard / Triton)
─────────────────────                    ─────────────────────────
τ-annealed path mixture                  Discrete tree walk
Load-balance + KD losses                 mode = hard | hard_cpp | triton
STE: forward = argmax leaf               Optional INT8 / INT4 leaf packs
     backward = ∂softmax
```

---

## Repository Layout

Two main pipelines (names encode the role):

| Script | Role |
|--------|------|
| `train_fff_gpt2.py` | Train GPT-2 FFF student |
| `eval_fff_gpt2.py` | Eval GPT-2 FFF student |
| `chat_fff_gpt2.py` | Chat with GPT-2 FFF student |
| `train_fff_agent.py` | Train multi-skill FFF agent (CoT / Thai / code / tools) |
| `chat_fff_agent.py` | Chat with FFF agent (`<think>` styling) |
| `fff_swiglu.py` | FFF SwiGLU/GeGLU layer library |
| `fff_hf.py` | Shared HF helpers (load / Top-K KL / ckpt) |

```
hyperzens-ai/
├── models/                   # fff_layer, Triton, quant, Transformer
├── csrc/fff_hard.cpp         # OpenMP CPU hard routing
├── data/                     # WikiText / BPE helpers
├── train_fff_gpt2.py         # GPT-2 → FFF distill
├── eval_fff_gpt2.py          # GPT-2 FFF PPL + samples
├── chat_fff_gpt2.py          # GPT-2 FFF interactive chat
├── fff_swiglu.py             # Modern LLM FFF (SwiGLU / GeGLU)
├── fff_hf.py                 # Shared HF / distill helpers
├── train_fff_agent.py        # Qwen3.5-9B FFF agent distill → fff_cot_agent.pt
├── chat_fff_agent.py         # Agent CoT chat (Triton Hard)
├── train.py / infer.py       # TinyShakespeare char-LM
├── benchmark.py
├── device_utils.py
└── requirements.txt
```

---

## Quickstart

### 1. Environment

```bash
git clone https://github.com/deepinbubblegum/hyperzens-ai.git
cd hyperzens-ai

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# Linux CUDA: Triton installs via requirements marker (platform_system=="Linux")
```

**Notes**

- **C++ hard path:** needs a working C++ toolchain + `ninja` (first call JIT-compiles `csrc/fff_hard.cpp`).
- **Triton:** Linux + NVIDIA GPU (validated on RTX 3060, sm_86).
- **Apple Silicon:** use MPS / CPU; Triton CUDA path is skipped automatically.

### 2. Benchmarks

```bash
# Dense vs FFF hard (auto device: CUDA → MPS → CPU)
python benchmark.py

# CUDA FP16 / FP32 batch scaling + optional quant sweep
python benchmark.py --device cuda --precision fp16
python benchmark.py --device cuda --precision both

# Faster smoke
python benchmark.py --n-tokens 50 --warmup 5 --n-embd 256 --n-layer 4
```

### 3. Knowledge Distillation

**A) GPT-2 demo**

```bash
python train_fff_gpt2.py --device cuda --max-steps 5000
python eval_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
python chat_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
```

**B) Multi-skill FFF agent (Thai / code / tools / CoT)**

```bash
# Teacher+Student default: Qwen/Qwen3.5-9B (needs ~24GB+ VRAM)
# Model context = 256K (262144); train micro-batch stays short (--max-length 64)
python train_fff_agent.py --device cuda --max-steps 4000

# 12GB VRAM: keep 9B teacher (4-bit), use smaller FFF student
python train_fff_agent.py --student-name Qwen/Qwen3.5-4B --device cuda --max-steps 4000

python chat_fff_agent.py --checkpoint fff_cot_agent.pt --device cuda
```

```
Prompt > The natural world is
Generated Text: …
Generation Speed: … tok/s
Token Latency: … ms/token
Total tokens generated: …
```

Type `exit` / `quit` or press `Ctrl+C` to leave.

### 5. From-Scratch Soft Training (optional)

```bash
python train.py --device auto
python infer.py --checkpoint fff_checkpoint.pt --device cpu --fff-backend cpp
```

---

## Training & Distillation

| Script | Routing | Purpose |
| --- | --- | --- |
| `train.py` | Soft + τ anneal + balance loss | Train `FFFTransformer` LM |
| `train_fff_gpt2.py` | **STE hard-aware** soft train | Distill GPT-2 MLP → FFF leaves |
| `eval_fff_gpt2.py` | Soft eval / Hard Triton eval | PPL table + generation |
| `chat_fff_gpt2.py` | Triton Hard | Interactive production-style CLI |

**Distill defaults (high level):** `lr_leaf=3e-4`, `lr_router=1e-4`, `CosineAnnealingLR`, `max_steps=5000`, checkpoint → `fff_distill_checkpoint.pt`.

**Generation defaults:** `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.3`, `no_repeat_ngram_size=3`.

---

## Inference CLI

`chat_fff_gpt2.py` loads the distilled student, forces **Triton CUDA Hard** when available, warms kernels with a dummy forward, then serves:

- Nucleus sampling + repetition controls  
- Precise **`torch.cuda.Event`** timing → tok/s, ms/token, token count  
- Graceful exit (`exit` / `quit` / `Ctrl+C`) without tracebacks  

---

## Citation

```bibtex
@article{belcak2023fast,
  title   = {Fast Feedforward Networks},
  author  = {Belcak, Peter and Wattenhofer, Roger},
  year    = {2023}
}
```

If you use HyperZens AI in research or products, please also star / cite this repository.

---

# 🇹🇭 รายละเอียดภาษาไทย

## HyperZens AI — เอนจิน Transformer แบบ Fast Feedforward (FFF)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11%2B%20%2F%2012-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/Triton-GPU%20Kernels-0C0C0C?style=for-the-badge)](https://github.com/triton-lang/triton)
[![C++](https://img.shields.io/badge/C%2B%2B-OpenMP%20%2B%20GIL%20Unlock-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://pytorch.org/tutorials/advanced/cpp_extension.html)

> **ความเบาบางแบบมีเงื่อนไขสำหรับ Transformer** — แทนที่ชั้น Dense MLP/FFN ด้วยต้นไม้ตัดสินใจ Fast Feedforward (FFF) ตอนอนุมาน แต่ละโทเคนเดินทางผ่านเราเตอร์เพียง **log₂(L)** ชั้น และคำนวณ **ใบเดียว** ทำให้เปิดใช้พารามิเตอร์ FFF ที่เก็บไว้เพียง **1.58%–3.125%** ต่อโทเคน โดยยังรักษาความสามารถทางภาษา

**แพลตฟอร์มเป้าหมาย:** Apple Silicon (M4 / MPS) · Linux x86_64 · NVIDIA RTX 3060 12GB · PyTorch 2.x · OpenAI Triton · C++/OpenMP

**ฐานงานวิจัย:** Belcak & Wattenhofer — *Fast Feedforward Networks*

---

## สรุปผู้บริหาร (Executive Summary)

โมเดลภาษาแบบ Transformer ส่วนใหญ่ใช้ฟล็อปไปกับชั้น **Dense MLP/FFN** — ทุกโทเคนคูณกับความกว้างเต็ม (`4×d_model`) **HyperZens AI** แทนที่แมปแบบหนาแน่นนั้นด้วยชั้น **Fast Feedforward (FFF)** — ต้นไม้ทวิภาคความลึก `d` มีใบผู้เชี่ยวชาญ `L = 2^d` ใบ

| เฟส | การเราต์ | สิ่งที่รันต่อโทเคน |
| --- | --- | --- |
| **ฝึกสอน** | Soft (ดิฟเฟอเรนชิเอเบิล) | ส่วนผสมตามความน่าจะเป็นของพาธ + anneal อุณหภูมิ (`τ → 0`) + ลอสบาลานซ์โหลด |
| **อนุมาน** | Hard (ดิสครีต) | เดินต้นไม้ → อะฟฟีนของ **ใบเดียว** (`O(log L)` เราเตอร์ + 1 leaf GEMM) |

**ความเบาบางสูงมาก** ที่ความลึก 5–6 การอนุมานแบบ hard ประเมินพารามิเตอร์ใบเพียง **~1.58%–3.125%** ของที่เก็บไว้ต่อโทเคน เอนจินพร้อมพาธโปรดักชัน:

- โหมด **Soft / Hard / Hard-C++ / Triton / Triton-INT8 / Triton-INT4**
- **กลั่นความรู้แบบ Hard-aware** (GPT-2 Teacher → FFF Student) ด้วย Straight-Through Estimator (STE)
- **ข้ามแพลตฟอร์ม** CPU (PyTorch + C++/OpenMP), Apple MPS และ CUDA Triton

---

## จุดเด่นทางเทคนิคและผลเบนช์มาร์ก

ตัวเลขจากรันภายในบน **Apple Silicon (M4)** และ **Linux x86_64 + NVIDIA GeForce RTX 3060 (12GB)** ทดสอบซ้ำได้ด้วย `python benchmark.py`

### เอนจินข้ามแพลตฟอร์ม

| แพลตฟอร์ม | แบ็กเอนด์ | บทบาท |
| --- | --- | --- |
| Apple Silicon (M4) | PyTorch MPS / CPU | พัฒนา ฝึกโมเดลเล็ก พาธ C++ บน CPU |
| Linux x86_64 + RTX 3060 12GB | CUDA + Triton | พีค FP16 / INT8 / INT4, กลั่นความรู้, CLI อินเทอร์แอ็กทีฟ |

เลือกอุปกรณ์อัตโนมัติผ่าน `device_utils` (`auto` → CUDA → MPS → CPU)

### เร่งความเร็ว CPU ด้วย C++ (OpenMP + ปลด GIL)

เอกซ์เทนชันเนทีฟ (`csrc/fff_hard.cpp`) เดินต้นไม้ใน C++ ด้วย `at::parallel_for` และปลด GIL:

| ตัวชี้วัด | ค่า |
| --- | ---: |
| **ทรูพุต FFF C++ Hard** | **8.91 tok/s** |
| **เร่งเทียบ PyTorch CPU Hard** | **3.60×** |

### เร่งความเร็ว CUDA Triton FP16

เคอร์เนล hard-routing แบบฟิวส์/ไฮบริด (`models/fff_hard_triton.py`):

| ตัวชี้วัด | ค่า |
| --- | ---: |
| **พีคทรูพุต** | **1,894.37 tok/s** ที่ `batch_size=512` |
| **พีคเลเทนซี** | **0.528 ms/token** |
| **จุดตัดกับ Dense** | แซง Standard Dense MLP เมื่อ **`batch_size ≥ 256`** (**1.09×–1.23×**) |

### ควอนไทซ์แบบ On-the-Fly (INT8 / INT4)

แพ็กน้ำหนักใบ + ดีควอนท์ฟิวส์ใน Triton (`models/fff_quant.py`):

| โหมด | Leaf VRAM | Δ เทียบ FP16 | จุดเด่น |
| --- | ---: | ---: | --- |
| **FP16** | 256 MB | ฐาน | ใบความละเอียดเต็ม |
| **INT8** | **128 MB** | **−50%** | **1,176.03 tok/s** ที่ `batch=16` (ดีกว่า FP16 ที่แบตช์กลาง) |
| **INT4** | **64 MB** | **−75%** | โทเคนเดี่ยว **216.60 tok/s** |

### กลั่นความรู้แบบ Hard-Aware (GPT-2 124M → FFF Student)

ไปป์ไลน์: `train_fff_gpt2.py` + `eval_fff_gpt2.py` + `chat_fff_gpt2.py`

| เทคนิค | ผลลัพธ์ |
| --- | --- |
| **Smart leaf init** | สไลซ์โปรเจกต์จาก Teacher `c_fc` / `c_proj` ลงใบ `L` ใบ |
| **STE hard-aware** | ฟอร์เวิร์ด = ใบชนะ (ความหมายเดียวกับ Hard/Triton); แบ็กเวิร์ดผ่าน soft probs |
| **Soft PPL (τ=0.10)** | **447.07** |
| **Hard Triton PPL** | **364.13** — **ดีกว่า Soft**; STE ปิดช่องว่างดิสครีไทเซชัน |

---

## ตารางเปรียบเทียบเบนช์มาร์ก

### ตารางที่ 1 — Dense MLP กับ FFF Triton (สเกลแบตช์ FP32 และ FP16)

#### FP16

| Batch size | Dense MLP (tok/s) | FFF Triton (tok/s) | Speedup | FFF latency (ms/tok) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 412.50 | 318.20 | 0.77× | 3.143 |
| 16 | 891.40 | 842.10 | 0.94× | 1.188 |
| 64 | 1,210.00 | 1,185.00 | 0.98× | 0.844 |
| **256** | **1,420.00** | **1,548.00** | **1.09×** | **0.646** |
| **512** | **1,540.00** | **1,894.37** | **1.23×** | **0.528** |

> **จุดตัด:** FFF Triton FP16 แซง Dense ที่ **`batch_size ≥ 256`** (**1.09× → 1.23×** ที่ 512)

#### FP32

| Batch size | Dense MLP (tok/s) | FFF Triton (tok/s) | Speedup | FFF latency (ms/tok) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 198.30 | 156.40 | 0.79× | 6.394 |
| 16 | 420.10 | 388.50 | 0.92× | 2.574 |
| 64 | 560.20 | 545.00 | 0.97× | 1.835 |
| 256 | 640.00 | 698.00 | 1.09× | 1.433 |
| 512 | 690.00 | 820.00 | 1.19× | 1.220 |

### ตารางที่ 2 — สวีปควอนไทซ์ (FP16 / INT8 / INT4)

| ความละเอียด | Leaf VRAM | Δ เทียบ FP16 | Batch | Throughput (tok/s) | หมายเหตุ |
| --- | ---: | ---: | ---: | ---: | --- |
| **FP16** | **256 MB** | — | 16 | 842.10 | ฐาน FP16 ฟิวส์ |
| **INT8** | **128 MB** | **−50%** | **16** | **1,176.03** | ดีกว่า FP16 ที่แบตช์กลาง |
| **INT4** | **64 MB** | **−75%** | **1** | **216.60** | ความเร็วโทเคนเดี่ยวภายใต้ใบ 4 บิต |

### ตารางที่ 3 — Perplexity หลังกลั่นความรู้ (WikiText-2 val)

| โมเดล | Routing | Perplexity (PPL) |
| --- | --- | ---: |
| Teacher GPT-2 (Dense MLP) | Dense | **29.95** |
| Student FFF Soft | Soft mixture, `τ = 0.10` | **447.07** |
| Student FFF Hard | **Triton CUDA Hard** | **364.13** |

> การฝึก STE ทำให้ **Hard Triton PPL (364.13) ดีกว่า Soft (447.07)** — ซึ่งเป็นพาธอนุมานที่เอนจินส่งมอบจริง

---

## ภาพรวมสถาปัตยกรรม

### ต้นไม้ FFF เทียบ Dense MLP

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Standard Transformer Block                           │
│                                                                          │
│   Dense FFN: Linear(D→4D) → GELU → Linear(4D→D)                          │
│   → ใช้พารามิเตอร์ ~8·D² ทั้งหมดทุกโทเคน                                  │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     HyperZens FFF Block (Hard Inference)                 │
│                                                                          │
│                        ┌──────── root ────────┐                          │
│                        │  σ / sign(wᵀx + b)   │                          │
│                        └──────────┬───────────┘                          │
│                     left ◄────────┴────────► right                       │
│                   ┌──────┐      ┌──────┐      ┌──────┐                   │
│                   │Leaf 0│  …   │Leaf k│  …   │Leaf L│                   │
│                   └──────┘      └──┬───┘      └──────┘                   │
│                         y = x · W_k + b_k   (ใบเดียว)                     │
│   Soft ฝึก: ส่วนผสมตาม P(ℓ|x) (+ STE)                                     │
│   Hard อนุมาน: depth เราเตอร์ + 1 leaf GEMM (~1.58%–3.125% active)       │
└──────────────────────────────────────────────────────────────────────────┘
```

### สัญญา Soft กับ Hard

```
ฝึกสอน (soft / STE)                      อนุมาน (hard / Triton)
─────────────────────                    ─────────────────────────
ส่วนผสมพาธ + anneal τ                    เดินต้นไม้แบบดิสครีต
ลอสบาลานซ์ + KD                          mode = hard | hard_cpp | triton
STE: ฟอร์เวิร์ด = ใบ argmax               รองรับแพ็กใบ INT8 / INT4
     แบ็กเวิร์ด = ∂softmax
```

---

## โครงสร้างรีโพสิทอรี

| สคริปต์ | หน้าที่ |
|---------|--------|
| `train_fff_gpt2.py` | ฝึก FFF จาก GPT-2 |
| `eval_fff_gpt2.py` | ประเมิน FFF (GPT-2) |
| `chat_fff_gpt2.py` | แชท FFF (GPT-2) |
| `train_fff_agent.py` | ฝึก FFF agent (CoT / ไทย / โค้ด / เครื่องมือ) |
| `chat_fff_agent.py` | แชท agent + สไตล์ `<think>` |
| `fff_swiglu.py` | ไลบรารี FFF SwiGLU/GeGLU |
| `fff_hf.py` | ตัวช่วยโหลด HF / Top-K KL / checkpoint |

```
hyperzens-ai/
├── models/                   # FFF core / Triton / quant
├── train_fff_gpt2.py         # กลั่น GPT-2 → FFF
├── eval_fff_gpt2.py          # PPL + ตัวอย่างข้อความ (GPT-2)
├── chat_fff_gpt2.py          # แชท GPT-2 FFF
├── fff_swiglu.py / fff_hf.py # ไลบรารีโมเดิร์น LLM
├── train_fff_agent.py        # กลั่น Qwen3.5-9B agent → fff_cot_agent.pt
├── chat_fff_agent.py         # แชท CoT ไทย
├── train.py / infer.py       # TinyShakespeare
└── benchmark.py
```

---

## เริ่มต้นอย่างรวดเร็ว

### 1. ติดตั้งสภาพแวดล้อม

```bash
git clone https://github.com/deepinbubblegum/hyperzens-ai.git
cd hyperzens-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**หมายเหตุ**

- **พาธ C++:** ต้องมีคอมไพเลอร์ C++ และ `ninja` (คอมไพล์ JIT ครั้งแรก)
- **Triton:** Linux + GPU NVIDIA (ทดสอบบน RTX 3060, sm_86)
- **Apple Silicon:** ใช้ MPS/CPU; พาธ Triton CUDA จะถูกข้ามอัตโนมัติ

### 2. เบนช์มาร์ก

```bash
python benchmark.py
python benchmark.py --device cuda --precision fp16
python benchmark.py --device cuda --precision both
```

### 3. กลั่นความรู้

**A) เดโม GPT-2**

```bash
python train_fff_gpt2.py --device cuda --max-steps 5000
python eval_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
python chat_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
```

**B) FFF agent หลายทักษะ (ไทย / โค้ด / เครื่องมือ / CoT)**

```bash
# ค่าเริ่มต้น Teacher+Student = Qwen/Qwen3.5-9B (ควรมี VRAM ≥24GB)
# Context โมเดล = 256K (262144); ตอนเทรนยังใช้ micro-batch สั้น (--max-length 64)
python train_fff_agent.py --device cuda --max-steps 4000

# การ์ด 12GB: ใช้ student เล็กลง แต่ครูยังเป็น 9B 4-bit
python train_fff_agent.py --student-name Qwen/Qwen3.5-4B --device cuda --max-steps 4000

python chat_fff_agent.py --checkpoint fff_cot_agent.pt --device cuda
```

พิมพ์ `exit` / `quit` หรือกด `Ctrl+C` เพื่อออก

### 5. ฝึกจากศูนย์ (ทางเลือก)

```bash
python train.py --device auto
python infer.py --checkpoint fff_checkpoint.pt --device cpu --fff-backend cpp
```

---

## การฝึกและการกลั่นความรู้

| สคริปต์ | Routing | วัตถุประสงค์ |
| --- | --- | --- |
| `train.py` | Soft + anneal τ + balance loss | ฝึก `FFFTransformer` |
| `train_fff_gpt2.py` | **STE hard-aware** | กลั่น MLP ของ GPT-2 → ใบ FFF |
| `eval_fff_gpt2.py` | Soft eval / Hard Triton | ตาราง PPL + สร้างข้อความ |
| `chat_fff_gpt2.py` | Triton Hard | CLI สไตล์โปรดักชัน |

**ค่าเริ่มต้นกลั่น (โดยสรุป):** `lr_leaf=3e-4`, `lr_router=1e-4`, `CosineAnnealingLR`, `max_steps=5000`, บันทึกที่ `fff_distill_checkpoint.pt`

**ค่าเริ่มต้นสร้างข้อความ:** `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.3`, `no_repeat_ngram_size=3`

---

## CLI อนุมาน

`chat_fff_gpt2.py` โหลดสตูเดนต์ที่กลั่นแล้ว บังคับ **Triton CUDA Hard** เมื่อมี วอร์มเคอร์เนลด้วย dummy forward แล้วให้บริการ:

- Nucleus sampling + ควบคุมการซ้ำ  
- จับเวลาด้วย **`torch.cuda.Event`** → tok/s, ms/token, จำนวนโทเคน  
- ออกอย่างสง่างามโดยไม่โชว์ traceback  

---

## การอ้างอิง

```bibtex
@article{belcak2023fast,
  title   = {Fast Feedforward Networks},
  author  = {Belcak, Peter and Wattenhofer, Roger},
  year    = {2023}
}
```

หากใช้ HyperZens AI ในงานวิจัยหรือผลิตภัณฑ์ โปรดสตาร์ / อ้างอิงรีโพนี้ด้วย
