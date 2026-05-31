# 🧭 LLM Crash Course — Bird's-Eye Visual Map

> One-page mental model of the 8-hour "Build Your Own LLM" workshop.  
> Skim top-to-bottom. Each section answers: **what is it, why does it exist, what's the one thing to remember.**  
> [go.justinangel.ai/neuralnet](go.justinangel.ai/neuralnet)

---

## 🗺️ The Whole Map

```
                    ┌─────────────────────────────────────────┐
                    │        BUILDING AN LLM (6 domains)      │
                    └─────────────────────────────────────────┘
                                     │
   ┌────────────┬──────────────┬─────┴──────┬──────────────┬──────────────┐
   ▼            ▼              ▼            ▼              ▼              ▼
 1.NN        2.TRAINING    3.STABILITY  4.TEXT→NUM     5.ATTENTION    6.LIFECYCLE
 BASICS      MECHANICS     TRICKS       PIPELINE       /TRANSFORMER   PRE→FT→RL
   │            │              │            │              │              │
 perceptron   loss          init         tokenizer       Q/K/V         pretrain
 activation   backprop      normaliz.    embedding       multi-head    fine-tune
 GPU          optimizer     residual     pos-encoding    GQA/MQA       RLHF
              LR/batch      regulariz.   softmax         transformer   eval/sample
```

**Mental model:** A perceptron stacked into a network (1) is *trained* by chasing loss downhill (2). Deep nets only train if you stabilize them (3). Text becomes numbers (4), attention mixes those numbers (5), and the full lifecycle is pretrain → fine-tune → align → evaluate (6).

---

## 1️⃣ Neural Network Foundations

### Perceptron — the atom

```
    inputs        weights       bias    activation
    x1 ──w1──┐
    x2 ──w2──┼──► Σ (wᵢxᵢ) + b ──► f(·) ──► output
    x3 ──w3──┘
```
A perceptron = weighted sum + nonlinear squish. Stack them → MLP. Stack MLPs with attention → transformer.

### MLP Anatomy — Hidden vs. Activation Layers

A **Multi-Layer Perceptron** = input layer + ≥1 *hidden* layer + output layer, with *activation* applied between them.

```
   Input layer        Hidden layer 1       Hidden layer 2        Output layer
   (the data)         (linear + act)       (linear + act)        (linear [+ act])

   x₁ ●───┐           ●───┐                ●───┐                  ●  ŷ₁
   x₂ ●───┼─►Σ+b─►f─► ●───┼─►Σ+b─►f─►      ●───┼─►Σ+b────►        ●  ŷ₂
   x₃ ●───┘           ●───┘                ●───┘                  
        W₁ matrix         W₂ matrix             W₃ matrix
        (input→h1)        (h1→h2)               (h2→output)
```

#### "Hidden layer" vs "Activation layer" — same thing or not?

People use the terms loosely. Strictly:

| Term | What it actually means |
|---|---|
| **Hidden layer** | A layer of neurons that's neither input nor output. Each neuron does `Σ(wᵢxᵢ) + b` (the **linear** / **dense** / **fully-connected** transformation). Called "hidden" because users only see input and output. |
| **Activation layer** | The nonlinear function `f(·)` applied **after** a hidden layer's linear transform. Sometimes broken out as a separate "layer" in framework code (e.g. `nn.ReLU()`), but mathematically it's just the squish step inside the hidden layer. |
| **Hidden units / neurons / width** | How many neurons are in one hidden layer (e.g. "768 hidden units"). |
| **Depth** | How many hidden layers are stacked. |

In PyTorch you literally write them as separate modules:
```python
nn.Linear(in, hidden)   # ← "hidden layer" (the dense transform)
nn.ReLU()               # ← "activation layer" (the nonlinearity)
nn.Linear(hidden, out)  # ← next hidden/output layer
```

> **Plain English:** "Hidden layer" usually refers to the *whole sandwich* (linear + activation). "Activation layer" zooms in on just the nonlinear slice.

#### Dimensional flow — how shape changes through an MLP

This trips people up. Watch the dimensions carefully:

```
   input vector     W₁: [4 × 8]      W₂: [8 × 8]       W₃: [8 × 2]
   x ∈ ℝ⁴           bias b₁: [8]     bias b₂: [8]      bias b₃: [2]
                                                              
   shape [4]   ──►  shape [8]   ──►  shape [8]    ──►  shape [2]
                    (h1 = ReLU(xW₁+b₁))             (ŷ = h₂W₃+b₃)
   ▲                ▲                               ▲
   │                │                               │
   "input dim       "hidden dim"                    "output dim"
    = features"      (a hyperparameter you choose)   (= classes / next layer)
```

**Key rule:** activations don't change shape — they apply elementwise.
**Linear layers do** — `nn.Linear(in_features, out_features)` reshapes `[..., in]` → `[..., out]`.

For a batch of B examples:
```
   X         W₁         X·W₁+b₁     ReLU       output
   [B, 4]  · [4, 8]  =  [B, 8]   →  [B, 8]   →  ...
```
The leading batch dim rides through unchanged. Only the **last dim** (features) gets transformed.

#### Worked numerical example

Input vector `x = [1, 2, 3, 4]`, hidden dim 8, output dim 2:

```
   x: [4]  ─► Linear(4→8) ─► z₁: [8]  ─► ReLU ─► h₁: [8]
                                                  │
                                                  ▼
                          Linear(8→8) ─► z₂: [8]  ─► ReLU ─► h₂: [8]
                                                              │
                                                              ▼
                                              Linear(8→2) ─► ŷ: [2]
```
Param count: `(4·8+8) + (8·8+8) + (8·2+2) = 40 + 72 + 18 = 130`. Most params live in the wide middle.

### Do SOTA LLMs still have hidden + activation layers?

**Yes — they're everywhere, just renamed and reshaped.** A modern transformer is mostly hidden + activation layers wired up cleverly. The two main places:

| Where in transformer | What it is | Modern shape |
|---|---|---|
| **FFN (feed-forward network) inside each block** | Two linear layers with an activation between them — literally a 2-layer MLP per token | Llama: `Linear(d→4d) → SwiGLU → Linear(4d→d)` |
| **Q/K/V/output projections in attention** | Linear layers (no activation between) | `Linear(d→d)` each |
| **LM head** | Final linear projection to vocab | `Linear(d→vocab)` |

The FFN is **2/3 of every transformer block's parameters**. So when people say "LLMs are mostly MLPs glued together with attention," this is what they mean.

See [§5 Transformer block](#5️⃣-attention--transformer) and the FFN dimensional walkthrough below for how SOTA-specific shapes (the `4×` expansion, SwiGLU's gated form) build on this base anatomy.

### 🧮 Workshop example: `A * B % C` — teaching an MLP modular multiplication

A canonical "first MLP" exercise: train a small dense network to compute **(A × B) mod C** from raw numerical inputs. It's the simplest task that's *just* nonlinear enough to require a real network — perfect for seeing every concept (forward, loss, backprop, hidden layers) in action.

#### The task

```
   inputs:  A, B, C  (e.g. A=7, B=8, C=5)
   target:  (A · B) % C  =  56 % 5  =  1
   model:   "given A, B, C → predict the remainder"
```

Why it's a good teaching example:
- **Nonlinear** (modulo wraps around) → linear regression can't solve it.
- **Tiny** → trains in seconds on a laptop.
- **Verifiable** → ground truth is exact arithmetic.
- **Bridges to LLMs** → same forward/backward/optimizer loop, just smaller.

#### Architecture (`20 → 100 → 100 → 1`)

```
   ┌───────────────┐    ┌───────────────┐   ┌───────────────┐   ┌──────┐
   │ INPUT LAYER   │    │ HIDDEN 1      │   │ HIDDEN 2      │   │ OUT  │
   │ 20 features   │    │ 100 neurons   │   │ 100 neurons   │   │ 1    │
   │ (encoded A,B,C)│   │ + ReLU        │   │ + ReLU        │   │ value│
   └───────────────┘    └───────────────┘   └───────────────┘   └──────┘
        [B, 20]   ──►        [B, 100]    ──►   [B, 100]      ──►  [B, 1]
                W₁: [20×100]            W₂: [100×100]      W₃: [100×1]
                +b₁: [100]              +b₂: [100]         +b₃: [1]
```

Why **20 inputs** for only 3 numbers? Because raw integers are bad input — they're typically **one-hot or binary-encoded** so the net can pick up bit-patterns. E.g. each of A, B, C as a 7-bit binary vector (≈21 dims), or as one-hot over a small range. The exact encoding is a hyperparameter; **20** is just the workshop's chosen width.

#### Dimensional walk-through

```
   batch x         W₁          z₁         ReLU       h₁          W₂          z₂        ReLU       h₂          W₃          ŷ
   [B, 20]    ·   [20,100]  =  [B,100]   ────►       [B,100]  ·  [100,100] = [B,100]   ────►      [B,100]  ·  [100,1]  = [B,1]
                  +b₁[100]                                       +b₂[100]                                     +b₃[1]
```

Param count:
- W₁: `20·100 + 100 = 2,100`
- W₂: `100·100 + 100 = 10,100`  ← bulk of the model
- W₃: `100·1 + 1 = 101`
- **Total: ~12,300 params** — a toy compared to GPT-2's 124M.

#### How it trains (full loop reuses every concept so far)

```
   ┌─ generate (A,B,C) random batch                       ← data
   │
   │  forward:  x → Linear → ReLU → Linear → ReLU → Linear → ŷ
   │  loss:     MSE(ŷ, (A·B) % C)                          ← § Loss Functions
   │  backward: ∂loss/∂W via chain rule                    ← § Backprop
   │  step:     W ← W - lr · grad   (AdamW)                ← § Optimizer
   │
   └─ repeat ~10K steps until loss plateaus
```

Hyperparameters typical for the demo: `lr=1e-3`, `batch=256`, ReLU, He init.

#### What this exercise teaches you (and how it maps to LLMs)

| Workshop concept | Where it shows up here | LLM equivalent |
|---|---|---|
| Hidden layer | the two `Linear(100)` blocks | FFN's up-projection |
| Activation | ReLU between layers | SwiGLU in Llama FFN |
| Loss | MSE for regression | Cross-entropy for next-token |
| Backprop | autograd through 3 matmuls | autograd through 70+ layers |
| Optimizer | AdamW | AdamW (same!) |
| Init | He (matches ReLU) | He / Xavier variants |
| Param count | ~12K | 7B+ |

> **The lesson:** an LLM is *exactly* this loop, scaled up. Same forward/loss/backward/step, just with **embeddings + attention + many more layers** in the middle. If you understand `A*B%C`, you understand the skeleton of GPT.

### Activation Functions — the "squish"

#### Where they fit in the perceptron

```
   inputs       weights      bias        ⭐ activation here ⭐
   x1 ──w1──┐
   x2 ──w2──┼──► Σ (wᵢxᵢ) + b ──► f(·) ──► output
   x3 ──w3──┘   └─── linear ───┘  └─nonlinear─┘
```

The weighted sum is **linear**. Stack 100 linear layers and you still get… a single linear layer (math collapses). The activation function is the **nonlinearity** that makes deep networks more expressive than shallow ones. Without it, no matter how many layers you stack, the model can only draw straight-line decision boundaries.

> **Key insight:** activations are what turn neural nets from glorified linear regression into universal function approximators.

#### The original use case: binary classification

The earliest perceptron (1958) used a **step function** — fire (1) or don't (0):

```
   step(x):    0 ────┐
                     │
                     └──── 1
                     ↑
                  threshold

   "Is this email spam?"  Σ(weights·features) > 0 → 1 (yes), else → 0 (no)
```

Problem: step has zero gradient everywhere → backprop can't learn. So we replaced it with **smooth** S-curves (sigmoid, tanh) that approximate a step but are differentiable. Then ReLU showed up and changed everything.

#### The four shapes you need to recognize

```
   Sigmoid σ(x)         Tanh                  ReLU                 SwiGLU
   range: 0→1           range: -1→1           range: 0→∞           gated ReLU
                                                                    
        1 ─────             1 ─────                  ╱                 ╱
       ╱                   ╱                       ╱                 ╱
      ╱                   ╱                      ╱                  ╱
   ──╱                ──╱                     ──╱──             ──╱─·gate
      0 (smooth)        -1 (zero-centered)       0 (kills neg)      smooth+selective
```

#### Detailed comparison

| Fn | Formula | Range | Use today | Why it failed / wins |
|---|---|---|---|---|
| **Step** | `1 if x>0 else 0` | {0,1} | 🪦 historical | zero gradient → can't train |
| **Sigmoid** | `1/(1+e⁻ˣ)` | (0,1) | classifier final layer, gates | saturates → vanishing gradient in deep nets |
| **Tanh** | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | (-1,1) | RNN era (LSTM gates) | zero-centered (better than sigmoid) but still vanishes |
| **ReLU** | `max(0, x)` | [0,∞) | default MLPs, CNNs | cheap, sparse, no vanishing on positive side. **Killed sigmoid.** |
| **GELU** | `x·Φ(x)` (Gaussian) | ≈ReLU smoothed | **GPT-2, BERT** | smoother than ReLU → better gradients |
| **SwiGLU** | `Swish(xW₁) ⊙ (xW₂)` | gated | **Llama, modern LLMs** ✅ | learnable gate decides what passes |
| **ReLU²** | `max(0,x)²` | [0,∞) | research | smoother gradient, fewer dead neurons |

#### How activation choice improves LLM performance

| Era | Activation | Result |
|---|---|---|
| 1980s | Sigmoid/Tanh | nets >3 layers wouldn't train (vanishing gradients) |
| 2012 (AlexNet) | **ReLU** | unlocked deep CNNs → ImageNet revolution |
| 2018 (GPT-2/BERT) | **GELU** | smoother than ReLU → better LLM convergence |
| 2023+ (Llama) | **SwiGLU** | ~1-2% better perplexity at same compute |

The pattern: each new activation **lets gradients flow further through deeper stacks**, so we can build bigger, better models without training collapsing.

#### Why ReLU killed sigmoid in practice

```
   Sigmoid derivative:  max ≈ 0.25  ── multiply 30 of these → 0.25³⁰ ≈ 10⁻¹⁸  💀
   ReLU derivative:     0 or 1     ── multiply 30 of these → still 1            ✅
```

For a 70-layer LLM, sigmoid would zero out gradients before they reach the bottom layers. ReLU passes them through cleanly on the active side.

#### Where each lives in a modern LLM

```
   ┌────────────────── Transformer block ──────────────────┐
   │                                                       │
   │   Attention:       softmax (a sigmoid cousin)         │  ← probability
   │   FFN inner:       SwiGLU / GELU                      │  ← main nonlinearity
   │   Output head:     softmax over vocab                 │  ← probability
   │                                                       │
   └───────────────────────────────────────────────────────┘
```

Sigmoid/tanh aren't gone — they live inside softmax and gating mechanisms. ReLU's family (GELU, SwiGLU) does the heavy lifting in the FFN.

> **Keep this:** Activations = the nonlinearity that makes "deep" matter. Modern LLMs use **SwiGLU** in FFNs because its smooth, gated shape moves gradients cleanly through 70+ layers — every percent of perplexity comes from details like this.

### GPU Performance
`torch.compile` → **CUDA kernels** → **Triton** (Python-ish kernel DSL).
Rule of thumb: matrix multiplies dominate; fuse ops; mind memory bandwidth.

---

## 2️⃣ Training Mechanics — how it learns

### The Loop
```
   ┌──► forward pass ──► prediction ──► loss ──┐
   │                                            │
   │                                            ▼
   │                                     compute gradients
   │                                            │
   └─── update weights ◄── optimizer ◄──────────┘
                  (SGD, AdamW, Muon...)
```

### Loss Functions
- **Cross-Entropy** — for classification / next-token (LLM default)
- **MSE / RMSE** — for regression
- **Residual** — raw difference

### 🔁 Backpropagation — the heart of training

**What it is:** the algorithm that figures out *how much each weight contributed to the loss*, so we know which direction to nudge it. It's the chain rule from calculus, applied mechanically over a graph of operations.

#### The two-pass dance

```
   ┌────────────── FORWARD PASS ──────────────┐
   │ inputs ─► layer1 ─► layer2 ─► ... ─► loss│   compute prediction & loss
   └──────────────────────────────────────────┘
   ┌────────────── BACKWARD PASS ─────────────┐
   │ grad ◄─ layer1 ◄─ layer2 ◄─ ... ◄─ loss  │   propagate ∂loss/∂w to every weight
   └──────────────────────────────────────────┘
              then: w ← w - lr · ∂loss/∂w
```

#### Why we need the chain rule

A deep net is just nested functions: `loss = L(f₃(f₂(f₁(x, w₁), w₂), w₃))`.
To know how `w₁` (way at the bottom) affects `loss` (way at the top), multiply the local derivatives along the path:

```
   ∂loss     ∂loss     ∂f₃     ∂f₂     ∂f₁
  ────── = ────── · ────── · ────── · ──────
   ∂w₁      ∂f₃      ∂f₂     ∂f₁      ∂w₁
```

Each layer only needs to know **its own local derivative** + the gradient flowing in from above. That locality is what makes backprop scale to billions of params.

#### Tiny worked example (one neuron)

```
   x=2 ──[w=3]──► z=6 ──[ReLU]──► a=6 ──[loss=(a-y)²]──► L=16     (target y=2)
```

Forward: `z = w·x = 6`, `a = ReLU(z) = 6`, `L = (a-y)² = 16`.

Backward (just multiply local derivatives):
```
   ∂L/∂a = 2(a-y)        = 8         ← loss layer
   ∂a/∂z = 1 (z>0, ReLU) = 1         ← activation
   ∂z/∂w = x             = 2         ← linear

   ∂L/∂w = 8 · 1 · 2 = 16            ← chain them
```
Update: `w ← 3 - 0.01·16 = 2.84`. Loss next step is lower. Done.

#### Computational graph (what PyTorch builds)

Every tensor op gets recorded as a node. `loss.backward()` walks the graph in reverse, applying each op's known derivative rule. You write forward code; autograd writes the backward pass for you.

```
       x       w
        \     /
         (mul)         ← node remembers "I'm a multiply"
           │
           z
           │
        (ReLU)
           │
           a    y
            \  /
           (sub)
             │
           (sqr)
             │
           loss ──► .backward() walks UP, multiplying gradients
```

#### The four things that can go wrong

| Problem | Cause | Fix |
|---|---|---|
| **Vanishing gradients** | many <1 derivatives multiplied → ~0 deep down | ReLU + He init + residuals + RMSNorm |
| **Exploding gradients** | many >1 derivatives multiplied → ∞ | gradient clipping (max norm 1.0) |
| **Dead neurons** | ReLU stuck at 0 → derivative 0 → no update ever | He init, lower LR, leaky/SwiGLU variants |
| **Gradient leak in long seqs** | softmax/attention saturates | scaled dot-product (÷√d), pre-norm |

#### How it connects to everything else

- **Loss function** decides what gradient enters the graph at the top.
- **Optimizer** (AdamW/Muon) decides *how* to use the gradient (momentum, adaptive LR, weight decay).
- **Init** + **norm** + **residuals** exist precisely to keep these multiplied gradients well-behaved across 70+ layers.
- **Batch size** averages gradients over examples → less noise per step.
- **Backprop runs once per training step**; for a frontier LLM that's millions of steps over trillions of tokens.

> **One-liner to keep:** Forward computes loss, backward computes ∂loss/∂each-weight via chain rule on the recorded graph, optimizer takes a small step against that gradient. Repeat ~10⁷ times. That's "training."

### Optimizer Family Tree
```
SGD ── + momentum ── + adaptive LR (RMSProp/AdaGrad)
                              │
                              └──► Adam ──► AdamW ──► Muon (newest)
```
**Default:** AdamW. Modern frontier: Muon.

### Hyperparameters that matter most
| Knob | Effect |
|---|---|
| Learning Rate | Too high = explode, too low = stuck. **Use warmup + decay** |
| Batch Size | Bigger = stabler gradients, more memory |
| Warmup/Decay | Ramp LR up, then cosine decay down |

### Loss Landscape
```
   ╱╲      ╱╲           ← local minima trap SGD
  ╱  ╲    ╱  ╲   ╱╲
 ╱    ╲__╱    ╲_╱  ╲___ ← global minimum (rarely reached)
```

---

## 3️⃣ Stability Tricks — why deep nets even work

These four tricks are what let you stack 72 layers without collapse.

```
┌─────────────────────────────────────────────┐
│  STABILITY = Init + Norm + Residuals + Reg  │
└─────────────────────────────────────────────┘
```

### 🎲 Initialization
| Activation | Use this init | Year |
|---|---|---|
| ReLU | **He / Kaiming** — `stdDev = √(2/n_in)` | 2015 |
| Tanh/Sigmoid | **Xavier / Glorot** — `stdDev = √(2/(n_in+n_out))` | 2010 |

> Wrong init → activations explode 💥 or collapse to zero 🕳️ over many layers.

Distributions cheat sheet:
- **Uniform**: flat — equal chance everywhere
- **Normal**: bell — 68/95/99.7 within 1/2/3 SD

### 🧮 Normalization

```
   x ──► (x - mean) / √(var + ε) ──► γx + β
         ↑                          ↑
    center+scale            learned shift+scale
```

| Type | Normalizes across | Used in |
|---|---|---|
| **BatchNorm** | batch dim | CNNs |
| **LayerNorm** | feature dim per token | original Transformer |
| **RMSNorm** | scales only (no centering) | **modern LLMs** ✅ |

**Pre-norm** (norm before sublayer) > **Post-norm** for deep stacks.
Modern recipe: **Pre-RMSNorm**.

### 🔗 Residuals
```
   x ──┬──► [sublayer] ──► +  ──► out
       └─────── skip ──────┘
```
Adds input back to output. Smooths loss landscape, lets gradients flow → enables 72+ layers.
Variants: additive, scaled, gated, concat, attention-based.

### 🛡️ Regularization
| Technique | What | LLM today? |
|---|---|---|
| **Dropout** (10%) | randomly zero neurons | ❌ rare (single-pass training) |
| **Gradient Clip** (max 1.0) | cap gradient norm | ✅ yes |
| **Weight Decay** (10%) | penalize big weights | ✅ yes (in AdamW) |

---

## 4️⃣ Text → Numbers Pipeline

```
"Hello world"
     │
     ▼   tokenizer (BPE)
[15496, 995]
     │
     ▼   embedding lookup (vocab × 768)
[[0.12, -0.4, ...],   ← 768-dim vector per token
 [0.88,  0.1, ...]]
     │
     ▼   + positional encoding
context-aware vectors → into transformer
```

### 🔤 Tokenizers
| Type | Granularity | Vocab | Use |
|---|---|---|---|
| Character | per char | tiny | inefficient |
| Word | per word | huge | OOV problems |
| **BPE** | subword (merge frequent pairs) | ~50K | **GPT-2/3/4** ✅ |
| WordPiece | subword (max likelihood) | ~30K | BERT |
| SentencePiece | subword + spaces | ~32K | multilingual |

GPT-2 vocab: **50,257** tokens. `tiktoken` is OpenAI's library.

### 🧬 Embeddings
- Each token → vector in **768-dim** (GPT-2) up to 12K+ (frontier).
- **PCA** lets you visualize ~semantic clusters in 2D.
- **Tied embeddings**: input embedding matrix == output (unembedding) matrix.

### 📍 Positional Encoding
| Type | How |
|---|---|
| **Absolute Sinusoidal** | fixed sin/cos per position (original Transformer) |
| **RoPE** | rotates Q/K vectors — encodes *relative* position (modern LLMs) ✅ |

### 🎰 SoftMax — logits → probabilities
```
   logits        exp(x)          softmax (sum=1)
   [2.0]   ──►   [7.39]   ──►   [0.66]  ◄── winner
   [1.0]         [2.72]         [0.24]
   [0.1]         [1.10]         [0.10]
```
`softmax(x)ᵢ = exp(xᵢ) / Σ exp(xⱼ)` — winner-take-most.

---

## 5️⃣ Attention & Transformer

### 🔑 Q, K, V — the three "lenses" on each token

Before the attention block makes sense, you need to know what Q/K/V actually *are*. They are three **different views of the same input**, produced by three small linear layers.

#### The library card-catalog analogy

Imagine each token in the sentence is a person at a library:

| Lens | Role | Plain English |
|---|---|---|
| **Q (Query)** | "What am I looking for?" | The thing this token wants to find in others |
| **K (Key)** | "What am I about?" | A label describing this token, used for matching |
| **V (Value)** | "What do I actually contain?" | The content this token will share if attended to |

Attention then asks: **"For each token's Query, which other tokens' Keys match best? Pull their Values."**

```
   token i's Query  ──► compared against ──► every token's Key  ──► gives weights
                                                                        │
                                                                        ▼
                                                    weighted sum of Values
                                                                        │
                                                                        ▼
                                                    new representation for token i
```

#### Where Q, K, V come from — three linear projections

Same input vector `x`, three different learned weight matrices:

```
                    Input X (one row per token)
                    shape: [seq_len, d_model]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         X · W_Q         X · W_K         X · W_V       ← three Linear layers
         [d×d]           [d×d]           [d×d]            (each is just nn.Linear)
              │               │               │
              ▼               ▼               ▼
              Q               K               V
        [seq, d]        [seq, d]        [seq, d]
       "what I want"  "what I am"     "what I share"
```

So when the slide says:
```
   Query = Input · Query_Weights
   Key   = Input · Key_Weights
   Value = Input · Value_Weights
```
that's literally three matrix multiplies — three "hidden layers" in MLP language (linear, no activation).

#### Concrete dimensional example (GPT-2 small, single head)

```
   X: [seq_len=10, d_model=768]
   W_Q, W_K, W_V: each [768, 768]    ← learned parameters

   Q = X · W_Q   →  [10, 768]
   K = X · W_K   →  [10, 768]
   V = X · W_V   →  [10, 768]
```

For **multi-head**, that 768 gets split into `n_heads=12` heads of `d_head=64` each, run in parallel, then concatenated.

#### Why three different matrices, not one?

Each weight matrix is **learned to extract a different aspect** of the same token:
- `W_Q` learns "what does this token need from context?"
- `W_K` learns "what's worth advertising about this token?"
- `W_V` learns "what info does this token carry?"

If they were the same matrix, every token's "what I want" would equal "what I am" — the model couldn't ask different questions than it answers.

#### Tiny worked example

```
   "the cat sat"  →  3 tokens, each becomes a vector of size d=4 (toy)
   X = [[1, 0, 1, 0],   ← "the"
        [0, 2, 0, 1],   ← "cat"
        [1, 1, 0, 0]]   ← "sat"

   W_Q = [[1,0,0,1],    Q = X·W_Q = [[1,0,1,1],
          [0,1,1,0],                  [0,2,2,1],
          [0,0,1,0],                  [1,1,1,1]]
          [1,0,0,1]]

   ... similarly K and V via their own W matrices
```

Q, K, V each end up as `[3 tokens × 4 dims]` — same shape as input, but each row now means something different.

#### How Q, K, V get used (preview of next subsection)

```
   scores = Q · Kᵀ            ← every Query dotted with every Key  → [seq, seq]
   weights = softmax(scores / √d_head)
   output = weights · V       ← weighted blend of Values
```
That `Q·Kᵀ` matrix is the famous "attention pattern" you see visualized as a heatmap.

> **Keep this:** Q, K, V are three learned linear projections of the same input. Q says what each token *wants*, K says what each token *offers*, V is what gets *shared*. Attention is matching Q against K, then pulling V accordingly.

### The Attention block
```
                Input X
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       Q=XWq    K=XWk    V=XWv
        │         │         │
        └─► Q·Kᵀ ◄┘         │
              │             │
              ▼             │
         scale + mask        │
         (causal: hide       │
          future tokens)     │
              │             │
              ▼             │
           softmax           │
              │             │
              └─────► × V ◄─┘
                      │
                      ▼
                  output
```

### Multi-Head Causal Attention
Run N attention "heads" in parallel → concat → project.
- Different heads learn different things (punctuation, prev-word, syntax, etc.)

### Variants (KV-cache savings)
| | Q heads | K/V heads | Used by |
|---|---|---|---|
| **MHA** | N | N | original |
| **GQA** | N | N/g (grouped) | Llama-2/3 |
| **MQA** | N | 1 | PaLM |
| **MLA** | N | low-rank latent | DeepSeek |

### Full Transformer Block (×12 in GPT-2, ×80+ in frontier)
```
   ┌─────────────────────────────┐
   │   x                         │
   │   │                         │
   │   ├─►RMSNorm─►MultiHeadAttn │
   │   │                  │      │
   │   ◄──────────────────┘ (residual)
   │   │                         │
   │   ├─►RMSNorm─►FFN(SwiGLU)   │
   │   │                  │      │
   │   ◄──────────────────┘ (residual)
   │   │                         │
   │   ▼ (to next layer)         │
   └─────────────────────────────┘
```

End of stack: `RMSNorm → unembedding → softmax → next-token probs`.

### 🥪 Inside the FFN — where SOTA "hidden layers" actually live

The FFN inside every transformer block is **a 2-layer MLP applied per token**. This is where most of the model's parameters and "knowledge" live.

```
   per-token vector
   x: [d_model]                                                  
       │                                                         
       ▼   W_up  [d_model × d_ff]            ← ⭐ HIDDEN LAYER 1 ⭐
   h₁: [d_ff]   (typically 4× wider!)        ← "expansion"
       │                                                         
       ▼   activation (GELU / SwiGLU)        ← ⭐ ACTIVATION ⭐
   h₂: [d_ff]                                                    
       │                                                         
       ▼   W_down  [d_ff × d_model]          ← HIDDEN LAYER 2
   out: [d_model]   (back to original size)  ← "projection"
```

#### Dimensional caveat — the **expand-then-project** pattern

| Layer | Real name | Input dim | Output dim |
|---|---|---|---|
| **GPT-2 small** | FFN up-proj | 768 | **3072** (4×) |
| | activation (GELU) | 3072 | 3072 (no change) |
| | FFN down-proj | 3072 | 768 |
| **Llama-2 7B** | FFN up-proj (SwiGLU) | 4096 | **11008** (~2.7× w/ gate) |
| | activation (SwiGLU) | 11008 | 11008 |
| | FFN down-proj | 11008 | 4096 |

**Why expand to 4×?** More hidden width = more "scratch space" for the model to compute features in, before compressing back to the residual stream's dim. Empirically tuned — 4× is the sweet spot for GPT/BERT, ~2.7× for SwiGLU (because the gate doubles the params for free).

#### SwiGLU twist (why modern FFNs have *3* matrices, not 2)

Vanilla FFN: `out = (ReLU(x · W_up)) · W_down`  → 2 matrices.

SwiGLU FFN (Llama, modern LLMs):
```
   out = ( Swish(x · W_gate)  ⊙  (x · W_up) ) · W_down
                ↑                    ↑
           "what to keep"      "what to pass"
                  └── elementwise multiply ──┘
```
Three matrices: a gate, an up-projection, and a down-projection. The gate **learns which features to let through** at each token — sharper, more selective than plain ReLU/GELU.

#### Param count check (per layer, GPT-2 small)
- Attention (Q,K,V,O proj): `4 · 768² ≈ 2.36M`
- FFN: `2 · 768 · 3072 ≈ 4.72M`  ← **~2× attention**
- Per block total: ~7M; ×12 blocks = ~85M of the 124M total.

> **Key takeaway:** Every "hidden layer" idea you learned for MLPs maps directly onto modern LLMs — just with `d_model = 768..16384`, a 4× expansion ratio, and SwiGLU instead of ReLU. Cross-ref [§1 MLP Anatomy](#mlp-anatomy--hidden-vs-activation-layers).

---

## 6️⃣ LLM Lifecycle

```
  📚 PRETRAIN ──► 🎯 FINE-TUNE ──► 🤝 RLHF ──► 📊 EVAL ──► 🚀 DEPLOY
   (trillions      (instructions    (align with    (perplexity,
    of tokens,      via LoRA/        human         Lambada,         ┌──► sample
    next-token      QLoRA/full)      preference)   benchmarks)      │    Top-K
    prediction)                                                     │    Top-P
                                                                    │    temperature
                                                                    └──► generate
```

### 📚 Pretraining
- **Objective**: next-token prediction with cross-entropy.
- **Scaling Laws** (Chinchilla): performance ∝ f(data, params, compute). Balance all three.

#### ⚠️ Caveat: Why modern LLMs **don't use dropout** anymore

The workshop's claim is **correct**, and there's a 2025 paper literally titled *"Drop Dropout on Single-Epoch Language Model Pretraining"* confirming it empirically.

```
   Old era (≤ ~2019)              Modern era (Llama-2/3, GPT-3+)
   ────────────────                ──────────────────────────────
   • Small dataset                 • Internet-scale corpus (trillions of tokens)
   • Many epochs (10s–100s)        • ~1 epoch — see each token once
   • Risk: memorize → overfit      • Risk: underfit (data > params)
   • Dropout @ 10% needed          • Dropout HURTS performance ❌
```

**Why dropout helped in the old days:** when you re-show the same training data many times, the network can memorize specific examples. Randomly zeroing 10% of neurons each pass forces it to generalize instead of memorize.

**Why it stopped helping:** with web-scale data and a single epoch, **the model never sees the same token twice**, so there's nothing to memorize. Dropout just adds noise that throws away useful signal — equivalent to training on less data.

| Era | Model | Dropout? |
|---|---|---|
| 2018 | GPT-2 | 10% (everywhere) |
| 2019 | BERT | 10% |
| 2020 | GPT-3 | small / phasing out |
| 2023 | **Llama-2** | **❌ none** |
| 2024+ | **Llama-3, modern frontier** | **❌ none** |

The 2025 ACL paper tested Pythia 160M/1.4B with varying dropout rates on single-epoch pretraining and found **downstream performance is consistently better with dropout = 0** across language modeling, BLiMP, SQuAD, and MNLI. Models without dropout were also easier to model-edit.

**Where dropout still appears:**
- 🎯 **Fine-tuning** (small datasets, multi-epoch) — dropout makes a comeback
- 🔧 **LoRA adapters** — small param count, multi-epoch → dropout helps
- 🧪 **Smaller research models** trained for many epochs

> **Keep this:** Dropout is a *small-data, multi-epoch* tool. Pretraining a frontier LLM is the opposite regime (huge-data, single-epoch), so dropout is gone. Other regularization survives: **gradient clipping** and **weight decay** still used.

### 🎯 Fine-tuning
| Method | What | Cost |
|---|---|---|
| **Full FT** | update all weights | 💰💰💰 |
| **LoRA** | low-rank adapter matrices | 💰 |
| **QLoRA** | LoRA on quantized base | 🪙 |

### 🤝 RLHF (alignment)
```
  prompt ──► model ──► response A
                  └──► response B   ──► human picks ──► reward model ──► PPO/DPO
```
Reward signal teaches the model human preferences.

### 📏 Evaluation
- **Perplexity**: exp(loss) — lower is better
- **Lambada / HellaSwag / MMLU**: benchmark suites
- **Task-specific**: code, math, reasoning

### 🔄 Autoregressive Generation — how LLMs actually "talk"

**Autoregressive** = each output token is predicted from **all previous tokens** (its own past outputs included). The model generates **one token at a time**, feeds it back as input, and repeats. "Auto" = self, "regressive" = predicting from prior values.

```
  step 1:  "The cat"                          ─► model ─► "sat"
  step 2:  "The cat sat"                      ─► model ─► "on"
  step 3:  "The cat sat on"                   ─► model ─► "the"
  step 4:  "The cat sat on the"               ─► model ─► "mat"
  step 5:  "The cat sat on the mat"           ─► model ─► <EOS>  ⏹
           └──────── grows each step ────────┘
```

Mathematically, the model factorizes a sentence's probability as a chain:

```
   P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁..xₙ₋₁)
```

Each forward pass produces `P(next | everything-so-far)`; sampling picks one token; loop.

#### Why it matters

| Concept | Connection |
|---|---|
| **Causal mask in attention** | Forces token *t* to only see tokens *≤ t* — this is what makes the model *autoregressive by construction*, both at training and inference |
| **Next-token prediction loss** | Pretraining objective is literally "predict xₜ from x₍<ₜ₎" for every position in parallel |
| **KV cache** | Since past tokens never change, we cache their K and V tensors — that's why GQA/MQA matter for inference speed |
| **Latency scales with output length** | N output tokens = N sequential forward passes (can't parallelize across time) |
| **Sampling knobs apply per step** | Temperature, top-k, top-p shape the distribution at *each* token, not the whole sequence |

#### Contrast: autoregressive vs. other paradigms
- **Autoregressive (GPT, Llama)** — left-to-right, one token at a time. Great for open-ended generation.
- **Masked (BERT)** — predict missing tokens with full bidirectional context. Great for understanding, not generation.
- **Diffusion / non-autoregressive** — generate all tokens in parallel via iterative denoising. Faster but lower quality (still research).

> **Keep this:** Every chatbot reply you've ever seen is a `for` loop calling the model once per token, each call conditioned on the growing prefix. The "streaming" UX you see is just that loop yielding tokens as they're sampled.

### 🎲 Sampling at inference time *(applied at every autoregressive step)*
| Knob | Effect |
|---|---|
| **Temperature** | <1 = sharper (deterministic), >1 = wilder |
| **Top-K** | sample from K highest-prob tokens |
| **Top-P (nucleus)** | sample from smallest set whose cumulative prob ≥ P |

```
  greedy   ─► always argmax       (boring, repetitive)
  temp=0.7 ─► softens distribution (balanced)
  top-p=0.9─► chops the tail        (diverse but safe)
```

### 💾 Model File Formats
| Format | Notes |
|---|---|
| `.pth` | PyTorch native (⚠️ pickle = unsafe) |
| `.safetensors` | safe, fast (HuggingFace default) ✅ |
| `.onnx` | cross-framework |
| `.gguf` | llama.cpp / local CPU/GPU |
| `.coreml` | Apple devices |

---

## 🔑 Five Things to Remember

1. **A transformer is just**: tokenize → embed → (attention + FFN + residual + norm) × N → softmax.
2. **Deep training works because of 4 tricks**: He/Xavier init, RMSNorm pre-norm, residuals, AdamW + clipping.
3. **Modern stack**: BPE tokenizer, RoPE positions, SwiGLU activation, GQA attention, RMSNorm pre-norm.
4. **Lifecycle**: pretrain (cross-entropy next-token) → fine-tune (LoRA) → align (RLHF/DPO) → eval → sample (top-p + temp).
5. **LLMs are *grown, not built*** — the workshop's "summoning ghosts" framing. We discover behavior empirically via ablation, not derivation.

---

## 🧠 Mind Map (compressed)

```
LLM
├── Architecture
│   ├── Tokenizer (BPE, ~50K vocab)
│   ├── Embeddings (768d) + RoPE
│   ├── Transformer × N
│   │   ├── Pre-RMSNorm
│   │   ├── Multi-Head Attention (Q/K/V, GQA)
│   │   ├── Residuals
│   │   └── FFN (SwiGLU)
│   └── Unembedding + Softmax
├── Training
│   ├── Loss: cross-entropy
│   ├── Backprop (autograd graph)
│   ├── Optimizer: AdamW (+ Muon)
│   ├── LR warmup + cosine decay
│   └── Gradient clip + weight decay
├── Init: He (ReLU) / Xavier (Tanh)
└── Lifecycle
    ├── Pretrain (next-token, scaling laws)
    ├── Fine-tune (LoRA/QLoRA)
    ├── RLHF / DPO
    ├── Eval (perplexity, Lambada, MMLU)
    └── Inference (temp, top-k, top-p)
```
