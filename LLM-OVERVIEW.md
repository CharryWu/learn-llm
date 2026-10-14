<!-- markdownlint-disable MD033 -- collapsible <details> symbol atlases -->
# 🧭 LLM Crash Course — Bird's-Eye Visual Map

> One-page mental model of the 8-hour "Build Your Own LLM" workshop.  
> Skim top-to-bottom. Each section answers: **what is it, why does it exist, what's the one thing to remember.**  
> Workshop: [go.justinangel.ai/neuralnet](https://go.justinangel.ai/neuralnet) · Deep dives: the [zhihu articles](#further-reading--zhihu-deep-dives) in this repo · 中文口语版: [LLM-OVERVIEW-zh.md](LLM-OVERVIEW-zh.md).

---

## 🗺️ The Whole Map

```text
                    ┌─────────────────────────────────────────┐
                    │        BUILDING AN LLM (7 domains)      │
                    └─────────────────────────────────────────┘
                                     │
   ┌──────────┬───────────┬──────────┼──────────┬───────────┬───────────┐
   ▼          ▼           ▼          ▼          ▼           ▼           ▼
 1.NN       2.TRAINING  3.STABILITY 4.TEXT→NUM 5.ATTENTION 6.LIFECYCLE 7.SYSTEMS
 BASICS     MECHANICS   TRICKS      PIPELINE   /TRANSFORMER PRE→FT→RL   GPU/SCALE
   │          │           │          │          │           │           │
 perceptron loss        init       tokenizer  Q/K/V       pretrain    kernels
 MLP        backprop    normaliz.  embedding  causal mask fine-tune   pipeline
 activation optimizer   residual   pos-enc.   multi-head  RLHF/DPO    parallel
 softmax    LR/batch    regulariz. RoPE       FFN/block   eval/sample checkpoint
```

**Mental model:** A perceptron stacked into a network (1) is *trained* by chasing loss downhill (2). Deep nets only train if you stabilize them (3). Text becomes numbers (4), attention mixes those numbers (5), and the full lifecycle is pretrain → fine-tune → align → evaluate → generate (6). Making it fit on real hardware is (7).

### How to read this doc

| If you want to… | Read in this order |
| --- | --- |
| Understand one forward pass end-to-end | [Tokenizers](#tokenizers) → [Embeddings](#embeddings) → [Positional encoding](#positional-encoding--from-one-hot-to-sinusoidal-to-rope) → [Attention](#scaled-dot-product-attention) → [FFN](#inside-the-ffn--where-sota-hidden-layers-live) → [Softmax](#softmax--logits--probabilities) |
| Understand why training works | [Loss](#loss-functions) → [Backprop](#backpropagation--the-heart-of-training) → [Optimizer](#optimizers--sgd--adam--adamw) → [Init](#initialization) → [Norm](#normalization) → [Residuals](#residual-connections) |
| Understand ChatGPT-style training | [Pretraining](#pretraining) → [Fine-tuning](#fine-tuning-sft-and-lora) → [RLHF](#rlhf--reward-model--ppo-and-dpo) → [Sampling](#sampling-at-inference-time) |

Every formula has a collapsed **🔍 Symbol atlas** under it — click to expand a symbol-by-symbol explanation.

Recurring idea worth spotting: **"keep the variance ≈ 1"** shows up in [init](#initialization), [normalization](#normalization), and the [÷√d_k in attention](#why-divide-by-d_k). Same problem, three fixes.

---

## 1️⃣ Neural Network Foundations

### Perceptron — the atom

```text
    inputs        weights       bias    activation
    x1 ──w1──┐
    x2 ──w2──┼──► Σ (wᵢxᵢ) + b ──► f(·) ──► output
    x3 ──w3──┘   └─── linear ───┘  └nonlinear┘
```

$$
y = f\Big(\sum_i w_i x_i + b\Big) = f(\mathbf{w}^\top \mathbf{x} + b)
$$

<details>
<summary>🔍 Symbol atlas — perceptron</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x_i$ | the $i$-th input feature (a pixel, one embedding dimension, …) | the raw evidence the neuron looks at | inside the sum, paired with its own weight: every input gets its own vote |
| $w_i$ | weight of input $i$ (learned) | encodes how much input $i$ matters; sign = for/against | multiplies $x_i$ directly, so it scales exactly that one input |
| $\sum_i$ | sum over all inputs $i$ | merges all weighted votes into one score | wraps $w_i x_i$, so the many inputs collapse into one number *before* any decision |
| $b$ | bias (learned) | shifts the firing threshold; lets the neuron output something even when every $x_i = 0$ | added *outside* the sum: one bias per neuron, not one per input |
| $f$ | activation function | adds the nonlinearity ([why](#why-we-need-them-at-all)) | outermost: applied once to the finished score — it is the "decision" step |
| $\mathbf{w}^\top \mathbf{x}$ | dot product of the weight vector and the input vector (bold = vector) | the same sum, written compactly | $^\top$ turns column $\mathbf{w}$ into a row, so row × column = one number |
| $y$ | the neuron's output | gets passed on to the next layer | left side: the quantity being defined |

**Read it as:** "score the inputs by importance ($w_i x_i$), add them up ($\sum$), shift by a bias ($b$), then decide ($f$)."

</details>

A perceptron = weighted sum + nonlinear squish. Stack them → MLP. Stack MLPs with attention → transformer.

### MLP anatomy — hidden vs. activation layers

A **Multi-Layer Perceptron** = input layer + ≥1 *hidden* layer + output layer, with *activation* applied between them.

```text
   Input layer        Hidden layer 1       Hidden layer 2        Output layer
   (the data)         (linear + act)       (linear + act)        (linear [+ act])

   x₁ ●───┐           ●───┐                ●───┐                  ●  ŷ₁
   x₂ ●───┼─►Σ+b─►f─► ●───┼─►Σ+b─►f─►      ●───┼─►Σ+b────►        ●  ŷ₂
   x₃ ●───┘           ●───┘                ●───┘
        W₁ matrix         W₂ matrix             W₃ matrix
        (input→h1)        (h1→h2)               (h2→output)
```

Every layer is the same two-step recipe — a linear map, then an elementwise nonlinearity:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} \qquad A^{[l]} = g^{[l]}\big(Z^{[l]}\big), \qquad A^{[0]} = x
$$

<details>
<summary>🔍 Symbol atlas — one layer</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $l$ | layer index | says *which* layer's numbers we mean | written as $^{[l]}$ in square brackets so it is not mistaken for a power |
| $A^{[l-1]}$ | output (activations) of the previous layer | the input to this layer | right of $W^{[l]}$; the index $l-1$ chains layer $l$ to the one before it |
| $W^{[l]}$ | weight matrix of layer $l$, shape [neurons in $l$, neurons in $l-1$] | mixes every input into every neuron | left-multiplies $A^{[l-1]}$: row $j$ of $W^{[l]}$ holds neuron $j$'s weights |
| $b^{[l]}$ | bias vector, one entry per neuron | gives each neuron its own threshold | added after the product; same shape as $Z^{[l]}$ |
| $Z^{[l]}$ | pre-activation (the linear result) | the "linear half" of the layer | defined first and stored separately, because backprop needs $g'(Z^{[l]})$ |
| $g^{[l]}$ | activation function of layer $l$ | the nonlinear half | wraps $Z^{[l]}$; applied elementwise *after* the linear step |
| $A^{[l]}$ | output of layer $l$ | becomes the input of layer $l+1$ | left side of the 2nd equation; reappears on the right of the next layer's formula |
| $A^{[0]} = x$ | base case: the raw input | lets the first layer use the same formula | written last as the starting point of the chain |

Code usually stores a batch as rows and writes `x @ W` (row-vector convention). It is the same math, transposed.

</details>

![Three-layer network: each block is W,b then an activation](zhihu/images/9dac34bd5cbb.jpg)

Data flow through a 3-layer net: $x \to Z^{[1]} \to A^{[1]} \to \dots \to A^{[3]} \to$ LOSS. The [backprop](#backpropagation--the-heart-of-training) section walks this same picture backwards. Source: [ResNet note](<zhihu/Transformer学习笔记四: ResNet (残差网络).md>).

#### "Hidden layer" vs "Activation layer" — same thing or not?

People use the terms loosely. Strictly:

| Term | What it actually means |
| --- | --- |
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

```text
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

```text
   X         W₁         X·W₁+b₁     ReLU       output
   [B, 4]  · [4, 8]  =  [B, 8]   →  [B, 8]   →  ...
```

The leading batch dim rides through unchanged. Only the **last dim** (features) gets transformed. (Same rule later for `[batch, seq, d_model]` tensors in a transformer.)

#### Worked numerical example

Input vector `x = [1, 2, 3, 4]`, hidden dim 8, output dim 2:

```text
   x: [4]  ─► Linear(4→8) ─► z₁: [8]  ─► ReLU ─► h₁: [8]
                                                  │
                                                  ▼
                          Linear(8→8) ─► z₂: [8]  ─► ReLU ─► h₂: [8]
                                                              │
                                                              ▼
                                              Linear(8→2) ─► ŷ: [2]
```

Param count: `(4·8+8) + (8·8+8) + (8·2+2) = 40 + 72 + 18 = 130`. Most params live in the wide middle.

> The same anatomy reappears inside every transformer block as the FFN — see [Do SOTA LLMs still have hidden + activation layers?](#do-sota-llms-still-have-hidden--activation-layers).

### Activation functions — the "squish"

#### Why we need them at all

The weighted sum is **linear**. Stack 100 linear layers and you still get… a single linear layer, because matrix products collapse:

$$
W_2 (W_1 x) = (W_2 W_1)\thinspace  x = W' x
$$

<details>
<summary>🔍 Symbol atlas — why stacking linear layers collapses</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | input vector | the thing both layers process | rightmost: matrices act right-to-left, so $x$ is touched first |
| $W_1$ | first layer's weights | the first linear map | directly next to $x$: it acts first |
| $W_2$ | second layer's weights | the second linear map | outside the parentheses: it acts on the result $W_1 x$ |
| $(W_2 W_1)$ | the product of the two matrices | shows we may regroup (associativity) | the regrouping *is* the argument: two steps become one |
| $W'$ | one equivalent matrix | proves the 2-layer net equals a 1-layer net | the final form: a single linear layer |

Insert any nonlinear $g$ between them and the regrouping fails: $W_2\thinspace  g(W_1 x) \ne (W_2 W_1) x$. That is the whole job of the activation.

</details>

The activation function is the **nonlinearity** that makes deep networks more expressive than shallow ones. Without it, no matter how many layers you stack, the model can only draw straight-line decision boundaries.

> **Key insight:** activations are what turn neural nets from glorified linear regression into universal function approximators.

#### The original use case: binary classification

The earliest perceptron (1958) used a **step function** — fire (1) or don't (0):

```text
   step(x):    0 ────┐
                     │
                     └──── 1
                     ↑
                  threshold

   "Is this email spam?"  Σ(weights·features) > 0 → 1 (yes), else → 0 (no)
```

Problem: step has zero gradient everywhere → [backprop](#backpropagation--the-heart-of-training) can't learn. So we replaced it with **smooth** S-curves (sigmoid, tanh) that approximate a step but are differentiable. Then ReLU showed up and changed everything.

#### The shapes you need to recognize

```text
   Sigmoid σ(x)         Tanh                  ReLU                 SwiGLU
   range: 0→1           range: -1→1           range: 0→∞           gated ReLU

        1 ─────             1 ─────                  ╱                 ╱
       ╱                   ╱                       ╱                 ╱
      ╱                   ╱                      ╱                  ╱
   ──╱                ──╱                     ──╱──             ──╱─·gate
      0 (smooth)        -1 (zero-centered)       0 (kills neg)      smooth+selective
```

![Common activation functions and their saturated regions](zhihu/images/70458ac36c28.jpg)

Sigmoid/tanh go flat at both ends — the **saturated regime**, where the gradient ≈ 0. Source: [LayerNorm note](<zhihu/Transformer学习笔记三: 为什么Transformer要用LayerNorm:Batch Normalization & Layer Normalization (批量&层标准化).md>).

#### Detailed comparison

| Fn | Formula | Range | Use today | Why it failed / wins |
| --- | --- | --- | --- | --- |
| **Step** | `1 if x>0 else 0` | {0,1} | 🪦 historical | zero gradient → can't train |
| **Sigmoid** | `1/(1+e⁻ˣ)` | (0,1) | classifier final layer, gates | saturates → vanishing gradient in deep nets |
| **Tanh** | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | (-1,1) | RNN era (LSTM gates) | zero-centered (better than sigmoid) but still vanishes |
| **ReLU** | `max(0, x)` | [0,∞) | default MLPs, CNNs | cheap, sparse, no vanishing on positive side. **Killed sigmoid.** |
| **GELU** | `x·Φ(x)` (Gaussian CDF) | ≈ReLU smoothed | **GPT-2, BERT** | smoother than ReLU → better gradients |
| **SwiGLU** | `Swish(xW₁) ⊙ (xW₂)` | gated | **Llama, modern LLMs** ✅ | learnable gate decides what passes ([details](#swiglu-twist--why-modern-ffns-have-3-matrices)) |
| **ReLU²** | `max(0,x)²` | [0,∞) | research | smoother gradient, fewer dead neurons |

#### Why ReLU killed sigmoid in practice

The derivative of sigmoid is never bigger than a quarter:

$$
\sigma'(x) = \sigma(x)\big(1-\sigma(x)\big) \le \tfrac{1}{4}
$$

<details>
<summary>🔍 Symbol atlas — sigmoid derivative</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | the input to the sigmoid (a pre-activation $Z$) | the point where we measure the slope | argument of both sides |
| $\sigma(x)$ | sigmoid $1/(1+e^{-x})$, always in $(0,1)$ | the activation being differentiated | appears on the right: the slope is expressed via the output itself |
| $\sigma'(x)$ | slope of the sigmoid at $x$ | the factor backprop multiplies by when it passes through this activation | left side: the quantity of interest |
| $1-\sigma(x)$ | distance of the output from its ceiling 1 | makes the slope vanish near the top | multiplied with $\sigma(x)$: near 0 the first factor kills the slope, near 1 the second does |
| $\le \tfrac{1}{4}$ | upper bound on the slope | the best case is still a 4× shrink per layer | $p(1-p)$ peaks at $p = \tfrac12$, giving $\tfrac12 \cdot \tfrac12 = \tfrac14$ |

</details>

[Backprop](#why-gradients-vanish-or-explode) multiplies one such factor per layer:

```text
   Sigmoid derivative:  max ≈ 0.25  ── multiply 30 of these → 0.25³⁰ ≈ 10⁻¹⁸  💀
   ReLU derivative:     0 or 1     ── multiply 30 of these → still 1            ✅
```

For a 70-layer LLM, sigmoid would zero out gradients before they reach the bottom layers. ReLU passes them through cleanly on the active side.

#### How activation choice improves LLM performance

| Era | Activation | Result |
| --- | --- | --- |
| 1980s | Sigmoid/Tanh | nets >3 layers wouldn't train (vanishing gradients) |
| 2012 (AlexNet) | **ReLU** | unlocked deep CNNs → ImageNet revolution |
| 2018–19 (BERT/GPT-2) | **GELU** | smoother than ReLU → better LLM convergence |
| 2023+ (Llama) | **SwiGLU** | ~1-2% better perplexity at same compute |

The pattern: each new activation **lets gradients flow further through deeper stacks**, so we can build bigger, better models without training collapsing.

#### Where each lives in a modern LLM

```text
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

### Softmax — logits → probabilities

Softmax is the "activation" of the **output layer** and of **attention weights**. It turns any vector of scores (logits) into a probability distribution:

$$
\text{softmax}(z) _ i = \frac{e^{z_i}}{\sum _ {j} e^{z_j}}
$$

<details>
<summary>🔍 Symbol atlas — softmax</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $z$ | vector of logits (raw scores), one per class / vocab token | the unnormalized input; any real numbers | inside the function call: softmax looks at the *whole* vector (all classes compete) |
| $i$ | the class we are asking about | picks one output entry | subscript outside the parentheses: input is a vector, output entry $i$ is one number |
| $z_i$ | logit of class $i$ | that class's raw score | in the numerator's exponent: only class $i$'s own score goes on top |
| $e^{z_i}$ | exponential of the logit | makes it positive and stretches gaps (+1 logit = ×2.72 weight) | numerator: the "unnormalized weight" of class $i$ |
| $j$ | a running index over **all** classes | lets the sum visit every class | a different letter from $i$, because $i$ stays fixed while $j$ sweeps |
| $\sum_j e^{z_j}$ | total weight of all classes | normalizer, so the outputs add to 1 | denominator: dividing by the total turns weights into shares |

**Read it as:** "class $i$'s share of the total exponentiated score." Exp (instead of dividing raw scores) handles negative logits, keeps the order, is smooth, and cancels nicely with the $\log$ in [cross-entropy](#loss-functions).

</details>

```text
   logits        exp(x)          softmax (sum=1)
   [2.0]   ──►   [7.39]   ──►   [0.66]  ◄── winner
   [1.0]         [2.72]         [0.24]
   [0.1]         [1.10]         [0.10]
```

Winner-take-most: `exp` makes everything positive and stretches gaps, dividing by the sum makes it add to 1.

Three facts you'll meet again later:

1. **Temperature** — divide logits by $T$ before softmax: $p_i = \dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$. $T\lt 1$ sharpens, $T\gt 1$ flattens. That's the [sampling knob](#sampling-at-inference-time).
2. **Saturation** — the gradient of softmax is $\dfrac{\partial\thinspace \text{softmax}(z) _ i}{\partial z_j} = s_i(\delta _ {ij} - s_j)$.
   If one logit dominates ($s_i \to 1$, others $\to 0$), **every** entry goes to 0 → no gradient. Huge logits are bad. That's exactly why attention [divides by √d_k](#why-divide-by-d_k).
3. **Numerical trick** — implementations compute `softmax(z - max(z))`; the result is identical but `exp` never overflows.

<details>
<summary>🔍 Symbol atlas — temperature and softmax gradient</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $T$ | temperature, $T \gt 0$ | controls how peaked the distribution is | divides every logit *before* $\exp$, scaling all gaps by $1/T$: $T \to 0$ ⇒ argmax, $T \to \infty$ ⇒ uniform |
| $p_i$ | resulting probability of token $i$ | what [sampling](#sampling-at-inference-time) draws from | left side of the temperature formula |
| $s_i$ | shorthand for $\text{softmax}(z) _ i$ | keeps the gradient formula short | outer factor: the gradient is proportional to the probability itself |
| $\dfrac{\partial s_i}{\partial z_j}$ | how output $i$ moves when logit $j$ is nudged | needed for backprop through softmax | a full matrix (Jacobian): every output depends on every logit via the shared denominator |
| $\delta _ {ij}$ | Kronecker delta: 1 if $i=j$, else 0 | separates "my own logit" from "someone else's" | inside the parentheses: $i=j$ gives $s_i(1-s_i) \gt 0$; $i \ne j$ gives $-s_i s_j \lt 0$ |
| $s_j$ | probability of class $j$ | how much class $j$ "steals" from class $i$ | subtracted: the more confident $j$ already is, the more it takes |
| $\max(z)$ | the largest logit | subtracting it avoids `exp` overflow | it cancels: $e^{z_i - c} / \sum_j e^{z_j - c} = e^{z_i}/\sum_j e^{z_j}$ |

**Saturation in one line:** if $s_i \approx 1$ then $(1 - s_i) \approx 0$; if $s_i \approx 0$ the outer $s_i$ is $\approx 0$. Either way the gradient dies.

</details>

Deeper notes: [softmax.md](softmax.md). Paired with [cross-entropy](#loss-functions), the gradient becomes beautifully simple: $p - y$.

---

## 2️⃣ Training Mechanics — how it learns

### The loop

```text
   ┌──► forward pass ──► prediction ──► loss ──┐
   │                                            │
   │                                            ▼
   │                                     compute gradients (backprop)
   │                                            │
   └─── update weights ◄── optimizer ◄──────────┘
                  (SGD, AdamW, Muon...)
```

### Loss functions

The loss is one number that says "how wrong". Training = push it down.

| Loss | Formula | Used for |
| --- | --- | --- |
| **Cross-Entropy** | $\mathcal{L} = -\sum_c y_c \log p_c = -\log p _ {\text{target}}$ | classification, **next-token prediction (LLM default)** |
| **MSE** | $\mathcal{L} = \frac{1}{N}\sum_i (\hat{y} _ i - y_i)^2$ | regression (e.g. the [`A*B%C` demo](#workshop-example-a--b--c--teaching-an-mlp-modular-multiplication)) |
| **RMSE** | $\sqrt{\text{MSE}}$ | regression, same units as $y$ |
| **Residual** | $y - \hat{y}$ | raw difference (building block of the above) |

<details>
<summary>🔍 Symbol atlas — loss table</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\mathcal{L}$ | the loss: one number, lower = better | what training minimizes | left side of every formula |
| $c$ | class index | lets the sum visit every class | under $\sum_c$ |
| $y_c$ | target probability of class $c$ (one-hot: 1 for the right class, 0 otherwise) | the truth | multiplies $\log p_c$, acting as a **selector**: only the true class's term survives |
| $p_c$ | the model's probability for class $c$ ([softmax](#softmax--logits--probabilities) output) | the model's belief | inside the $\log$ |
| $\log$ | natural log | maps $(0,1] \to (-\infty, 0]$ and turns products of probabilities into sums | wraps $p_c$: small $p$ ⇒ very negative $\log p$ ⇒ big loss |
| leading $-$ | sign flip | logs of probabilities are $\le 0$; the minus makes the loss $\ge 0$ | in front of everything |
| $p _ {\text{target}}$ | probability given to the correct class | what's left after the selector | right-most form of cross-entropy |
| $N$ | number of examples | averaging makes the loss independent of batch size | in $\frac{1}{N}$ in front of the sum |
| $\hat{y} _ i$ | prediction for example $i$ (hat = "estimated") | the model's answer | first term of the difference |
| $y_i$ | ground truth for example $i$ | the correct answer | subtracted; squaring makes errors positive and punishes big misses more |
| $\sqrt{\text{MSE}}$ | root of the MSE | brings the error back to $y$'s units | the outer root |

</details>

**Cross-entropy, intuitively:** with a one-hot target, everything cancels except the probability the model gave to the right answer. Loss = "how surprised were you by the truth?"

```text
   p(correct token) = 0.9   → loss = −ln 0.9  ≈ 0.11   (barely surprised)
   p(correct token) = 0.1   → loss = −ln 0.1  ≈ 2.30   (very surprised)
   p(correct token) = 0.001 → loss = −ln 0.001 ≈ 6.9   (shocked)
```

For a language model, average that over every position in the sequence:

$$
\mathcal{L} _ {\text{LM}} = -\frac{1}{T}\sum _ {t=1}^{T} \log P _ \theta(x_t \mid x _ {\lt t})
$$

<details>
<summary>🔍 Symbol atlas — language-model loss</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\mathcal{L} _ {\text{LM}}$ | language-modeling loss | the pretraining objective | left side |
| $T$ | number of predicted positions (**not** temperature) | averaging makes long and short sequences comparable | in $\frac{1}{T}$ and as the upper limit of the sum |
| $t$ | position index | names one prediction | the sum runs $t = 1 \dots T$: **every** position is its own training example |
| $x_t$ | the real token at position $t$ | the "correct class" | left of the bar $\mid$: what is predicted |
| $x _ {\lt t}$ | all tokens before position $t$ | the context | right of the bar ("given"); strictly $\lt$ so the answer is hidden — the [causal mask](#causal-mask--no-peeking-at-the-future) enforces it |
| $P _ \theta(\cdot \mid \cdot)$ | the model's predicted probability | the model's belief about the next token | subscript $\theta$: it depends on the parameters — exactly what training changes |
| $\theta$ | all model parameters | the things gradient descent updates | subscript of $P$ |
| $-\log$ | cross-entropy of one position | the "surprise" at the real token | applied inside the sum, then averaged |

**Read it as:** "on average over positions, how surprised was the model by the real next token, given everything before it?"

</details>

**Why it pairs with softmax:** the gradient of cross-entropy w.r.t. the logits is just

$$
\frac{\partial \mathcal{L}}{\partial z} = p - y
$$

<details>
<summary>🔍 Symbol atlas — softmax + cross-entropy gradient</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\dfrac{\partial \mathcal{L}}{\partial z}$ | gradient of the loss w.r.t. the logits (a vector) | the signal backprop sends into the network from the very top | left side |
| $p$ | predicted probabilities (softmax output) | what the model believed | first term |
| $y$ | one-hot target | what was true | subtracted from $p$ |
| $p - y$ | per class: $p - 1 \lt 0$ for the right class, $p \gt 0$ for the others | descent moves *against* it ⇒ raises the right logit, lowers the rest | the softmax's $\exp$ and the loss's $\log$ cancel, leaving a plain difference |

</details>

"predicted distribution minus target" — push the right token up, everything else down, proportionally. No saturation problem. Exponentiate the loss and you get [perplexity](#evaluation). Deeper notes: [cross-entropy.md](cross-entropy.md), [likelihood.md](likelihood.md) (minimizing cross-entropy = maximizing likelihood).

### Backpropagation — the heart of training

**What it is:** the algorithm that figures out *how much each weight contributed to the loss*, so we know which direction to nudge it. It's the chain rule from calculus, applied mechanically over a graph of operations.

#### The two-pass dance

```text
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

$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial f_3}\cdot\frac{\partial f_3}{\partial f_2}\cdot\frac{\partial f_2}{\partial f_1}\cdot\frac{\partial f_1}{\partial w_1}
$$

<details>
<summary>🔍 Symbol atlas — chain rule</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\dfrac{\partial \mathcal{L}}{\partial w_1}$ | how the loss changes when $w_1$ is nudged | the number the optimizer needs | left side: the goal |
| $f_1, f_2, f_3$ | outputs of layers 1, 2, 3 | intermediate stops on the path from $w_1$ to $\mathcal{L}$ | each is a denominator once and the next numerator once — that hand-off links the factors |
| each fraction | the local derivative of one step | what each layer can compute on its own | ordered from the loss downwards — the order the backward pass runs |
| $\cdot$ | multiplication (for vectors: products of Jacobian matrices) | combines the local slopes into the total slope | between the factors |

</details>

Each layer only needs to know **its own local derivative** + the gradient flowing in from above. That locality is what makes backprop scale to billions of params.

#### Tiny worked example (one neuron)

```text
   x=2 ──[w=3]──► z=6 ──[ReLU]──► a=6 ──[loss=(a-y)²]──► L=16     (target y=2)
```

Forward: `z = w·x = 6`, `a = ReLU(z) = 6`, `L = (a-y)² = 16`.

Backward (just multiply local derivatives):

```text
   ∂L/∂a = 2(a-y)        = 8         ← loss layer
   ∂a/∂z = 1 (z>0, ReLU) = 1         ← activation
   ∂z/∂w = x             = 2         ← linear

   ∂L/∂w = 8 · 1 · 2 = 16            ← chain them
```

Update: `w ← 3 - 0.01·16 = 2.84`. Loss next step is lower. Done.

#### Why gradients vanish or explode

Apply the chain rule to the [3-layer picture above](#mlp-anatomy--hidden-vs-activation-layers). Every layer contributes one "activation slope × weight" factor:

$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}}
= \frac{\partial \mathcal{L}}{\partial A^{[3]}}\thickspace
\underbrace{g^{[3]\prime} W^{[3]}} _ {\text{layer 3}}\thickspace
\underbrace{g^{[2]\prime} W^{[2]}} _ {\text{layer 2}}\thickspace
g^{[1]\prime}\thinspace \frac{\partial Z^{[1]}}{\partial W^{[1]}}
$$

<details>
<summary>🔍 Symbol atlas — vanishing / exploding product</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $W^{[1]}$ | first-layer weights | the parameter farthest from the loss — the worst case | left side |
| $\dfrac{\partial \mathcal{L}}{\partial A^{[3]}}$ | gradient arriving from the loss | the starting point of the backward pass | first factor = top of the network |
| $g^{[l]\prime}$ | slope of layer $l$'s activation at $Z^{[l]}$ | the cost of passing back through the activation | first half of each underbraced pair |
| $W^{[l]}$ | layer $l$'s weights | the cost of passing back through the linear map ($\partial Z^{[l]} / \partial A^{[l-1]} = W^{[l]}$) | second half of each pair |
| underbraced pair | one $g'W$ per layer crossed | makes the per-layer factor visible | the number of pairs = depth, which is why depth amplifies the problem |
| $\dfrac{\partial Z^{[1]}}{\partial W^{[1]}}$ | equals the input $x$ | local derivative of the first linear layer w.r.t. its own weights | last factor: the bottom of the path |

Strictly these are matrix / elementwise products; "factor $\gt 1$ or $\lt 1$" means their size (norm).

</details>

- If most factors $g' W \gt 1$ → the product blows up → **exploding gradients**.
- If most factors $g' W \lt 1$ → the product shrinks to 0 → **vanishing gradients** (sigmoid's $g' \le 1/4$ makes this the default).

With 70+ layers this product is the whole story of "why deep nets are hard". Every trick in [§3](#3️⃣-stability-tricks--why-deep-nets-even-work) exists to keep these factors near 1.

#### Computational graph (what PyTorch builds)

Every tensor op gets recorded as a node. `loss.backward()` walks the graph in reverse, applying each op's known derivative rule. You write forward code; autograd writes the backward pass for you. More: [backprop.md](backprop.md).

```text
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
| --- | --- | --- |
| **Vanishing gradients** | many <1 derivatives multiplied → ~0 deep down | [ReLU](#activation-functions--the-squish) + [He init](#initialization) + [residuals](#residual-connections) + [RMSNorm](#normalization) |
| **Exploding gradients** | many >1 derivatives multiplied → ∞ | [gradient clipping](#regularization) (max norm 1.0) |
| **Dead neurons** | ReLU stuck at 0 → derivative 0 → no update ever | He init, lower LR, leaky/SwiGLU variants |
| **Saturated attention** | softmax gets huge logits → gradient ≈ 0 | [scaled dot-product (÷√d)](#why-divide-by-d_k), pre-norm |

#### How it connects to everything else

- **[Loss function](#loss-functions)** decides what gradient enters the graph at the top.
- **[Optimizer](#optimizers--sgd--adam--adamw)** decides *how* to use the gradient (momentum, adaptive LR, weight decay).
- **[Init](#initialization)** + **[norm](#normalization)** + **[residuals](#residual-connections)** exist precisely to keep these multiplied gradients well-behaved across 70+ layers.
- **Batch size** averages gradients over examples → less noise per step.
- **Backprop runs once per training step**; for a frontier LLM that's millions of steps over trillions of tokens.

> **One-liner to keep:** Forward computes loss, backward computes ∂loss/∂each-weight via chain rule on the recorded graph, optimizer takes a small step against that gradient. Repeat ~10⁷ times. That's "training."

### Optimizers — SGD → Adam → AdamW

```text
SGD ── + momentum ── + adaptive LR (RMSProp/AdaGrad)
                              │
                              └──► Adam ──► AdamW ──► Muon (newest)
```

| Optimizer | Update rule (intuition) |
| --- | --- |
| **SGD** | $w \leftarrow w - \eta\thinspace  g$ — step straight down the slope |
| **+ Momentum** | $m \leftarrow \beta m + g$, $w \leftarrow w - \eta m$ — a rolling ball that keeps its speed through small bumps |
| **Adam** | momentum $m$ **plus** a per-parameter scale $v$ (running average of $g^2$): $w \leftarrow w - \eta\thinspace \dfrac{\hat m}{\sqrt{\hat v} + \epsilon}$ — parameters with big noisy gradients take smaller steps |
| **AdamW** | Adam + **decoupled** weight decay: $\dots - \eta \lambda w$ — shrink weights directly, not through the gradient |
| **Muon** | orthogonalizes the update matrix for 2-D weights — newer, used at frontier scale |

<details>
<summary>🔍 Symbol atlas — optimizer updates</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $w$ | one parameter (or a tensor of them) | the thing being trained | left of $\leftarrow$ |
| $\leftarrow$ | assignment ("replace with") | marks an update, not an equation | between the new and the old value |
| $\eta$ | learning rate | sets the step size | multiplies the whole step, scaling every update |
| $g$ | gradient $\nabla_w \mathcal{L}$ on the current mini-batch | points to steepest *increase* | subtracted, so we go downhill |
| $m$ | momentum (1st moment): running average of $g$ | smooths noisy gradients, keeps direction | replaces the raw $g$ in the step |
| $\beta$ | decay of the running average, e.g. 0.9 (Adam's $\beta_1$) | how much past velocity is kept | multiplies the old $m$: older gradients fade geometrically |
| $v$ | 2nd moment: running average of $g^2$ | typical gradient size per parameter | under the square root in the denominator |
| $\hat{m}, \hat{v}$ | bias-corrected $m, v$ (divided by $1 - \beta^t$) | both start at 0 and would be too small early | hats = "corrected estimate" used in the step |
| $\sqrt{\hat{v}}$ | typical gradient magnitude | normalizes each step | denominator: big-gradient parameters take proportionally smaller steps, so each moves about $\eta$ |
| $\epsilon$ | tiny constant, e.g. $10^{-8}$ | avoids division by zero | added to the denominator |
| $\lambda$ | weight-decay coefficient | pulls weights toward 0 | multiplies $w$, **not** $g$: decoupled from Adam's scaling — the "W" in AdamW |

</details>

**Default:** AdamW. Modern frontier: Muon.

### Hyperparameters that matter most

| Knob | Effect |
| --- | --- |
| Learning Rate | Too high = explode, too low = stuck. **Use warmup + decay** |
| Batch Size | Bigger = stabler gradients, more memory |
| Warmup/Decay | Ramp LR up, then cosine decay down |

#### Warmup + cosine decay

$$
\text{lr}(t) =
\begin{cases}
\dfrac{t}{T _ {\text{warmup}}}\thinspace \text{lr} _ {\max} & t \le T _ {\text{warmup}} \cr
\text{lr} _ {\min} + \tfrac12(\text{lr} _ {\max}-\text{lr} _ {\min})\Big(1+\cos\big(\pi\thinspace \tfrac{t-T _ {\text{warmup}}}{T-T _ {\text{warmup}}}\big)\Big) & t \gt  T _ {\text{warmup}}
\end{cases}
$$

<details>
<summary>🔍 Symbol atlas — warmup + cosine schedule</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $t$ | current training step | the input of the schedule | appears in both cases |
| $T _ {\text{warmup}}$ | number of warmup steps | where the ramp ends | the boundary between the two cases |
| $T$ | total training steps | where the decay ends | in the decay-progress denominator |
| $\text{lr} _ {\max}, \text{lr} _ {\min}$ | peak and floor learning rate | the range the schedule moves in | $\text{lr} _ {\max}$ scales the ramp; their gap scales the cosine |
| $\dfrac{t}{T _ {\text{warmup}}}$ | warmup progress, $0 \to 1$ | a linear ramp | multiplies $\text{lr} _ {\max}$: LR grows from 0 to the peak |
| $\dfrac{t - T _ {\text{warmup}}}{T - T _ {\text{warmup}}}$ | decay progress, $0 \to 1$ | measures how far into the decay we are | inside the cosine |
| $\cos(\pi \cdot)$ | goes from $1$ to $-1$ as progress goes $0 \to 1$ | gives a smooth curve | $\pi$ makes it cover exactly half a period |
| $\tfrac12(1 + \cos)$ | maps that to $1 \to 0$ | a smooth "fade" factor | multiplies the gap $(\text{lr} _ {\max} - \text{lr} _ {\min})$, then $\text{lr} _ {\min}$ is added back as the floor |
| `cases` brace | piecewise definition | different rules before and after warmup | first row for $t \le T _ {\text{warmup}}$, second afterwards |

</details>

![Warm-up learning rate schedule](zhihu/images/035f514e01ed.jpg)

Why warm up? Early in training the gradients are large and unreliable (especially with [Post-LN](#pre-norm-vs-post-norm)); a tiny LR lets [AdamW](#optimizers--sgd--adam--adamw)'s running statistics settle before taking big steps. Source: [LayerNorm note](<zhihu/Transformer学习笔记三: 为什么Transformer要用LayerNorm:Batch Normalization & Layer Normalization (批量&层标准化).md>).

### Loss landscape

```text
   ╱╲      ╱╲           ← local minima trap SGD
  ╱  ╲    ╱  ╲   ╱╲
 ╱    ╲__╱    ╲_╱  ╲___ ← global minimum (rarely reached)
```

[Residuals](#residual-connections) and [normalization](#normalization) both make this surface smoother — which is the modern explanation for why they help.

### Workshop example: `A * B % C` — teaching an MLP modular multiplication

A canonical "first MLP" exercise: train a small dense network to compute **(A × B) mod C** from raw numerical inputs. It's the simplest task that's *just* nonlinear enough to require a real network — perfect for seeing every concept so far ([MLP](#mlp-anatomy--hidden-vs-activation-layers), [loss](#loss-functions), [backprop](#backpropagation--the-heart-of-training), [optimizer](#optimizers--sgd--adam--adamw)) in action.

#### The task

```text
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

```text
   ┌────────────────┐   ┌───────────────┐   ┌───────────────┐   ┌──────┐
   │ INPUT LAYER    │   │ HIDDEN 1      │   │ HIDDEN 2      │   │ OUT  │
   │ 20 features    │   │ 100 neurons   │   │ 100 neurons   │   │ 1    │
   │ (encoded A,B,C)│   │ + ReLU        │   │ + ReLU        │   │ value│
   └────────────────┘   └───────────────┘   └───────────────┘   └──────┘
        [B, 20]   ──►        [B, 100]    ──►   [B, 100]      ──►  [B, 1]
                W₁: [20×100]            W₂: [100×100]      W₃: [100×1]
                +b₁: [100]              +b₂: [100]         +b₃: [1]
```

Why **20 inputs** for only 3 numbers? Because raw integers are bad input — they're typically **one-hot or binary-encoded** so the net can pick up bit-patterns. E.g. each of A, B, C as a 7-bit binary vector (≈21 dims), or as one-hot over a small range. The exact encoding is a hyperparameter; **20** is just the workshop's chosen width. (Same idea as [tokens → embeddings](#embeddings): never feed raw IDs as magnitudes.)

#### Dimensional walk-through

```text
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

```text
   ┌─ generate (A,B,C) random batch                       ← data
   │
   │  forward:  x → Linear → ReLU → Linear → ReLU → Linear → ŷ
   │  loss:     MSE(ŷ, (A·B) % C)                          ← Loss functions
   │  backward: ∂loss/∂W via chain rule                    ← Backprop
   │  step:     W ← W - lr · grad   (AdamW)                ← Optimizer
   │
   └─ repeat ~10K steps until loss plateaus
```

Hyperparameters typical for the demo: `lr=1e-3`, `batch=256`, ReLU, He init.

#### What this exercise teaches you (and how it maps to LLMs)

| Workshop concept | Where it shows up here | LLM equivalent |
| --- | --- | --- |
| Hidden layer | the two `Linear(100)` blocks | [FFN](#inside-the-ffn--where-sota-hidden-layers-live)'s up-projection |
| Activation | ReLU between layers | SwiGLU in Llama FFN |
| Loss | MSE for regression | Cross-entropy for next-token |
| Backprop | autograd through 3 matmuls | autograd through 70+ layers |
| Optimizer | AdamW | AdamW (same!) |
| Init | He (matches ReLU) | He / Xavier variants |
| Param count | ~12K | 7B+ |

> **The lesson:** an LLM is *exactly* this loop, scaled up. Same forward/loss/backward/step, just with **embeddings + attention + many more layers** in the middle. If you understand `A*B%C`, you understand the skeleton of GPT.

---

## 3️⃣ Stability Tricks — why deep nets even work

These four tricks are what let you stack 72 layers without the [vanishing/exploding product](#why-gradients-vanish-or-explode) killing training.

```text
┌─────────────────────────────────────────────┐
│  STABILITY = Init + Norm + Residuals + Reg  │
└─────────────────────────────────────────────┘
   init        → start with variance ≈ 1
   norm        → re-center/re-scale every layer, every step
   residuals   → give the gradient a "+1" highway
   regularize  → stop weights/gradients getting extreme
```

### Initialization

**Goal:** keep the variance of activations the same from layer to layer at step 0. For one neuron with $n _ {in}$ inputs (independent, zero-mean):

$$
\text{Var}(z) = n _ {in}\thinspace \text{Var}(w)\thinspace \text{Var}(x)
$$

So pick $\text{Var}(w) = 1/n _ {in}$ to make the factor 1. ReLU zeroes half its inputs, so it needs twice that.

| Activation | Use this init | Formula | Year |
| --- | --- | --- | --- |
| ReLU | **He / Kaiming** | $\text{std} = \sqrt{2/n _ {in}}$ | 2015 |
| Tanh/Sigmoid | **Xavier / Glorot** | $\text{std} = \sqrt{2/(n _ {in}+n _ {out})}$ | 2010 |

<details>
<summary>🔍 Symbol atlas — initialization</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $z$ | pre-activation of one neuron, $\sum_i w_i x_i$ | the signal whose size we want to keep stable | left side |
| $\text{Var}$ | variance: average squared distance from the mean | measures the signal's "size" | applied to each quantity |
| $n _ {in}$ | fan-in: number of inputs to the neuron | each input adds its own variance | multiplies: a sum of $n _ {in}$ independent terms has $n _ {in}$ times the variance |
| $\text{Var}(w), \text{Var}(x)$ | spread of weights and of inputs | the two sources of signal size | multiplied: for independent zero-mean variables, $\text{Var}(wx) = \text{Var}(w)\thinspace \text{Var}(x)$ |
| $\text{std}$ | standard deviation $= \sqrt{\text{Var}}$ | what init functions take as input | left side of the table formulas |
| $2$ (He) | compensation factor | ReLU zeroes about half its inputs, halving the variance | in the numerator: doubles the weight variance back |
| $n _ {out}$ | fan-out: neurons in the next layer | the backward pass sums over $n _ {out}$, not $n _ {in}$ | Xavier uses their average: $\frac{2}{n _ {in}+n _ {out}}$ balances forward and backward |

</details>

> Wrong init → activations explode 💥 or collapse to zero 🕳️ over many layers. Same "variance ≈ 1" logic as [÷√d_k in attention](#why-divide-by-d_k).

Distributions cheat sheet:

- **Uniform**: flat — equal chance everywhere
- **Normal**: bell — 68/95/99.7 within 1/2/3 SD

### Normalization

Init only fixes step 0. As weights update, each layer's input distribution keeps drifting (the original "Internal Covariate Shift" story). Normalization re-centers and re-scales at **every** step:

```text
   x ──► (x - mean) / √(var + ε) ──► γx + β
         ↑                          ↑
    center+scale            learned shift+scale
```

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \odot \hat{x} + \beta
$$

<details>
<summary>🔍 Symbol atlas — normalization</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | the vector being normalized (for LN: one token's features) | the raw activations | numerator |
| $\mu$ | mean of the chosen group of numbers | the "center" | subtracted ⇒ result centered at 0 |
| $\sigma^2$ | variance of the same group | the "spread" | under the root in the denominator: dividing by the standard deviation gives unit scale |
| $\epsilon$ | tiny constant (e.g. $10^{-5}$) | avoids dividing by ~0 when all values are equal | inside the root, added to $\sigma^2$ |
| $\hat{x}$ | normalized vector: mean 0, variance 1 | intermediate result | hat = "standardized version of $x$" |
| $\gamma$ | learned per-feature scale | lets the network re-introduce any scale it needs | multiplies *after* normalizing |
| $\odot$ | elementwise product | each feature gets its own $\gamma$ | between $\gamma$ and $\hat{x}$ |
| $\beta$ | learned per-feature shift (**not** the momentum $\beta$) | re-introduces an offset | added last |
| $y$ | output of the norm layer | what the next sublayer sees | left side of the 2nd equation |

BatchNorm, LayerNorm and RMSNorm share this shape; they differ only in *which* numbers $\mu$ and $\sigma$ are computed over (next table).

</details>

$\gamma, \beta$ are learned so the network can undo the normalization if it wants — normalizing must not cost expressiveness.

#### BatchNorm vs LayerNorm vs RMSNorm — which axis?

The only difference is **which numbers you average over**:

```text
   activations tensor [batch B, seq S, features d]

   BatchNorm:  for each feature j, average over (B × S)   ← mixes different sentences
   LayerNorm:  for each token,     average over its d features   ← one token at a time
   RMSNorm:    like LayerNorm, but skip the mean (scale only)
```

![LayerNorm: statistics are computed per sample across features](zhihu/images/1beda6be4e67.jpg)

LayerNorm computes $\mu, \sigma$ within one sample, then applies learned $\gamma, \beta$. Source: [LayerNorm note](<zhihu/Transformer学习笔记三: 为什么Transformer要用LayerNorm:Batch Normalization & Layer Normalization (批量&层标准化).md>).

| Type | Normalizes across | Formula | Used in |
| --- | --- | --- | --- |
| **BatchNorm** | batch dim | $\mu_j = \frac{1}{m}\sum _ {i=1}^{m} z^{(i)} _ j$ over the batch | CNNs |
| **LayerNorm** | feature dim per token | $\mu = \frac{1}{d}\sum _ {k=1}^{d} x_k$ within one token | original Transformer, BERT, GPT-2 |
| **RMSNorm** | scales only (no centering) | $y = \gamma \odot \dfrac{x}{\sqrt{\frac{1}{d}\sum_k x_k^2 + \epsilon}}$ | **modern LLMs** ✅ |

<details>
<summary>🔍 Symbol atlas — BatchNorm / LayerNorm / RMSNorm statistics</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $j$ | feature index | BatchNorm keeps separate statistics per feature | subscript on $\mu_j$ and $z_j$ |
| $m$ | batch size | the number of values BN averages | BN divides by $m$: it averages **across examples** |
| $z^{(i)} _ j$ | feature $j$ of example $i$ | one entry of the activation table | superscript $(i)$ = which example (summed), subscript $j$ = which feature (fixed) |
| $d$ | number of features (hidden size) | the number of values LN averages | LN/RMSNorm divide by $d$: they average **across one token's features** |
| $x_k$ | $k$-th feature of the token | one value being averaged | summed over $k = 1 \dots d$ |
| $\frac{1}{d}\sum_k x_k^2$ | mean of squares | its root is the RMS, the "typical size" | under the root in RMSNorm's denominator — no mean subtracted, no $\beta$ added |
| $\gamma, \epsilon$ | learned scale and tiny constant | same roles as in the formula above | $\gamma$ in front, $\epsilon$ inside the root |

</details>

**Why not BatchNorm for text?**

- Averaging "the 1st token of 32 different sentences" is meaningless; sentence lengths differ and padding pollutes the statistics.
- BN behaves differently at train vs test time (running averages); LN computes the same thing for one token either way.
- BN couples examples in a batch — painful with micro-batches in [pipeline parallelism](#pipeline-parallelism--fitting-a-model-across-gpus).

**Why RMSNorm?** Dropping the mean saves compute and works just as well in practice — the re-scaling is what matters.

> Plot twist from the BN literature: it probably works less by fixing covariate shift and more by **smoothing the loss landscape** (Santurkar et al., 2018).

#### Pre-norm vs Post-norm

![Post-LN (a) vs Pre-LN (b) transformer layer](zhihu/images/90dbd1c5c910.jpg)

$$
\text{Post-LN: } x _ {l+1} = \text{LN}\big(x_l + F(x_l)\big)
\qquad\qquad
\text{Pre-LN: } x _ {l+1} = x_l + F\big(\text{LN}(x_l)\big)
$$

<details>
<summary>🔍 Symbol atlas — Pre-LN vs Post-LN</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x_l$ | the residual stream entering layer $l$ | the input that both versions transform | on the right of both formulas |
| $F$ | the sublayer (attention or FFN) | the part that computes something new | inside both formulas |
| $\text{LN}$ | LayerNorm (or RMSNorm) | stabilizes scale | **its position is the only difference** between the two formulas |
| $x _ {l+1}$ | output, fed to the next layer | the updated stream | left side |
| $\text{LN}(x_l + F(x_l))$ | Post-LN: the norm **wraps the sum** | the original 2017 design | the norm sits on the main path, so every layer rescales the whole stream and the identity path is broken |
| $x_l + F(\text{LN}(x_l))$ | Pre-LN: the norm only touches the **sublayer's input** | the modern design | $x_l$ is added untouched ⇒ a clean highway $x _ {l+1} = x_l + \dots$ from bottom to top |

</details>

- **Post-LN** (original 2017 Transformer) puts the norm *on* the residual path → gradients near the output are large, so it **needs [LR warmup](#warmup--cosine-decay)** or it diverges.
- **Pre-LN** leaves a clean identity path $x_l \to x _ {l+1}$ → gradient norms stay even across layers → trains without delicate warmup and goes deeper.

**Pre-norm** (norm before sublayer) > **Post-norm** for deep stacks. Modern recipe: **Pre-RMSNorm**. Source: [LayerNorm note §3](<zhihu/Transformer学习笔记三: 为什么Transformer要用LayerNorm:Batch Normalization & Layer Normalization (批量&层标准化).md>).

### Residual connections

```text
   x ──┬──► [sublayer] ──► +  ──► out
       └─────── skip ──────┘
```

$$
H(x) = F(x) + x
$$

<details>
<summary>🔍 Symbol atlas — residual block</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | input of the block | both the thing processed and the thing passed through | appears **twice**: fed through $F$, and added directly — that second copy *is* the skip connection |
| $F(x)$ | what the layers compute: the residual (a correction) | the only part the layers must learn | first term on the right |
| $H(x)$ | output of the block (the mapping we actually want) | what the next block receives | left side |
| $+$ | elementwise addition | merges the correction with the original | requires $F(x)$ and $x$ to have the same shape (else a projection $W_s x$ is used) |

**Read it as:** the layers only have to learn the *change* $F = H - x$. Doing nothing means $F = 0$, which is easy.

</details>

**Problem it solves — degradation:** even when a deep net *can* train, a 56-layer plain net does worse than a 20-layer one — on the **training** set, so it's not overfitting:

![Deeper plain nets have higher training and test error](zhihu/images/4bd38f2b31e2.jpg)

**Intuition:** a deeper net should never be worse — the extra layers could just copy their input (identity). But learning "do nothing" with weights is surprisingly hard. With a skip connection, "do nothing" just means $F(x) = 0$, which is easy (weights near 0).

![Residual block: F(x) + x](zhihu/images/4df520a1ba48.jpg)

**Gradient view** — why it fixes the [vanishing product](#why-gradients-vanish-or-explode):

$$
\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + I
$$

<details>
<summary>🔍 Symbol atlas — residual gradient</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\dfrac{\partial H}{\partial x}$ | how the block passes gradient backwards | the factor this block contributes to the backprop product | left side |
| $\dfrac{\partial F}{\partial x}$ | gradient through the layers | may be tiny (the vanishing problem) | first term |
| $I$ | identity matrix — the derivative of $x$ w.r.t. itself | comes from the skip path | **added, not multiplied**: a product of $(I + \partial F/\partial x)$ terms can't collapse to 0 |

</details>

The $+I$ term means every block passes gradient through **at least unchanged**, no matter how small $\partial F / \partial x$ gets. Stack 80 blocks and there is still a direct gradient highway from the loss to layer 1.

Adds input back to output. Smooths loss landscape, lets gradients flow → enables 72+ layers (ResNet trained 1000+).
Variants: additive, scaled, gated, concat, attention-based. Source: [ResNet note](<zhihu/Transformer学习笔记四: ResNet (残差网络).md>).

### Regularization

| Technique | What | Formula | LLM today? |
| --- | --- | --- | --- |
| **Dropout** (10%) | randomly zero neurons | $h' = \dfrac{m \odot h}{1-p},\ m_i \sim \text{Bernoulli}(1-p)$ | ❌ rare in pretraining ([why](#why-modern-llms-dont-use-dropout-anymore)) |
| **Gradient Clip** (max 1.0) | cap gradient norm | $g \leftarrow g \cdot \min\negthinspace \big(1, \tfrac{c}{\lVert g\rVert}\big)$ | ✅ yes |
| **Weight Decay** (10%) | penalize big weights | $w \leftarrow w - \eta\lambda w$ (inside [AdamW](#optimizers--sgd--adam--adamw)) | ✅ yes |

<details>
<summary>🔍 Symbol atlas — regularization</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $h$ | activations of a layer | what dropout perturbs | inside the masked product |
| $m$ | random 0/1 mask vector (**not** batch size or momentum) | decides which units survive this step | $\odot$ multiplies elementwise: 0 entries switch units off |
| $p$ | drop probability (e.g. 0.1) | sets how much is dropped | inside the Bernoulli and the rescale |
| $\text{Bernoulli}(1-p)$ | coin flip giving 1 with probability $1-p$ | generates each mask entry | after $\sim$ ("is drawn from") |
| $\frac{1}{1-p}$ | rescaling ("inverted dropout") | keeps the expected value of $h'$ equal to $h$, so nothing changes at test time | divides the masked vector |
| $g$ | gradient vector | what clipping limits | on both sides of the clip rule |
| $\lVert g \rVert$ | its length (L2 norm) | measures how big the step would be | denominator of $c / \lVert g \rVert$ |
| $c$ | maximum allowed norm (e.g. 1.0) | the cap | numerator of the ratio |
| $\min(1, \cdot)$ | the smaller of 1 and the ratio | small gradients pass unchanged; big ones shrink to length $c$, direction kept | outer function of the scale factor |
| $\eta \lambda w$ | weight-decay step | shrinks $w$ by the factor $(1 - \eta\lambda)$ each step | subtracted from $w$ |

</details>

#### Why modern LLMs don't use dropout anymore

The workshop's claim is **correct**, and there's a 2025 paper literally titled *"Drop Dropout on Single-Epoch Language Model Pretraining"* confirming it empirically.

```text
   Old era (≤ ~2019)              Modern era (Llama-2/3, GPT-3+)
   ────────────────                ──────────────────────────────
   • Small dataset                 • Internet-scale corpus (trillions of tokens)
   • Many epochs (10s–100s)        • ~1 epoch — see each token once
   • Risk: memorize → overfit      • Risk: underfit (data > params)
   • Dropout @ 10% needed          • Dropout HURTS performance ❌
```

**Why dropout helped in the old days:** when you re-show the same training data many times, the network can memorize specific examples. Randomly zeroing 10% of neurons each pass forces it to generalize instead of memorize.

**Why it stopped helping:** with web-scale data and a single epoch, **the model never sees the same token twice**, so there's nothing to memorize. Dropout just adds noise that throws away useful signal — equivalent to training on less data.

| Year | Model | Dropout? |
| --- | --- | --- |
| 2018 | BERT | 10% |
| 2019 | GPT-2 | 10% (everywhere) |
| 2020 | GPT-3 | small / phasing out |
| 2023 | **Llama-2** | **❌ none** |
| 2024+ | **Llama-3, modern frontier** | **❌ none** |

The 2025 ACL paper tested Pythia 160M/1.4B with varying dropout rates on single-epoch pretraining and found **downstream performance is consistently better with dropout = 0** across language modeling, BLiMP, SQuAD, and MNLI. Models without dropout were also easier to model-edit.

**Where dropout still appears:**

- 🎯 **[Fine-tuning](#fine-tuning-sft-and-lora)** (small datasets, multi-epoch) — dropout makes a comeback
- 🔧 **LoRA adapters** — small param count, multi-epoch → dropout helps
- 🧪 **Smaller research models** trained for many epochs

> **Keep this:** Dropout is a *small-data, multi-epoch* tool. Pretraining a frontier LLM is the opposite regime (huge-data, single-epoch), so dropout is gone. Other regularization survives: **gradient clipping** and **weight decay** still used.

---

## 4️⃣ Text → Numbers Pipeline

```text
"Hello world"
     │
     ▼   tokenizer (BPE)
[15496, 995]
     │
     ▼   embedding lookup (vocab × 768)
[[0.12, -0.4, ...],   ← 768-dim vector per token
 [0.88,  0.1, ...]]
     │
     ▼   + positional information (added, or RoPE inside attention)
position-aware vectors → into transformer
```

### Tokenizers

A tokenizer has four parts: **trainer** (learns the vocab from a corpus) → **vocabulary** → **encoder** (text → IDs) → **decoder** (IDs → text).

| Type | Granularity | Vocab | Use |
| --- | --- | --- | --- |
| Character | per char | tiny | sequences too long, chars carry little meaning |
| Word | per word | huge | OOV problems (`<UNK>`), `love/loving/loved` all separate |
| **BPE** | subword (merge most *frequent* pairs) | ~50K | **GPT-2/3/4** ✅ |
| WordPiece | subword (merge pair with max *likelihood gain*) | ~30K | BERT |
| Unigram LM | subword (start big, *prune* least useful) | ~32K | SentencePiece, T5, multilingual |

GPT-2 vocab: **50,257** tokens. `tiktoken` is OpenAI's library.

#### BPE training in 4 lines

```text
corpus counts:   low</w>:5   lower</w>:2   newest</w>:6   widest</w>:3
start:           split every word into characters (+ end marker </w>)
repeat:          count all adjacent pairs → merge the most frequent one
   e.g. (e,s) appears 6+3=9 times → merge to "es"; then (es,t) → "est"; then (est,</w>) …
stop:            when vocab hits the target size (e.g. 50K)
```

Encoding a new word = greedily apply the learned merges; anything unknown falls back to smaller pieces (byte-level BPE never needs `<UNK>`).

#### What WordPiece and Unigram change

**WordPiece** merges the pair that most increases corpus likelihood (assuming subwords are independent), not just the most frequent one:

$$
\text{score}(x, y) = \log \frac{P(xy)}{P(x)\thinspace P(y)}
$$

<details>
<summary>🔍 Symbol atlas — WordPiece merge score</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x, y$ | two adjacent subword tokens | the candidates for a merge | arguments of the score |
| $xy$ | the merged token | what we'd add to the vocab | numerator |
| $P(\cdot)$ | probability of a token ≈ its count / total count | measures how common a token is | applied to the merged and the separate tokens |
| $P(x)\thinspace P(y)$ | probability of seeing $x$ then $y$ **if independent** | the "by chance" baseline | denominator |
| ratio | $\gt 1$ ⇒ they co-occur more than chance | compares reality with the baseline | numerator over denominator |
| $\log$ | turns the ratio into $\log P(xy) - (\log P(x) + \log P(y))$ | equals the gain in corpus log-likelihood from merging | outermost |

This quantity is the pointwise mutual information (PMI) between $x$ and $y$.

</details>

Intuition: merge pairs that appear together **more than chance** — `q`+`u` scores high, `e`+`s` less so even if frequent.

**Unigram LM** goes the other way: start from a huge vocab, repeatedly drop the subwords whose removal hurts likelihood least. Because one word can have several segmentations, it picks the most probable one:

$$
P(\vec{x}) = \prod _ {i=1}^{m} P(x_i), \qquad \vec{x}^ * = \arg\max _ {\vec{x} \in \mathcal{S}(X)} P(\vec{x})
$$

<details>
<summary>🔍 Symbol atlas — Unigram LM segmentation</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $X$ | a raw word or sentence | the text to split | inside $\mathcal{S}(X)$ |
| $\vec{x}$ | one segmentation: the subword sequence $(x_1, \dots, x_m)$ | a candidate answer | arrow = sequence; argument of $P$ and of the argmax |
| $m$ | number of pieces in this segmentation | more pieces = more factors | upper limit of the product |
| $P(x_i)$ | probability of subword $x_i$ (learned with EM) | the score of one piece | factor in the product |
| $\prod$ | product | "unigram" = pieces assumed independent | multiplies piece probabilities into a sequence probability |
| $\mathcal{S}(X)$ | set of all valid segmentations of $X$ | the search space | under the argmax |
| $\arg\max$ | the *argument* (segmentation) that maximizes, not the max value | we want the split itself | in front of $P(\vec{x})$ |
| $\vec{x}^ *$ | the best segmentation (star = optimal) | the tokenizer's output | left side of the 2nd equation |

Because every extra piece multiplies in another number $\lt 1$, the product naturally prefers fewer, more common pieces.

</details>

(solved with the Viterbi algorithm). Source: [Tokenization note](<zhihu/Transformer学习笔记五: Subword Tokenization (子词分词器).md>).

### Embeddings

- An embedding table is a matrix $E \in \mathbb{R}^{V \times d}$. Token ID $i$ → row $i$. (Mathematically: one-hot vector × $E$, but implemented as a lookup.)
- Each token → vector in **768-dim** (GPT-2) up to 12K+ (frontier).
- **PCA** lets you visualize ~semantic clusters in 2D.
- **Tied embeddings**: input embedding matrix == output (unembedding) matrix, so logits = $h E^\top$ ("which token's embedding is my final vector closest to?").
- Embeddings **add up**: BERT's input is token + segment + position embeddings, summed elementwise:

<details>
<summary>🔍 Symbol atlas — embeddings</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $E$ | embedding matrix | stores one learned vector per token | the object being described |
| $\mathbb{R}^{V \times d}$ | real matrix with $V$ rows and $d$ columns | states the shape | after $\in$ ("is a member of") |
| $V$ | vocabulary size (50,257 for GPT-2) | one row per token | first dimension |
| $d$ | model dimension (768 for GPT-2) | vector length | second dimension |
| $i$ | token ID | the tokenizer's output | selects row $i$ |
| $h$ | final hidden vector at one position, length $d$ | the model's summary of the context | left factor of $hE^\top$ |
| $E^\top$ | transposed embedding matrix, $d \times V$ | reuses the input table as the output layer | $h E^\top$ dots $h$ with **every** token's embedding ⇒ $V$ logits |

</details>

![BERT input = token + segment + position embeddings](zhihu/images/393335f7e21a.jpg)

Source: [BERT note](<zhihu/BERT学习笔记一: 基于论文精读的模型详解.md>).

### Positional encoding — from one-hot to sinusoidal to RoPE

**The problem:** attention has no built-in sense of order — `Q·Kᵀ` treats tokens as a *set*, not a *sequence*. "Dog bites man" and "man bites dog" would look identical without positional info. We need to inject "where am I in the sentence" into each token vector.

![Self-attention sees all tokens at once — no notion of order](zhihu/images/2a2d39a8e155.jpg)

Each output $b_i$ looks at all inputs $a_1..a_4$ in parallel; nothing tells it that $a_2$ comes after $a_1$. Source: [Positional Encoding note](<zhihu/Transformer学习笔记一: Positional Encoding (位置编码).md>).

A good position signal should be: **bounded**, **unique per position**, **smooth** (nearby positions look similar), **consistent across lengths**, and able to **extrapolate** to unseen lengths. Watch each step fix one more of these.

#### Step 1 — Integer or one-hot position (naive, doesn't work)

```text
   integer:   pos = 0, 1, 2, … 5000        ← unbounded, dwarfs the embedding values
   one-hot:
   pos 0: [1,0,0,0,0]
   pos 1: [0,1,0,0,0]
   pos 2: [0,0,1,0,0]
```

Problems: integers are unbounded; normalizing to [0,1] makes "one step" mean different things for different lengths. One-hot vectors are huge (size = max length), all equally dissimilar (no notion of "5 is close to 6"), and can't go past the max length. Binary codes are bounded but **jump** discontinuously between neighbors.

#### Step 2 — Learned absolute embedding (better, still limited)

Just like token embeddings, learn a lookup table: `pos_embedding[0], pos_embedding[1], ...`. Compact and trainable (BERT, GPT-2 do this), but still **can't extrapolate** beyond the longest position seen in training (position 5000 has literally no learned vector if you only trained up to 4096).

#### Step 3 — Sinusoidal (the original Transformer paper's fix)

**Intuition:** instead of learning a table, *compute* a unique fixed pattern per position using sine/cosine waves of different frequencies — like a clock (or odometer) with many hands spinning at different speeds.

$$
PE _ {(t,\thinspace 2k)} = \sin(\omega_k t), \qquad PE _ {(t,\thinspace 2k+1)} = \cos(\omega_k t), \qquad \omega_k = \frac{1}{10000^{2k/d}}
$$

<details>
<summary>🔍 Symbol atlas — sinusoidal positional encoding</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $PE$ | positional-encoding matrix [positions × $d$] | holds the position signal | left side |
| $t$ | position in the sequence | what we encode | multiplied by $\omega_k$: position becomes an **angle** |
| $2k$, $2k+1$ | an even and the following odd dimension | dimensions come in pairs sharing one frequency | second subscript: even ⇒ $\sin$, odd ⇒ $\cos$ |
| $k$ | pair index, $0 \dots d/2 - 1$ | picks the frequency | subscript of $\omega$ |
| $\omega_k$ | angular frequency of pair $k$ | how fast that dial turns as $t$ grows | multiplies $t$ inside $\sin$ / $\cos$ |
| $10000$ | base constant | sets the longest wavelength ($2\pi \cdot 10000$) | base of the power |
| $2k/d$ | goes from 0 to ~1 across dimensions | spreads frequencies over the dimensions | exponent: frequencies fall geometrically from 1 down to $1/10000$ |
| $d$ | model dimension | PE must match the embedding size to be added | denominator of the exponent |

</details>

![Each dimension is a dial; right-hand dials turn more slowly](zhihu/images/e00f72c5405e.jpg)

Each row is one position $t$, each column one dimension. Like binary counting, the low dimensions flip fast and the high dimensions flip slowly — but smoothly. Wavelengths grow from $2\pi$ to $2\pi \cdot 10000$.

![Sinusoidal PE heatmap, 50 positions × 128 dims](zhihu/images/ce98128f8b77.jpg)

Left columns oscillate quickly with position; right columns barely change. Every row is a unique "barcode" with values in [-1, 1].

**Why sin *and* cos in pairs?** So that shifting position is a **rotation**. For each frequency, moving from $t$ to $t + \Delta$ is a fixed linear map that doesn't depend on $t$:

$$
\begin{pmatrix} \sin\omega(t+\Delta) \cr \cos\omega(t+\Delta) \end{pmatrix} =
\begin{pmatrix} \cos\omega\Delta & \sin\omega\Delta \cr -\sin\omega\Delta & \cos\omega\Delta \end{pmatrix}
\begin{pmatrix} \sin\omega t \cr \cos\omega t \end{pmatrix}
$$

<details>
<summary>🔍 Symbol atlas — shift = rotation</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\omega$ | frequency of one sin/cos pair | the same for both entries of the pair | in every entry |
| $t$ | current position | the starting point | right-hand vector: the pair at position $t$ |
| $\Delta$ | offset (how far we shift) | the relative distance | **only** inside the matrix |
| left vector | the pair at position $t + \Delta$ | the result | left side |
| $2 \times 2$ matrix | rotation by the angle $\omega\Delta$ | depends on $\Delta$ only ⇒ one fixed matrix shifts **any** position | between the two vectors |
| entry layout | cos on the diagonal, ±sin off it | comes from the angle-addition formulas | rows read off $\sin(a+b) = \sin a\cos b + \cos a \sin b$ and $\cos(a+b) = \cos a \cos b - \sin a \sin b$ |

</details>

Consequence — the dot product of two positions depends **only on their distance**:

$$
PE_t^\top PE _ {t+\Delta} = \sum_k \cos(\omega_k \Delta)
$$

<details>
<summary>🔍 Symbol atlas — PE dot product</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $PE_t$ | encoding vector of position $t$ | one of the two positions compared | first vector |
| $^\top$ | transpose | row × column = one number (dot product = similarity) | on the first vector |
| $PE _ {t+\Delta}$ | encoding of the position $\Delta$ further on | the other position | second vector |
| $\sum_k$ | over frequency pairs | each pair contributes one term | outside the cosine |
| $\cos(\omega_k\Delta)$ | one pair's contribution: $\sin\sin + \cos\cos = \cos(\text{angle difference})$ | shows only the angle *difference* matters | inside the sum |
| missing $t$ | $t$ cancelled out | the similarity depends only on distance | and $\cos$ is even, so $\pm\Delta$ give the same value: distance yes, direction no |

</details>

![Dot product between positions vs offset k](zhihu/images/a954d3bfc60b.jpg)

Symmetric in $\Delta$: it knows "how far", not "which direction". And once $PE$ passes through $W_Q, W_K$, even this clean distance pattern gets scrambled — the motivation for RoPE.

**Why sin/cos specifically?**

- **Bounded** (always between -1 and 1) — doesn't blow up for long sequences.
- **Unique per position** — the combination across all frequencies never repeats.
- **Smooth/continuous** — nearby positions get similar vectors (unlike one-hot), so the model can generalize "close in position."
- **Relative shifts are linear** (the rotation above).
- Fixed, **not learned** — zero extra parameters, computable for any position.

#### Step 4 — RoPE (Rotary Position Embedding) — what SOTA models use today

**Key shift in idea:** instead of *adding* a position vector to the token embedding (steps 1-3), RoPE **rotates** the Query and Key vectors by an angle proportional to their position, right before the attention dot-product. It reuses the exact rotation from step 3 — but applies it to $q$ and $k$ instead of to the input.

```text
   token embedding "cat" at position 5   ──rotate by θ·5──►  rotated Q/K vector
   token embedding "cat" at position 9   ──rotate by θ·9──►  rotated Q/K vector
```

Split $q$ (and $k$) into 2-D pairs; rotate pair $k$ by angle $m\theta_k$ at position $m$ (same $\theta_k = 10000^{-2k/d}$ as sinusoidal). With $R_m$ the block-diagonal rotation matrix:

$$
q_m = R_m W_Q x_m, \quad k_n = R_n W_K x_n
\quad\Longrightarrow\quad
q_m^\top k_n = x_m^\top W_Q^\top\thinspace  R _ {n-m}\thinspace  W_K\thinspace  x_n
$$

because $R_m^\top R_n = R _ {n-m}$ (rotating back by $m$ then forward by $n$). The attention score sees **only the offset $n - m$**, and — unlike step 3 — this survives the projection matrices.

<details>
<summary>🔍 Symbol atlas — RoPE</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x_m, x_n$ | token vectors at positions $m$ (query side) and $n$ (key side) | the two tokens being compared | rightmost in each product |
| $W_Q, W_K$ | query / key projection matrices | turn tokens into queries and keys | applied first, *then* rotated |
| $R_m$ | block-diagonal rotation: 2-D pair $k$ rotated by $m\theta_k$ | injects position $m$ | multiplies **after** $W_Q$: position acts on the projected query |
| $\theta_k$ | frequency of pair $k$ ($10000^{-2k/d}$, as in sinusoidal) | gives each pair its own speed | inside the rotation angle |
| $q_m, k_n$ | rotated query and key | what attention actually uses | left sides of the first two equations |
| $\Longrightarrow$ | "therefore" | separates definitions from the result | between them |
| $q_m^\top k_n$ | the attention score (before $\div\sqrt{d_k}$) | what we want to understand | left side of the result |
| $x_m^\top W_Q^\top$ | equals $(W_Q x_m)^\top$ | the query written as a row | transposing a product reverses its order |
| $R_m^\top R_n = R _ {n-m}$ | undo rotation $m$, then apply rotation $n$ | turns two absolute rotations into one relative one | the transpose of a rotation is its inverse |
| $R _ {n-m}$ | rotation by the **offset** | position enters only as relative distance | sits **between** $W_Q^\top$ and $W_K$, so it survives the projections |

</details>

**Why rotation instead of addition?**

- Attention naturally learns "how far apart" two tokens are, not "which absolute slots" they occupy.
- Rotation preserves vector length, so it doesn't distort token content.
- No separate position vector to store/add — folded directly into Q/K computation.
- Caveat: RoPE alone still degrades past the trained length; long-context models extend it with tricks like position interpolation / NTK / YaRN scaling of $\theta$.

| Method | Encodes | Learned? | Extrapolates? | Used by |
| --- | --- | --- | --- | --- |
| One-hot | absolute | ❌ no | ❌ no | toy example only |
| Learned absolute | absolute | ✅ yes | ❌ no | BERT, GPT-2 |
| Sinusoidal | absolute (relative via rotation) | ❌ no (fixed formula) | ⚠️ somewhat | original Transformer (2017) |
| **RoPE** | **relative** | ❌ no (fixed formula, applied to Q/K) | ⚠️ better, ✅ with θ-scaling | **Llama, GPT-NeoX, modern LLMs** ✅ |

> **Keep this:** Positional encoding evolved from "stamp a position label on each token" (one-hot → learned → sinusoidal) to "bake relative distance directly into the attention math" (RoPE). Sinusoidal already had the rotation trick; RoPE moved it to where the dot-product happens.

---

## 5️⃣ Attention & Transformer

### Why attention — the RNN bottleneck

An RNN reads tokens one at a time: token 4 has to wait for tokens 1–3, and information from token 1 must survive 3 hops to reach token 4.

| | RNN | Self-attention |
| --- | --- | --- |
| Sequential steps (parallelism) | $O(n)$ — must wait | $O(1)$ — all tokens at once |
| Path length between two tokens | $O(n)$ — info fades | $O(1)$ — direct link |
| Cost per layer | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ — the price of all-pairs |

<details>
<summary>🔍 Symbol atlas — complexity</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $O(\cdot)$ | big-O: how cost grows, ignoring constants | compares scaling, not exact time | wraps each cost |
| $n$ | sequence length | the variable we scale | in every entry |
| $d$ | model dimension | the width of each vector | in every cost entry |
| $O(n \cdot d^2)$ | RNN: $n$ sequential steps, each a $d \times d$ matrix-vector product | shows why RNNs are slow on long text | $n$ to the first power: linear in length, but sequential |
| $O(n^2 \cdot d)$ | attention: $n \times n$ query-key pairs, each a length-$d$ dot product | the price of all-pairs links | $n^2$: doubling the context = 4× the attention cost |

</details>

That $O(n^2)$ is why long context is expensive and why [KV-cache tricks](#attention-variants-kv-cache-savings) matter.

### Q, K, V — the three "lenses" on each token

Before the attention block makes sense, you need to know what Q/K/V actually *are*. They are three **different views of the same input**, produced by three small linear layers.

#### The library card-catalog analogy

Imagine each token in the sentence is a person at a library:

| Lens | Role | Plain English |
| --- | --- | --- |
| **Q (Query)** | "What am I looking for?" | The thing this token wants to find in others |
| **K (Key)** | "What am I about?" | A label describing this token, used for matching |
| **V (Value)** | "What do I actually contain?" | The content this token will share if attended to |

Attention then asks: **"For each token's Query, which other tokens' Keys match best? Pull their Values."**

![Self-attention for token a²: q² is matched against every key, weights α′ blend the values](zhihu/images/1ecfc3b9b900.jpg)

Token $a^2$'s query is compared with every key $k^1..k^4$, giving weights $\alpha' _ {2,i}$; the output is the weighted sum $b^2 = \sum_i \alpha' _ {2,i} v^i$. Source: [Self-Attention note](<zhihu/Transformer学习笔记二: Self-Attention (自注意力机制).md>).

#### Where Q, K, V come from — three linear projections

Same input $X$, three different learned weight matrices:

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V
$$

<details>
<summary>🔍 Symbol atlas — Q, K, V projections</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $X$ | input matrix [$n$ tokens, $d _ {\text{model}}$], one row per token | the shared source of all three | on the left: every **row** (token) is multiplied by the same matrix ⇒ same projection for every token |
| $W^Q, W^K, W^V$ | learned matrices [$d _ {\text{model}}$, $d_k$] | three different "lenses" | on the right of $X$; superscripts are **labels**, not powers |
| $Q, K, V$ | results [$n$, $d_k$]: row $i$ = token $i$'s query / key / value | the inputs to attention | left sides |

</details>

![X times W^Q, W^K, W^V gives Q, K, V](zhihu/images/6c4e61c48ddf.jpg)

```text
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

That's literally three matrix multiplies — three "hidden layers" in [MLP language](#mlp-anatomy--hidden-vs-activation-layers) (linear, no activation).

#### Concrete dimensional example (GPT-2 small, single head)

```text
   X: [seq_len=10, d_model=768]
   W_Q, W_K, W_V: each [768, 768]    ← learned parameters

   Q = X · W_Q   →  [10, 768]
   K = X · W_K   →  [10, 768]
   V = X · W_V   →  [10, 768]
```

For **[multi-head](#multi-head-attention)**, that 768 gets split into `n_heads=12` heads of `d_head=64` each, run in parallel, then concatenated.

#### Why three different matrices, not one?

Each weight matrix is **learned to extract a different aspect** of the same token:

- `W_Q` learns "what does this token need from context?"
- `W_K` learns "what's worth advertising about this token?"
- `W_V` learns "what info does this token carry?"

If they were the same matrix, every token's "what I want" would equal "what I am" — the model couldn't ask different questions than it answers.

#### Tiny worked example

```text
   "the cat sat"  →  3 tokens, each becomes a vector of size d=4 (toy)
   X = [[1, 0, 1, 0],   ← "the"
        [0, 2, 0, 1],   ← "cat"
        [1, 1, 0, 0]]   ← "sat"

   W_Q = [[1,0,0,1],    Q = X·W_Q = [[1,0,1,1],
          [0,1,1,0],                  [1,2,2,1],
          [0,0,1,0],                  [1,1,1,1]]
          [1,0,0,1]]

   ... similarly K and V via their own W matrices
```

Q, K, V each end up as `[3 tokens × 4 dims]` — same shape as input, but each row now means something different.

> **Keep this:** Q, K, V are three learned linear projections of the same input. Q says what each token *wants*, K says what each token *offers*, V is what gets *shared*. Attention is matching Q against K, then pulling V accordingly.

### Scaled dot-product attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\negthinspace \left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Read it right-to-left as three steps:

```text
   scores  = Q · Kᵀ            ← every Query dotted with every Key  → [seq, seq]
   weights = softmax(scores / √d_k)   ← each ROW sums to 1
   output  = weights · V       ← weighted blend of Values           → [seq, d_v]
```

![softmax(QKᵀ/√d_k) gives a seq×seq score matrix, then multiplies V](zhihu/images/fe757a3975b1.jpg)

Per token, that's just the picture above in formula form: $b_i = \sum_j \alpha _ {ij} v_j$ with $\alpha _ {ij} = \text{softmax} _ j\big(q_i \cdot k_j / \sqrt{d_k}\big)$. That `Q·Kᵀ` matrix is the famous "attention pattern" you see visualized as a heatmap.

<details>
<summary>🔍 Symbol atlas — scaled dot-product attention</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $Q$ | queries [$n$, $d_k$] | what each token is looking for | first factor |
| $K^\top$ | keys transposed [$d_k$, $n$] | what each token offers | transposing makes $QK^\top$ an [$n$, $n$] table of every query · every key |
| $\sqrt{d_k}$ | square root of the key dimension | keeps scores at variance ~1 ([why](#why-divide-by-d_k)) | divides the scores **before** softmax |
| $\text{softmax}$ | applied row by row | turns scores into weights summing to 1 per query | wraps the scaled scores |
| $V$ | values [$n$, $d_v$] | the content to be mixed | rightmost: weights × $V$ = weighted average of value rows |
| $b_i$ | output for token $i$ | the new representation | per-token view, left side |
| $\alpha _ {ij}$ | how much token $i$ attends to token $j$ | one attention weight | row $i$, column $j$ of the weight matrix |
| $\text{softmax} _ j$ | softmax over $j$ (the keys) | normalizes across what token $i$ looks at | its subscript says which index is normalized |
| $q_i \cdot k_j$ | similarity of token $i$'s question and token $j$'s label | the raw match score | inside the softmax |
| $v_j$ | token $j$'s content | what gets passed along | multiplied by $\alpha _ {ij}$ and summed over $j$ |

</details>

#### Why divide by √d_k?

Suppose each entry of $q$ and $k$ is roughly independent with mean 0 and variance 1. Their dot product sums $d_k$ such terms:

$$
q \cdot k = \sum _ {m=1}^{d_k} q_m k_m \quad\Rightarrow\quad \text{Var}(q \cdot k) = d_k
$$

<details>
<summary>🔍 Symbol atlas — variance argument</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $q, k$ | one query and one key vector | the two vectors being scored | left side |
| $q_m, k_m$ | their $m$-th components ($m$ = dimension index here, not position) | the per-dimension contributions | multiplied pairwise inside the sum |
| $d_k$ | vector length | how many terms are summed | upper limit of the sum |
| $\text{Var}(q \cdot k)$ | spread of the score | what makes softmax saturate if large | each $q_m k_m$ has variance 1, so a sum of $d_k$ of them has variance $d_k$ |
| $\Rightarrow$ | "implies" | links the sum to its variance | between the two statements |

Dividing by $\sqrt{d_k}$ divides the variance by $d_k$, bringing it back to 1.

</details>

With $d_k = 64$, scores have standard deviation 8 → softmax becomes nearly one-hot → its gradient $s_i(\delta _ {ij}-s_j) \approx 0$ ([softmax saturation](#softmax--logits--probabilities)) → attention stops learning. Dividing by $\sqrt{d_k}$ brings the variance back to 1. Same "keep variance ≈ 1" idea as [He init](#initialization).

### Causal mask — no peeking at the future

A decoder (GPT) must predict token $t$ using only tokens $\le t$. So before softmax, add a mask $M$ that is $0$ where allowed and $-\infty$ where the key is in the future:

$$
\text{Attention}(Q,K,V) = \text{softmax}\negthinspace \left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V,
\qquad
M _ {ij} = \begin{cases} 0 & j \le i \cr -\infty & j \gt  i \end{cases}
$$

<details>
<summary>🔍 Symbol atlas — causal mask</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $M$ | mask matrix [$n$, $n$] | blocks attention to future tokens | added inside the softmax |
| $i$ | row = query position (the token doing the looking) | identifies whose view we are masking | first subscript |
| $j$ | column = key position (the token being looked at) | identifies what is visible | second subscript |
| $j \le i$ | key is at or before the query | allowed context | first case ⇒ $0$: adding 0 leaves the score unchanged |
| $j \gt  i$ | key is in the future | must stay hidden | second case ⇒ $-\infty$: $e^{-\infty} = 0$ weight |
| $+ M$ | adding the mask to the scores | implements the hiding | **after** scaling and **before** softmax, so the remaining weights still sum to 1 |

</details>

![Causal mask: 1s mark future positions that get replaced by a huge negative number](zhihu/images/138a5506a098.jpg)

Row $i$ = token $i$. The 1s (upper triangle) mark future positions; their scores become $-\infty$ (in code `-1e9`), so $e^{-\infty} = 0$ after softmax. Source: [Self-Attention note](<zhihu/Transformer学习笔记二: Self-Attention (自注意力机制).md>).

This one triangle is what makes a model **[autoregressive](#autoregressive-generation--how-llms-actually-talk)** — and it lets training compute the loss for **all positions in parallel** while each position still only sees its past.

### The attention block

```text
                Input X
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       Q=XWq    K=XWk    V=XWv
        │         │         │
        └─► Q·Kᵀ ◄┘         │
              │             │
              ▼             │
         scale ÷√d_k        │
         + causal mask      │
         (hide future)      │
              │             │
              ▼             │
           softmax          │
              │             │
              └─────► × V ◄─┘
                      │
                      ▼
                  output
```

### Multi-head attention

Run $h$ attention "heads" in parallel, each with its own smaller $W^Q_i, W^K_i, W^V_i$, then concatenate and mix with $W^O$:

$$
\text{head} _ i = \text{Attention}(XW_i^Q,\thinspace  XW_i^K,\thinspace  XW_i^V), \qquad
\text{MultiHead}(X) = \text{Concat}(\text{head} _ 1, \dots, \text{head} _ h)\thinspace  W^O
$$

with $d_k = d_v = d _ {\text{model}} / h$, so the total compute equals one big head.

<details>
<summary>🔍 Symbol atlas — multi-head attention</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $h$ | number of heads | more heads = more parallel patterns | upper index of the concat; divides $d _ {\text{model}}$ |
| $i$ | head index | names one head | subscript on each head's matrices |
| $W_i^Q, W_i^K, W_i^V$ | head $i$'s own projections [$d _ {\text{model}}$, $d_k$] | each head can look for a different pattern | inside head $i$'s attention call |
| $\text{head} _ i$ | output of head $i$ [$n$, $d_v$] | one head's view | left side; listed inside $\text{Concat}$ |
| $\text{Concat}$ | glue the heads side by side ⇒ [$n$, $h \cdot d_v$] | collects all heads | wraps the head list |
| $W^O$ | output projection [$d _ {\text{model}}$, $d _ {\text{model}}$] | mixes information across heads | multiplies the concatenation last |
| $d_k = d _ {\text{model}}/h$ | per-head width | keeps the total cost equal to one big head | stated under the formula |

</details>

![8-head attention: split, attend per head, concatenate, multiply by W^O](zhihu/images/696dfd8df49b.jpg)

- Different heads learn different things (coreference "it → animal", previous word, syntax, punctuation, …).

![Each colour is one head attending from "it" — heads focus on different words](zhihu/images/55291d070d8f.jpg)

### Attention variants (KV-cache savings)

During [generation](#autoregressive-generation--how-llms-actually-talk) we cache every past token's K and V. Its size per token is $2 \times n _ {\text{layers}} \times n _ {\text{kv}} \times d _ {\text{head}} \times \text{bytes}$. Example (7B-class, 32 layers, 32 KV heads × 128, fp16): ≈ 0.5 MB per token → **2 GB for a 4K context**, per sequence. Fewer K/V heads = proportionally smaller cache:

| | Q heads | K/V heads | Cache vs MHA | Used by |
| --- | --- | --- | --- | --- |
| **MHA** | N | N | 1× | original, GPT-2/3 |
| **GQA** | N | N/g (grouped) | 1/g | Llama-2 70B, Llama-3 |
| **MQA** | N | 1 | 1/N | PaLM |
| **MLA** | N | low-rank latent | ≪ | DeepSeek |

<details>
<summary>🔍 Symbol atlas — KV-cache size</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $2$ | one K and one V | both are cached | leading factor |
| $n _ {\text{layers}}$ | number of layers | every layer keeps its own cache | multiplies everything |
| $n _ {\text{kv}}$ | number of K/V heads | the knob GQA/MQA turn down | multiplies the per-head size |
| $d _ {\text{head}}$ | width of each head | numbers stored per head per token | per-head size |
| $\text{bytes}$ | bytes per number (fp16 = 2) | converts numbers to memory | last factor |

Multiply by the context length (and batch size) for the total cache.

</details>

### Full transformer block (×12 in GPT-2, ×80+ in frontier)

Everything from [§3](#3️⃣-stability-tricks--why-deep-nets-even-work) shows up here: [pre-norm](#pre-norm-vs-post-norm), [residuals](#residual-connections), plus attention and the [FFN](#inside-the-ffn--where-sota-hidden-layers-live).

```text
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

$$
h = x + \text{MHA}\big(\text{RMSNorm}(x)\big), \qquad
\text{out} = h + \text{FFN}\big(\text{RMSNorm}(h)\big)
$$

<details>
<summary>🔍 Symbol atlas — transformer block</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | residual stream entering the block [$n$, $d$] | the running representation | added back untouched ([pre-norm](#pre-norm-vs-post-norm)) |
| $\text{RMSNorm}$ | normalizes the sublayer input | keeps sublayer inputs at a stable scale | innermost: applied before each sublayer |
| $\text{MHA}$ | multi-head attention | mixes information **between** tokens | first sublayer |
| $h$ | stream after attention (**not** the number of heads) | the halfway result | left of the 1st equation, input of the 2nd |
| $\text{FFN}$ | feed-forward network | processes **each token on its own** | second sublayer |
| $x + \dots$, $h + \dots$ | residual additions | gradient highway and "learn only the change" | outermost in each equation |
| $\text{out}$ | block output | input of the next block | left of the 2nd equation |

</details>

End of stack: `RMSNorm → unembedding → softmax → next-token probs`.

### Inside the FFN — where SOTA "hidden layers" live

The FFN inside every transformer block is **a 2-layer [MLP](#mlp-anatomy--hidden-vs-activation-layers) applied per token**. This is where most of the model's parameters and "knowledge" live.

$$
\text{FFN}(x) = \text{GELU}(x W _ {\text{up}} + b_1)\thinspace  W _ {\text{down}} + b_2
$$

<details>
<summary>🔍 Symbol atlas — FFN</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x$ | one token's vector [$d _ {\text{model}}$] | the FFN works per token | leftmost input |
| $W _ {\text{up}}$ | [$d _ {\text{model}}$, $d _ {ff}$] | expands to the wider hidden size | first matrix |
| $b_1$ | bias [$d _ {ff}$] | per-unit threshold | added before the activation |
| $\text{GELU}$ | activation | the nonlinearity | between the two matrices |
| $W _ {\text{down}}$ | [$d _ {ff}$, $d _ {\text{model}}$] | projects back to the residual width | second matrix |
| $b_2$ | bias [$d _ {\text{model}}$] (Llama drops biases) | output offset | added last |

</details>

```text
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

#### Dimensional caveat — the expand-then-project pattern

| Model | Layer | Input dim | Output dim |
| --- | --- | --- | --- |
| **GPT-2 small** | FFN up-proj | 768 | **3072** (4×) |
| | activation (GELU) | 3072 | 3072 (no change) |
| | FFN down-proj | 3072 | 768 |
| **Llama-2 7B** | FFN gate + up-proj (SwiGLU) | 4096 | **11008** (~2.7×) |
| | activation (SwiGLU) | 11008 | 11008 |
| | FFN down-proj | 11008 | 4096 |

**Why expand to 4×?** More hidden width = more "scratch space" for the model to compute features in, before compressing back to the residual stream's dim. Empirically tuned — 4× is the sweet spot for GPT/BERT.

#### SwiGLU twist — why modern FFNs have 3 matrices

Vanilla FFN: `out = (ReLU(x · W_up)) · W_down`  → 2 matrices.

SwiGLU FFN (Llama, modern LLMs):

$$
\text{FFN} _ {\text{SwiGLU}}(x) = \big(\text{Swish}(xW _ {\text{gate}}) \odot xW _ {\text{up}}\big)\thinspace  W _ {\text{down}}, \qquad \text{Swish}(z) = z\thinspace \sigma(z)
$$

<details>
<summary>🔍 Symbol atlas — SwiGLU</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $W _ {\text{gate}}$, $W _ {\text{up}}$ | two parallel [$d$, $d _ {ff}$] matrices | one computes the gate, one the content | both applied to the same $x$ |
| $\text{Swish}(z) = z\thinspace \sigma(z)$ | smooth ReLU-like curve | turns the gate projection into "how open" each feature is | wraps only the gate branch |
| $\sigma$ | sigmoid | squashes the gate into $(0,1)$ inside Swish | inside the Swish definition |
| $\odot$ | elementwise product | gate value $j$ scales content feature $j$ | between the two branches |
| $W _ {\text{down}}$ | [$d _ {ff}$, $d$] | back to model width | applied last |

</details>

```text
   out = ( Swish(x · W_gate)  ⊙  (x · W_up) ) · W_down
                ↑                    ↑
           "what to keep"      "what to pass"
                  └── elementwise multiply ──┘
```

Three matrices: a gate, an up-projection, and a down-projection. The gate **learns which features to let through** at each token — sharper, more selective than plain ReLU/GELU.

**Where 11008 comes from:** three matrices instead of two would add 50% params, so the hidden width is shrunk to keep the budget equal: $3 \cdot d \cdot d _ {ff} = 2 \cdot d \cdot 4d \Rightarrow d _ {ff} = \tfrac{8}{3}d$. For $d = 4096$ that's ≈ 10923, rounded up to a hardware-friendly multiple → **11008**.

<details>
<summary>🔍 Symbol atlas — SwiGLU width</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $3 \cdot d \cdot d _ {ff}$ | parameters of three $d \times d _ {ff}$ matrices (SwiGLU) | the new cost | left side |
| $2 \cdot d \cdot 4d$ | parameters of a vanilla FFN (two matrices, $4\times$ width) | the budget to match | right side |
| $\tfrac{8}{3}d$ | the solution for $d _ {ff}$ | the SwiGLU width at equal parameters | after $\Rightarrow$ |

</details>

#### Param count check (per layer, GPT-2 small)

- Attention (Q,K,V,O proj): `4 · 768² ≈ 2.36M`
- FFN: `2 · 768 · 3072 ≈ 4.72M`  ← **~2× attention**
- Per block total: ~7M; ×12 blocks = ~85M of the 124M total (the rest is mostly the 50,257 × 768 embedding table).

#### Do SOTA LLMs still have hidden + activation layers?

**Yes — they're everywhere, just renamed and reshaped.** A modern transformer is mostly hidden + activation layers wired up cleverly:

| Where in transformer | What it is | Modern shape |
| --- | --- | --- |
| **FFN (feed-forward network) inside each block** | Two linear layers with an activation between them — literally a 2-layer MLP per token | Llama: `Linear(d→~2.7d) → SwiGLU → Linear(~2.7d→d)` |
| **Q/K/V/output projections in attention** | Linear layers (no activation between) | `Linear(d→d)` each |
| **LM head** | Final linear projection to vocab | `Linear(d→vocab)` |

The FFN is **2/3 of every transformer block's parameters**. So when people say "LLMs are mostly MLPs glued together with attention," this is what they mean.

> **Key takeaway:** Every "hidden layer" idea you learned for MLPs maps directly onto modern LLMs — just with `d_model = 768..16384`, a 4× (or 8/3× SwiGLU) expansion ratio, and SwiGLU instead of ReLU.

### Encoder vs decoder — BERT vs GPT

The same block can be wired two ways, and the only real difference is the [causal mask](#causal-mask--no-peeking-at-the-future):

![ELMo vs GPT vs BERT: bidirectional LSTM, left-to-right transformer, bidirectional transformer](zhihu/images/8cc4ab578689.jpg)

| | **Encoder (BERT)** | **Decoder (GPT, Llama)** |
| --- | --- | --- |
| Attention | bidirectional — sees left **and** right | causal — sees only the left |
| Pretraining task | **Masked LM**: hide 15% of tokens, predict them (+ next-sentence prediction) | **Next-token prediction** |
| Loss | $-\sum _ {i \in \text{masked}} \log P(x_i \mid \tilde{x})$ | $-\sum_t \log P(x_t \mid x _ {\lt t})$ |
| Good at | understanding (classification, retrieval, embeddings) | generation (chat, code) |

<details>
<summary>🔍 Symbol atlas — MLM vs LM loss</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\text{masked}$ | set of positions chosen for masking (15%) | only hidden tokens are scored | under the sum |
| $x_i$ | the original token at masked position $i$ | the answer | left of the bar |
| $\tilde{x}$ | the corrupted input (with `[MASK]` / random tokens) | what BERT actually sees | right of the bar: **both sides** visible, just with holes |
| $x _ {\lt t}$ | prefix only | what GPT sees | right of the bar in the decoder loss (see [LM loss](#loss-functions)) |

</details>

BERT's masking detail: of the chosen 15%, 80% → `[MASK]`, 10% → random token, 10% → unchanged — so the model can't rely on seeing `[MASK]`, which never appears at fine-tune time. Source: [BERT note](<zhihu/BERT学习笔记一: 基于论文精读的模型详解.md>), [GPT note](<zhihu/ChatGPT技术解析系列之: GPT1, GPT2与GPT3.md>).

Decoders won for LLMs: next-token prediction uses **every** token as a training signal and doubles as a generation procedure.

---

## 6️⃣ LLM Lifecycle

```text
  📚 PRETRAIN ──► 🎯 FINE-TUNE ──► 🤝 RLHF ──► 📊 EVAL ──► 🚀 DEPLOY
   (trillions      (instructions    (align with    (perplexity,
    of tokens,      via LoRA/        human         Lambada,         ┌──► sample
    next-token      QLoRA/full)      preference)   benchmarks)      │    Top-K
    prediction)                                                     │    Top-P
                                                                    │    temperature
                                                                    └──► generate
```

### Pretraining

- **Objective**: next-token prediction with [cross-entropy](#loss-functions), $\mathcal{L} = -\frac{1}{T}\sum_t \log P _ \theta(x_t \mid x _ {\lt t})$, computed for every position in parallel thanks to the [causal mask](#causal-mask--no-peeking-at-the-future).
- **Scaling Laws** (Chinchilla): performance ∝ f(data, params, compute). Balance all three — rule of thumb ~20 training tokens per parameter.
- **No dropout** — single-epoch regime, see [why](#why-modern-llms-dont-use-dropout-anymore).

#### From GPT-1 to GPT-3: scale replaces fine-tuning

![Timeline: Transformer → GPT-1 → BERT → GPT-2 → GPT-3](zhihu/images/baf512f34db7.jpg)

| Model | Year | Params | Big idea |
| --- | --- | --- | --- |
| GPT-1 | 2018 | 117M | pretrain on raw text, then fine-tune per task |
| GPT-2 | 2019 | 1.5B | **zero-shot**: with enough good data, skip fine-tuning |
| GPT-3 | 2020 | 175B | **few-shot / in-context learning**: put examples in the prompt |

![Zero-shot vs one-shot vs few-shot prompting](zhihu/images/4d329b7d29be.jpg)

Zero-shot = task description only; few-shot = description + a few examples, all inside the prompt, **no weight updates**. Source: [GPT note](<zhihu/ChatGPT技术解析系列之: GPT1, GPT2与GPT3.md>).

### Fine-tuning (SFT and LoRA)

**SFT (supervised fine-tuning)** = keep the same next-token loss, but on curated `(prompt, ideal response)` pairs — "show the model what a good answer looks like". InstructGPT used ~13K human-written demonstrations. (GPT-1 also kept the LM loss as a helper: $L = L _ {\text{task}} + \lambda L _ {\text{LM}}$.)

<details>
<summary>🔍 Symbol atlas — GPT-1 combined loss</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $L _ {\text{task}}$ | supervised loss of the downstream task | the main goal | first term |
| $L _ {\text{LM}}$ | language-model loss on the same text | keeps general language skill | second term |
| $\lambda$ | weight of the auxiliary term | balances the two goals | multiplies only $L _ {\text{LM}}$ |

</details>

| Method | What | Cost |
| --- | --- | --- |
| **Full FT** | update all weights | 💰💰💰 |
| **LoRA** | freeze $W$, learn a low-rank update $\Delta W = BA$ | 💰 |
| **QLoRA** | LoRA on a 4-bit quantized base | 🪙 |

**LoRA intuition:** the change fine-tuning needs is "low-rank" — a few directions matter. So instead of updating a $d \times k$ matrix, learn two thin ones:

$$
W' = W + BA, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll d
$$

For $d = k = 4096$, $r = 8$: $2 \cdot 4096 \cdot 8 = 65{,}536$ trainable params instead of 16.7M — **0.4%**.

<details>
<summary>🔍 Symbol atlas — LoRA</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $W$ | frozen pretrained weight [$d$, $k$] | keeps the base model's knowledge | first term, never trained |
| $W'$ | effective weight after fine-tuning | what the model uses | left side |
| $B$ | trainable [$d$, $r$] (initialized to 0, so $W' = W$ at the start) | half of the low-rank update | left factor |
| $A$ | trainable [$r$, $k$] | the other half | right factor |
| $r$ | rank, e.g. 8 | the bottleneck width | shared inner dimension of $B$ and $A$ |
| $\ll$ | "much smaller than" | states the whole point: $r$ is tiny | in $r \ll d$ |
| $BA$ | product [$d$, $k$]: same shape as $W$, rank $\le r$ | the update itself | added to $W$ |
| $\Delta W$ | the change fine-tuning makes | names the update | in the table, equal to $BA$ |
| $2 \cdot 4096 \cdot 8$ | $d \cdot r + r \cdot k$ trainable numbers | the cost of LoRA | compared with $d \cdot k$ for full fine-tuning |

</details>

### RLHF — reward model + PPO (and DPO)

![InstructGPT's 3 steps: SFT, reward model, PPO](zhihu/images/52c847ffe406.jpg)

Step 1 = [SFT](#fine-tuning-sft-and-lora). Step 2 = train a reward model on human rankings. Step 3 = optimize the policy against it with PPO. Source: [InstructGPT note](<zhihu/ChatGPT技术解析系列之: 训练框架InstructGPT.md>).

```text
  prompt ──► model ──► response A
                  └──► response B   ──► human picks ──► reward model ──► PPO/DPO
```

**RL vocabulary, mapped to an LLM:**

| RL term | In RLHF |
| --- | --- |
| Agent / **policy** $\pi(a \mid s)$ | the LLM itself |
| State $s$ | prompt + tokens generated so far |
| Action $a$ | the next token (or whole response) |
| Reward $r$ | reward-model score for the finished response |

<details>
<summary>🔍 Symbol atlas — RL notation</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\pi$ | policy: a probability distribution over actions | the LLM's next-token distribution | the function name |
| $a$ | action | what the policy picks | left of the bar |
| $s$ | state | what the policy sees | right of the bar: the action is chosen **given** the state |
| $r$ | reward | the feedback signal | one number per response |

</details>

More background: [RL / MDP note](<zhihu/强化学习解析一: 马尔可夫决策过程 (MDP) 和有模型学习.md>).

**Step 2 — reward model.** Humans rank $K$ answers; every pair (winner $y_w$, loser $y_l$) becomes a training example:

$$
\mathcal{L} _ {\text{RM}}(\theta) = -\thinspace \mathbb{E} _ {(x,\thinspace  y_w,\thinspace  y_l)}\Big[\log \sigma\big(r _ \theta(x, y_w) - r _ \theta(x, y_l)\big)\Big]
$$

<details>
<summary>🔍 Symbol atlas — reward-model loss</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\theta$ | reward-model parameters | what this loss trains | argument of $\mathcal{L}$ and subscript of $r$ |
| $x$ | prompt | the shared question | first argument of both scores |
| $y_w, y_l$ | preferred ("win") and rejected ("lose") responses | the human's choice | subscripts mark the winner and loser |
| $r _ \theta(x, y)$ | scalar score of response $y$ to prompt $x$ | the reward model's output | inside the gap |
| $r _ \theta(x, y_w) - r _ \theta(x, y_l)$ | score gap | only the **difference** matters: adding a constant to all scores changes nothing | argument of $\sigma$ |
| $\sigma$ | sigmoid | turns the gap into "probability the human prefers $y_w$" (Bradley–Terry model) | wraps the gap |
| $-\log$ | cross-entropy for the event "$y_w$ wins" | makes it a loss to minimize | outside $\sigma$ |
| $\mathbb{E} _ {(x, y_w, y_l)}$ | average over all labeled comparison pairs | trains on the whole dataset | subscript lists what is averaged over |

The InstructGPT paper also divides by $\binom{K}{2}$, the number of pairs from one ranking of $K$ answers, so prompts with many answers don't dominate.

</details>

Intuition: only the **gap** in scores matters; $\sigma$ turns the gap into "probability the human prefers $y_w$". It's [cross-entropy](#loss-functions) on pairwise comparisons.

**Step 3 — PPO with a KL leash.** Maximize reward, but don't drift too far from the SFT model (or it learns to game the reward model):

$$
\text{objective}(\phi) = \mathbb{E} _ {(x,y) \sim \pi _ \phi^{\text{RL}}}\Big[r _ \theta(x,y) - \beta \log \frac{\pi _ \phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}\Big] + \gamma\thinspace \mathbb{E} _ {x \sim D _ {\text{pretrain}}}\big[\log \pi _ \phi^{\text{RL}}(x)\big]
$$

- term 1: reward-model score — higher is better
- term 2: KL penalty — $\beta$ = how far the policy may wander from SFT
- term 3 ("ptx"): keep some pretraining loss so general abilities don't regress

<details>
<summary>🔍 Symbol atlas — PPO-ptx objective</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\phi$ | parameters of the policy being trained | what this objective updates | argument of the objective |
| $\pi _ \phi^{\text{RL}}$ | the policy (the LLM being optimized) | generates the responses | in the 1st expectation's subscript and the numerator of the ratio |
| $\pi^{\text{SFT}}$ | the frozen SFT model | the reference point | denominator of the log ratio |
| $(x, y) \sim \pi _ \phi^{\text{RL}}$ | prompts from the dataset, responses **sampled from the current policy** | RL learns from its own outputs | subscript of the 1st expectation |
| $r _ \theta(x, y)$ | reward-model score | what we want high | first term inside the brackets |
| $\beta$ | KL coefficient | how strongly drifting is punished | multiplies the log ratio |
| $\log \dfrac{\pi _ \phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}$ | per-sample KL estimate | positive when the policy makes $y$ more likely than SFT did | subtracted: a penalty |
| $\gamma$ | weight of the pretraining term | balances skill retention vs reward | multiplies the 2nd expectation |
| $D _ {\text{pretrain}}$ | original pretraining data | the text whose likelihood we protect | subscript of the 2nd expectation |
| $\log \pi _ \phi^{\text{RL}}(x)$ | LM log-likelihood of pretraining text | keeps general skills | added: also maximized |

</details>

**DPO** skips the separate reward model and PPO loop: it plugs the same pairwise idea directly into the policy, using log-prob ratios against a frozen reference model as an implicit reward:

$$
\mathcal{L} _ {\text{DPO}} = -\thinspace \mathbb{E}\left[\log \sigma\negthinspace \left(\beta \log \frac{\pi _ \theta(y_w \mid x)}{\pi _ {\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi _ \theta(y_l \mid x)}{\pi _ {\text{ref}}(y_l \mid x)}\right)\right]
$$

<details>
<summary>🔍 Symbol atlas — DPO</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\pi _ \theta$ | the policy being trained | what DPO updates | numerators |
| $\pi _ {\text{ref}}$ | frozen reference model (usually SFT) | the anchor | denominators |
| $\beta \log \dfrac{\pi _ \theta(y \mid x)}{\pi _ {\text{ref}}(y \mid x)}$ | implicit reward of $y$ | how much more likely the policy made $y$ than the reference | appears once for $y_w$, once for $y_l$ |
| $\beta$ | strength (plays the KL role) | limits drift from the reference | scales both log ratios |
| winner − loser | gap between the two implicit rewards | same structure as the reward model | argument of $\sigma$ |
| $\sigma$, $-\log$, $\mathbb{E}$ | sigmoid, cross-entropy, average | same roles as in $\mathcal{L} _ {\text{RM}}$ | wrapping the gap from inside out |

</details>

Same shape as $\mathcal{L} _ {\text{RM}}$ — simpler, more stable, now very common.

### Evaluation

- **Perplexity**: $\text{PPL} = e^{\mathcal{L}}$ — the exponentiated [cross-entropy](#loss-functions). Lower is better. Intuition: "the model is as unsure as if choosing uniformly among PPL tokens" (loss 2.3 → PPL ≈ 10).
- **Lambada / HellaSwag / MMLU**: benchmark suites
- **Task-specific**: code, math, reasoning. For code, **pass@k** = chance that at least one of $k$ samples passes the unit tests, estimated from $n \ge k$ samples with $c$ correct:

$$
\text{pass@}k = \mathbb{E} _ {\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

<details>
<summary>🔍 Symbol atlas — perplexity and pass@k</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $\text{PPL} = e^{\mathcal{L}}$ | exponent undoes the log | if $\mathcal{L}$ is the average $-\log p$, then $e^{\mathcal{L}}$ = 1 / (geometric mean of $p$) = "effective number of choices" | $\mathcal{L}$ sits in the exponent |
| $n$ | samples generated per problem (e.g. 200) | a large pool for a stable estimate | top of both binomials |
| $c$ | how many of them pass the tests | the number of good samples | subtracted in $n - c$ |
| $k$ | sampling budget we evaluate | the "@k" | bottom of both binomials |
| $\binom{n}{k}$ | ways to choose $k$ samples out of $n$ | all possible picks | denominator |
| $\binom{n-c}{k}$ | ways to choose $k$ samples **all from the $n - c$ wrong ones** | the all-wrong picks | numerator |
| ratio | probability that all $k$ picks are wrong | the failure probability | inside the brackets |
| $1 - \dots$ | probability that at least one is right | the success probability | flips the failure probability |
| $\mathbb{E} _ {\text{problems}}$ | average over the benchmark | one score for the whole test set | outermost |

</details>

Source: [Codex note](<zhihu/ChatGPT技术解析系列之: 赋予GPT写代码能力的Codex.md>).

### Autoregressive generation — how LLMs actually "talk"

**Autoregressive** = each output token is predicted from **all previous tokens** (its own past outputs included). The model generates **one token at a time**, feeds it back as input, and repeats. "Auto" = self, "regressive" = predicting from prior values.

```text
  step 1:  "The cat"                          ─► model ─► "sat"
  step 2:  "The cat sat"                      ─► model ─► "on"
  step 3:  "The cat sat on"                   ─► model ─► "the"
  step 4:  "The cat sat on the"               ─► model ─► "mat"
  step 5:  "The cat sat on the mat"           ─► model ─► <EOS>  ⏹
           └──────── grows each step ────────┘
```

Mathematically, the model factorizes a sentence's probability as a chain:

$$
P(x_1, x_2, \dots, x_n) = \prod _ {t=1}^{n} P(x_t \mid x_1, \dots, x _ {t-1})
$$

<details>
<summary>🔍 Symbol atlas — autoregressive factorization</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $x_1 \dots x_n$ | the tokens of a sequence | what we assign a probability to | arguments on the left |
| $P(x_1, \dots, x_n)$ | probability of the whole sequence | the joint probability | left side |
| $\prod_t$ | product over positions | chain rule of probability: any joint probability splits this way | right side, outermost |
| $P(x_t \mid x_1, \dots, x _ {t-1})$ | probability of token $t$ given its prefix | exactly what one forward pass outputs | factor in the product |
| $t = 1$ term | $P(x_1)$ with an empty prefix (in practice: given a start token) | the first factor | lower limit of the product |

Take $-\log$ of both sides and divide by $n$: you get the [LM loss](#loss-functions).

</details>

Each forward pass produces `P(next | everything-so-far)`; sampling picks one token; loop.

#### Why it matters

| Concept | Connection |
| --- | --- |
| **[Causal mask in attention](#causal-mask--no-peeking-at-the-future)** | Forces token *t* to only see tokens *≤ t* — this is what makes the model *autoregressive by construction*, both at training and inference |
| **[Next-token prediction loss](#pretraining)** | Pretraining objective is literally "predict xₜ from x₍<ₜ₎" for every position in parallel |
| **[KV cache](#attention-variants-kv-cache-savings)** | Since past tokens never change, we cache their K and V tensors — that's why GQA/MQA matter for inference speed |
| **Latency scales with output length** | N output tokens = N sequential forward passes (can't parallelize across time) |
| **[Sampling knobs](#sampling-at-inference-time) apply per step** | Temperature, top-k, top-p shape the distribution at *each* token, not the whole sequence |

#### Contrast: autoregressive vs. other paradigms

- **Autoregressive (GPT, Llama)** — left-to-right, one token at a time. Great for open-ended generation.
- **Masked (BERT)** — predict missing tokens with full bidirectional context. Great for understanding, not generation. See [encoder vs decoder](#encoder-vs-decoder--bert-vs-gpt).
- **Diffusion / non-autoregressive** — generate all tokens in parallel via iterative denoising. Faster but lower quality (still research).

> **Keep this:** Every chatbot reply you've ever seen is a `for` loop calling the model once per token, each call conditioned on the growing prefix. The "streaming" UX you see is just that loop yielding tokens as they're sampled.

### Sampling at inference time

Applied at every autoregressive step, on top of the [softmax](#softmax--logits--probabilities) output:

| Knob | Effect | Rule |
| --- | --- | --- |
| **Temperature** $T$ | <1 = sharper (deterministic), >1 = wilder | $p_i \propto e^{z_i / T}$ |
| **Top-K** | sample from K highest-prob tokens | keep the $K$ largest $p_i$, renormalize |
| **Top-P (nucleus)** | sample from smallest set whose cumulative prob ≥ P | sort by $p_i$, keep until $\sum p_i \ge P$, renormalize |

<details>
<summary>🔍 Symbol atlas — sampling</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $T$ | temperature | sharpens or flattens the distribution | divides each logit |
| $z_i$ | logit of token $i$ | the raw score | in the exponent |
| $\propto$ | "proportional to" — then normalize to sum 1 | hides the softmax denominator | between $p_i$ and $e^{z_i/T}$ |
| $p_i$ | probability of token $i$ | what we sample from | left side; also what top-k / top-p sort by |
| $K$ | how many top tokens to keep | cuts the long tail by count | in the top-k rule |
| $P$ (top-p) | cumulative-probability threshold, e.g. 0.9 (**not** the probability function) | cuts the tail by mass | in the top-p rule |

</details>

```text
  greedy   ─► always argmax       (boring, repetitive)
  temp=0.7 ─► softens distribution (balanced)
  top-p=0.9─► chops the tail        (diverse but safe)
```

Why sampling at all? See [stochastic.md](stochastic.md): greedy decoding loops and repeats; a little randomness gives more natural text.

### Model file formats

| Format | Notes |
| --- | --- |
| `.pth` | PyTorch native (⚠️ pickle = unsafe) |
| `.safetensors` | safe, fast (HuggingFace default) ✅ |
| `.onnx` | cross-framework |
| `.gguf` | llama.cpp / local CPU/GPU |
| `.coreml` | Apple devices |

---

## 7️⃣ Systems — GPUs and scaling

### GPU performance

`torch.compile` → **CUDA kernels** → **Triton** (Python-ish kernel DSL).
Rule of thumb: matrix multiplies dominate; fuse ops; mind memory bandwidth (moving data often costs more than the math).

### Pipeline parallelism — fitting a model across GPUs

When the model doesn't fit on one GPU, put consecutive layers on different GPUs. Naively, only one GPU works at a time:

![Naive model parallelism: one GPU busy at a time, the rest idle](zhihu/images/2af8fe0bff59.jpg)

With $K$ GPUs, the idle fraction (the **bubble**) is $\frac{K-1}{K}$ — 75% wasted with 4 GPUs.

**GPipe's fix:** split each mini-batch into $M$ **micro-batches** and stream them through like an assembly line:

![Pipeline with 4 micro-batches: GPUs overlap, the bubble shrinks](zhihu/images/e57bc8318fd7.jpg)

$$
\text{bubble fraction} = \frac{K-1}{K+M-1}
$$

<details>
<summary>🔍 Symbol atlas — pipeline bubble</summary>

| Symbol | Meaning | Why it appears | Why in this position |
| --- | --- | --- | --- |
| $K$ | number of GPUs (pipeline stages) | how deep the pipeline is | in numerator and denominator |
| $M$ | number of micro-batches | how much work flows through | only in the denominator |
| $K - 1$ | steps spent filling and draining the pipeline | the idle time | numerator: doesn't grow with $M$ |
| $K + M - 1$ | total steps to push $M$ micro-batches through $K$ stages | the total time | denominator: grows with $M$, so the idle share shrinks |
| $M = 1$ | no micro-batching | recovers the naive case | gives $\frac{K-1}{K}$ |

</details>

$K = 4$: $M = 1$ → 75%; $M = 16$ → $3/19 \approx$ 16%. GPipe found $M \ge 4K$ makes the bubble negligible.

**Activation checkpointing (re-materialization):** backprop needs every layer's activations, which eats memory. Instead, keep only each stage's input and **recompute** the forward pass during backward — trade ~30% more compute for far less memory. (This micro-batching is also why [BatchNorm](#batchnorm-vs-layernorm-vs-rmsnorm--which-axis) is awkward at scale and LayerNorm isn't.)

Source: [Pipeline parallelism note](<zhihu/图解大模型训练之: 流水线并行 (Pipeline Parallelism), 以Gpipe为例.md>).

---

## 🔑 Five Things to Remember

1. **A transformer is just**: tokenize → embed → (attention + FFN + residual + norm) × N → softmax.
2. **Deep training works because of 4 tricks**: He/Xavier init, RMSNorm pre-norm, residuals, AdamW + clipping — all keeping the [gradient product](#why-gradients-vanish-or-explode) near 1.
3. **Modern stack**: BPE tokenizer, RoPE positions, SwiGLU activation, GQA attention, RMSNorm pre-norm.
4. **Lifecycle**: pretrain (cross-entropy next-token) → fine-tune (LoRA) → align (RLHF/DPO) → eval → sample (top-p + temp).
5. **LLMs are *grown, not built*** — the workshop's "summoning ghosts" framing. We discover behavior empirically via ablation, not derivation.

---

## 🧮 Formula Cheat Sheet

Each formula's **🔍 Symbol atlas** (what every symbol means, why it's there, why it sits where it does) is the collapsible block right under it in its section.

| Concept | Formula | Section |
| --- | --- | --- |
| Neuron / layer | $A^{[l]} = g(W^{[l]}A^{[l-1]} + b^{[l]})$ | [MLP](#mlp-anatomy--hidden-vs-activation-layers) |
| Softmax | $e^{z_i} / \sum_j e^{z_j}$ | [Softmax](#softmax--logits--probabilities) |
| Cross-entropy (LM) | $-\frac{1}{T}\sum_t \log P(x_t \mid x _ {\lt t})$ | [Loss](#loss-functions) |
| Softmax + CE gradient | $p - y$ | [Loss](#loss-functions) |
| Gradient step | $w \leftarrow w - \eta \nabla_w \mathcal{L}$ | [Optimizers](#optimizers--sgd--adam--adamw) |
| He init | $\text{std} = \sqrt{2/n _ {in}}$ | [Init](#initialization) |
| LayerNorm | $\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ | [Norm](#normalization) |
| RMSNorm | $\gamma \odot x / \text{RMS}(x)$ | [Norm](#normalization) |
| Residual | $H(x) = F(x) + x$ | [Residuals](#residual-connections) |
| Sinusoidal PE | $\sin / \cos(t / 10000^{2k/d})$ | [Positional encoding](#step-3--sinusoidal-the-original-transformer-papers-fix) |
| RoPE | $q_m^\top k_n = x_m^\top W_Q^\top R _ {n-m} W_K x_n$ | [RoPE](#step-4--rope-rotary-position-embedding--what-sota-models-use-today) |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k} + M)V$ | [Attention](#scaled-dot-product-attention) |
| SwiGLU FFN | $(\text{Swish}(xW_g) \odot xW_u)W_d$ | [FFN](#swiglu-twist--why-modern-ffns-have-3-matrices) |
| LoRA | $W + BA$ | [Fine-tuning](#fine-tuning-sft-and-lora) |
| Reward model | $-\log\sigma(r(x,y_w) - r(x,y_l))$ | [RLHF](#rlhf--reward-model--ppo-and-dpo) |
| Perplexity | $e^{\mathcal{L}}$ | [Evaluation](#evaluation) |
| Temperature | $p_i \propto e^{z_i/T}$ | [Sampling](#sampling-at-inference-time) |
| Pipeline bubble | $(K-1)/(K+M-1)$ | [Systems](#pipeline-parallelism--fitting-a-model-across-gpus) |

---

## 🧠 Mind Map (compressed)

```text
LLM
├── Architecture
│   ├── Tokenizer (BPE, ~50K vocab)
│   ├── Embeddings (768d) + RoPE
│   ├── Transformer × N
│   │   ├── Pre-RMSNorm
│   │   ├── Multi-Head Attention (Q/K/V, ÷√d_k, causal mask, GQA)
│   │   ├── Residuals
│   │   └── FFN (SwiGLU, ~8/3× expansion)
│   └── Unembedding + Softmax
├── Training
│   ├── Loss: cross-entropy
│   ├── Backprop (autograd graph)
│   ├── Optimizer: AdamW (+ Muon)
│   ├── LR warmup + cosine decay
│   └── Gradient clip + weight decay
├── Init: He (ReLU) / Xavier (Tanh)
├── Lifecycle
│   ├── Pretrain (next-token, scaling laws)
│   ├── Fine-tune (SFT, LoRA/QLoRA)
│   ├── RLHF (reward model + PPO) / DPO
│   ├── Eval (perplexity, Lambada, MMLU, pass@k)
│   └── Inference (KV cache, temp, top-k, top-p)
└── Systems
    ├── Kernels (torch.compile, Triton)
    └── Pipeline parallelism (micro-batches, checkpointing)
```

---

## Further reading — zhihu deep dives

Local copies (with images) of the 猛猿 zhihu column, in [zhihu/](zhihu/):

| Article | Deepens |
| --- | --- |
| [Transformer 笔记一: Positional Encoding](<zhihu/Transformer学习笔记一: Positional Encoding (位置编码).md>) | [Positional encoding](#positional-encoding--from-one-hot-to-sinusoidal-to-rope) |
| [Transformer 笔记二: Self-Attention](<zhihu/Transformer学习笔记二: Self-Attention (自注意力机制).md>) | [Q/K/V](#q-k-v--the-three-lenses-on-each-token), [scaled attention](#scaled-dot-product-attention), [mask](#causal-mask--no-peeking-at-the-future), [multi-head](#multi-head-attention) |
| [Transformer 笔记三: BatchNorm & LayerNorm](<zhihu/Transformer学习笔记三: 为什么Transformer要用LayerNorm:Batch Normalization & Layer Normalization (批量&层标准化).md>) | [Normalization](#normalization), [pre-norm](#pre-norm-vs-post-norm), [warmup](#warmup--cosine-decay) |
| [Transformer 笔记四: ResNet](<zhihu/Transformer学习笔记四: ResNet (残差网络).md>) | [Vanishing gradients](#why-gradients-vanish-or-explode), [residuals](#residual-connections) |
| [Transformer 笔记五: Subword Tokenization](<zhihu/Transformer学习笔记五: Subword Tokenization (子词分词器).md>) | [Tokenizers](#tokenizers) |
| [BERT 笔记一](<zhihu/BERT学习笔记一: 基于论文精读的模型详解.md>) | [Embeddings](#embeddings), [encoder vs decoder](#encoder-vs-decoder--bert-vs-gpt) |
| [ChatGPT 系列: GPT1, GPT2 与 GPT3](<zhihu/ChatGPT技术解析系列之: GPT1, GPT2与GPT3.md>) | [Pretraining](#pretraining) |
| [ChatGPT 系列: InstructGPT](<zhihu/ChatGPT技术解析系列之: 训练框架InstructGPT.md>) | [Fine-tuning](#fine-tuning-sft-and-lora), [RLHF](#rlhf--reward-model--ppo-and-dpo) |
| [ChatGPT 系列: Codex](<zhihu/ChatGPT技术解析系列之: 赋予GPT写代码能力的Codex.md>) | [Evaluation (pass@k)](#evaluation) |
| [强化学习解析一: MDP](<zhihu/强化学习解析一: 马尔可夫决策过程 (MDP) 和有模型学习.md>) | [RLHF vocabulary](#rlhf--reward-model--ppo-and-dpo) |
| [图解大模型训练: 流水线并行](<zhihu/图解大模型训练之: 流水线并行 (Pipeline Parallelism), 以Gpipe为例.md>) | [Pipeline parallelism](#pipeline-parallelism--fitting-a-model-across-gpus) |
