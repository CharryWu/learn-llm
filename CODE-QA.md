# 🧪 Code Q&A — Running Log

> Deeper line-by-line annotations for code snippets from the LLM workshop.
> Each entry: snippet → line-by-line breakdown → focus question.

---

## Q1: What is `return_tensors='pt'` in the tokenizer?

### Snippet

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

self_prompt = "Complete the phrase: the meaning of life is "
tokens = tokenizer.encode(self_prompt, return_tensors='pt')
```

### Line-by-line

#### `from transformers import GPT2LMHeadModel, GPT2Tokenizer`

- `transformers` is HuggingFace's library — the de facto Python interface to thousands of pretrained models.
- `GPT2Tokenizer` — converts text ↔ token IDs using GPT-2's BPE vocabulary (50,257 tokens).
- `GPT2LMHeadModel` — GPT-2 transformer **with a Language Modeling head** stuck on top.
  - "LM head" = the final linear layer that projects the 1024-dim hidden state back to vocab-sized logits (one number per possible next token).
  - Without this head you'd just get hidden states, not next-token predictions.
  - Other variants exist: `GPT2Model` (no head), `GPT2ForSequenceClassification` (classification head), etc.

#### `tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")`

- `from_pretrained` downloads (or loads from cache `~/.cache/huggingface/`) the tokenizer files: `vocab.json`, `merges.txt`, special tokens config.
- `"gpt2-medium"` — picks one of the four GPT-2 sizes:

  | Name | Params | Layers | Hidden dim | Heads |
  |---|---|---|---|---|
  | `gpt2` (small) | 124M | 12 | 768 | 12 |
  | **`gpt2-medium`** | **355M** | **24** | **1024** | **16** |
  | `gpt2-large` | 774M | 36 | 1280 | 20 |
  | `gpt2-xl` | 1.5B | 48 | 1600 | 25 |

- All four sizes share the **same** tokenizer/vocabulary — only the weights differ. So tokenizer choice is just naming consistency.

#### `model = GPT2LMHeadModel.from_pretrained("gpt2-medium")`

- Downloads the **weights** (`pytorch_model.bin` or `model.safetensors`, ~1.5 GB for medium) and instantiates the architecture.
- Returns a `torch.nn.Module` in **eval mode by default** (dropout off, etc., when used via `.generate()`).
- Lives on **CPU** until you call `.to('cuda')` or `.to('mps')`.

#### `self_prompt = "Complete the phrase: the meaning of life is "`

- Plain Python string. Note the **trailing space** — GPT-2's BPE treats `" is"` and `"is"` as different tokens. Trailing space hints to the model "next token starts a new word."

#### `tokens = tokenizer.encode(self_prompt, return_tensors='pt')`

This is the focus line. Two things happen:

**(a) Tokenization** — text → list of integer IDs via BPE:

```
"Complete the phrase: the meaning of life is "
        │
        ▼  BPE merges
["Complete", " the", " phrase", ":", " the", " meaning", " of", " life", " is", " "]
        │
        ▼  vocab lookup
[20988,    262,    9546,     25,  262,   3616,    286,  1204,   318,   220]
```

**(b) `return_tensors='pt'`** — wrap that list in a **PyTorch tensor with a batch dimension**:

```
                          tokenizer.encode(...)
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
  return_tensors=None       return_tensors='pt'        return_tensors='tf' / 'np'
  (default)
  ─────────────             ─────────────                ─────────────
  Python list               torch.Tensor                 tf.Tensor / np.ndarray
  [20988, 262, ...]         tensor([[20988,262,...]])
  shape: 1-D, len N         shape: [1, N]  ← batch dim!  same shape, other framework
```

#### Why the batch dimension matters

Every PyTorch model expects `[batch_size, seq_len]`, even for a single prompt. Without `'pt'`:

```python
ids = tokenizer.encode(prompt)              # list[int], len N
tokens = torch.tensor(ids).unsqueeze(0)     # → shape [1, N]  (manual)
```

With `'pt'`:

```python
tokens = tokenizer.encode(prompt, return_tensors='pt')  # → shape [1, N]  (one step)
```

Skip the batch dim and the model errors out with a shape mismatch on the first matmul.

#### Valid values for `return_tensors`

| Value | Returns | Use when |
|---|---|---|
| `'pt'` | `torch.Tensor` | PyTorch model (most common) ✅ |
| `'tf'` | `tf.Tensor` | TensorFlow / Keras |
| `'np'` | `numpy.ndarray` | inspection, custom pipelines |
| `'jax'` | `jnp.ndarray` | JAX / Flax |
| `None` | `list[int]` | manual handling, debugging |

#### Mental rule of thumb
> If the next line is `model(tokens)` or `model.generate(tokens)` and the model is PyTorch → use `'pt'`. Always.

#### Quick sanity check you can run

```python
print(tokens)                    # tensor([[20988, 262, 9546, 25, 262, 3616, 286, 1204, 318, 220]])
print(tokens.shape)              # torch.Size([1, 10])  ← [batch=1, seq_len=10]
print(tokens.dtype)              # torch.int64
print(tokenizer.decode(tokens[0]))  # back to original string
```

---

## Q2: `tokenizer(...).input_ids` vs `tokenizer.encode(...)` — what's the difference?

```python
# Way A — call the tokenizer object, grab .input_ids
tokens = tokenizer(self_prompt, return_tensors='pt').input_ids

# Way B — call the .encode() method directly
tokens = tokenizer.encode(self_prompt, return_tensors='pt')
```

### Short answer
**For this snippet, both produce the identical tensor.** They differ in *what else* they return alongside the IDs.

- `tokenizer(...)` (calling the object) is the **full pipeline**. Returns a `BatchEncoding` dict-like object with `input_ids`, `attention_mask`, and sometimes `token_type_ids`. This is what you feed to the model with `**inputs` unpacking.
- `tokenizer.encode(...)` is a **convenience shortcut** that returns *only* the token ID tensor.

### Visual

```
                          input string
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
   tokenizer(text, return_tensors='pt')        tokenizer.encode(text, return_tensors='pt')
        │                                             │
        ▼                                             ▼
   BatchEncoding {                                tensor([[20988, 262, ...]])
     'input_ids':      tensor([[20988,...]]),    ← just the IDs, nothing else
     'attention_mask': tensor([[1,1,1,...]]),
     # 'token_type_ids' for BERT-family
   }
        │
        └─► .input_ids  ──► same tensor as encode() returns
```

### What you get back

| | `tokenizer(text)` | `tokenizer.encode(text)` |
|---|---|---|
| Return type | `BatchEncoding` (dict-like) | `list[int]` or `Tensor` |
| Has `input_ids` | ✅ via `.input_ids` | itself **is** the ids |
| Has `attention_mask` | ✅ | ❌ |
| Has `token_type_ids` | ✅ for BERT-style | ❌ |
| Supports batched input (list of strings) | ✅ natural | ✅ via `.batch_encode_plus` (older API) |
| Supports padding/truncation kwargs | ✅ first-class | ✅ but more awkward |
| Recommended by HF docs | ✅ **yes** | legacy-ish but still fine |

### Why `attention_mask` matters (and why `.encode()` skips it)

When you pad multiple prompts to the same length, you need a mask telling the model which positions are real vs. padding:

```
prompts = ["hi", "hello world"]

tokenizer(prompts, padding=True, return_tensors='pt')
  → input_ids:      [[31373,    0,     0],     ← "hi" + 2 pads
                     [31373, 995,  220]]       ← "hello world"
  → attention_mask: [[    1,    0,     0],     ← attend to position 0 only
                     [    1,    1,     1]]     ← attend to all three
```

For a **single unpadded prompt** (your snippet), the mask is all 1s and harmless to drop — that's why both ways work.

### Idiomatic usage

```python
# Way A — preferred for real model inference (passes mask along)
inputs = tokenizer(self_prompt, return_tensors='pt')
output = model.generate(**inputs)               # unpacks input_ids + attention_mask

# Way B — fine for quick exploration / single prompts
tokens = tokenizer.encode(self_prompt, return_tensors='pt')
output = model.generate(tokens)                 # works, no mask passed
```

### When the difference actually bites you
1. **Batched prompts of different lengths** → without `attention_mask`, padding tokens leak into attention scores and corrupt outputs.
2. **`.generate()` with `pad_token_id` set** → HF will warn if you pass `input_ids` only and there's padding.
3. **Models that need `token_type_ids`** (BERT, RoBERTa for some tasks) — `.encode()` won't give them to you.

### Rule of thumb
> Use `tokenizer(...)` + `**inputs` unpacking by default. Reach for `.encode()` only for single-prompt exploratory code where you just want the ID tensor.

---
