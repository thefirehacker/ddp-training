# A Detailed Beginner Tutorial for `train_gpt2.py`

**Last reconciled with monorepo [`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) at commit `b73e1affdc61bf55c7b8744e9fbce3b168741d8f`.** Update this banner when the trainer changes materially.

**Choose your path**

- **~30 minutes — concepts + map of the real file:** [quickstart-train-gpt2.md](quickstart-train-gpt2.md) (read this first if you only have one sitting).
- **Full depth — Python, PyTorch, and line-by-line habits:** continue below (this document).

This tutorial is designed for students who know basic Python and only have a primary understanding of GPT-2. The goal is not just to read the file, but to understand **why each part exists**, **what concept it teaches**, and **how all the pieces fit together**.

---

## For [First Break AI](https://thefirehacker.github.io/firstbreakai/) learners

This lesson supports the **First Break AI** cohort ([site](https://thefirehacker.github.io/firstbreakai/)): a free, practice-first path toward training, inference, and shipping AI products. You do not need a specific degree—what matters is working through real code and building a portfolio.

**What this tutorial adds to the cohort**

- Connects **course fundamentals** (Python patterns like `@dataclass`, tensors, training loops) to a **real speedrun trainer** used in the NanoGPT speedrun line of work.
- Gives a **single coherent picture** of GPT-style models: embeddings, blocks, attention, **projections** (`c_attn`, `c_proj`, `lm_head`), and why **multi-head** attention exists.
- Prepares you to read Tyler Romero’s write-up and code without getting lost in naming: the paper *Attention Is All You Need* uses symbols like \(W^Q, W^K, W^V\); this codebase uses OpenAI/GPT-2 names like `c_attn` and `c_proj`. Both describe **learned linear maps**.

**Primary references (read alongside this file)**

- Tyler Romero — [NanoGPT speedrun worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) (motivation and journey).
- Upstream trainer (historical reference for a given step): [Tyler’s `train_gpt2.py` at commit `b3c32f8`](https://github.com/tyler-romero/nanogpt-speedrun/blob/b3c32f8937c1f4655c5eb9607970e03e351a6c08/src/train_gpt2.py) (your instructor may pin a different commit for a specific “step”).
- **This repository’s** live trainer (what you run locally / on Modal):  
  `nanogpt-speedrun/src/train_gpt2.py` — it follows the same ideas but may include **extra optimizations** (e.g. FlexAttention, rotary embeddings, Muon, profiling). When the text below differs from your file on a line-by-line detail, **trust your checked-in file** and use this tutorial for concepts.

**Prerequisites**

- Comfortable with: variables, functions, classes, imports, running a script from the terminal.
- High-level idea of GPT-2: tokens → embeddings → layers of self-attention + MLP → next-token logits.

**After this lesson you should be able to**

1. Explain what `@dataclass` is doing for `GPTConfig` / hyperparameters.
2. Trace one token position through **Q/K/V**, **heads**, attention output, and **`c_proj`**.
3. Name what **`c_proj`** is in plain language (a learned linear map / “projection” after attention).
4. Describe why multiple **heads** are used instead of one giant attention.
5. Locate where **distributed training (DDP)** and the **data loader** fit in the script.

**Cohort resources (accounts, path, showcase)**

- [Roadmap](https://thefirehacker.github.io/firstbreakai/roadmap.html) — where this lesson sits in the broader path.
- [Checklist](https://thefirehacker.github.io/firstbreakai/checklist.html) — Hugging Face, GitHub, and setup you may need before running jobs.
- **Suggested artifact after this lesson:** a short write-up (README section or blog draft) explaining **one** of: `CausalSelfAttention` + `c_proj`, or **why AdamW + Muon split**, or **one training step** from data load → loss → backward → optimizers; plus optional screenshot of a successful `torchrun` or Modal run.

---

## Truth table: paper vs OpenAI-style names vs this repo’s `train_gpt2.py`

Use this so you never confuse “the minimal GPT-2 story” with “what our file actually imports.”

| Concept | Common paper symbol | Typical GPT-2 / NanoGPT name | In **this** [`train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) |
|--------|---------------------|------------------------------|------------------------------------------------------------------------|
| Q/K/V bundle for all heads | \(W^Q, W^K, W^V\) | `c_attn` (`Linear(n_embd, 3*n_embd)`) | Same pattern in `CausalSelfAttention` |
| Attention output projection | \(W^O\) | `c_proj` | `nn.Linear(n_embd, n_embd)` after heads concatenate |
| Next-token logits | output embedding | `lm_head` | `Linear(n_embd, vocab_size)`; **weight-tied** with `wte` |
| Multi-head split | \(h\) heads of dim \(d/h\) | `n_head`, reshape to `(B, H, T, head_dim)` | `config.n_head`, `head_dim = n_embd // n_head` |
| Positional encoding | sinusoid table (original paper) | learned `wpe` **or** RoPE | **RoPE:** `Rotary`, `apply_rotary_emb` on Q and K |
| Causal mask | lower-triangular | causal | **Plus** document mask (EOT boundaries) + sliding window via **FlexAttention** `block_mask` |
| Normalization | LayerNorm (paper) | LayerNorm / RMSNorm | **`norm(x)`** → `F.rms_norm` (not a standalone `rmsnorm()` helper at top of file) |
| Optimizer | Adam (paper) | AdamW | **Two optimizers:** `AdamW` on **`lm_head` only**; **Muon** on **transformer blocks** `transformer.h` |
| LR schedule | — | cosine / warmup | `LambdaLR` + `get_lr`: warmup → flat → **warmdown** |

**Historical / other repos:** Tyler’s older steps or Karpathy `llm.c` `train_gpt2.py` may use a **single AdamW**, no RoPE, and a top-level `rmsnorm` helper. Treat those as pedagogical cousins, not a line-for-line match to this file.

---

We will go in a teaching order rather than strict file order:

1. Python basics used in the file
2. PyTorch basics used in the file
3. One-page **GPT-2 architecture** overview (before diving into attention math)
4. Transformer basics needed before the code makes sense (including **heads** and **projections**)
5. Walk through the model code piece by piece
6. Walk through the data loader
7. Walk through DDP and the training loop
8. Build a complete mental model of what happens during one training step

---

# 1. What this file is

`train_gpt2.py` is not just a model definition.
It is a **complete training program**.

That means it does all of the following:

* defines a GPT-style transformer
* defines how token data is loaded from binary files
* sets up multi-GPU training with PyTorch DDP
* performs training and validation
* logs metrics
* manages optimizer and learning rate schedule

So students should expect this file to mix together:

* Python language features
* neural-network building blocks
* transformer architecture ideas
* GPU/distributed systems ideas
* training-loop engineering

## 1.1 Which file is “the” trainer?

For **First Break AI** exercises in this monorepo, the trainer you execute is:

- **`nanogpt-speedrun/src/train_gpt2.py`**

Tyler’s public repo and blog often refer to the same path relative to that project: **`src/train_gpt2.py`**. A pinned snapshot for comparison is:

- [github.com/tyler-romero/nanogpt-speedrun @ `b3c32f8` …/src/train_gpt2.py](https://github.com/tyler-romero/nanogpt-speedrun/blob/b3c32f8937c1f4655c5eb9607970e03e351a6c08/src/train_gpt2.py)

Your local copy may differ slightly (extra features, refactors). **Concepts** (config, attention, loss, DDP) stay the same; **names and imports** are what you verify in your editor.

---

# 2. Python basics first

Before touching attention or DDP, students should be comfortable with the Python features this file uses.

## 2.1 Imports

At the top of the file, we see imports like:

* `os`, `sys`, `glob`, `time`, `math`
* `argparse`
* `dataclass`
* `numpy`
* `torch`

This immediately tells us the script needs:

* operating-system access
* command-line arguments
* math helpers
* file matching (`glob`)
* numeric arrays (`numpy`)
* tensor computation and training (`torch`)

A useful teaching habit is to ask students:

> “What does each import suggest this script is going to do?”

That turns imports from noise into clues.

---

## 2.2 What is a function?

A function is a named block of reusable code.

Example:

```python
def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)
```

This means:

* the function is called `rmsnorm`
* it takes inputs `x0` and optional `eps`
* it performs some computation
* it returns a value

Students should understand this before reading model code, because nearly every deep learning file is built from functions and methods.

---

## 2.3 What is a class?

A class is a blueprint for building objects.

In this file, classes are used for things like:

* attention layer
* MLP layer
* transformer block
* GPT model
* data loader

Example:

```python
class MLP(nn.Module):
```

This says:

* we are defining a class called `MLP`
* it inherits from `nn.Module`
* it is meant to behave like a PyTorch neural-network module

A class usually contains:

* **state**: variables stored inside the object
* **behavior**: methods that define what the object does

---

## 2.4 What is `@dataclass`?

One of the most important Python basics in this file is:

```python
from dataclasses import dataclass
```

and then:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

### What problem does `dataclass` solve?

Suppose you want a simple object whose job is only to hold configuration values.
Without `dataclass`, you would often write:

```python
class GPTConfig:
    def __init__(self, block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
```

That is repetitive boilerplate.

A `dataclass` lets Python write most of that constructor code automatically.

### What does it give us?

With `@dataclass`, Python automatically creates:

* an `__init__` method
* a readable representation when printed
* convenient field handling

So this works:

```python
cfg = GPTConfig()
print(cfg.n_layer)
```

and so does this:

```python
cfg = GPTConfig(n_layer=24, n_embd=1024)
```

### Why is this useful in ML code?

Because ML training scripts have many configuration values:

* vocabulary size
* number of layers
* number of heads
* embedding size
* sequence length

A dataclass keeps these grouped neatly.

### Teaching intuition

Tell students:

> A dataclass is like a neat labeled container for settings.

It is not “the model.” It is the bag of instructions the model uses.

### Extra pattern in this repo’s `train_gpt2.py`

The checked-in trainer may define **two** dataclasses inside `main()`:

- **`Hyperparameters`** — training knobs (batch size, paths, learning rates, etc.).
- **`GPTConfig`** — model shape (`n_layer`, `n_head`, `n_embd`, …), sometimes with `__post_init__` to assert `n_embd % n_head == 0` and set `head_dim`.

To update a config **without mutating** the original object, the code often uses:

```python
import dataclasses
model_config = dataclasses.replace(model_config, n_layer=24)
```

That is the **immutable-style** way to override fields when you sweep or test.

---

## 2.5 What is `if __name__ == "__main__"`?

Near the bottom, the file uses:

```python
if __name__ == "__main__":
```

This is Python’s standard “entry point” pattern.

It means:

* if the file is run directly, execute the code inside this block
* if the file is only imported from somewhere else, do not run that training setup automatically

This is important because the file defines many classes and functions first, but only starts training inside the main block.

---

## 2.6 What is `argparse`?

The file uses:

```python
parser = argparse.ArgumentParser()
parser.add_argument(...)
args = parser.parse_args()
```

This is how Python scripts accept values from the command line.

For example, if the user runs:

```bash
torchrun --nproc_per_node=2 src/train_gpt2.py --batch_size 4 --sequence_length 1024
```

then `argparse` reads those values and stores them in `args`.

So later the script can do:

```python
B, T = args.batch_size, args.sequence_length
```

### Why this matters

It means the same script can be reused with different settings without editing the code every time.

---

# 3. PyTorch basics before transformer code

Now that Python basics are clear, students need a few PyTorch basics.

## 3.1 What is a tensor?

A tensor is the basic data object in PyTorch.

You can think of it as a generalized array.

Examples:

* scalar: one number
* vector: 1D list of numbers
* matrix: 2D grid of numbers
* higher-rank tensor: 3D, 4D, etc.

In language-model training, tensors often store:

* token IDs
* embeddings
* hidden states
* attention matrices
* gradients

---

## 3.2 What is `nn.Module`?

PyTorch models and layers usually inherit from `nn.Module`.

Example:

```python
class CausalSelfAttention(nn.Module):
```

This gives the class PyTorch features such as:

* parameter registration
* device movement (`.cuda()`)
* train/eval mode
* compatibility with optimizers

You can think of `nn.Module` as the standard base class for trainable building blocks.

---

## 3.3 What is `forward()`?

A PyTorch module defines how input becomes output using a method called `forward`.

Example:

```python
def forward(self, x):
```

This says:

* when the layer receives input `x`
* this is the computation it should perform

In practice, when you write:

```python
y = module(x)
```

PyTorch internally calls:

```python
y = module.forward(x)
```

---

## 3.4 What are parameters?

Parameters are the trainable numbers inside the model.

For example, `nn.Linear` contains weights.
`nn.Embedding` also contains weights.

These weights are updated during training.

So when we say “the model learns,” what we really mean is:

> the optimizer changes the parameters so the model’s predictions improve.

---

## 3.5 What does `loss.backward()` do?

This is a crucial beginner concept.

* the model computes a loss
* `backward()` computes gradients of that loss with respect to parameters
* those gradients are stored in `.grad`
* the optimizer later uses those gradients to update weights

So training follows this core pattern:

1. forward pass
2. compute loss
3. backward pass
4. optimizer step

That pattern is the heartbeat of the whole script.

---

# 3.5 GPT-2 architecture: one picture before the details

This section answers: “What are we even building?” before we dive into Q/K/V and heads.

## Decoder-only transformer (what GPT-2 is)

The famous paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) describes an **encoder–decoder** architecture (e.g. for translation). **GPT-2 is not that full stack.** It is a **decoder-only** model: a stack of blocks that each token passes through **left-to-right**, with **causal** (no peeking at future tokens) self-attention.

Think of the flow:

```text
token IDs  →  token embeddings  →  [Block 1] → [Block 2] → … → [Block L]  →  logits over vocabulary
```

Each **Block** (simplified) is:

```text
x  →  norm  →  attention  →  add to x  →  norm  →  MLP  →  add to x  →  next block
```

That is the **pre-norm + residual** pattern used in GPT-2–style code (exact order and norm type can vary; this repo uses RMSNorm-style `norm` in the blocks).

## What each layer “does” in plain language

| Piece | Role |
|--------|------|
| **Token embedding (`wte`)** | Turns each token ID into a vector in \(\mathbb{R}^{d_{\text{model}}}\). |
| **Positional info** | Original GPT-2 used learned position embeddings; many modern trainers use **rotary** (RoPE) in attention—your file may use `Rotary` + `apply_rotary_emb` on Q and K. |
| **Self-attention** | Lets each position gather context from **earlier** positions (causal). |
| **MLP** | Processes each position **after** mixing; often “4× wider” then back (e.g. `4 * n_embd` → `n_embd`). |
| **LM head (`lm_head`)** | Final linear map from hidden size → **vocab size** = scores for next-token prediction. |

## “Why attention *and* an MLP?”

- **Attention**: *where to look* (routing information between positions).
- **MLP**: *transform each position* after that routing (rich non-linear computation per token).

Both are needed; neither alone replaces the other.

---

# 4. Transformer basics students need first

Before opening the model classes, students need a simple conceptual map of a GPT-style transformer.

A GPT-like model does this:

1. convert token IDs into vectors (embeddings)
2. add position information (embeddings and/or rotary)
3. pass vectors through many transformer blocks (attention + MLP each)
4. turn final hidden vectors into vocabulary logits
5. compare logits to the correct next token using cross entropy

The file implements exactly this pipeline.

---

## 4.1 What is a token?

A token is a chunk of text used by the model.
It may be:

* a full word
* part of a word
* punctuation
* special symbol

The model does not directly understand raw characters or meanings. It operates on token IDs.

So the sentence is first converted into numbers.

---

## 4.2 What is an embedding?

A token ID by itself is just an integer like `15496`.
That number has no geometry and no useful meaning for neural computation.

An embedding maps that token ID to a dense vector.

Example idea:

* token ID `15496` → vector of length 768

This vector is a learned representation.

### Intuition

A token embedding is like the model’s learned coordinate for that token inside a high-dimensional semantic space.

---

## 4.3 What is a position embedding?

If the model only sees token embeddings, then these two sequences contain the same tokens:

* `dog bites man`
* `man bites dog`

But their order is different.

Transformers need a way to know position.
So the file adds a **position embedding** to each token embedding.

That lets the model know whether a token is first, second, third, and so on.

---

## 4.4 What is self-attention?

Self-attention is the mechanism that lets each token look at other tokens in the same sequence and decide which ones are important.

Example sentence:

> “The cat sat on the mat because it was warm.”

When the model processes `it`, it may need to pay attention to earlier words to build a useful representation.

Self-attention is how that selective “looking back” happens.

---

## 4.5 What does “causal” mean in causal self-attention?

GPT is an autoregressive language model.
It predicts the next token.

That means when predicting position `t`, the model is allowed to use:

* tokens before `t`
* token `t` itself (depending on internal representation)

But it is **not** allowed to see future tokens.

This restriction is called **causal masking**.

So “causal self-attention” means:

> each token can attend only to itself and earlier tokens, never to later ones.

This preserves the next-token prediction setup.

### Supplemental: attention and KV cache (video)

This file is about **training** a transformer; it recomputes attention over the full sequence each step. **KV cache** is mainly an **inference** idea: when generating text token by token, you can **reuse** past keys and values instead of recomputing them for all prior positions—big speedup at decode time.

For a visual explanation of attention mechanics and how **KV cache** fits in, see this recommended talk:

- [Attention and KV cache (YouTube)](https://www.youtube.com/watch?v=80bIUggRJf4)

Watching it alongside **§4.6–§4.8** below will connect “what the tensors mean” to “what we cache when serving a model.”

---

## 4.6 What is a head?

This is one of the most important concepts to explain carefully.

In multi-head attention, the model does not perform just one single attention pattern.
It performs several attention patterns in parallel.
Each of these parallel attention channels is called a **head**.

### Intuition

You can think of a head as one “attention specialist.”

One head might learn to focus on:

* recent nearby tokens
* matching brackets
* subject–verb relationships
* punctuation boundaries
* repeated names

That is only intuition, but it is a useful beginner mental model.

### Why multiple heads?

If we used only one big attention mechanism, the model would have one single way of relating tokens.
With many heads, the model can look at the sequence in multiple ways simultaneously.

### In this file

If:

* embedding size `C = 768`
* number of heads `n_head = 12`

then each head gets:

```text
head_size = 768 / 12 = 64
```

So instead of one 768-dimensional attention operation, the model does 12 smaller 64-dimensional attention operations in parallel.

### Why split like this?

Because each head can learn a different pattern of relationships.
After that, the results are combined again.

### Why not use a single “fat” attention instead of many heads?

You *could* in principle run one attention with full width \(d_{\text{model}}\). Splitting into \(h\) heads of size \(d_{\text{model}}/h\) is the standard design because:

1. **Capacity to learn different relationships** — each head can specialize (syntax, long-range, local, etc.); empirically this works better than one monolithic pattern.
2. **Same parameter budget as one big bilinear interaction** — the total size of Q/K/V projections is structured so the model can represent multiple interaction modes in parallel.
3. **Hardware-friendly** — attention is run per head; libraries (and `scaled_dot_product_attention` / fused kernels) are optimized for this layout.

See also **§4.8** for how head outputs are merged via **`c_proj`**.

---

## 4.7 What are query, key, and value?

These names sound abstract at first, so use a practical interpretation.

For each token representation:

* **query** = what this position is looking for
* **key** = what this position offers as matching information
* **value** = the information this position contributes if selected

### Analogy

Imagine a library search system.

* Query: “What am I searching for?”
* Key: “What topics does this book match?”
* Value: “What content should I retrieve if I choose this book?”

In attention, each token creates a query, key, and value vector.
Then queries compare against keys to decide which values matter.

### Very important idea

Queries and keys decide **where to look**.
Values decide **what information to pull back**.

---

## 4.8 What is a “projection” (`c_attn`, `c_proj`, `lm_head`)?

Students often ask: *The “Attention Is All You Need” paper talks about matrices \(W^Q, W^K, W^V\). Where is “projection” in the paper, and why does this code say `c_proj`?*

### Short answer

In this codebase, a **projection** almost always means a **learned linear layer** (`nn.Linear`): it maps a vector (or batch of vectors) from one dimension to another by a matrix multiply plus optional bias. In math, that is a linear map; in engineering language people still say **projection** when the output is another vector space of the same or different size.

### Map paper notation → GPT-2 / NanoGPT-style code

| Idea in the paper | Typical symbol | In `train_gpt2.py` (naming) |
|-------------------|----------------|-----------------------------|
| Project input to queries, keys, values for all heads at once | \(W^Q, W^K, W^V\) (often separate) | One `nn.Linear(n_embd, 3 * n_embd)` named **`c_attn`** — output is split into Q, K, V |
| Project attention output back to model width | \(W^O\) (output projection) | **`c_proj`**: `nn.Linear(n_embd, n_embd)` after heads are concatenated |
| Map to vocabulary logits | (output embedding / unembedding) | **`lm_head`**: `Linear(n_embd, vocab_size)`; often **weight-tied** with token embeddings |

So **`c_proj`** is exactly the **output projection** \(W^O\) from the paper’s multi-head attention block: it mixes the concatenated head outputs back into a single \(d_{\text{model}}\)-dimensional vector per token.

**Why `c_attn`?** The OpenAI GPT-2 reference code uses **c** for “conv”-shaped linear layers (historical naming from the original TensorFlow). You can read **`c_attn`** as “the linear layer that computes the attention Q/K/V bundle.”

### Mental model for one token position

1. **Input** \(x \in \mathbb{R}^{d_{\text{model}}}\).
2. **`c_attn`**: \(x \mapsto [q \| k \| v]\) (concatenated; then split and reshaped into heads).
3. **Attention**: combines heads using Q/K/V (with causal mask, and optionally rotary).
4. **`c_proj`**: mixes the concatenated head outputs back to \(d_{\text{model}}\).
5. Later, **`lm_head`**: maps hidden state to logits over vocabulary for next-token prediction.

### Why this matters for First Break AI

When you read errors or shapes in the trainer, **every `nn.Linear` is a place dimensions change**. Naming (`c_proj` vs `lm_head`) tells you *which* stage of the pipeline you are in: inside the block vs at the output.

---

## 4.9 What is the MLP in a transformer block?

A transformer block has two big sub-parts:

1. attention
2. MLP

Attention mixes information **across tokens**.
The MLP then processes each token representation **independently at that position**.

So after attention lets a token gather context, the MLP transforms that enriched representation.

---

## 4.10 What is a residual connection?

A residual connection is when the input is added back to the output of a sublayer:

```python
x = x + sublayer(...)
```

### Why do this?

Because it helps information flow through deep networks.
Instead of forcing each layer to completely replace the representation, the layer only has to learn a useful adjustment.

### Intuition

Rather than saying:

> “Throw away the old state and start over,”

residual connections say:

> “Keep the old state, and add an improvement on top.”

---

## 4.11 What is normalization?

Normalization helps keep activations numerically well-scaled during training.

If activations grow too large or too unstable, training becomes harder.
So models often normalize hidden representations before or after major operations.

In this file, the normalization used is RMSNorm.

---

# 5. Walk through the model code from small pieces to big pieces

Now the students are ready to read the model code.

---

## 5.1 Normalization: `norm` (RMSNorm) in this repo

**Important:** Older tutorials and minimal `train_gpt2.py` snapshots may define a **`rmsnorm(x0)`** helper function at the top of the file. **This monorepo’s** trainer uses a small **`norm`** helper and PyTorch’s built-in RMSNorm-style op:

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

`Block` applies it before attention and before the MLP:

```python
x = x + self.attn_scale * self.attn(norm(x), block_mask)
x = x + self.mlp(norm(x))
```

### What this does

`F.rms_norm` rescales the last dimension so the root-mean-square (RMS) of the vector is controlled—same *idea* as the manual `rmsnorm` snippet you may see elsewhere, but implemented as one API call.

### Why use RMSNorm?

It is a lighter normalization choice than some older alternatives.
Its role here is to keep activations stable before attention and MLP.

### Beginner intuition

Think of RMSNorm as a way to keep a vector from being “too loud” or “too quiet.”

If you read external blog posts that paste a `def rmsnorm(...):` block, mentally map it to **`norm` / `F.rms_norm`** here.

---

## 5.2 `CausalSelfAttention`

This class implements multi-head causal self-attention.

### Repository note (read your actual file)

In **this monorepo’s** [`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py), attention may use **FlexAttention** (`flex_attention` + block masks for document boundaries and sliding windows) and **rotary embeddings** (`Rotary`, `apply_rotary_emb`), not only `F.scaled_dot_product_attention`. The **ideas** below (Q/K/V split, heads, `c_proj`) are the same; the **exact** `forward` code on your disk is the source of truth.

If you are comparing against [Tyler’s pinned `train_gpt2.py`](https://github.com/tyler-romero/nanogpt-speedrun/blob/b3c32f8937c1f4655c5eb9607970e03e351a6c08/src/train_gpt2.py), the structure may match that snapshot more closely than the evolving local file.

### Constructor

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
self.n_head = config.n_head
self.n_embd = config.n_embd
```

### What is `self.c_attn`?

This is one linear layer that produces Q, K, and V together.

If input has size `n_embd`, output has size `3 * n_embd`.
That means one matrix multiply produces:

* query part
* key part
* value part

This is efficient.

### What is `self.c_proj`?

After all heads finish attention, their outputs are recombined into one vector.
This projection mixes that combined information back into model dimension.

---

### Forward pass: line by line

```python
B, T, C = x.size()
```

This extracts tensor shape:

* `B` = batch size
* `T` = sequence length
* `C` = embedding dimension

### What is batch size?

Batch size is how many sequences are processed together.

### What is sequence length?

The number of token positions in each example.

### What is embedding dimension?

The width of each token representation vector.

---

```python
qkv = self.c_attn(x)
q, k, v = qkv.split(self.n_embd, dim=2)
```

This does one linear transform, then splits the result into three equal parts.

So now we have:

* `q`: query tensor
* `k`: key tensor
* `v`: value tensor

Each still has shape related to `(B, T, C)`.

---

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

### What does “reshape into heads” mean?

Originally each token has one embedding vector of width `C`.
But multi-head attention wants to split that width across multiple heads.

Suppose:

* `C = 768`
* `n_head = 12`

Then each head gets `64` features.

So instead of treating the tensor as:

```text
(B, T, 768)
```

we reinterpret it as:

```text
(B, T, 12, 64)
```

That means:

* batch size `B`
* sequence length `T`
* 12 separate heads
* each head has 64-dimensional vectors

### Why transpose?

After reshaping, the code does `.transpose(1, 2)` to make the shape:

```text
(B, n_head, T, head_size)
```

This layout is convenient because attention is computed per head.

### Beginner intuition

Reshaping into heads means:

> take one big representation and split it into several smaller “attention specialists,” each working in parallel.

---

### Minimal mental model (any implementation)

In the **simplest** PyTorch path you may see in courses:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

That is the core attention operation: compare Q to K, softmax, apply to V, with an optional causal mask.

### What happens conceptually?

For each token position:

1. compare its query with keys from allowed positions
2. turn those comparisons into attention weights
3. use those weights to combine the corresponding value vectors

### Why is it called “dot product” attention?

Because similarity between query and key is measured using dot products.

### Why “scaled”?

Because the dot products are scaled to keep the values numerically stable.

### Why causal?

Because GPT must not look at future tokens when training for next-token prediction.

### What **this repo** does instead

[`train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) uses **`flex_attention`** with a **`block_mask`** built from:

- **document boundaries** (tokens after `<|endoftext|>` do not attend across documents), and  
- **causal** + **sliding-window** constraints,

after **rotary embeddings** on Q and K. So the *math* is still “attention,” but the *kernel* is FlexAttention, not only `scaled_dot_product_attention`. When reading the source, follow `flex_attention(q, k, v, block_mask=block_mask)`.

---

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

### What is happening here?

Earlier, we split into many heads.
Now we are putting them back together.

If each head produced output of shape roughly:

```text
(B, n_head, T, head_size)
```

then after transpose and reshape we return to:

```text
(B, T, C)
```

This means all head outputs are concatenated side by side into one full-width representation per token.

### What does `contiguous()` do?

Some tensor operations like transpose change how data is viewed in memory.
`contiguous()` ensures the data is laid out in a form suitable for reshaping.

For beginners, you can say:

> it prepares the tensor memory layout so `.view(...)` works correctly.

---

```python
y = self.c_proj(y)
y = y / math.sqrt(24)
```

The projection mixes information after all heads are combined.
The division is an implementation detail of this trainer.

### Teaching note

It is okay to tell beginners:

* the important architectural idea is the projection
* the scaling detail is a tuning/stability choice in this specific implementation

---

## 5.3 `MLP`

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
```

This is the feed-forward part of the transformer block.

### Why expand to `4 * n_embd`?

A common transformer pattern is:

1. expand hidden width
2. apply nonlinearity
3. project back down

This gives the model more expressive capacity inside each block.

### Forward

```python
x = self.c_fc(x)
x = F.gelu(x)
x = self.c_proj(x)
```

### What is GELU?

`GELU` is a nonlinear activation function.
A neural network needs nonlinearities, otherwise many stacked layers would collapse into something much too simple.

### Intuition

The MLP is where each token’s representation gets transformed in a richer nonlinear way after attention has mixed context into it.

---

## 5.4 `Block`

```python
class Block(nn.Module):
```

A transformer block combines attention and MLP.

### Constructor

```python
self.attn = CausalSelfAttention(config)
self.mlp = MLP(config)
```

### Forward

**This repo** scales the attention residual (deep-stability trick):

```python
x = x + self.attn_scale * self.attn(norm(x), block_mask)
x = x + self.mlp(norm(x))
return x
```

with `attn_scale = 1 / (2 * n_layer) ** 0.5`. Minimal tutorials often use `x + attn(...)` without that scale, and omit `block_mask`.

### First line (minimal form)

```python
x = x + self.attn(norm(x))
```

This means:

1. normalize `x`
2. send normalized `x` into attention
3. add the attention output back to the original `x`

This is a residual update.

### Second line

```python
x = x + self.mlp(norm(x))
```

Now do the same idea for the MLP.

### Big picture

Each block says:

* first let tokens exchange information using attention
* then transform each token representation using the MLP
* use residual connections both times

### Beginner intuition

A block is one refinement stage.
After each block, token representations become more context-aware and more useful.

---

## 5.5 `GPTConfig`

This is the configuration container.

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

### What do these mean?

* `block_size`: maximum sequence length
* `vocab_size`: number of token IDs in the vocabulary
* `n_layer`: number of transformer blocks
* `n_head`: number of attention heads per block
* `n_embd`: width of hidden representation

### Why these matter

These values control model size, memory use, and compute cost.

---

## 5.6 `GPT`

This is the full language model.

### Constructor

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

### What is `wte`?

`wte` means token embedding table.
It maps token IDs to embedding vectors.

### What is `wpe`?

`wpe` means position embedding table.
It maps positions `0, 1, 2, ...` to learned vectors.

### What is `h`?

A list of transformer blocks.
If `n_layer=12`, this creates 12 blocks.

### What is `lm_head`?

This is the final linear layer that turns hidden states into logits over the vocabulary.

Each token position ends with a vector of size `n_embd`.
`lm_head` converts that into a vector of size `vocab_size`, one score for each possible next token.

---

### Weight tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

### What is weight tying?

It means the token embedding matrix and the output projection matrix share the same weights.

### Why do this?

This is a standard trick in language models that:

* reduces parameter count
* often works well empirically

### Beginner intuition

The same learned token-space geometry used to read tokens into vectors is reused when mapping vectors back to token scores.

---

### `_init_weights`

This initializes embeddings.

Students do not need to obsess over every initialization detail on the first pass.
The important lesson is:

> models need initial parameter values before training starts.

---

### `forward(self, idx, targets=None, return_logits=True)`

This is the full model computation.

#### Step 1: check shape

```python
b, t = idx.size()
assert t <= self.config.block_size
```

* `idx` is the input tensor of token IDs
* shape is `(batch, time)`
* sequence length must fit within maximum context window

---

#### Step 2: create positions

```python
pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
```

This creates position IDs:

```text
0, 1, 2, ..., t-1
```

These will be used to look up position embeddings.

---

#### Step 3: token embeddings

```python
tok_emb = self.transformer.wte(idx)
```

If `idx` has shape `(b, t)`, then `tok_emb` has shape:

```text
(b, t, n_embd)
```

because each token ID is replaced by its embedding vector.

---

#### Step 4: position embeddings

```python
pos_emb = self.transformer.wpe(pos)
```

This gives a tensor of shape:

```text
(t, n_embd)
```

Each position gets a learned vector.

---

#### Step 5: add them

```python
x = tok_emb + pos_emb
```

This combines token identity information and token position information.

### Why addition?

Because we want each token representation to carry both:

* what token it is
* where it occurs in the sequence

---

#### Step 6: pass through blocks

```python
for block in self.transformer.h:
    x = block(x)
```

This applies the transformer blocks one after another.

### Intuition

At every layer, token representations become more context-aware.
Later layers can represent more abstract relationships than earlier ones.

---

#### Step 7: final normalization

```python
x = norm(x)
```

One more normalization before output.

---

#### Step 8: logits and loss

If targets are provided:

```python
logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

### What are logits?

Logits are raw unnormalized scores for each vocabulary token.
They are not probabilities yet.

If vocabulary size is 50304, then for each token position the model outputs 50304 scores.

### What does cross entropy do?

Cross entropy compares the predicted logits against the true next token IDs.
It measures how wrong the model is.
Lower loss is better.

### Why flatten with `.view(-1, ...)`?

Because cross entropy here expects a simple 2D tensor of predictions and a 1D tensor of targets.
So batch and time dimensions are flattened together.

---

#### Inference-time shortcut

If `targets` is `None`:

```python
logits = self.lm_head(x[:, [-1], :])
```

This computes logits only for the last position.

### Why?

During generation, we usually only need the prediction for the newest token.
So this saves work.

---

# 6. How next-token prediction works

This concept is central enough to isolate.

The model is trained to predict the next token.

If the token sequence is:

```text
[A, B, C, D, E]
```

then the training pair is roughly:

* input: `[A, B, C, D]`
* target: `[B, C, D, E]`

This is exactly what the data loader creates.

So the model learns patterns like:

* after this context, what token usually comes next?

That is the core GPT objective.

---

# 7. The custom distributed data loader

Now let us understand the data side.

---

## 7.1 Why not raw text?

The script reads `.bin` files containing tokenized data.

Why?

Because pre-tokenized binary files are faster to load during training than raw text that must be repeatedly processed.

This matters in speedrun-style training.

---

## 7.2 `_peek_data_shard`

This function only reads the header of a binary shard.

It checks:

* magic number
* version
* number of tokens

### What is a magic number?

A magic number is a fixed value stored in a file header that helps verify the file format.

It is like a “format signature.”

If it does not match, the file may be wrong or corrupted.

---

## 7.3 `_load_data_shard`

This reads the full shard.

It first reads the header, then reads token data as `uint16`.

### Why `uint16`?

Because token IDs here can fit in 16 bits, which saves space.

---

## 7.4 `DistributedDataLoader`

This class manages token shards and makes sure each DDP process reads its own slice.

### Constructor inputs

* filename pattern
* batch size `B`
* sequence length `T`
* `process_rank`
* `num_processes`

### Why does a DDP-aware loader matter?

If every GPU process read the exact same examples, distributed training would waste compute.
We want different processes to work on different pieces of data.

---

## 7.5 `reset()`

```python
self.current_position = self.process_rank * self.B * self.T
```

### What does this do?

It offsets where each process starts reading.

For example, with two processes:

* rank 0 starts at one location
* rank 1 starts further ahead

That helps avoid identical batches.

---

## 7.6 `next_batch()`

```python
buf = self.tokens[self.current_position : self.current_position+B*T+1]
```

This reads exactly enough tokens to create:

* input batch of size `B*T`
* target batch shifted by one token

### Why `+1`?

Because to create shifted targets, we need one extra token.

---

```python
x = (buf[:-1]).view(B, T)
y = (buf[1:]).view(B, T)
```

This is the next-token setup.

### Important explanation

`x` is every token except the last one.
`y` is every token except the first one.
So `y` is `x` shifted by one position.

That is exactly how the model learns next-token prediction.

---

### Advancing the position

```python
self.current_position += B * T * self.num_processes
```

### Why multiply by `num_processes`?

Because after one step, all DDP processes together have consumed that many tokens in total.
So each process jumps ahead enough to avoid overlapping the other processes’ slices.

---

# 8. DDP: Distributed Data Parallel

Now we reach the systems part.

This is where many students get intimidated, so the explanation should be very slow.

---

## 8.1 What problem does DDP solve?

Suppose one GPU is too slow.
A natural idea is to use multiple GPUs.

In **data parallelism**, each GPU:

1. gets a copy of the same model
2. receives a different batch of data
3. computes gradients on its own batch
4. synchronizes gradients with the others
5. updates so all model copies stay identical

PyTorch’s DDP automates this pattern.

---

## 8.2 Why one process per GPU?

In PyTorch DDP, the common design is:

* one Python process controls one GPU

This is why the script reads rank information from environment variables.

---

## 8.3 `torchrun`

The file says `torchrun` sets the needed env variables.

That means `torchrun` launches multiple worker processes and gives each one metadata like:

* global rank
* local rank
* world size

---

## 8.4 `init_process_group(backend='nccl')`

This initializes the distributed communication system.

### What is `nccl`?

NCCL is NVIDIA’s communication library for GPU collectives.
It is commonly used for fast multi-GPU synchronization.

### Why initialize a process group?

Because the processes need to discover each other and coordinate communication.

---

## 8.5 Rank, local rank, and world size

```python
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
```

### `RANK`

The global ID of the process across the whole distributed job.

### `LOCAL_RANK`

The GPU index on the current machine.

### `WORLD_SIZE`

The total number of processes participating.

### Example

If you have 2 GPUs on one machine:

* process 0 → rank 0, local rank 0
* process 1 → rank 1, local rank 1
* world size = 2

---

## 8.6 Setting device

```python
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
```

Each process binds itself to its assigned GPU.

This is essential, otherwise both processes might try to use the same device incorrectly.

---

## 8.7 Master process

```python
master_process = ddp_rank == 0
```

Why have a master process?
Because you usually want only one process to:

* print logs
* write checkpoints
* initialize external logging like W&B

Otherwise you would get duplicate outputs from every process.

---

## 8.8 Wrapping model in DDP

```python
ddp_model = DDP(model, device_ids=[ddp_local_rank])
```

### What does this do?

It tells PyTorch:

* this model replica belongs to one distributed worker
* synchronize gradients with other workers during training

### Important beginner mental model

DDP does **not** mean one giant model split across GPUs.
It means:

> each GPU has a full copy of the model, but sees different data.

That is data parallelism.

---

# 9. Gradient accumulation

This is another crucial concept.

---

## 9.1 Why do we need it?

Sometimes the effective batch size you want is too large to fit in memory at once.

So instead of doing:

* one huge forward pass
* one huge backward pass

we do:

* several smaller forward/backward passes
* accumulate gradients across them
* perform one optimizer step at the end

That is gradient accumulation.

---

## 9.2 In the script

The script computes:

```python
tokens_per_fwdbwd = B * T * ddp_world_size * grad_accum_steps
```

This tells us the effective number of tokens contributing to one optimizer update.

### Meaning of each term

* `B`: micro-batch size per process
* `T`: sequence length
* `ddp_world_size`: number of processes / GPUs
* `grad_accum_steps`: number of accumulation rounds before optimizer step

So total effective batch in tokens is:

```text
B × T × world_size × grad_accum_steps
```

That is one of the most important formulas in the file.

---

## 9.3 Chunking into micro-batches

Inside training:

```python
for i, (micro_x, micro_y) in enumerate(zip(x.chunk(grad_accum_steps, dim=0), y.chunk(grad_accum_steps, dim=0))):
```

### What is happening?

The loader produced a larger batch.
Then the code splits it into smaller chunks along the batch dimension.

Each chunk is one micro-batch.

### Why do this?

Because smaller micro-batches fit into GPU memory more easily.

---

# 10. Mixed precision

The script uses:

```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
```

and then:

```python
with ctx:
```

### What is mixed precision?

It means some operations run in lower precision, such as `bfloat16`, instead of always using `float32`.

### Why?

Because this often:

* speeds up training
* reduces memory usage

### Beginner intuition

Think of it as using a more compact numeric format where safe, while still keeping training stable enough.

---

# 11. `torch.compile`

```python
model = torch.compile(model)
```

### What does this mean?

PyTorch tries to optimize execution of the model for better speed.

### What should beginners know?

You do not need to understand compiler internals.
The key idea is:

> the same model logic may run faster after PyTorch transforms and optimizes it.

---

# 12. Optimizer and learning rate schedule

This section matches **this monorepo’s** [`train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py). Many minimal GPT-2 tutorials use **one AdamW** for all parameters; **this trainer uses two optimizers** on disjoint parameter sets.

---

## 12.1 Two optimizers: AdamW + Muon

```python
optimizer1 = torch.optim.AdamW(
    model.lm_head.parameters(),
    lr=args.emb_learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
)
optimizer2 = Muon(
    model.transformer.h.parameters(),
    lr=args.learning_rate,
    momentum=0.95,
)
optimizers = [optimizer1, optimizer2]
```

| Optimizer | Parameters | Role (intuition) |
|-----------|------------|------------------|
| **AdamW** | `lm_head` only (output layer; tied with embeddings for weight tying) | Standard adaptive update with weight decay on the “word prediction” side. |
| **Muon** | All **transformer blocks** `transformer.h` (attention + MLP stacks) | Momentum + orthogonalized updates on 2D-ish weight matrices (see Muon reference in the file). |

### What does an optimizer do?

After gradients are computed, each optimizer updates **its** subset of parameters.

So the loop is:

* compute gradients with `backward()` (possibly scaled and averaged after accumulation)
* **both** `optimizer1.step()` and `optimizer2.step()` run each training step
* `scheduler.step()` for each (see below)

### Why not one AdamW for everything?

Speedrun-style training often splits **how** embeddings/head vs blocks are optimized. Muon is specialized for certain matrix parameters; embeddings/output head are usually kept on Adam(W). The file follows that pattern.

### Historical note

If you read [Tyler’s pinned snapshot](https://github.com/tyler-romero/nanogpt-speedrun/blob/b3c32f8937c1f4655c5eb9607970e03e351a6c08/src/train_gpt2.py) or older NanoGPT material, you may see a **single** `AdamW`. That is still valid pedagogy; **this repo’s** script is the more specialized version above.

---

## 12.2 Learning rate

The learning rate controls how large each parameter update is.

* **AdamW** uses `emb_learning_rate` for the head.
* **Muon** uses `learning_rate` for the blocks.

If either is too large, training may become unstable.
If too small, training may be too slow.

---

## 12.3 Schedulers

The script builds **one `LambdaLR` per optimizer**, both using the same `get_lr(it)`:

```python
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
```

`get_lr(it)` implements:

1. **Warmup** for the first `warmup_iters` steps (scale ramps from 0 to 1).
2. **Constant** multiplier `1.0` until the warmdown region.
3. **Warmdown** over the last `warmdown_iters` steps (linear decay to 0).

So it is **warmup → plateau → warmdown**, not only “cosine decay from the start.”

### What is warmup?

Start with smaller effective learning rates and ramp up.
This often helps early training stability.

### What is warmdown?

Reduce the multiplier near the end of training so updates shrink as you approach the final iterations.

---

# 13. Training loop in plain English

Now we can explain the loop very clearly.

---

## 13.1 Setup

Inside `main()`, the script:

* parses minimal CLI (`--disable_wandb`, `--notes`) and builds **`Hyperparameters`** + **`GPTConfig`** dataclasses
* initializes **NCCL** DDP and sets device from `RANK` / `LOCAL_RANK`
* **logs** the full source of `train_gpt2.py`, environment, and `nvidia-smi` (rank 0)
* builds `GPT`, moves to GPU, **`torch.compile`**, wraps in **DDP**
* builds train/val **`DistributedDataLoader`s**
* creates **AdamW + Muon** and **LambdaLR** schedulers
* optionally starts **memory snapshot** or **PyTorch profiler**
* starts the training timer

---

## 13.2 For each training step

At a high level, one iteration does this (see **§14** for accumulation detail):

1. Optionally run **validation** (every `val_loss_every` steps), `ddp_model.eval()`, accumulate val loss, **wandb.log**, check **speedrun target** `SPEEDRUN_TARGET`
2. **`ddp_model.train()`**
3. **Gradient accumulation:** loop `train_accumulation_steps` times:

   * `with ctx:` autocast **forward** `loss = ddp_model(x, y)` (this model returns **loss only**)
   * advance `x, y = train_loader.next_batch()`
   * `loss.backward()` with `ddp_model.no_sync()` on non-last micro-steps
4. **Average gradients** across accumulation: `p.grad /= train_accumulation_steps` for all parameters
5. **Step both optimizers** and **both schedulers**
6. **`ddp_model.zero_grad(set_to_none=True)`**
7. Log train loss, tokens/sec, **wandb** (rank 0)

**This repo does not use gradient clipping** (`clip_grad_norm_` does not appear). Other trainers often do; do not assume it here.

That is the whole training rhythm.

---

## 13.3 Validation

Validation runs with `ddp_model.eval()` and forward passes under **bf16 autocast** (`ctx`), matching this file:

```python
ddp_model.eval()
val_loader.reset()
for _ in range(val_steps):
    x_val, y_val = val_loader.next_batch()
    with ctx:
        loss = ddp_model(x_val, y_val)
```

### Why `eval()`?

It switches the model into evaluation mode.
Some modules behave differently in train vs eval mode.

### Why autocast instead of `torch.no_grad()` here?

The source comments note that `no_grad()` can interact badly with **`torch.compile`** for this model, so the trainer keeps gradients *technically* possible but **does not call `backward()`** during validation—only the forward loss. For a from-scratch trainer you might use `torch.no_grad()`; **trust this file’s pattern.**

---

## 13.4 Training mode

```python
ddp_model.train()
```

This switches the model back into training mode.

---

## 13.5 Forward and backward under autocast

```python
with ctx:
    loss = ddp_model(x, y)
loss.backward()
```

(`x`, `y` are full batch tensors; accumulation uses `no_sync` on non-final backward steps—see §14.)

### What is happening?

* run forward pass
* compute scalar **loss** (the `GPT.forward` in this repo returns **loss only**)
* compute gradients

Because this is inside the gradient-accumulation loop, gradients from multiple micro-batches add together before the optimizer step.

---

## 13.6 Gradient clipping

**Not used** in this repository’s `train_gpt2.py`.

Other tutorials often show:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

If you add clipping later, you would apply it **after** accumulation and **before** `optimizer.step()`—but the speedrun script relies on Muon/AdamW and scaling without clipping.

---

## 13.7 Optimizer steps

```python
for optimizer, scheduler in zip(optimizers, schedulers):
    optimizer.step()
    scheduler.step()
```

**Two** optimizers run every training step (AdamW on `lm_head`, Muon on blocks).

---

## 13.8 Zero gradients

```python
ddp_model.zero_grad(set_to_none=True)
```

### Why do this?

Because PyTorch accumulates gradients by default.
If you do not clear them, gradients from the next training step would keep stacking on top.

That would usually be wrong outside intentional accumulation.

---

# 14. What exactly happens in one full optimizer step?

This is the single most useful summary for students.

Suppose:

* 2 GPUs
* `B = 4`
* `T = 1024`
* `grad_accum_steps = 4`

Then one **global** optimizer update looks like this:

1. Each GPU process has its own copy of the model.
2. Each process loads different token slices via `DistributedDataLoader`.
3. Each process runs **`train_accumulation_steps`** micro-batches (using `ddp_model.no_sync()` except on the last backward to avoid extra all-reduces).
4. After the last backward in the accumulation window, DDP has synchronized gradients across GPUs.
5. Gradients are **averaged** by dividing each `p.grad` by `train_accumulation_steps`.
6. **Both** `AdamW` and `Muon` `step()`, and both `LambdaLR` schedulers `step()`.
7. `ddp_model.zero_grad(set_to_none=True)`.
8. Move to next batch (next `x, y` was already prefetched during the loop).

**No** `clip_grad_norm_` in this file.

So the model update is based on information from:

* multiple micro-batches (gradient accumulation)
* multiple GPUs (DDP)
* **two** optimizer steps per iteration (head vs blocks)

That is the core systems idea of this trainer.

---

# 15. The file’s execution flow from top to bottom

This map matches **[`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py)** (not every older blog snapshot).

1. Read entire script into `code` for logging; import stdlib, **`optuna`**, **`wandb`**, **`flex_attention`**, profiler, etc.
2. Define **memory snapshot** helpers (`start_record_memory_history`, `export_memory_snapshot`, …) and **`maybe_profile`**.
3. Define **`Muon`** optimizer and Newton–Schulz helpers.
4. Define **`norm`**, **`Rotary`**, **`CausalSelfAttention`**, **`MLP`**, **`Block`**, **`GPT`** (with **`get_attn_mask` / `get_block_mask`** for FlexAttention).
5. Define **`DistributedDataLoader`** and shard readers.
6. Define **`main()`** (nested **`Hyperparameters`** + **`GPTConfig`** dataclasses, minimal **`argparse`**).
7. Init **DDP** (`dist.init_process_group`), ranks, **`print0`**, logging path, optional **wandb**.
8. Log full source + environment + **`nvidia-smi`**.
9. Build **`GPT`**, `.cuda()`, **`torch.compile`**, **`DDP`**, bf16 **`autocast` ctx**.
10. Build train/val loaders; peek one batch.
11. Create **`AdamW` (`lm_head`)** + **`Muon` (`transformer.h`)** and **`LambdaLR`** schedulers.
12. Training loop: validation intervals, **`SPEEDRUN_TARGET`**, accumulation + backward, **dual optimizer step**, **wandb**, optional profiler step.
13. **`dist.destroy_process_group()`**, **`wandb.finish()`**, free GPU memory; return **`val_loss`** (e.g. for Optuna).
14. **`if __name__ == "__main__": main()`**

If students can narrate that flow, they understand the file structurally.

---

# 16. Common beginner confusions to clear up explicitly

## Confusion 1: Is DDP the same as model parallelism?

No.
DDP means each GPU gets a full copy of the model and different data.
Model parallelism means different parts of the model are split across devices.

This file uses data parallelism, not model parallelism.

---

## Confusion 2: Why do we need heads if we already have one embedding vector?

Because one single attention mechanism gives the model only one way to compare tokens.
Multiple heads let the model learn multiple attention patterns in parallel.

---

## Confusion 3: Why are Q, K, V all needed?

Because attention needs three roles:

* query: what am I looking for?
* key: what pattern do I match?
* value: what information do I provide if selected?

Without this structure, attention would not have the same flexible matching-and-retrieval behavior.

---

## Confusion 4: Why shift targets by one?

Because GPT is trained for next-token prediction.
It must learn to predict the next token from previous context.

---

## Confusion 5: Why both DDP and gradient accumulation?

Because they solve different things.

* DDP scales across GPUs.
* gradient accumulation lets you simulate larger effective batches than fit in memory at once.

They can be used together, and this file does exactly that.

---

# 17. Final mental model

If students remember only one paragraph, let it be this:

This file builds a **decoder-only** GPT-style model that maps token IDs to **token embeddings** (weight-tied with **`lm_head`**), runs **rotary** multi-head **FlexAttention** inside **transformer blocks** (each block: **RMSNorm** → attention → residual → **RMSNorm** → MLP → residual), applies **logit softcap** and **cross-entropy** loss, loads **FineWeb-style** `.bin` shards via a **distributed** data loader, trains with **DDP** + **gradient accumulation**, and updates parameters with **two** optimizers—**AdamW** on the output head and **Muon** on the stack of blocks—under a **warmup / flat / warmdown** LR schedule, logging to **wandb** and optionally **profiling** GPU memory and traces.

---

## 17.5 Verifiable self-check (First Break AI)

Answer from memory first, then skim **§4.8**, **§12**, and **§15**.

| # | Question | Pass if you can… |
|---|----------|-------------------|
| 1 | What is **`c_proj`**? | Say: linear map \(W^O\) after heads concatenate; mixes head outputs back to `n_embd`. |
| 2 | Why **multiple heads**? | Say: parallel attention patterns; head dim = `n_embd / n_head`. |
| 3 | Which **two** optimizers, and **on which parameters**? | **AdamW** → `lm_head`; **Muon** → `transformer.h`. |
| 4 | Does this repo use **gradient clipping**? | **No** (`clip_grad_norm_` not used). |
| 5 | What is **`flex_attention`** doing here vs plain SDPA? | Say: same attention math family; **block_mask** adds causal + document + window constraints. |
| 6 | What does **`train_accumulation_steps`** change? | Effective batch / how many micro-backwards before one global optimizer step (with grad scaling). |

**Office hours prep:** bring one **stuck** line from [`train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) and one **answered** question from the table above.

---

# 18. Best teaching strategy for this file

Teach it in three passes.

## Pass 1: Python and structure

Focus only on:

* dataclass
* classes
* functions
* argparse
* main block
* file control flow

## Pass 2: model and data

Focus only on:

* embeddings
* heads
* QKV
* attention
* MLP
* residuals
* next-token targets
* data loader

## Pass 3: systems and training

Focus only on:

* DDP
* rank/local rank/world size
* gradient accumulation and `no_sync`
* mixed precision (`autocast`)
* **AdamW + Muon** (two optimizers, two schedulers)
* validation, **wandb**, **SPEEDRUN_TARGET**
* optional profiling / memory snapshots

That order prevents beginners from mixing too many abstraction layers at once.

---

# 19. A simple classroom summary

You can summarize the file like this to students:

> This script is a full training engine. It defines a decoder-only GPT with FlexAttention and rotary, loads token batches from shards, trains with DDP and gradient accumulation, updates the head with AdamW and the blocks with Muon, and logs validation and speed toward a fixed loss target.

That is the real job of `train_gpt2.py` in **this** repository.

---

## Appendix: Reconciliation log (vs older tutorial drafts)

This deep tutorial originally followed a **minimal** `train_gpt2.py` narrative (top-level `rmsnorm`, single AdamW, `scaled_dot_product_attention` only, gradient clipping, simple `main`). The following were **aligned** to [`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) at `b73e1affdc61bf55c7b8744e9fbce3b168741d8f`:

- **Normalization:** `norm` / `F.rms_norm` instead of a file-level `rmsnorm` helper.
- **Block:** `attn_scale`, `block_mask`, and `norm` in `forward`.
- **Attention:** FlexAttention + masks + RoPE, not SDPA-only.
- **Optimizers:** AdamW (`lm_head`) + Muon (`transformer.h`); dual `LambdaLR`.
- **Training loop:** no `clip_grad_norm_`; `loss = ddp_model(x, y)`; accumulation + `no_sync`; validation under `ctx` not `no_grad`.
- **Execution map §15 and mental model §17:** rewritten for memory/profiler/Muon/wandb/optuna hooks.

Historical single-optimizer stories remain valid for **other** repos; see the **truth table** at the top.
