# Quick path: `train_gpt2.py` in one sitting (~30–40 minutes)

**Audience:** [First Break AI](https://thefirehacker.github.io/firstbreakai/) learners who know basic Python and have a rough idea what GPT-2 does.

**Full tutorial:** [tutorial.md](tutorial.md) (deep dive, same repo).

**Source of truth:** [`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py) at monorepo commit `b73e1affdc61bf55c7b8744e9fbce3b168741d8f` (see banner in `tutorial.md`).

---

## 1. What this file is

One script that **defines the model**, **loads token data**, **runs multi-GPU DDP training**, **logs to Weights & Biases**, and optionally **profiles** memory and performance. You are not reading a toy `model.py` in isolation—you are reading a **training system**.

---

## 2. Data flow (three sentences)

1. **Input:** integer token IDs `(B, T)` from `.bin` shards (FineWeb-style).
2. **Middle:** embeddings → stack of **transformer blocks** (each: RMSNorm → **FlexAttention** + RoPE → residual → RMSNorm → MLP → residual) → final RMSNorm → **softcapped logits** → **cross-entropy** loss.
3. **Output:** scalar loss; backward → **AdamW** (head) + **Muon** (blocks) → repeat.

---

## 3. The two optimizers (high signal)

| Optimizer | What it updates | Why split |
|-----------|------------------|-----------|
| `AdamW` | `lm_head` only | Standard adaptive + weight decay on the **vocabulary projection** (tied with embeddings). |
| `Muon` | `transformer.h` (all blocks) | Orthogonalized momentum for **2D** weight matrices in the stack—speedrun recipe. |

If you remember nothing else: **not** “one AdamW for the whole model.”

---

## 4. Attention in this repo vs “course minimal”

- **Concept:** same as everywhere—Q, K, V, heads, causal masking.
- **Implementation:** **Rotary** on Q/K; **FlexAttention** with a **block mask** = causal + **document** (EOT) + **sliding window**—not only `scaled_dot_product_attention`.
- **Extra (attention + KV cache):** [YouTube — attention and KV cache](https://www.youtube.com/watch?v=80bIUggRJf4) (inference-focused; training path in `tutorial.md` §4.5 supplement).

---

## 5. Execution order (read `main()`)

1. `Hyperparameters` + `GPTConfig` dataclasses; `dist.init_process_group`.
2. Log `train_gpt2.py` source + env + `nvidia-smi` (rank 0).
3. Build `GPT` → `.cuda()` → `torch.compile` → `DDP`.
4. Loaders; `AdamW` + `Muon` + `LambdaLR` ×2.
5. Loop: validation (optional), train with **accumulation** + `no_sync`, **grad average**, **dual** `step()`, `zero_grad`, wandb.

---

## 6. Cohort links (setup + showcase)

- [Roadmap](https://thefirehacker.github.io/firstbreakai/roadmap.html)
- [Checklist](https://thefirehacker.github.io/firstbreakai/checklist.html) (HF, GitHub, etc.)

**Suggested artifact:** 3–5 bullet “readme” explaining **one** of: dual optimizers, FlexAttention masks, or one training step; plus optional screenshot of a successful run.

---

## 7. Self-check (answer without peeking)

1. What does `c_proj` do after attention?
2. Name the two optimizers and which parameters each sees.
3. Does this file use gradient clipping?
4. What does `train_accumulation_steps` change about how often weights update?

---

## 8. Next step

Open [tutorial.md](tutorial.md) from **§2** onward for Python basics, then **§5** for line-by-line model reading, or jump to **§12–§15** for optimizers and execution flow aligned with this repo.
