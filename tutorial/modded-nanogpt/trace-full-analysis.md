# Perfetto Trace Full Analysis
**Trace:** `modal_16.1781887331130109650.pt.trace.json` (181 MB)  
**Duration:** 1s 847ms 693µs 645ns  
**Run Stats:** step_avg_ms=67.8, val_loss=3.27781, peak_mem_alloc=30933 MiB

---

## Three Questions Answered

### Q1: Does the trace capture data loading?

**Short answer: No — data loading is NOT captured inside the profiler window.**

The profiler is only enabled on `master_process` (rank 0) and is configured with:
```python
_profiler_schedule = schedule(skip_first=2, wait=1, warmup=1, active=3, repeat=1)
```
This skips 2 steps, waits 1, warms up 1, then captures only 3 active steps. The `profiler_context.step()` is called at line 1992 (`<module>` scope) **after** each training step:

```python
# train_gpt.py lines 1986-1992
for idx in range(grad_accum_steps):
    inputs, targets, ... = train_loader.send(...)    # data load  ← INSIDE step
    model(...).backward()                             # forward + backward
training_manager.step_optimizers(step)               # optimizer
profiler_context.step()                              # profiler tick AFTER
```

The `train_loader.send()` IS inside the profiler window (you CAN see `train_gpt.py(1483): distributed_data_generator` in Image 10), but the actual disk I/O happens in a background thread (`DataPreloader._load` at line 1396) that ran during the previous step. By the time `.send()` is called, the shard is already in CPU RAM — so what you see in the trace is just CPU index math + `get_bigram_hash` + non-blocking H2D transfer, all sub-millisecond.

The `train_gpt.py(1992): <module>` green bar you see spanning each step is the profiler's frame — it covers the entire training loop body. It does NOT mean data loading is excluded; it means the profiler is active for all code inside that step.

---

### Q2: Why does `distributed_c10d.py` appear in the trace at profiler startup?

In Image 9 (very start of profiling, around ns 360000–385000) you see this stack:

```
train_gpt.py(1992): <module>
  torch/profiler/profiler.py(882): step
    torch/profiler/profiler.py(916): _transit_action
      torch/profiler/profiler.py(229): start_trace
        torch/profiler/profiler.py(393): _get_distributed_info
          torch/distributed/distributed_c10d.py(1387): get_backend
```

This is the **profiler itself** calling `_get_distributed_info()` to record metadata about the distributed setup (backend, world size, ranks) into the trace header. It calls `get_backend()` to confirm NCCL is the backend. This is not your training code — it is the profiler annotating itself before the first captured step begins.

The `distributed_c10d.py` entries you see **during actual training steps** (Image 6 — `_launch_reduce` → `c10d_logger` → `distributed_c10d.py(4501)` → `nccl_reduce_scatter_base`) are completely different — those are the live gradient all-reduce calls over NCCL during the backward pass.

---

### Q3: What is `train_gpt.py(1483): distributed_data_generator` and `get_bigram_hash` doing in the trace?

In Image 10 (around ns 2540000–2565000) you see:

```
train_gpt.py(1992): <module>
  <built-in method send of generator object>
    train_gpt.py(1483): distributed_data_generator
      train_gpt.py(1352): next_batch
```

`train_gpt.py(1483)` is the `yield` point inside `distributed_data_generator`. When `train_loader.send(args)` is called at line 1987, Python resumes the generator from the previous `yield`, runs the next loop iteration (calls `next_batch`, builds input/target tensors, computes bigram hash), then yields the batch back. The trace captures this generator resumption frame.

`get_bigram_hash` (line 1404) is called at line 1481 **inside** the generator on CPU, before the tensors are moved to GPU:
```python
# train_gpt.py lines 1481-1488
_bigram_inputs = get_bigram_hash(_inputs)           # CPU: XOR hash of adjacent tokens
new_params = yield (
    _inputs.to(device="cuda", non_blocking=True),
    _targets.to(device="cuda", non_blocking=True),
    _cum_lengths.to(device="cuda", non_blocking=True),
    _bigram_inputs.to(device="cuda", non_blocking=True)
)
```

It XORs adjacent token IDs (`rand_int_1 * x[i] XOR rand_int_2 * x[i-1]`) to produce a bigram feature index. This feeds the model's `bigram_lambdas` in the attention residual mix — a learned per-bigram skip connection. The tiny teal bars at the bottom of the trace row are the non-blocking H2D (CPU→GPU) DMA transfers that happen after the yield.

---

## Full Execution Flow (startup → training step)

### Phase 1: Process Group Init (before profiler — not in trace)
```
train_gpt.py(1760-1764):
  dist.init_process_group(backend="nccl", device_id=device)
  dist.barrier()
```

### Phase 2: Model + Optimizer Setup (not in trace)
```
train_gpt.py(1853): GPT(...).cuda()           # construct model on GPU
train_gpt.py(1869): dist.broadcast(param, 0)  # sync rank-0 weights to all GPUs
train_gpt.py(1871): torch.compile(model)      # torch.dynamo compilation
train_gpt.py(1872): TrainingManager(model)    # build NorMuonAndAdam optimizer
```

### Phase 3: Kernel Warmup (not in trace)
```
train_gpt.py(1881): train_loader = distributed_data_generator(train_files, ...)
train_gpt.py(1888): for step in warmup_steps:  # {0, 1, 2} + schedule transitions
train_gpt.py(1897):   inputs, ... = train_loader.send(...)
train_gpt.py(1898):   model(...).backward()     # triggers dynamo graph capture + triton compilation
train_gpt.py(1899):   training_manager.step_optimizers(step)
train_gpt.py(1900-1904): reset model state, delete loaders
```

### Phase 4: Profiler Setup + Training Loop Start
```
train_gpt.py(1910): train_loader = distributed_data_generator(...)   # fresh loader
train_gpt.py(1914-1930): profile(skip_first=2, wait=1, warmup=1, active=3)
train_gpt.py(1929): profiler_context.__enter__()
train_gpt.py(1938): for step in range(train_steps + 1):
```

The profiler fires at step `skip_first + wait + warmup = 4`. Steps 0-3 run outside the profiler window.

---

### Phase 5: Per-Step Training (IN TRACE — what you see in Perfetto)

Each captured step corresponds to one repetition of this block:

**5a. Data batch preparation**
```
train_gpt.py(1987): train_loader.send(send_args)
  → resumes generator at train_gpt.py(1483)
    train_gpt.py(1352): next_batch()          # slices start/end offsets from shard BOS index
    get_bigram_hash(_inputs)                  # CPU XOR hash: line 1404
    yield tensors.to("cuda", non_blocking=True)  # H2D DMA (tiny teal bars in trace)
```

**5b. Forward pass — `train_gpt.py(1199): GPT.forward`**

This is the largest span in the trace. Under `torch.compile` the whole forward is fused into a `Torch-Compiled Region`.

```
train_gpt.py(1988): model(inputs, targets, cum_seqlens, bigram_inputs, ...)
  torch/dynamo/eval_frame.py(453): __call__        # dynamo trampoline
    nn.Module: OptimizedModule_0
      nn.Module: GPT_0
        Torch-Compiled Region: N/0                 # all of forward fused by inductor
          train_gpt.py(1199): forward()

            # --- Embeddings ---
            train_gpt.py(1229): self.embed(input_seq)           # token embed lookup [T, dim]
            train_gpt.py(1230): self.bigram_embed(bigram_input_seq)  # bigram skip embed [T, dim]
            train_gpt.py(1233): value_embed(input_seq) × 5     # value embeddings for ve layers

            # --- Smear gate (1-token forward blending) ---
            train_gpt.py(1239): smear_gate(x[1:])              # learned blend of x[t-1] into x[t]

            # --- Per-layer transformer loop (11 layers) ---
            for i in range(11):
              train_gpt.py(1281): Block[i].forward(x, attn_args, qkvo_w, c_fc, c_proj)
                CausalSelfAttention.forward / PairedHeadCausalSelfAttention.forward
                  train_gpt.py(950):  q,k,v = F.linear(x, qkvo_w)        # QKV projection
                  train_gpt.py(951):  q,k = RMSNorm(q), RMSNorm(k)       # QK norm
                  train_gpt.py(952):  yarn.rotary(q), yarn.rotary(k)      # RoPE / YaRN
                  train_gpt.py(963):  flash_attn_varlen_func(...)          # FA3 varlen attention
                  train_gpt.py(967):  y * sigmoid(attn_gate)              # gated attention output
                  train_gpt.py(969):  F.linear(y, qkvo_w[dim*3:])         # output projection
                MLP block:
                  F.linear(x, c_fc)   → F.gelu → F.linear(x, c_proj)     # fused MLP

              # skip connections at layers 3 (save) and 6 (inject)
              train_gpt.py(1269-1270): skip gate + add

            # --- Output ---
            train_gpt.py(1288): x -= backout_lambda * x_backout   # backout from layer 7
            train_gpt.py(1289): RMSNorm(x)
            train_gpt.py(1290): lm_head(x)                        # final linear [T, vocab]
            train_gpt.py(1294): FusedSoftcappedCrossEntropy.apply(...)
              # 23 * sigmoid((logits+5)/7.5) softcap → cross entropy
              # returns per-token loss vector (MTP multi-token prediction weighted)

          /tmp/torchinductor_root/mi/cmib4stnx...  # actual compiled triton kernel binary
```

What you see in the trace: a single wide `Torch-Compiled Region: N/0` block that is internally many GPU kernels. The tall narrow bars at the bottom of the compiled region row are individual triton kernel launches (attention, matmuls, norms, elementwise fusions).

---

**5c. Backward pass — `torch/tensor.py(575): backward`**

Autograd runs the backward graph that AOT autograd constructed at compile time.

```
torch/tensor.py(575): backward
  torch/autograd/__init__.py(252): backward
    torch/autograd/graph.py(856): _engine_run_backward
      <built-in run_backward>
        torch/functorch/aot_autograd/runtime_wrappers.py(313): runtime_wrapper
          torch/functorch/aot_autograd/utils.py(124): call_func_at_runtime_with_args
            torch/functorch/aot_autograd/runtime_wrappers.py(517): wrapper
              torch/functorch/aot_autograd/runtime_wrappers.py(721): inner_fn
                torch/inductor/output_code.py(611): __call__
                  /tmp/torchinductor_root/...  # compiled backward triton kernels
                    CompiledFunction              # visible in trace as "CompiledFunction" span
```

This is the `torch/tensor.py(575): backward` span you see in Image 2. It runs in parallel to the gradient reduce launches below (CUDA streams overlap).

---

**5d. Gradient all-reduce — `train_gpt.py(433): _launch_reduce` (overlapped with backward)**

```
train_gpt.py(433): NorMuonAndAdam._launch_reduce(param, param.grad)
  torch/distributed/c10d_logger.py(80): wrapper
    distributed_c10d.py(4501): reduce_scatter_tensor / all_reduce (async_op=True)
      nccl::_reduce_scatter_base_   # for NorMuon (sharded) params — ZeRO-style
      nccl::all_reduce              # for Adam (replicated) params
        [NCCL kernel running on GPU — async, does not block CPU]
```

NorMuon params use `reduce_scatter` because each GPU will own only a `param_size / world_size` shard after the scatter. Adam params (embed, lm_head) use `all_reduce` (or `reduce_scatter` + later `all_gather`). Both calls return a `Future` immediately; the CPU continues launching more backward kernels while NCCL communicates.

---

**5e. Optimizer step — `train_gpt.py(1684): step_optimizers`**

```
train_gpt.py(1684): step_optimizers(step)
  get_lr(step), get_muon_momentum(step)        # compute LR/momentum schedule scalars
  train_gpt.py(1696): self.optimizer.step(do_adam=True/False)
    torch/utils/_contextlib.py(120): decorate_context   # @torch.no_grad()
      train_gpt.py(559): NorMuonAndAdam.step()

        # Phase 1: launch remaining reduces not already fired during backward
        for param in scatter_order:
          train_gpt.py(433): _launch_reduce(param, param.grad)   # async NCCL

        # Phase 2: process each param in work_order
        for param in work_order:
          future.wait()                         # block until reduce_scatter done

          if optim == "normuon":
            train_gpt.py(696): _normuon_update(param, grad_chunk, p_cfg, rank)
              grad_chunk.float()                # upcast to FP32 for momentum
              momentum_buffer.lerp_(grad_chunk, 1 - momentum)   # EMA of gradients
              train_gpt.py(151): polar_express(updated_grads)
                # Polar Express Sign Method (Newton-Schulz orthogonalization, 5 iters)
                # X / (||X|| * 1.02 + 1e-6)             -- normalize spectral norm
                # loop 5x: A = X @ X.T; B = b*A + c*A@A; X = a*X + B@X  -- polynomial
                @torch.compile(dynamic=False, fullgraph=True)   # its own compiled region
                  Torch-Compiled Region: 3/0    # visible separately in trace
                    XXT, ba_plus_cAA, baddbmm × 5 iters
              train_gpt.py(764): _apply_normuon_variance_reduction(v_chunk, ...)
                @torch.compile(dynamic=False, fullgraph=True)
                  v_mean = v_chunk.float().square().mean()    # RMS of gradient
                  second_momentum_buffer.lerp_(v_mean, 1-beta2)  # Adam-style var tracking
                  step_size = second_momentum_buffer.rsqrt()      # adaptive scale
                  v_chunk *= final_scale                          # normalize update magnitude
              p_slice.add_(update, alpha=-1.0)   # apply update to param shard

          elif optim == "adam":
            train_gpt.py(655): _adam_update(param, grad_chunk, p_cfg, rank)
              NorMuonAndAdam._adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, ...)
                @torch.compile(dynamic=False, fullgraph=True)
                  exp_avg  = beta1*exp_avg + (1-beta1)*g           # 1st moment
                  exp_avg_sq = beta2*exp_avg_sq + (1-beta2)*g*g   # 2nd moment
                  update = exp_avg / (sqrt(exp_avg_sq) + eps) * step_size
                  mask = (update * p_slice) > 0                    # cautious weight decay
                  p_slice -= update + p_slice * mask * eff_wd

          if comms == "sharded":
            train_gpt.py(466): _launch_gather(param, p_slice)
              dist.all_gather_into_tensor(param, p_slice, async_op=True)  # gather shards back

        # Phase 3: wait for all_gather futures
        for fut in gather_futures:
          fut.wait()                            # block until params are re-assembled on all GPUs

        # tied embed: copy lm_head.data.T -> embed.data (while lm_head gather completes)
        train_gpt.py(638): embed_param.data.copy_(lm_param.data.T)
```

**5f. Profiler tick**
```
train_gpt.py(1992): profiler_context.step()    # advances schedule; flushes trace after 3 active steps
```

---

## Overall Analysis

### Thread structure visible in trace
| Thread | What it does |
|--------|-------------|
| **main thread (python 16)** | All training: forward, backward, optimizer step |
| **W&B threads** | `wandb_run.py: check_stop_status / _loop_check_status` — heartbeat to W&B servers, non-blocking |
| **tqdm monitor** | `tqdm/_monitor.py(60): run` — progress bar refresh, non-blocking |

These background threads appear in the upper purple/teal bands you see at the top of the flame chart. They run between training steps and do not block GPU work.

### Key findings

**1. torch.compile is fully active.**  
The `Torch-Compiled Region: N/0` spans and `/tmp/torchinductor_root/mi/cmib4stnx...` triton binaries confirm forward and backward are fused. The warmup phase ran the real compilation; the profiled steps are all compiled steady-state.

**2. Gradient comms overlap with backward (correct).**  
`_launch_reduce` fires with `async_op=True` per parameter. In Image 6, `nccl::_reduce_scatter_base_` appears alongside backward kernels — this is the intended overlap. If you see NCCL only starting after all backward kernels finish, that would be a bubble to fix.

**3. Data loading is not the bottleneck.**  
The `distributed_data_generator` + `next_batch` + `get_bigram_hash` spans are sub-millisecond (visible in Image 10 as a narrow span before the forward kernel block). The `DataPreloader` already loaded the shard in a background thread — by the time `.send()` is called, data is in CPU RAM.

**4. Polar Express is compiled inside the optimizer.**  
`Torch-Compiled Region: 3/0` inside `_normuon_update` is the Newton-Schulz orthogonalization for Muon. It is on the critical path of the optimizer step.

**5. Peak memory: 30.2 GiB allocated / 45.9 GiB reserved.**  
With `reduce_scatter` sharding, each GPU holds `param_size / world_size` of optimizer state for NorMuon params. The 15 GiB gap between allocated and reserved is PyTorch's memory pool fragmentation buffer.

### What to investigate next

- **NCCL timing:** Use Perfetto SQL (`:` key → type a query): `SELECT ts, dur, name FROM slice WHERE name LIKE '%nccl%' ORDER BY ts` — measure whether comms duration fits inside the backward window or spills past it.
- **Gaps between steps:** In the zoomed-out view (Image 3), look for white space between the repeating forward+backward blocks. Any gap is idle GPU time — either CPU scheduling delay or a stalled `.send()`.
- **all_gather duration:** The param re-gather after optimizer update is on the critical path. If it's long, that's where ZeRO communication overhead lives for this run.
- **Torch-Compiled Region recompiles:** If you see region labels like `0/1`, `0/2` (second number > 0), it means dynamo recompiled due to shape changes. All should be `N/0` in steady state.
