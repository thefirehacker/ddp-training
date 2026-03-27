# What is a storyboard

A storyboard is a learning tool where TimeCapsule creates a full research story for learners. We start with a builder/researcher/problem-solver `A` trying to solve problem `X`. We define the skills `Y` that `A` must develop and the tools/actions `Z` that `A` uses. The storyboard is then broken into waypoints where `A` applies `Y` and `Z` to move forward. Waypoints are not optimized for speed; they capture real workflow. In many cases, `A` uses existing `Y` and `Z` to create new `Y` (and sometimes new `Z`).

## Storyboard rules

- `A` is a learner on [First Break AI Roadmap - Step 2](https://thefirehacker.github.io/firstbreakai/roadmap.html).
- Every storyboard must define:
  - `Problem X`
  - `Skills Y`
  - `Tools/Actions Z`
  - `Waypoint sequence`
- If two problems are unrelated, they must be separate storyboards.
- Each board follows this logic: bottleneck -> intuition -> mechanism -> what changed.

---

## Storyboard 1: Sinusoidal positional encoding

### A profile
`A` can run local inference and inspect logits/attention maps, but cannot yet reason about why order information is needed.

### Problem X (bottleneck)
Pure attention has no native sense of token order.

### Research intuition
Inject position without adding recurrence/convolution so training stays parallel.

### Mechanism built
Add sinusoidal positional vectors to token embeddings at input.

### What changed
The model gains sequence order information while preserving Transformer parallelism.

### Skills Y
- Distinguish architecture capability vs inductive bias.
- Explain absolute position encoding in plain language.
- Trace how embedding + position enters the block stack.

### Tools/Actions Z
- Read architecture sections of the Transformer paper.
- Plot or inspect sinusoidal patterns for different positions.
- Run small prompts and compare behavior with/without position signals (conceptual experiment design).

### Waypoint sequence
1. Observe failure mode: order-sensitive sentences collapse without order cues.
2. Learn why attention weights alone do not encode absolute index.
3. Add sinusoidal position vectors and validate that order-sensitive behavior improves.
4. Record new `Y`: ability to explain why sequence models need explicit order signals.

### Waypoint diagram

```mermaid
flowchart TD
  sb1X["Problem X: no native order in attention"]
  sb1wp1["WP1: observe order-sensitive failures"]
  sb1wp2["WP2: see why plain attention lacks absolute index"]
  sb1wp3["WP3: add sinusoidal PE and validate"]
  sb1wp4["WP4: new Y — explain need for order signal"]
  sb1X --> sb1wp1 --> sb1wp2 --> sb1wp3 --> sb1wp4
```

---

## Storyboard 2: Relative positions (Shaw + Transformer-XL framing)

### A profile
`A` understands absolute position embeddings but sees they are awkward for distance-based language patterns.

### Problem X (bottleneck)
Absolute position is not the right primitive for many pairwise relations.

### Research intuition
Attention is pairwise, so positional information should be pairwise too (relative offsets/distances).

### Mechanism built
Inject relative position representations into attention computations; in Transformer-XL, pair this with recurrence-compatible relative formulation.

### What changed
Models become better at representing local/relative relationships and handling cross-segment context logic.

### Skills Y
- Differentiate absolute vs relative position signals.
- Explain why pairwise scoring aligns with relative distance.
- Reason about context fragmentation at segment boundaries.

### Tools/Actions Z
- Compare attention score formulas conceptually (absolute-additive vs relative-aware).
- Build toy examples using "token k steps left" reasoning.
- Inspect segment boundary effects in language modeling setups.

### Waypoint sequence
1. Spot mismatch: absolute index does not directly encode "how far apart."
2. Reformulate position as relation between query and key positions.
3. Validate improved handling of relative dependencies.
4. Record new `Y`: ability to choose positional primitives based on task structure.

### Waypoint diagram

```mermaid
flowchart TD
  sb2X["Problem X: absolute index is wrong primitive"]
  sb2wp1["WP1: spot mismatch vs distance-based language"]
  sb2wp2["WP2: reformulate as query-key relative offset"]
  sb2wp3["WP3: validate relative dependencies"]
  sb2wp4["WP4: new Y — pick primitives for task structure"]
  sb2X --> sb2wp1 --> sb2wp2 --> sb2wp3 --> sb2wp4
```

---

## Storyboard 3: RoPE (Rotary Position Embedding)

### A profile
`A` knows relative position helps, and now asks whether position can be encoded inside Q/K geometry itself.

### Problem X (bottleneck)
Position methods feel bolted on instead of native to attention geometry.

### Research intuition
Encode position by rotating Q/K vectors so dot products naturally carry relative information.

### Mechanism built
Apply position-dependent rotations in paired dimensions of queries and keys.

### What changed
Relative-position effects emerge directly in attention scores with strong practical performance.

### Skills Y
- Explain geometric encoding of position in attention space.
- Connect dot-product behavior to relative-position sensitivity.
- Compare additive-bias methods vs geometric-transformation methods.

### Tools/Actions Z
- Walk through a 2D rotation intuition for one Q/K pair.
- Inspect attention score behavior conceptually as distance increases.
- Map RoPE usage to real model code paths.

### Waypoint sequence
1. Identify friction with external position tags.
2. Move position into Q/K transformation.
3. Validate that attention scoring now carries relative structure more natively.
4. Record new `Y`: ability to reason about representation geometry, not only array shape.

### Waypoint diagram

```mermaid
flowchart TD
  sb3X["Problem X: position feels bolted onto attention"]
  sb3wp1["WP1: friction with external position tags"]
  sb3wp2["WP2: move position into Q and K transforms"]
  sb3wp3["WP3: validate relative structure in scores"]
  sb3wp4["WP4: new Y — geometry not only array shape"]
  sb3X --> sb3wp1 --> sb3wp2 --> sb3wp3 --> sb3wp4
```

---

## Storyboard 4: ALiBi (Attention with Linear Biases)

### A profile
`A` is focused on inference length extension and sees models degrade beyond training context.

### Problem X (bottleneck)
Many positional methods extrapolate poorly to longer lengths.

### Research intuition
Do not encode position in token representation; bias attention scores directly by distance.

### Mechanism built
Add a linear distance-based bias term to attention scores (head-dependent slope pattern).

### What changed
Improved length extrapolation behavior with lower complexity overhead.

### Skills Y
- Explain extrapolation vs interpolation in context length.
- Understand where to inject inductive bias: embeddings vs scores.
- Evaluate recency priors and their trade-offs.

### Tools/Actions Z
- Compare perplexity/quality across train length vs test length.
- Plot bias magnitude vs distance for intuition.
- Run long-sequence sanity checks in inference settings.

### Waypoint sequence
1. Detect extrapolation failure with existing positional encoding.
2. Shift positional signal from embeddings to score bias.
3. Evaluate longer-context behavior with minimal architecture change.
4. Record new `Y`: ability to design for deployment constraints, not only training fit.

### Waypoint diagram

```mermaid
flowchart TD
  sb4X["Problem X: poor length extrapolation"]
  sb4wp1["WP1: detect failure beyond train context"]
  sb4wp2["WP2: move bias from embeddings to scores"]
  sb4wp3["WP3: evaluate long context with small change"]
  sb4wp4["WP4: new Y — deploy constraints not only train fit"]
  sb4X --> sb4wp1 --> sb4wp2 --> sb4wp3 --> sb4wp4
```

---

## Storyboard 5: RoPE extension track (YaRN + LongRoPE)

### A profile
`A` wants to keep RoPE benefits but extend context safely for long-document inference.

### Problem X (bottleneck)
RoPE-based models often degrade sharply beyond trained context length.

### Research intuition
Do not discard RoPE; rescale/interpolate positional geometry more carefully and progressively.

### Mechanism built
Use efficient context-extension recipes (e.g., interpolation/rescaling strategies and staged extension) to preserve short-context quality while extending long-context behavior.

### What changed
Longer usable context windows become feasible with less retraining cost than full-from-scratch alternatives.

### Skills Y
- Diagnose long-context failure modes in RoPE models.
- Understand controlled rescaling of positional geometry.
- Balance short-context retention vs long-context extension.

### Tools/Actions Z
- Run targeted long-context eval suites.
- Compare baseline RoPE vs extension recipe checkpoints.
- Use progressive extension experiments (short -> medium -> long) with quality tracking.

### Waypoint sequence
1. Confirm baseline RoPE degradation beyond train window.
2. Apply context-extension recipe with controlled scaling.
3. Evaluate short-context regression and long-context gains.
4. Record new `Y`: ability to evolve strong mechanisms instead of replacing them blindly.

### Waypoint diagram

```mermaid
flowchart TD
  sb5X["Problem X: RoPE breaks past trained length"]
  sb5wp1["WP1: confirm degradation beyond train window"]
  sb5wp2["WP2: apply extension recipe and rescaling"]
  sb5wp3["WP3: check short vs long tradeoffs"]
  sb5wp4["WP4: new Y — evolve mechanism not replace blindly"]
  sb5X --> sb5wp1 --> sb5wp2 --> sb5wp3 --> sb5wp4
```

---

## Storyboard 6: NoPE (No positional encoding)

### A profile
`A` now questions assumptions and tests whether explicit positional machinery is always necessary.

### Problem X (bottleneck)
Positional methods may be over-engineered for some settings; unclear what bias is truly required.

### Research intuition
Causal masking and optimization dynamics may allow implicit positional information to emerge.

### Mechanism built
Train without explicit positional encodings and analyze learned behavior and length generalization.

### What changed
Reframed the field question from "which positional encoding should we add?" to "what positional bias is minimally necessary?"

### Skills Y
- Form null-hypothesis style experiments in model design.
- Distinguish emergent behavior from explicitly injected structure.
- Reason about limits of implicit position learning.

### Tools/Actions Z
- Ablation runs with and without explicit positional encoding.
- Diagnostic probes for positional information in activations.
- Length-generalization tests under controlled decoding settings.

### Waypoint sequence
1. Remove explicit positional module as a deliberate ablation.
2. Measure what positional behavior still emerges.
3. Identify boundaries where explicit methods are still needed.
4. Record new `Y`: ability to challenge default architecture assumptions.

### Waypoint diagram

```mermaid
flowchart TD
  sb6X["Problem X: unclear if explicit PE is needed"]
  sb6wp1["WP1: ablate explicit positional module"]
  sb6wp2["WP2: measure emergent positional behavior"]
  sb6wp3["WP3: find where explicit methods still win"]
  sb6wp4["WP4: new Y — question default assumptions"]
  sb6X --> sb6wp1 --> sb6wp2 --> sb6wp3 --> sb6wp4
```

---

## Storyboard 7: FlashAttention (separate, unrelated bottleneck)

### A profile
`A` can explain attention math but struggles with real latency/throughput bottlenecks on GPU.

### Problem X (bottleneck)
Attention runtime is heavily constrained by memory IO, not just FLOPs.

### Research intuition
Keep exact attention formula but redesign execution to reduce high-bandwidth memory traffic.

### Mechanism built
Use IO-aware tiling/fusion strategy so Q/K/V blocks are processed with far fewer expensive memory reads/writes.

### What changed
Large practical speedups with exact attention computation and improved long-context usability.

### Skills Y
- Separate algorithmic complexity from systems bottlenecks.
- Explain IO-aware optimization in model-serving context.
- Connect kernel-level design to user-visible latency.

### Tools/Actions Z
- Profile GPU memory traffic and kernel timelines.
- Benchmark baseline attention vs optimized kernels.
- Track throughput/latency trade-offs at different sequence lengths.

### Waypoint sequence
1. Measure that naive attention is memory-traffic dominated.
2. Reframe optimization target from approximation to IO movement.
3. Adopt tiled exact attention kernel and benchmark gains.
4. Record new `Y`: ability to find bottlenecks in implementation, not only equations.

### Waypoint diagram

```mermaid
flowchart TD
  sb7X["Problem X: wall clock limited by memory IO"]
  sb7wp1["WP1: profile traffic not only FLOPs"]
  sb7wp2["WP2: target IO not approximation first"]
  sb7wp3["WP3: adopt tiled exact attention kernel"]
  sb7wp4["WP4: new Y — bottleneck in implementation path"]
  sb7X --> sb7wp1 --> sb7wp2 --> sb7wp3 --> sb7wp4
```

---

## Storyboard 8: MQA/GQA for KV-cache efficiency (separate, unrelated bottleneck)

### A profile
`A` is now running token-by-token decoding and sees KV cache size/bandwidth dominate inference.

### Problem X (bottleneck)
In decoder inference, loading per-head K/V tensors is expensive in memory bandwidth and cache size.

### Research intuition
Keep multi-head query expressivity but share or group K/V heads to reduce cache overhead.

### Mechanism built
- **MQA:** share one K/V set across all query heads.
- **GQA:** share K/V within groups of query heads as a quality/efficiency middle ground.

### What changed
Substantial inference efficiency gains, making long-context decoding and serving more practical.

### Skills Y
- Model KV-cache memory costs from tensor shapes.
- Evaluate quality vs efficiency trade-offs in head sharing.
- Pick inference-oriented attention variants for deployment constraints.

### Tools/Actions Z
- Compute KV-cache footprint for MHA vs MQA vs GQA.
- Benchmark tokens/sec and latency under identical hardware.
- Run quality regression checks after conversion/uptraining.

### Waypoint sequence
1. Quantify decode-time memory-bandwidth bottleneck.
2. Reduce K/V redundancy via sharing/grouping strategy.
3. Re-test quality and serving metrics at target context lengths.
4. Record new `Y`: ability to redesign attention for inference bottlenecks.

### Waypoint diagram

```mermaid
flowchart TD
  sb8X["Problem X: KV cache bandwidth at decode"]
  sb8wp1["WP1: quantify decode KV cost"]
  sb8wp2["WP2: share or group K and V heads"]
  sb8wp3["WP3: re-test quality and serving metrics"]
  sb8wp4["WP4: new Y — attention for inference not training"]
  sb8X --> sb8wp1 --> sb8wp2 --> sb8wp3 --> sb8wp4
```

---

## Cross-board pattern for Step 2 learners

Each breakthrough follows the same research workflow:

1. Find a concrete bottleneck (`X`): order, length, IO, or KV bandwidth.
2. Form an intuition about structure (relative vs absolute, geometry vs additive tags, system IO vs formula).
3. Build a mechanism (`Z`) that encodes that intuition.
4. Measure what changed and extract reusable skills (`Y`) for the next bottleneck.

### Pattern diagram

```mermaid
flowchart TD
  crossX["Name bottleneck X"]
  crossIntuition["Form structural intuition"]
  crossZ["Build mechanism Z"]
  crossY["Measure impact and new skills Y"]
  crossX --> crossIntuition --> crossZ --> crossY
```