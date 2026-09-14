# Plan: Déjà View as a Submap Backbone for VGGT-SLAM 2.0

## 1. Motivation

Foundation-model SLAM systems (VGGT-SLAM, MASt3R-SLAM) replaced hand-crafted
feature matching (ORB) with feed-forward 3D reconstruction priors. This bought
robustness but inherited a new bottleneck: the backbone's compute/memory cost.

- VGGT-SLAM's own paper states usable video length is capped by VGGT's GPU
  memory requirements.
- VGGT-SLAM 2.0's own ablation identifies its main remaining failure mode as
  drift accumulating **between submap boundaries** — because large submaps are
  too expensive, you're forced into more, smaller submaps, and each boundary
  is a place the SL(4)/factor-graph alignment can go wrong.

The bottleneck is not "loop closure is missing" — VGGT-SLAM 2.0 already has
loop closure (submap-to-submap constraints from revisit detection, corrected
via global factor-graph optimization), same as MASt3R-SLAM (ASMK retrieval +
Gauss-Newton backend). The bottleneck is: **backbone cost forces small
submaps, and small submaps mean more seams.**

Déjà View (NVIDIA, arXiv 2605.30215) offers a candidate fix: it loops a single
shared transformer block K times instead of stacking many unique layers,
matching/beating VGGT-scale baselines at 8–10× fewer parameters and
comparable-or-lower compute, with K exposed as an inference-time knob.

## 2. What is / isn't SLAM here — scope clarity

- **Déjà View is not SLAM.** Like VGGT, it's a feed-forward, offline, batch
  multi-view reconstructor: given a fixed set of views, it jointly outputs
  poses/depth/pointmaps for all of them in one pass. No incremental state, no
  loop closure, no persistent map.
- **The SLAM-ness lives entirely in the wrapper**, not the backbone:
  1. *Sequential submap alignment* — chaining submap *t* → *t+1* as frames
     arrive (incremental pose/map extension; drifts over time on its own).
  2. *Loop closure* — detecting revisits to non-adjacent submaps and adding
     correction constraints the global optimizer uses to remove accumulated
     drift.
- Reusing VGGT-SLAM 2.0's existing factor-graph chaining + loop-closure
  detection unchanged, but building each submap with Déjà View instead of
  VGGT, **is** a working SLAM system. It is not a new SLAM architecture —
  the mechanism is borrowed wholesale. The contribution has to come from what
  changes *because the backbone changed*, not from the fact that it's SLAM.

## 3. Core hypotheses to test

### H1 (fixed-K, systems claim)
Replacing VGGT with Déjà View at a fixed K inside VGGT-SLAM 2.0's existing
submap-construction step preserves (or improves) VGGT-SLAM 2.0's accuracy
while reducing memory/compute cost, enabling either:
- larger submaps per fixed GPU-memory budget (→ fewer seams → less
  accumulated alignment drift), or
- the same submap size on cheaper hardware.

### H2 (adaptive-K, capability claim)
Exposing K as a per-submap compute dial lets the system spend extra
refinement iterations specifically at loop-closure candidates and
low-confidence/degenerate views (textureless regions, near-planar scenes),
and less at easy, high-confidence tracking segments — a capability no
fixed-cost backbone (VGGT, MASt3R) structurally has.

H2 is the more novel and interesting claim, but is gated behind H1: if H1
fails, an H2 failure is confounded (can't tell if it's the adaptive policy or
Déjà View simply not transferring to this data regime at all).

## 4. Experimental plan

### Phase 1 — Fixed-K drop-in (de-risking)
1. Swap Déjà View in as the per-submap reconstructor inside VGGT-SLAM 2.0's
   pipeline, keeping factor-graph alignment, loop-closure detection, and SL(4)
   correction unchanged.
2. Run at a fixed K (start with the paper's reported sweet spot) on VGGT-SLAM's
   existing benchmark sequences (e.g., Office loop dataset, EuRoC, TUM,
   7-Scenes) to get an apples-to-apples comparison against stock VGGT-SLAM 2.0.
3. Measure, per submap and end-to-end:
   - ATE / trajectory accuracy vs. stock VGGT-SLAM 2.0
   - Peak GPU memory during submap construction
   - Wall-clock latency per submap and per loop-closure event
   - Whether larger submaps become feasible at equal memory budget, and
     whether that reduces the number of seams / improves drift
4. Explicitly check the two risk points flagged earlier:
   - **Attention memory doesn't shrink from looping.** Parameter savings come
     from weight reuse, not shorter sequences — each of the K passes still
     attends over the same token count. Verify actual peak memory
     empirically; do not assume the 8–10× parameter reduction implies a
     comparable memory reduction.
   - **Distribution shift.** Déjà View was trained/evaluated on unordered,
     largely static multi-view benchmarks (indoor/outdoor/object-centric/
     driving). SLAM submaps are highly correlated, small-baseline, temporally
     continuous sliding windows — check whether reconstruction quality
     degrades relative to Déjà View's own reported benchmark numbers.
5. Check downstream compatibility: VGGT-SLAM 2.0 found a specific VGGT
   attention layer gives loop-closure retrieval-verification signal "for
   free." Déjà View collapses many distinct layers into one repeated block —
   re-identify whether any iteration's attention map carries an equivalent
   signal, or whether this needs to be rebuilt from scratch.

**Exit criterion for Phase 1:** Déjà View submaps at fixed K must match
stock VGGT-SLAM 2.0 accuracy at meaningfully lower memory/compute, or the
project stops here and is reported as a negative/mixed systems result.

### Phase 2 — Adaptive K (capability claim)
Only proceed if Phase 1 succeeds.

1. Define a per-submap difficulty signal to drive K, e.g.:
   - Loop-closure candidate detected (via retrieval) → high K
   - Low reconstruction confidence / near-planar or textureless regions → high K
   - Straight-line tracking, high-confidence regions → low K
2. Because Déjà View's recurrence does not converge to a fixed point (state
   norm grows; only direction stabilizes — "directional refinement"), define
   a calibrated stopping criterion for adaptive K rather than assuming a
   natural convergence point exists. This likely requires new calibration
   work beyond what the Déjà View paper reports for its fixed-benchmark
   evaluation.
3. Compare adaptive-K against fixed-K (Phase 1) and against stock VGGT-SLAM
   2.0 on the same metrics, plus total compute spent, to show whether
   targeted compute allocation beats uniform allocation at equal total
   budget.

**Exit criterion for Phase 2:** Adaptive-K should either (a) match fixed-K
accuracy at lower total compute, or (b) beat fixed-K accuracy at equal total
compute, specifically at loop-closure/degenerate submaps. Otherwise, report
as: fixed-K systems result stands, adaptive-K did not add value.

## 5. What this plan does and does not claim

- Does **not** claim a new SLAM architecture — the incremental alignment,
  factor graph, and loop closure are VGGT-SLAM 2.0's, unchanged.
- Does **not** address dynamic-scene robustness — Déjà View, like VGGT and
  MASt3R, is trained/evaluated on largely static/rigid multi-view data. This
  is an orthogonal, unsolved problem for this whole family of systems.
- Does **not** remove the fundamental reconstruction ambiguity for uncalibrated
  cameras (the 15-DoF projective ambiguity VGGT-SLAM's SL(4) machinery exists
  to correct) — that's a property of the problem, not the backbone.
- **Does** claim, if Phase 1 holds: a cheaper, drop-in submap backbone for an
  existing SLAM wrapper, with a plausible mechanism (fewer seams via cheaper/
  larger submaps) for why that improves the system's known weak point.
- **Does** claim, if Phase 2 holds: a genuinely new axis of control —
  test-time compute allocated by SLAM-specific difficulty — that no other
  feed-forward SLAM backbone currently exposes.

## 6. Open risks / unknowns going in

- Unclear whether Déjà View generalizes to streaming/sliding-window input
  distributions at all (its benchmarks are unordered, offline view sets).
- Unclear whether memory savings materialize in practice vs. only on paper
  (parameter count vs. attention/activation cost).
- Unclear whether any useful loop-closure signal survives the collapse from
  many distinct layers to one shared block.
- No published prior work combining these two systems — this is a novel
  combination, not a documented approach with existing benchmarks to point to.
