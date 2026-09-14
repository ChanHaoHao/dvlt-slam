# dvlt-slam — Déjà View as the submap backbone for VGGT-SLAM 2.0

A fork of [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) 2.0 with NVIDIA's
**Déjà View (DVLT)** swapped in as the per-submap reconstructor in place of
VGGT-1B. The factor graph, loop-closure detection and SL(4) correction are
untouched upstream code, so this stays a clean backbone comparison.

DVLT is the **default** backbone here (`--backbone vggt` still selects the
original path, which is what produced the baseline column below).

**Status:** Phase 1 (fixed-`K` drop-in) complete. Phase 2 (adaptive `K`) was
attempted and did not work; it is deliberately not in this repo.

## Headline result

TUM `freiburg1`, 9 sequences, matched settings (`w=16`, SL(4), `evo_ape -as`):

| | mean ATE RMSE | params |
|---|---|---|
| VGGT-SLAM 2.0 | 0.0370 m | ~1.2 B |
| **DVLT (this work)** | **0.0342 m** | **117 M** |

DVLT wins 7/9 sequences. A signed-rank test over the paired differences gives
**p = 0.164**, so the supported claim is **parity at ~10× fewer parameters**,
not superiority. Nine paired sequences cannot resolve a difference this small:
separating the two backbones at this effect size would take on the order of a
hundred, so the parity claim is the honest ceiling on this evidence.

Evaluation was since extended, at the same fixed `K=12`, to **21 TUM sequences
(fr1+fr2+fr3, 20 scored)** and **7-Scenes (7 scenes)**. Those runs have no VGGT
baseline — VGGT-1B does not fit in the 6 GB GPU used here — so they characterise
DVLT in isolation and do **not** bear on the parity claim above.
[`results/RESULTS.md`](results/RESULTS.md) is the authoritative table.

Loop-closure verification turns out to be decisive:

| config | mean ATE |
|---|---|
| loops on, ungated | 0.1374 m |
| loops off | 0.0530 m |
| **loops on, attention-gated** | **0.0342 m** |

VGGT-SLAM reads its loop-closure verification signal from one specific attention
layer (20 of 24). DVLT has no layer 20 — it loops a single shared block `K`
times — but the signal survives at iteration `k`, and **VGGT's 0.95 threshold
transfers unchanged**, with no recalibration. Because ungated loops are actively
harmful, `--lc_verify` defaults to `attn` here, not to upstream's `bypass`.

## What is here

This repo is upstream VGGT-SLAM @ `35327ac` with history removed, plus:

```
vggt_slam/dvlt_backbone.py    the adapter — DVLT behind VGGT's calling convention
main.py                       --backbone/--dvlt_k/--lc_verify (default: dvlt/attn)
dvlt/                         submodule, pinned @ 134b21f (nv-tlabs/dvlt)
evals/eval_tum_backbone.sh    TUM ATE harness (records failures as NaN, not 0.0)
                              SEQ_SET=fr1|fr23|all, RESUME=1 to skip done sequences
evals/eval_7scenes_backbone.sh  the same harness for 7-Scenes
evals/summarize_results.py    consolidates one config's runs into a table
evals/collect_all_results.py  regenerates results/RESULTS.md + all_metrics.csv
                              from logs/ — numbers are never hardcoded
evals/visualize_submaps.py    per-submap coloured reconstructions
evals/dense_pass.sh           dense point clouds per sequence
results/RESULTS.md            the authoritative ATE tables, with caveats
results/all_metrics.csv       one tidy row per (experiment, sequence)
results/vggt_baseline.csv     VGGT numbers — supplied, NOT produced by this repo
plan.md                       the original research plan
pyproject.toml, uv.lock       pinned environment (183 packages, torch 2.5.1+cu124)
UPSTREAM_README.md            VGGT-SLAM's own README, kept for attribution
```

`evals/eval_tum.sh` and `evals/process_logs_tum.py` are upstream's, left as they
were. Upstream's `setup.py` was merged into `pyproject.toml`: with a pyproject
present setuptools ignores setup.py, so keeping both would have silently
installed no packages at all.

## Reproducing

```bash
# 1. this repo, with the dvlt submodule at its pinned commit
git clone --recursive <this-repo> dvlt-slam
cd dvlt-slam
# (already cloned without --recursive? git submodule update --init)

# 2. one env, from the lock file (exactly the versions that produced the
#    results). Upstream's requirements.txt pins torch==2.3.1; that pin is stale —
#    everything works on 2.5.1, which DVLT requires.
uv sync                      # creates .venv from uv.lock
uv pip install pip           # torch.utils.collect_env shells out to `python -m pip`,
                             # and uv venvs ship without pip

# 3. `uv sync` above already installed THIS project (vggt_slam + evals) editable.
#    The other three are upstream trees, absent from the lock on purpose.
#    Order matters: `uv sync` prunes anything not in the lock, so run it FIRST —
#    re-running it later will silently uninstall these three again.
./setup.sh                   # clones third_party/{salad,vggt}; sam3 and
                             # perception_models are lazily imported behind --run_os
uv pip install --no-deps -e ./dvlt \
    -e ./third_party/salad -e ./third_party/vggt

# 4. SALAD weights — setup.sh does NOT fetch these, and the run dies without them
curl -L -o ~/.cache/torch/hub/checkpoints/dino_salad.ckpt \
  https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt

# 5. run — DVLT and attention-gated loop closure are the defaults
python main.py --image_folder office_loop
TUM_ROOT=/path/to/tum ./evals/eval_tum_backbone.sh dvlt 16 1
python evals/collect_all_results.py
```

## Gotchas worth knowing

- **DVLT emits `camera_to_worlds`; the pose encoder wants world-to-camera.**
  Getting this backwards yields a plausible but mirrored trajectory.
  `selftest_pose_roundtrip()` in the adapter guards it.
- **`target_tokens` is dead code** in `vggt_slam` — referenced once, only to be
  excluded from a cast. Nothing to reproduce.
- **The stock `evals/eval_tum.sh` scores a failed sequence as `0.0`**
  (`rmse=${rmse:-0}`), which silently *lowers* the mean. Our harness records NaN.
- **Peak memory is flat in `K`** and linear in the number of frames
  (~136 MiB/frame + ~700 MiB). `K` costs wall-clock, not memory.
- **One-shot ceiling on a 6 GB GPU is 40 frames**; `w=32` OOMs inside the SLAM
  pipeline, where SALAD and DINOv2 are also resident.

## Open items

1. VGGT baseline on the *same* hardware — the memory/throughput half of the
   claim is currently unmeasured, not confirmed. `results/vggt_baseline.csv` was
   supplied from other hardware.
2. More sequences for significance. fr2/fr3 and 7-Scenes are now done; EuRoC is
   not. Repeated runs alone will not get there, as the variance is between
   sequences.
3. Submap-size sweep on one machine. Current evidence suggests *smaller* submaps
   help VGGT (0.0370 at w=16 vs 0.0533 at w=32), which cuts against the
   "fewer seams" mechanism in `plan.md`.
4. Diagnose `freiburg1_360`, the one sequence driving the variance.

## Licences

This repo **redistributes upstream VGGT-SLAM source** (it is a history-stripped
fork), so upstream's BSD-2-Clause terms apply to that code and
[`LICENSE`](LICENSE) is retained unmodified. Only `vggt_slam/dvlt_backbone.py`,
`evals/*_backbone.sh`, `evals/collect_all_results.py` and `results/` are new
work.

| component | licence |
|---|---|
| VGGT-SLAM (this tree) | BSD-2-Clause |
| DVLT **code** (submodule) | Apache-2.0 |
| **DVLT weights** (`nvidia/dvlt`) | **NVIDIA License — non-commercial, research/evaluation only** |
| VGGT weights | non-commercial (VGGT-1B-Commercial is the exception) |

The weight licences are the binding constraint: **there is no commercial path
without separate agreements**, regardless of how this repo is licensed.
