#!/bin/bash
# 7-Scenes ATE evaluation, parameterized by backbone.
#
# VGGT-SLAM's README references evals/eval_7scenes.sh, but that script was never
# shipped in the repo. This is a replacement following MASt3R-SLAM's convention
# (which the README cites): seq-01 of each of the 7 scenes, scored with
# `evo_ape tum ... -as`, against MASt3R-SLAM's prebuilt ground truth.
#
# Ground-truth timestamps are integer frame indices, which is what VGGT-SLAM's
# sort_images_by_number() extracts from frame-000123.color.png -- so estimate and
# ground truth associate without any conversion.
#
# main.py's image glob already drops *depth* and *txt*, so pointing --image_folder
# at a raw seq-01 directory picks up only the .color.png frames.
#
# Usage:
#   ./evals/eval_7scenes_backbone.sh <backbone> [submap_size] [runs]
#   SCENES_ROOT=/path/to/7scenes ./evals/eval_7scenes_backbone.sh dvlt 16 1
#
# Environment:
#   RESUME      1 (default) skips scenes that already have a trajectory
#   DVLT_ARGS   extra DVLT flags, e.g. DVLT_ARGS="--lc_verify attn"
#
# Both backbones must be run over the same scenes for the comparison to mean
# anything.

set -u

backbone=${1:-dvlt}
submap_size=${2:-16}
n=${3:-1}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENES_ROOT=${SCENES_ROOT:-"${repo_dir}/../data/7scenes"}
DVLT_ARGS=${DVLT_ARGS:-}
PYTHON=${PYTHON:-"${repo_dir}/../.venv/bin/python"}
EVO_APE=${EVO_APE:-"${repo_dir}/../.venv/bin/evo_ape"}

out_dir="${repo_dir}/logs/${backbone}_7scenes_w${submap_size}"
log_path="${out_dir}/7scenes_results.txt"
mkdir -p "$out_dir"

scenes=(chess fire heads office pumpkin redkitchen stairs)

# Fail early rather than silently scoring a missing scene.
missing=0
for s in "${scenes[@]}"; do
    [ -d "${SCENES_ROOT}/${s}/seq-01" ] || { echo "MISSING frames: ${SCENES_ROOT}/${s}/seq-01"; missing=1; }
    [ -s "${SCENES_ROOT}/groundtruths/${s}.txt" ] || { echo "MISSING gt:     ${SCENES_ROOT}/groundtruths/${s}.txt"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "7-Scenes data incomplete under SCENES_ROOT=${SCENES_ROOT} — run data/download_7scenes.sh first."
    exit 1
fi

echo "backbone=${backbone} submap_size=${submap_size} runs=${n} scenes=${#scenes[@]}"
[ -n "$DVLT_ARGS" ] && echo "DVLT_ARGS=${DVLT_ARGS}"

# Rewritten each invocation: scoring re-runs over every scene, so appending
# would duplicate rows on resume.
echo "Run,Dataset,RMSE" > "$log_path"

for run in $(seq 1 "$n"); do
    echo "==== Run $run ===="

    for s in "${scenes[@]}"; do
        est_path="${out_dir}/${s}_run${run}.txt"
        if [ "${RESUME:-1}" = "1" ] && [ -s "$est_path" ]; then
            echo "-- skipping ${s} (already done: $(wc -l < "$est_path") poses)"
            continue
        fi
        echo "-- running ${s}"
        # shellcheck disable=SC2086
        "$PYTHON" "${repo_dir}/main.py" \
            --backbone "$backbone" \
            --image_folder "${SCENES_ROOT}/${s}/seq-01" \
            --max_loops 1 --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 \
            --submap_size "$submap_size" \
            --log_results --skip_dense_log --log_path "$est_path" \
            $DVLT_ARGS || echo "  run FAILED for ${s}"
    done

    for s in "${scenes[@]}"; do
        est_path="${out_dir}/${s}_run${run}.txt"
        gt_file="${SCENES_ROOT}/groundtruths/${s}.txt"

        if [ ! -s "$est_path" ]; then
            echo "${s}: no trajectory produced -> NaN"
            echo "$run,$s,NaN" >> "$log_path"
            continue
        fi

        ape_result=$("$EVO_APE" tum "$gt_file" "$est_path" -as 2>/dev/null)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        # A failed evo run must never be recorded as a perfect 0.0.
        rmse=${rmse:-NaN}

        echo "${s}: ${rmse}"
        echo "$run,$s,$rmse" >> "$log_path"
    done
done

echo
echo "results -> $log_path"
"$PYTHON" - "$log_path" <<'PY'
import sys, csv, math
rows = list(csv.DictReader(open(sys.argv[1])))
vals = []
for r in rows:
    try:
        v = float(r["RMSE"])
        if math.isfinite(v):
            vals.append(v)
    except ValueError:
        pass
print(f"{len(vals)}/{len(rows)} scenes scored")
if vals:
    print(f"mean ATE RMSE: {sum(vals)/len(vals):.4f} m")
if len(vals) != len(rows):
    print("WARNING: some scenes did not score - mean is over successful runs only")
PY
echo "SEVENSCENES_EVAL_COMPLETE"
