#!/bin/bash
# TUM freiburg1 ATE evaluation, parameterized by backbone.
#
# Same protocol as the stock evals/eval_tum.sh (same 9 sequences, same main.py
# flags, same evo_ape invocation), with two changes:
#   * the hardcoded "/home/<user>/Documents" placeholder becomes $TUM_ROOT
#   * --backbone is passed through, so vggt and dvlt are run identically
#
# Usage:
#   ./evals/eval_tum_backbone.sh <backbone> [submap_size] [runs]
#   TUM_ROOT=/path/to/tum ./evals/eval_tum_backbone.sh dvlt 16 1
#
# Environment:
#   SEQ_SET     fr1 (default) | fr23 | all  -- which sequences to evaluate
#   RESUME      1 (default) skips sequences that already have a trajectory;
#               0 forces a full re-run
#   DVLT_ARGS   extra flags for the DVLT backbone,
#               e.g. DVLT_ARGS="--dvlt_k 8 --lc_verify attn"
#   TUM_ROOT    dataset root (default ../data/tum)
#
# Both backbones must be evaluated over the SAME SEQ_SET for the comparison to
# mean anything -- a mismatched sequence set invalidates the mean.

set -u

backbone=${1:-dvlt}
submap_size=${2:-16}
n=${3:-1}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUM_ROOT=${TUM_ROOT:-"${repo_dir}/../data/tum"}
DVLT_ARGS=${DVLT_ARGS:-}
PYTHON=${PYTHON:-"${repo_dir}/../.venv/bin/python"}
EVO_APE=${EVO_APE:-"${repo_dir}/../.venv/bin/evo_ape"}

out_dir="${repo_dir}/logs/${backbone}_w${submap_size}"
log_path="${out_dir}/tum_results.txt"
mkdir -p "$out_dir"

fr1=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)
# fr2: longer trajectories. fr3: the structure/texture 2x2 of geometric vs
# photometric degeneracy. Dynamic (walking_*/sitting_*) deliberately excluded --
# every backbone here is trained on static scenes.
fr23=(
    rgbd_dataset_freiburg2_xyz
    rgbd_dataset_freiburg2_desk
    rgbd_dataset_freiburg2_360_hemisphere
    rgbd_dataset_freiburg2_large_no_loop
    rgbd_dataset_freiburg3_long_office_household
    rgbd_dataset_freiburg3_cabinet
    rgbd_dataset_freiburg3_structure_texture_far
    rgbd_dataset_freiburg3_structure_texture_near
    rgbd_dataset_freiburg3_structure_notexture_far
    rgbd_dataset_freiburg3_structure_notexture_near
    rgbd_dataset_freiburg3_nostructure_texture_far
    rgbd_dataset_freiburg3_nostructure_texture_near_withloop
)

case "${SEQ_SET:-fr1}" in
    fr1)  datasets=("${fr1[@]}") ;;
    fr23) datasets=("${fr23[@]}") ;;
    all)  datasets=("${fr1[@]}" "${fr23[@]}") ;;
    *)    echo "SEQ_SET must be fr1, fr23 or all (got '${SEQ_SET}')"; exit 1 ;;
esac
echo "SEQ_SET=${SEQ_SET:-fr1} (${#datasets[@]} sequences)"

# Fail early and loudly rather than silently scoring 0.0 on missing data.
missing=0
for dataset in "${datasets[@]}"; do
    [ -d "${TUM_ROOT}/${dataset}/rgb" ] || { echo "MISSING images:  ${TUM_ROOT}/${dataset}/rgb"; missing=1; }
    [ -f "${TUM_ROOT}/${dataset}/groundtruth.txt" ] || { echo "MISSING gt:      ${TUM_ROOT}/${dataset}/groundtruth.txt"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "TUM data incomplete under TUM_ROOT=${TUM_ROOT} — run data/download_tum.sh first."
    exit 1
fi

# Always rewritten: scoring re-runs over every selected sequence (it is cheap,
# operating on existing trajectories), so appending would duplicate rows on resume.
echo "Run,Dataset,RMSE" > "$log_path"

echo "backbone=${backbone} submap_size=${submap_size} runs=${n}"
echo "TUM_ROOT=${TUM_ROOT}"
[ -n "$DVLT_ARGS" ] && echo "DVLT_ARGS=${DVLT_ARGS}"

for run in $(seq 1 "$n"); do
    echo "==== Run $run ===="

    for dataset in "${datasets[@]}"; do
        est_path="${out_dir}/${dataset}_run${run}.txt"
        # Resume: a completed sequence is not redone. Long evaluations here get
        # killed by session teardowns, so re-running must cost time, not progress.
        # Set RESUME=0 to force a full re-run.
        if [ "${RESUME:-1}" = "1" ] && [ -s "$est_path" ]; then
            echo "-- skipping ${dataset} (already done: $(wc -l < "$est_path") poses)"
            continue
        fi
        echo "-- running ${dataset}"
        # shellcheck disable=SC2086
        "$PYTHON" "${repo_dir}/main.py" \
            --backbone "$backbone" \
            --image_folder "${TUM_ROOT}/${dataset}/rgb" \
            --max_loops 1 --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 \
            --submap_size "$submap_size" \
            --log_results --skip_dense_log --log_path "$est_path" \
            $DVLT_ARGS || echo "  run FAILED for ${dataset}"
    done

    for dataset in "${datasets[@]}"; do
        est_path="${out_dir}/${dataset}_run${run}.txt"
        gt_file="${TUM_ROOT}/${dataset}/groundtruth.txt"

        if [ ! -s "$est_path" ]; then
            echo "${dataset}: no trajectory produced -> NaN"
            echo "$run,$dataset,NaN" >> "$log_path"
            continue
        fi

        # -as: align (Umeyama) and correct scale, as the stock script does.
        ape_result=$("$EVO_APE" tum "$gt_file" "$est_path" -as 2>/dev/null)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        # A failed evo run must not be recorded as a perfect 0.0 score.
        rmse=${rmse:-NaN}

        echo "${dataset}: ${rmse}"
        echo "$run,$dataset,$rmse" >> "$log_path"
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
print(f"{len(vals)}/{len(rows)} sequences scored")
if vals:
    print(f"mean ATE RMSE: {sum(vals)/len(vals):.4f} m")
if len(vals) != len(rows):
    print("WARNING: some sequences did not score - mean is over successful runs only")
PY
