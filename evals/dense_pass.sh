#!/bin/bash
# Dense reconstruction pass over the TUM sequences.
#
# The ATE benchmark (eval_tum_backbone.sh) runs with --skip_dense_log, matching the
# stock protocol, so it produces trajectories but no point clouds. This pass
# re-runs each sequence with dense logging and saves a per-scene point cloud.
#
# Usage:  ./evals/dense_pass.sh [backbone] [submap_size]
#         DVLT_ARGS="--max_loops 0" ./evals/dense_pass.sh dvlt 16
#
# Outputs, per sequence, under outputs/tum/<sequence>/:
#   traj.txt          estimated trajectory (TUM format)
#   points_raw.pcd    full-resolution cloud straight from the solver
#   points_1cm.ply    1cm-voxel downsample, outliers removed (the shareable one)

set -u

backbone=${1:-dvlt}
submap_size=${2:-16}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUM_ROOT=${TUM_ROOT:-"${repo_dir}/../data/tum"}
DVLT_ARGS=${DVLT_ARGS:-}
PYTHON=${PYTHON:-"${repo_dir}/../.venv/bin/python"}
KEEP_RAW=${KEEP_RAW:-1}

datasets=(
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

for dataset in "${datasets[@]}"; do
    out_dir="${repo_dir}/outputs/tum/${dataset}"
    mkdir -p "$out_dir"
    echo "-- dense ${dataset}"

    # shellcheck disable=SC2086
    "$PYTHON" "${repo_dir}/main.py" \
        --backbone "$backbone" \
        --image_folder "${TUM_ROOT}/${dataset}/rgb" \
        --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 \
        --submap_size "$submap_size" \
        --log_results --log_path "${out_dir}/traj.txt" \
        $DVLT_ARGS > "${out_dir}/run.log" 2>&1 || { echo "   FAILED (see ${out_dir}/run.log)"; continue; }

    # main.py writes <log_path minus .txt>_points.pcd
    raw="${out_dir}/traj_points.pcd"
    if [ ! -s "$raw" ]; then echo "   no cloud produced"; continue; fi
    mv "$raw" "${out_dir}/points_raw.pcd"

    "$PYTHON" - "$out_dir" <<'PY'
import sys, os
import open3d as o3d

out = sys.argv[1]
raw = os.path.join(out, "points_raw.pcd")
pcd = o3d.io.read_point_cloud(raw)
n0 = len(pcd.points)
if n0 == 0:
    print("   empty cloud"); raise SystemExit
# Depth at frame borders throws far-field flyers that dominate the bounding box.
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.voxel_down_sample(0.01)
dst = os.path.join(out, "points_1cm.ply")
o3d.io.write_point_cloud(dst, pcd, write_ascii=False, compressed=True)
print(f"   {n0:,} -> {len(pcd.points):,} pts, {os.path.getsize(dst)/1e6:.1f} MB")
PY

    [ "$KEEP_RAW" = "0" ] && rm -f "${out_dir}/points_raw.pcd"
done

echo "DENSE_PASS_COMPLETE"
du -sh "${repo_dir}/outputs/tum" 2>/dev/null
