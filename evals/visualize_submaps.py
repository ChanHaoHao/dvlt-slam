"""Render a reconstruction with each submap drawn in its own colour.

Runs the VGGT-SLAM pipeline (unmodified solver) over a sequence, then draws:

  <name>_geometry.png   point cloud in natural colour, trajectory coloured per submap
  <name>_submaps.png    points themselves coloured per submap, showing the seams

Colouring by submap is the point: it makes the submap decomposition visible, so
seam placement and any misalignment between adjacent submaps can be seen directly
rather than inferred from an ATE number.

Usage:
    python evals/visualize_submaps.py --image_folder office_loop --name office \
        --submap_size 16 --lc_verify attn
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
from open3d.visualization import rendering
from tqdm.auto import tqdm

import slam.slam_utils as utils
from slam.solver import Solver


# Qualitative palette: distinct hues, readable on white, colour-blind friendly ordering.
PALETTE = np.array([
    [0.12, 0.47, 0.71], [0.89, 0.47, 0.06], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.22, 0.30, 0.55], [0.90, 0.62, 0.00],
])


def build_map(args):
    from slam.backbones.dvlt_backbone import DVLTBackbone

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=None,
        vis_imgs=False,
    )
    model = DVLTBackbone(inference_steps=args.dvlt_k, lc_verify=args.lc_verify)

    names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
             if "depth" not in os.path.basename(f).lower()
             and "txt" not in os.path.basename(f).lower()
             and "db" not in os.path.basename(f).lower()]
    names = utils.sort_images_by_number(names)
    print(f"{len(names)} images in {args.image_folder}")

    subset = []
    for name in tqdm(names):
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, args.min_disparity, False):
            subset.append(name)
        if len(subset) == args.submap_size + 1 or name == names[-1]:
            preds = solver.run_predictions(subset, model, args.max_loops, None, None)
            solver.add_points(preds)
            solver.graph.optimize()
            subset = [subset[-1]]
    return solver


def collect(solver):
    """Per-submap points, colours and camera centres, all in the optimised world frame."""
    graph = solver.graph
    submaps = []
    for i, sm in enumerate(solver.map.ordered_submaps_by_key()):
        pts = np.asarray(sm.get_points_in_world_frame(graph)).reshape(-1, 3)
        cols = np.asarray(sm.get_points_colors()).reshape(-1, 3)
        if cols.max() > 1.0:
            cols = cols / 255.0
        poses = np.asarray(sm.get_all_poses_world(graph))
        centres = poses[:, :3, 3] if poses.ndim == 3 else poses[:, :3]
        submaps.append(dict(idx=i, pts=pts, cols=cols, centres=centres,
                            colour=PALETTE[i % len(PALETTE)]))
    return submaps


def clean(pts, cols, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    if len(pcd.points) == 0:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd.voxel_down_sample(voxel)


def traj_geometry(submaps, radius):
    """Camera centres as spheres plus connecting lines, coloured by submap."""
    spheres, lines = [], []
    for sm in submaps:
        c = sm["centres"]
        for p in c:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            s.translate(p)
            s.paint_uniform_color(sm["colour"])
            s.compute_vertex_normals()
            spheres.append(s)
        if len(c) > 1:
            ls = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(c),
                o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(c) - 1)]),
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(sm["colour"], (len(c) - 1, 1)))
            lines.append(ls)
    return spheres, lines


def render(geoms, path, width=1500, height=1050, view=("persp", 0.8)):
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([1, 1, 1, 1])
    r.scene.scene.set_sun_light([-0.3, -0.8, -0.5], [1, 1, 1], 75000)

    mp = rendering.MaterialRecord(); mp.shader = "defaultUnlit"; mp.point_size = 2.5
    ml = rendering.MaterialRecord(); ml.shader = "unlitLine"; ml.line_width = 3.0
    # Unlit, so the camera spheres show their submap colour rather than a shaded version of it.
    mm = rendering.MaterialRecord(); mm.shader = "defaultUnlit"

    lo, hi = [], []
    for i, g in enumerate(geoms):
        if isinstance(g, o3d.geometry.PointCloud):
            mat = mp
        elif isinstance(g, o3d.geometry.LineSet):
            mat = ml
        else:
            mat = mm
        r.scene.add_geometry(f"g{i}", g, mat)
        bb = g.get_axis_aligned_bounding_box()
        lo.append(np.asarray(bb.min_bound))
        hi.append(np.asarray(bb.max_bound))

    # Open3D AABBs do not support +, so union them by hand.
    lo, hi = np.min(np.stack(lo), axis=0), np.max(np.stack(hi), axis=0)
    c = (lo + hi) * 0.5
    rad = float(np.linalg.norm(hi - lo)) * 0.5
    kind, zoom = view
    eye = c + (np.array([0.02, -1, 0.02]) if kind == "top" else np.array([1.1, -0.85, -1.1])) * rad * zoom
    r.scene.camera.look_at(c, eye, [0, 0, 1] if kind == "top" else [0, -1, 0])
    o3d.io.write_image(path, r.render_to_image())
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="outputs/figures")
    ap.add_argument("--submap_size", type=int, default=16)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--conf_threshold", type=float, default=25.0)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--lc_verify", default="attn")
    ap.add_argument("--dvlt_k", type=int, default=None)
    ap.add_argument("--voxel", type=float, default=0.012)
    ap.add_argument("--view", default="persp", choices=["persp", "top"])
    ap.add_argument("--zoom", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    solver = build_map(args)
    submaps = collect(solver)
    if not submaps:
        print("no submaps produced"); sys.exit(1)

    all_pts = np.concatenate([s["pts"] for s in submaps])
    all_cols = np.concatenate([s["cols"] for s in submaps])
    span = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0)))
    radius = max(span * 0.004, 1e-3)
    n_cams = sum(len(s["centres"]) for s in submaps)
    print(f"{len(submaps)} submaps, {n_cams} camera poses, {len(all_pts):,} points, span {span:.2f} m")

    spheres, lines = traj_geometry(submaps, radius)
    view = (args.view, args.zoom)

    # Figure 1: true-colour geometry with the trajectory coloured per submap.
    natural = clean(all_pts, all_cols, args.voxel)
    render([natural, *lines, *spheres],
           os.path.join(args.out_dir, f"{args.name}_geometry.png"), view=view)

    # Figure 2: the points themselves coloured per submap, exposing the seams.
    seg_cols = np.concatenate([np.tile(s["colour"], (len(s["pts"]), 1)) for s in submaps])
    segmented = clean(all_pts, seg_cols, args.voxel)
    render([segmented, *lines, *spheres],
           os.path.join(args.out_dir, f"{args.name}_submaps.png"), view=view)

    o3d.io.write_point_cloud(os.path.join(args.out_dir, f"{args.name}_cloud.ply"),
                             natural, write_ascii=False, compressed=True)
    with open(os.path.join(args.out_dir, f"{args.name}_stats.txt"), "w") as fh:
        fh.write(f"submaps {len(submaps)}\ncameras {n_cams}\n"
                 f"points_raw {len(all_pts)}\npoints_shown {len(natural.points)}\nspan_m {span:.3f}\n")


if __name__ == "__main__":
    main()
