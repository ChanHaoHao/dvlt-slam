"""Live SLAM demo, DVLT or VGGT backbone.

A thin driver: build the models, hand them to the Solver, then push frames at
it. All the concurrency lives inside Solver.start() / Solver.track() so that a
ROS node, a different camera backend, or this file are interchangeable callers.
"""

import time
import argparse

import cv2
import torch
from torchvision.transforms.functional import to_pil_image

import slam.slam_utils as utils
from slam.backbones import build_backbone
from slam.solver import Solver
from slam.cameras import BACKENDS

parser = argparse.ArgumentParser(description="Live SLAM demo (DVLT or VGGT backbone)")
parser.add_argument("--keyframe_folder", type=str, default="keyframes", help="Folder to save captured keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=list(BACKENDS.keys()), help="Camera backend (default: realsense)")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being built, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--backbone", type=str, default="dvlt", choices=["vggt", "dvlt"], help="Per-submap reconstruction backbone")
parser.add_argument("--dvlt_checkpoint", type=str, default="nvidia/dvlt", help="DVLT checkpoint: local dir, HTTPS URL, or HF Hub repo id")
parser.add_argument("--dvlt_k", type=int, default=None, help="DVLT refinement iterations K at inference. Default: the checkpoint's own inference_steps")
parser.add_argument("--lc_verify", type=str, default="attn", choices=["bypass", "attn"], help="How the DVLT backbone produces image_match_ratio for loop-closure verification")
parser.add_argument("--lc_attn_step", type=int, default=None, help="Iteration k to read attention from when --lc_verify=attn")
parser.add_argument("--queue_depth", type=int, default=1, help="Submaps allowed to queue for the GPU worker before track() starts dropping them")


def restart_camera(camera, retry_delay: float = 2.0, max_attempts: int = None):
    """Stop and re-start the camera, retrying until a device is streaming again.

    Recovers from a mid-session disconnect (e.g. a RealSense USB drop), where
    ``capture()`` raises. Returns the working camera, or None if it gave up.
    """
    try:
        camera.stop()
    except Exception:
        pass  # device may already be gone; ignore teardown errors

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        try:
            camera.start()
            print(f"[Camera] Reconnected after {attempt} attempt(s).")
            return camera
        except Exception as e:
            print(f"[Camera] Reconnect attempt {attempt} failed: {e}.")
            print(f"[Camera] Retrying in {retry_delay:.0f}s...")
            time.sleep(retry_delay)
    return None


def run_semantic_query_loop(solver, clip_model, clip_tokenizer, processor):
    """Interactive open-set semantic query loop, run after capture ends."""
    while True:
        query = input("\nEnter text query or q to quit: ").strip()
        if len(query) == 0 or query == "q":
            print("Exiting.")
            return

        text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
        with solver.map_lock:
            best_score, best_submap_id, best_frame_index = \
                solver.map.retrieve_best_semantic_frame(text_emb)
            found_submap = solver.map.get_submap(best_submap_id)
            best_img = found_submap.get_frame_at_index(best_frame_index)

        print("Score:", best_score)
        with torch.no_grad():
            best_img = to_pil_image(best_img)
            inference_state = processor.set_image(best_img)
            output = processor.set_text_prompt(state=inference_state, prompt=query)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
            print("Scores:", scores.cpu().numpy())

        masked_img = utils.overlay_masks(best_img, masks)
        masked_img.show()

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            with solver.map_lock:
                points = found_submap.get_points_in_mask(best_frame_index, mask, solver.graph)
            obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(points)
            solver.viewer.visualize_obb(
                center=obb_center, extent=obb_extent, rotation=obb_rotation,
                color=(255, 0, 0), line_width=8.0,
            )


def main():
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # When --run_os is set, SAM3/decord loads its own libxcb which poisons
    # OpenCV's XCB state and makes cv2.waitKey() hang. Skip cv2 display in that
    # case and print periodic status to the console instead.
    use_display = not args.run_os

    clip_model = clip_preprocess = clip_tokenizer = processor = None
    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.50)
        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)

    print(f"Initializing and loading {args.backbone.upper()} model...")
    model = build_backbone(
        backbone=args.backbone,
        device=device,
        dvlt_checkpoint=args.dvlt_checkpoint,
        dvlt_k=args.dvlt_k,
        lc_verify=args.lc_verify,
        lc_attn_step=args.lc_attn_step,
    )

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        model=model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
    )
    backbone = utils.backbone_name(model)
    # One name for both imshow calls: a mismatch opens two windows.
    window_name = f"{backbone}-SLAM Live"
    print(f"All models loaded ({backbone} backbone). Starting SLAM loop.")

    if args.run_os:
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, solver.map_lock)

    camera = BACKENDS[args.camera]()
    print(f"Initializing {args.camera} camera...")
    camera.start()

    # Warm up the camera — first few frames can be None
    print("Waiting for first camera frame...")
    first_frame = None
    while first_frame is None:
        try:
            first_frame = camera.capture()
        except Exception as e:
            print(f"[Camera] Error during warm-up ({e}). Reconnecting...")
            camera = restart_camera(camera)
            if camera is None:
                return
    if use_display:
        cv2.imshow(window_name, first_frame)
        cv2.waitKey(1)
    print("Camera ready.")

    solver.start(
        submap_size=args.submap_size,
        overlapping_window_size=args.overlapping_window_size,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        keyframe_dir=args.keyframe_folder,
        vis_map=args.vis_map,
        vis_flow=args.vis_flow,
        gpu_queue_depth=args.queue_depth,
    )

    last_status_frame = 0
    try:
        while True:
            try:
                img = camera.capture()
            except Exception as e:
                print(f"[Camera] Lost connection ({e}). Attempting to reconnect...")
                camera = restart_camera(camera)
                if camera is None:
                    break
                continue
            if img is None:
                continue

            solver.track(img, timestamp=time.time())
            stats = solver.get_stats()

            if use_display:
                display = img.copy()
                status = f"KFs: {stats['pending']}/{stats['target_size']}  Submaps: {stats['done']}"
                if stats["busy"]:
                    status += "  [SLAM running]"
                if stats["dropped"]:
                    status += f"  [dropped {stats['dropped']}]"
                cv2.putText(display, status, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif stats["frames"] - last_status_frame >= 30:
                busy_str = "  [SLAM running]" if stats["busy"] else ""
                print(f"[Camera] frame={stats['frames']}  KFs={stats['pending']}/{stats['target_size']}"
                      f"  submaps={stats['done']}  dropped={stats['dropped']}{busy_str}")
                last_status_frame = stats["frames"]

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        camera.stop()
        if use_display:
            cv2.destroyAllWindows()
        solver.shutdown()  # finishes the partial submap and joins the workers

    with solver.map_lock:
        print("Total number of submaps in map", solver.map.get_num_submaps())
        print("Total number of loop closures in map", solver.graph.get_num_loops())
        if not args.vis_map:
            solver.update_all_submap_vis()

    if args.run_os:
        run_semantic_query_loop(solver, clip_model, clip_tokenizer, processor)

    if args.log_results:
        with solver.map_lock:
            solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
            if not args.skip_dense_log:
                solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
