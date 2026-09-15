import os
import queue
import threading
from dataclasses import dataclass

import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images

from slam.slam_utils import compute_image_embeddings, Accumulator, backbone_name
from slam.loop_closure import ImageRetrieval
from slam.frame_overlap import FrameTracker
from slam.map import GraphMap
from slam.submap import Submap
from slam.graph import PoseGraph
from slam.scale_solver import estimate_scale_pairwise
from slam.viewer import Viewer

DEBUG = False

_SHUTDOWN = object()  # queue sentinel


@dataclass
class _SubmapWork:
    """One submap in flight between the GPU worker and the backend thread.

    Everything the backend needs that the GPU produced. Carried through the
    queue rather than parked on ``self`` so two submaps can be in flight
    without stepping on each other.
    """
    image_names: list
    images: object
    reconstruction: object
    retrieval_vectors: object
    semantic_vectors: object
    model: object


def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        vis_imgs: bool = False,
        model=None,
        clip_model=None,
        clip_preprocess=None,
        viewer: bool = True,
        viewer_port: int = 8080,
        scale_depth_percentile: float = 0.0):

        self.init_conf_threshold = init_conf_threshold
        # See add_edge: restricts the inter-submap scale estimate to the nearest
        # N% of confident overlap points. 0 disables (upstream behaviour).
        self.scale_depth_percentile = scale_depth_percentile
        self.vis_voxel_size = vis_voxel_size
        self.vis_imgs = vis_imgs

        # The solver owns its models, so a caller needs only Solver + track().
        # Still accepted per-call by run_predictions() for the batch path.
        self.model = model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

        # Keep the historical default for standalone entry points. ROS/RViz-only
        # callers pass viewer=False, avoiding an unnecessary Viser server/port.
        self.viewer = Viewer(port=viewer_port) if viewer else None

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.backbone_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

        # Guards every read and write of self.map / self.graph. The backend
        # thread is the only mutator; external readers (viewer panels, final
        # logging) take it to see a consistent map.
        self.map_lock = threading.RLock()

        self._running = False
        self._gpu_thread = None
        self._backend_thread = None
        self._gpu_q = None
        self._backend_q = None
        self._pending_keyframes = []
        self._latest_pose = None
        self._frame_count = 0
        self._keyframe_count = 0
        self._submaps_submitted = 0
        self._submaps_done = 0
        self._submaps_dropped = 0
        self._on_submap = None
        self._backend_active = False

    @property
    def vggt_timer(self):
        """Deprecated alias for ``backbone_timer``, kept for upstream scripts."""
        return self.backbone_timer

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.viewer is None:
            return
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        if self.viewer is None:
            return
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames() if self.vis_imgs else None
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        # Guarded here as well as in the setters: assembling world-frame points
        # for every submap is expensive, and with no viewer it is pure waste.
        if self.viewer is None:
            return
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        if self.viewer is None:
            return
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T # TODO Dominic check if we want to use P_prior here
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            prior_submap = self.map.get_submap(submap_id_prev)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
            good_mask = good_mask.reshape(-1)

            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points. 
                    good_mask = (prior_conf > 0).reshape(-1)

            # Depth band. estimate_scale_pairwise fits ONE scalar to the ratio
            # y_dist/x_dist over every confident point. That ratio is constant
            # only if the two submaps differ by a similarity; under a projective
            # difference it varies with depth, so the median lands wherever that
            # frame's median depth happened to be -- open road at one seam, a
            # building face at the next. Restricting every seam to the same
            # relative depth slice makes the bias consistent rather than
            # wandering, and a consistent scale bias is absorbed by Umeyama
            # alignment at scoring time.
            #
            # A percentile, not metres: the reconstruction is not metric, so an
            # absolute cut would mean a different physical distance per submap,
            # reintroducing exactly the inconsistency this is meant to remove.
            if self.scale_depth_percentile > 0:
                d_cur = np.linalg.norm(
                    current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3), axis=1)
                cutoff = np.percentile(d_cur[good_mask], self.scale_depth_percentile)
                banded = good_mask & (d_cur <= cutoff)
                if np.sum(banded) >= 100:
                    good_mask = banded
                else:
                    print(colored(
                        f"Depth band left only {np.sum(banded)} points; keeping the unbanded mask",
                        'yellow'))

            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            t1 = (P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output,
                  f"[{int(np.sum(good_mask))} pts]")
            scale_factor = scale_factor_est_output[0]
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            H_overlap_prior_overlap_current = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0] @ H_scale
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint.
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose) # TODO Dominic, no need to take the inverse twice, just use cam_to_world
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

    def add_points(self, pred_dict):
        """Fold one finished submap into the map and the pose graph.

        Args:
            pred_dict (dict): what ``_finalize_submap`` returns -- numpy, no batch
                dimension. The backbone geometry:
            {
                "images": (S, 3, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
                "detected_loops": [LoopMatch],
            }
                plus, only when a loop passed verification, the same geometry for
                the 2 loop frames under "*_lc", and "frames_lc"/"frames_lc_names".

            World points are not passed in; they are unprojected here from depth
            and the camera matrices.
        """
        # Unpack prediction dict
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, K_4x4)
        self.current_working_submap.set_conf_masks(conf)
        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"]) 
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]
            

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc)
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)
            self.map.add_submap(lc_submap)

            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    # ------------------------------------------------------------------
    # Submap reconstruction, split in two.
    #
    # _run_backbone is the GPU half and touches no map or graph state, so the
    # GPU worker can overlap it with the backend finishing the previous submap.
    # _finalize_submap is the map half and runs on the backend thread alone.
    # Anything that reads self.map belongs in the second, never the first.
    # ------------------------------------------------------------------

    def _run_backbone(self, image_names, model=None, clip_model=None, clip_preprocess=None):
        """GPU half: image loading, retrieval descriptors, backbone forward."""
        model = model if model is not None else self.model
        if model is None:
            raise RuntimeError(
                "Solver has no backbone. Pass model= to Solver(), or model= to run_predictions()."
            )
        clip_model = clip_model if clip_model is not None else self.clip_model
        clip_preprocess = clip_preprocess if clip_preprocess is not None else self.clip_preprocess

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        with self.backbone_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        retrieval_vectors = self.image_retrieval.get_batch_descriptors(images)

        semantic_vectors = None
        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                semantic_vectors = compute_image_embeddings(clip_model, clip_preprocess, image_names)

        with torch.no_grad():
            t1 = time.time()
            with self.backbone_timer:
                reconstruction = model.reconstruct(images)
            print(f"{backbone_name(model)} model inference took {time.time() - t1:.2f} seconds")

        return _SubmapWork(
            image_names=list(image_names),
            images=images,
            reconstruction=reconstruction,
            retrieval_vectors=retrieval_vectors,
            semantic_vectors=semantic_vectors,
            model=model,
        )

    def _finalize_submap(self, work, max_loops=1):
        """Map half: submap id, loop-closure detection and verification.

        Caller must hold ``self.map_lock``.
        """
        model = work.model
        images = work.images
        image_names = work.image_names
        recon = work.reconstruction

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(work.retrieval_vectors)
        new_submap.set_img_names(image_names)
        if work.semantic_vectors is not None:
            new_submap.set_all_semantic_vectors(work.semantic_vectors)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        # Check for loop closures and add retrieval vectors from new submap to the database
        verification = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)
                verification = model.verify_loop(lc_frames)
                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame), 
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        # The solver's own working dict: the backbone's geometry plus everything
        # this method derives. Distinct from what a backbone returns -- nothing
        # below this line is a model output.
        predictions = {
            "images": images,
            "extrinsic": recon.extrinsic,
            "intrinsic": recon.intrinsic,
            "depth": recon.depth,
            "depth_conf": recon.depth_conf,
            "detected_loops": detected_loops,
        }
        
        if verification is not None:
            if verification.match_score < 0.95:
                print(colored("Loop closure image match ratio too low, skipping loop closure", "red"))
                verification = None # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
            else:
                self.graph.increment_loop_closure()
                lc = verification.reconstruction
                predictions["extrinsic_lc"] = lc.extrinsic
                predictions["intrinsic_lc"] = lc.intrinsic
                predictions["depth_lc"] = lc.depth
                predictions["depth_conf_lc"] = lc.depth_conf

            
        # No squeeze(0) here any more: SubmapBackbone has no batch dimension, so
        # these are already (S, ...). Squeezing would silently drop the frame axis
        # of a single-frame submap.
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.float().cpu().numpy()

        if verification is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        return predictions

    def run_predictions(self, image_names, model=None, max_loops=1, clip_model=None, clip_preprocess=None):
        """Reconstruct one submap end to end, blocking.

        The batch path in main.py calls this. The live pipeline calls the two
        halves on separate threads instead.
        """
        work = self._run_backbone(image_names, model, clip_model, clip_preprocess)
        with self.map_lock:
            return self._finalize_submap(work, max_loops)


    # ------------------------------------------------------------------
    # Live pipeline.
    #
    # Three stages, mirroring ORB-SLAM3's Tracking / LocalMapping / LoopClosing:
    #
    #   track()          runs in the caller's thread. Keyframe gating only —
    #                    cheap enough to sit in a camera callback.
    #   _gpu_worker      one thread. _run_backbone, no map access.
    #   _backend_worker  one thread. The only mutator of map and graph.
    #
    # Queues are bounded. When the GPU worker falls behind, track() drops whole
    # submaps and counts them, rather than growing a backlog silently.
    # ------------------------------------------------------------------

    def start(self,
        submap_size: int = 16,
        overlapping_window_size: int = 1,
        max_loops: int = 1,
        min_disparity: float = 50.0,
        keyframe_dir: str = "keyframes",
        vis_map: bool = False,
        vis_flow: bool = False,
        gpu_queue_depth: int = 1,
        backend_queue_depth: int = 2,
        drop_when_full: bool = True):
        """Spawn the worker threads. Call before the first track().

        ``drop_when_full`` picks the back-pressure policy. True (live camera)
        drops whole submaps when the GPU worker is behind, keeping track()
        non-blocking. False (dataset replay) blocks track() instead, so every
        submap is processed and no frame is lost.
        """
        if self._running:
            raise RuntimeError("Solver pipeline is already running.")
        if self.model is None:
            raise RuntimeError("Solver has no backbone; construct it with Solver(..., model=...).")

        self._submap_size = submap_size
        self._overlap = overlapping_window_size
        self._target_size = submap_size + overlapping_window_size
        self._max_loops = max_loops
        self._min_disparity = min_disparity
        self._keyframe_dir = keyframe_dir
        self._vis_map = vis_map
        self._vis_flow = vis_flow
        self._drop_when_full = drop_when_full

        os.makedirs(keyframe_dir, exist_ok=True)

        self._gpu_q = queue.Queue(maxsize=gpu_queue_depth)
        self._backend_q = queue.Queue(maxsize=backend_queue_depth)
        self._running = True

        self._gpu_thread = threading.Thread(target=self._gpu_worker, name="dvlt-gpu", daemon=True)
        self._backend_thread = threading.Thread(target=self._backend_worker, name="dvlt-backend", daemon=True)
        self._gpu_thread.start()
        self._backend_thread.start()
        print(f"[Solver] Pipeline started (submap_size={submap_size}, overlap={overlapping_window_size}).")

    def set_submap_callback(self, fn):
        """Register fn(loop_closed: bool), called after each submap is merged.

        Runs on the backend thread with map_lock released, so the callback may
        take map_lock itself to read a consistent map. Lets a ROS node publish
        on map change instead of polling.
        """
        self._on_submap = fn

    def track(self, image, timestamp: float = None):
        """Feed one frame. Returns the latest world pose, or None before the first submap.

        Cheap: optical-flow keyframe gating and a disk write, nothing else. All
        reconstruction happens on the worker threads.
        """
        if not self._running:
            raise RuntimeError("Call start() before track().")

        self._frame_count += 1

        if self.flow_tracker.compute_disparity(image, self._min_disparity, self._vis_flow):
            # Submap.set_frame_ids parses a number out of the filename, so the
            # timestamp has to survive in the name for the pose logs to line up
            # with ground truth.
            name = f"{timestamp:.6f}.png" if timestamp is not None else f"frame_{self._keyframe_count:06d}.png"
            path = os.path.join(self._keyframe_dir, name)
            cv2.imwrite(path, image)
            self._keyframe_count += 1
            self._pending_keyframes.append(path)

        if len(self._pending_keyframes) >= self._target_size:
            self._submit(self._pending_keyframes)
            self._pending_keyframes = self._pending_keyframes[-self._overlap:]

        return self._latest_pose

    def _submit(self, image_names):
        if not self._drop_when_full:
            self._gpu_q.put(list(image_names))  # block until there is room
            self._submaps_submitted += 1
            return
        try:
            self._gpu_q.put_nowait(list(image_names))
            self._submaps_submitted += 1
        except queue.Full:
            self._submaps_dropped += 1
            print(colored(
                f"[Solver] GPU queue full, dropped submap "
                f"({self._submaps_dropped} dropped so far).", "yellow"))

    def _gpu_worker(self):
        while True:
            item = self._gpu_q.get()
            if item is _SHUTDOWN:
                self._backend_q.put(_SHUTDOWN)
                return
            try:
                work = self._run_backbone(item)
            except Exception:
                import traceback
                print(colored("[Solver] backbone failed, skipping submap:", "red"))
                traceback.print_exc()
                continue
            # Blocking on purpose: a slow backend stalls the GPU worker, which
            # backs the pressure up to track(), where it is visible as a drop.
            self._backend_q.put(work)

    def _backend_worker(self):
        while True:
            work = self._backend_q.get()
            if work is _SHUTDOWN:
                return
            self._backend_active = True
            try:
                with self.map_lock:
                    predictions = self._finalize_submap(work, self._max_loops)
                    self.add_points(predictions)
                    self.graph.optimize()
                    loop_closed = len(predictions["detected_loops"]) > 0
                    latest = self.map.get_latest_submap(ignore_loop_closure_submaps=True)
                    self._latest_pose = latest.get_all_poses_world(self.graph)[-1]
                    self._submaps_done += 1
                if self._vis_map:
                    with self.map_lock:
                        if loop_closed:
                            self.update_all_submap_vis()
                        else:
                            self.update_latest_submap_vis()
                if self._on_submap is not None:
                    try:
                        self._on_submap(loop_closed)
                    except Exception:
                        import traceback
                        print(colored("[Solver] submap callback raised:", "red"))
                        traceback.print_exc()
            except Exception:
                import traceback
                print(colored("[Solver] backend failed on a submap:", "red"))
                traceback.print_exc()
            finally:
                self._backend_active = False

    def flush(self) -> bool:
        """Submit the partial submap in hand. Returns whether there was one.

        Separate from shutdown() so a caller can drain the pipeline and keep
        running -- waiting for is_busy() to clear without flushing first would
        wait forever, because the tail is only submitted here.
        """
        if not self._running or len(self._pending_keyframes) <= self._overlap:
            return False  # a tail no longer than the overlap holds no new frames
        self._submit(self._pending_keyframes)
        self._pending_keyframes = self._pending_keyframes[-self._overlap:]
        return True

    def shutdown(self, flush: bool = True):
        """Stop the pipeline, optionally finishing the partial submap in hand."""
        if not self._running:
            return
        if flush:
            self.flush()
        self._pending_keyframes = []
        self._running = False

        self._gpu_q.put(_SHUTDOWN)
        self._gpu_thread.join()
        self._backend_thread.join()
        print(f"[Solver] Pipeline stopped ({self._submaps_done} submaps processed, "
              f"{self._submaps_dropped} dropped).")

    def is_busy(self) -> bool:
        """True while any frame already handed to track() has yet to reach the map.

        Covers four places work hides, not just the queues:
          * keyframes buffered in the frontend that shutdown() will still flush
          * the queues themselves
          * a submap in the GPU worker (off the queue, not yet on the next)
          * a submap in the backend, including its on_submap callback

        Counting only the queues reports "drained" while two submaps are still
        in flight, which is exactly wrong for anyone using this to decide that
        a replay has finished.
        """
        if not self._running:
            return False
        pending_tail = len(self._pending_keyframes) > getattr(self, "_overlap", 0)
        return (pending_tail
                or not self._gpu_q.empty()
                or not self._backend_q.empty()
                or self._backend_active
                or self._submaps_done < self._submaps_submitted)

    def get_stats(self) -> dict:
        return {
            "frames": self._frame_count,
            "keyframes": self._keyframe_count,
            "pending": len(self._pending_keyframes),
            "target_size": getattr(self, "_target_size", None),
            "submitted": self._submaps_submitted,
            "done": self._submaps_done,
            "dropped": self._submaps_dropped,
            "busy": self.is_busy(),
        }

    def get_latest_pose(self):
        """Latest world pose as a 4x4 SE(3), or None before the first submap lands.

        Cached by the backend rather than recomputed, so calling this at camera
        rate costs nothing.
        """
        return self._latest_pose
