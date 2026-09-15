"""ROS 2 node wrapping slam.Solver.

Thin by construction: the image callback converts and calls track(), and a
callback from the solver's backend thread publishes whenever the map changes.
All concurrency lives inside Solver, so this file owns no threads and no locks.

Frames
------
Poses are published as map -> camera. There is no odom frame because there is
no odometry source; the estimate jumps when a loop closes, which is correct and
visible. The trajectory is also up to scale (monocular), so anything consuming
it must align with scale, e.g. `evo_ape -as`.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import UInt64
from tf2_ros import TransformBroadcaster

from slam.backbones import build_backbone
from slam.solver import Solver

from dvlt_slam_ros.conversions import (
    image_msg_to_bgr, stamp_to_sec, sec_to_stamp,
    se3_to_transform, se3_to_pose_stamped, make_pointcloud2,
)

# Lossless replay: KEEP_ALL + RELIABLE makes DDS back-pressure the publisher
# when the node is inside a blocking track(), so a dataset run drops nothing.
REPLAY_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_ALL,
    depth=1,
)
# Live camera: never let a backlog build; the solver's own drop policy decides.
LIVE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
LATCHED = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


class DvltSlamNode(Node):
    def __init__(self):
        super().__init__("dvlt_slam")

        p = self.declare_parameters("", [
            ("image_topic", "/camera/image_raw"),
            ("mode", "replay"),                 # replay (lossless) | live (drop)
            ("map_frame", "map"),
            ("camera_frame", "camera"),
            # backbone
            ("backbone", "dvlt"),
            ("dvlt_checkpoint", "nvidia/dvlt"),
            ("dvlt_k", -1),                     # -1 = checkpoint default
            ("lc_verify", "attn"),
            ("lc_attn_step", -1),
            # slam
            ("submap_size", 16),
            ("overlapping_window_size", 1),
            ("max_loops", 1),
            ("min_disparity", 15.0),
            ("conf_threshold", 25.0),
            ("lc_thres", 0.95),
            ("keyframe_dir", "keyframes"),
            # output
            ("publish_cloud", True),
            ("cloud_voxel_size", 0.05),
            ("vis_map", False),          # also starts viser; see below
            ("viewer_port", 8080),
            ("log_path", ""),
            # >0: finish once no frame has arrived for this long AND the
            # pipeline has drained. 0 disables (run until Ctrl-C).
            ("auto_shutdown_idle", 0.0),
        ])
        self.cfg = {q.name: q.value for q in p}
        mode = self.cfg["mode"]
        if mode not in ("replay", "live"):
            raise ValueError(f"mode must be 'replay' or 'live', got {mode!r}")

        self.map_frame = self.cfg["map_frame"]
        self.camera_frame = self.cfg["camera_frame"]

        self.get_logger().info(f"Loading {self.cfg['backbone'].upper()} backbone...")
        model = build_backbone(
            backbone=self.cfg["backbone"],
            dvlt_checkpoint=self.cfg["dvlt_checkpoint"],
            dvlt_k=None if self.cfg["dvlt_k"] < 0 else self.cfg["dvlt_k"],
            lc_verify=self.cfg["lc_verify"],
            lc_attn_step=None if self.cfg["lc_attn_step"] < 0 else self.cfg["lc_attn_step"],
        )
        # vis_map does double duty: it starts the viser server AND makes the
        # backend push geometry to it. With it off, RViz is the only GUI and the
        # solver never builds viser scenes on the backend thread.
        self.solver = Solver(
            init_conf_threshold=self.cfg["conf_threshold"],
            lc_thres=self.cfg["lc_thres"],
            model=model,
            viewer=self.cfg["vis_map"],
            viewer_port=self.cfg["viewer_port"],
        )
        if self.cfg["vis_map"]:
            self.get_logger().info(
                f"viser viewer at http://localhost:{self.cfg['viewer_port']} "
                "(note: cloud building runs on the backend thread and slows it)")
        self.solver.set_submap_callback(self.on_submap)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, "~/pose", 10)
        self.path_pub = self.create_publisher(Path, "~/path", LATCHED)
        self.cloud_pub = self.create_publisher(PointCloud2, "~/cloud", LATCHED)
        # Lets a replay publisher pace itself instead of buffering ahead.
        self.consumed_pub = self.create_publisher(UInt64, "~/frames_consumed", 10)

        self.solver.start(
            submap_size=self.cfg["submap_size"],
            overlapping_window_size=self.cfg["overlapping_window_size"],
            max_loops=self.cfg["max_loops"],
            min_disparity=self.cfg["min_disparity"],
            keyframe_dir=self.cfg["keyframe_dir"],
            vis_map=self.cfg["vis_map"],
            drop_when_full=(mode == "live"),
        )

        # Subscribe last: the pipeline must be running before a frame can land,
        # and a subscriber appearing is what unblocks a waiting replay publisher.
        self.sub = self.create_subscription(
            Image, self.cfg["image_topic"], self.on_image,
            REPLAY_QOS if mode == "replay" else LIVE_QOS,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.last_frame_time = None
        idle = float(self.cfg["auto_shutdown_idle"])
        if idle > 0:
            self.create_timer(1.0, self._check_idle)

        self.get_logger().info(
            f"Listening on {self.cfg['image_topic']} in {mode} mode "
            f"({'drops allowed' if mode == 'live' else 'lossless, back-pressured'})."
        )

    def _check_idle(self):
        """End the run once the stream stops and every queued submap is merged.

        Frames stopping is not the end: the backend can still hold several
        submaps. Waiting on is_busy() too is what makes a replay produce the
        same map as a batch run.
        """
        if self.last_frame_time is None:
            return
        quiet = (self.get_clock().now() - self.last_frame_time).nanoseconds * 1e-9
        if quiet < float(self.cfg["auto_shutdown_idle"]):
            return
        # Submit the partial submap first: is_busy() counts it as outstanding,
        # so waiting on it without flushing never terminates.
        if self.solver.flush():
            self.get_logger().info("Stream idle; flushed the final partial submap.")
        if self.solver.is_busy():
            self.get_logger().info("Stream idle; draining pipeline...", throttle_duration_sec=5.0)
            return
        self.get_logger().info("Stream idle and pipeline drained. Finishing.")
        raise SystemExit(0)

    # -- frontend -----------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            img = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(f"{e}")
            return

        self.last_frame_time = self.get_clock().now()
        pose = self.solver.track(img, timestamp=stamp_to_sec(msg.header.stamp))
        self.consumed_pub.publish(UInt64(data=self.solver.get_stats()["frames"]))
        if pose is None:
            return  # no submap has landed yet

        self.tf_broadcaster.sendTransform(
            se3_to_transform(pose, msg.header.stamp, self.map_frame, self.camera_frame))
        self.pose_pub.publish(
            se3_to_pose_stamped(pose, msg.header.stamp, self.map_frame))

    # -- called on the solver's backend thread, map_lock released -----------
    def on_submap(self, loop_closed: bool):
        stamp = self.get_clock().now().to_msg()
        with self.solver.map_lock:
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = self.map_frame
            for submap in self.solver.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue  # loop-closure submaps are not part of the trajectory
                for pose, fid in zip(submap.get_all_poses_world(self.solver.graph),
                                     submap.get_frame_ids()):
                    path.poses.append(se3_to_pose_stamped(pose, sec_to_stamp(fid), self.map_frame))
            stats = self.solver.get_stats()
            cloud = self._build_cloud(stamp) if self.cfg["publish_cloud"] else None

        self.path_pub.publish(path)
        if cloud is not None:
            self.cloud_pub.publish(cloud)
        self.get_logger().info(
            f"submap {stats['done']} merged{' (LOOP CLOSED)' if loop_closed else ''} — "
            f"{len(path.poses)} poses, {stats['dropped']} dropped")

    def _build_cloud(self, stamp):
        """Whole map as one cloud. Caller must hold map_lock."""
        voxel = self.cfg["cloud_voxel_size"]
        pts, cols = [], []
        for submap in self.solver.map.get_submaps():
            p = submap.get_points_in_world_frame(self.solver.graph)
            c = submap.get_points_colors()
            if len(p):
                pts.append(p)
                cols.append(c)
        if not pts:
            return None
        pts = np.concatenate(pts, axis=0)
        cols = np.concatenate(cols, axis=0)

        if voxel and voxel > 0:
            # Dense submaps are millions of points each; sending them raw over
            # DDS stalls the backend thread. Voxel-grid to one point per cell.
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(voxel)
            pts = np.asarray(pcd.points, dtype=np.float32)
            cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        return make_pointcloud2(pts, cols, stamp, self.map_frame)

    def finish(self):
        self.solver.shutdown()
        log_path = self.cfg["log_path"]
        if log_path and self.solver.map.get_num_submaps() > 0:
            with self.solver.map_lock:
                self.solver.map.write_poses_to_file(log_path, self.solver.graph, kitti_format=False)
            self.get_logger().info(f"Wrote trajectory to {log_path}")
        elif log_path:
            self.get_logger().warn("Map is empty; no trajectory written.")
        stats = self.solver.get_stats()
        self.get_logger().info(
            f"Final: {stats['frames']} frames, {stats['keyframes']} keyframes, "
            f"{stats['done']} submaps, {stats['dropped']} dropped, "
            f"{self.solver.graph.get_num_loops()} loop closures")


def main():
    rclpy.init()
    node = DvltSlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
