"""Replay an image folder through DVLT-SLAM, losslessly.

    ros2 launch dvlt_slam_ros replay.launch.py \
        folder:=/path/to/office_loop log_path:=/tmp/ros_poses.txt
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    args = [
        DeclareLaunchArgument("folder", description="image folder (flat, or TUM with rgb/)"),
        DeclareLaunchArgument("log_path", default_value="", description="TUM-format trajectory output"),
        DeclareLaunchArgument("fps", default_value="10.0"),
        DeclareLaunchArgument("backbone", default_value="dvlt"),
        DeclareLaunchArgument("submap_size", default_value="16"),
        DeclareLaunchArgument("keyframe_dir", default_value="/tmp/dvlt_keyframes"),
        DeclareLaunchArgument("publish_cloud", default_value="true"),
        DeclareLaunchArgument("mode", default_value="replay", description="replay (lossless) | live (drop)"),
        # SLAM tuning. Defaults mirror slam_node's own declared defaults, so
        # omitting these changes nothing. min_disparity is the one that matters
        # per-dataset: 50 suits TUM's handheld indoor motion, but forward
        # vehicle motion produces a low-magnitude flow expansion field, so on
        # KITTI it starves the keyframe gate on fast straight segments.
        DeclareLaunchArgument("min_disparity", default_value="50.0",
                              description="optical-flow threshold for a new keyframe"),
        DeclareLaunchArgument("conf_threshold", default_value="25.0"),
        DeclareLaunchArgument("scale_depth_percentile", default_value="0.0",
                              description="restrict inter-submap scale to the nearest N% of overlap points; 0 = all"),
        DeclareLaunchArgument("lc_thres", default_value="0.95"),
        DeclareLaunchArgument("max_loops", default_value="1"),
        DeclareLaunchArgument("overlapping_window_size", default_value="1"),
        DeclareLaunchArgument("auto_shutdown_idle", default_value="5.0",
                              description="finish after this many quiet seconds; 0 = run until Ctrl-C"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="open RViz2 on the map cloud, trajectory and tf"),
        DeclareLaunchArgument("vis_map", default_value="false",
                              description="also serve the repo's viser viewer; costs backend time"),
    ]
    topic = "/camera/image_raw"
    slam = Node(
            package="dvlt_slam_ros", executable="slam_node", name="dvlt_slam",
            output="screen",
            parameters=[{
                "image_topic": topic,
                "mode": LaunchConfiguration("mode"),
                "backbone": LaunchConfiguration("backbone"),
                "submap_size": LaunchConfiguration("submap_size"),
                "keyframe_dir": LaunchConfiguration("keyframe_dir"),
                "publish_cloud": LaunchConfiguration("publish_cloud"),
                "log_path": LaunchConfiguration("log_path"),
                "auto_shutdown_idle": LaunchConfiguration("auto_shutdown_idle"),
                "vis_map": LaunchConfiguration("vis_map"),
                "min_disparity": LaunchConfiguration("min_disparity"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "scale_depth_percentile": LaunchConfiguration("scale_depth_percentile"),
                "lc_thres": LaunchConfiguration("lc_thres"),
                "max_loops": LaunchConfiguration("max_loops"),
                "overlapping_window_size": LaunchConfiguration("overlapping_window_size"),
            }],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", os.path.join(
            get_package_share_directory("dvlt_slam_ros"), "config", "dvlt_slam.rviz")],
    )
    publisher = Node(
            package="dvlt_slam_ros", executable="image_folder_publisher",
            name="image_folder_publisher", output="screen",
            parameters=[{
                "folder": LaunchConfiguration("folder"),
                "topic": topic,
                "fps": LaunchConfiguration("fps"),
                "consumed_topic": "/dvlt_slam/frames_consumed",
            }],
    )
    return LaunchDescription(args + [
        slam,
        publisher,
        rviz,
        # When replay finishes, tear the whole launch down. The SLAM node gets
        # SIGINT, which is what makes it flush the trajectory and join workers.
        # The SLAM node decides when the run is over: it waits for the stream
        # to stop AND the backend to drain. Keying off the publisher instead
        # would kill it with submaps still queued.
        RegisterEventHandler(OnProcessExit(target_action=slam,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
