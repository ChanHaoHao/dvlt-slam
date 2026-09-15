"""Publish an image folder as sensor_msgs/Image, for offline verification.

Handles both layouts in use here:
  flat  --  office_loop/frame_0001.jpg
  TUM   --  rgbd_dataset_freiburg1_desk/rgb/1305031102.175304.png

The header stamp is parsed from the filename, exactly as Submap.set_frame_ids
does, so a trajectory logged from the node carries the same timestamps as one
logged by main.py and the two are directly diffable.

Flow control: with max_lead > 0 the publisher waits for the node to report
frames consumed before running further ahead. Without it, a blocking track()
call makes the subscriber queue grow without bound on a KEEP_ALL profile.
"""

import glob
import os
import re

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt64

from dvlt_slam_ros.conversions import sec_to_stamp

RELIABLE_ALL = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_ALL,
    depth=1,
)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _natural_key(path):
    m = _NUM.search(os.path.basename(path))
    return float(m.group()) if m else 0.0


def find_images(folder):
    """Prefer a TUM-style rgb/ subfolder, else treat the folder as flat."""
    folder = os.path.expanduser(folder)
    rgb_dir = os.path.join(folder, "rgb")
    search = rgb_dir if os.path.isdir(rgb_dir) else folder
    files = []
    for e in _EXTS:
        files.extend(glob.glob(os.path.join(search, e)))
    # Match main.py's filtering so both paths see the same frame set.
    files = [f for f in files
             if not any(s in os.path.basename(f).lower() for s in ("depth", "txt", "db"))]
    return sorted(files, key=_natural_key), search


class ImageFolderPublisher(Node):
    def __init__(self):
        super().__init__("image_folder_publisher")
        self.declare_parameter("folder", "")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("max_lead", 30)
        self.declare_parameter("consumed_topic", "/dvlt_slam/frames_consumed")
        self.declare_parameter("exit_when_done", True)

        folder = self.get_parameter("folder").value
        if not folder:
            raise RuntimeError("parameter 'folder' is required")
        self.files, search = find_images(folder)
        if not self.files:
            raise RuntimeError(f"no images found in {search}")

        self.max_lead = int(self.get_parameter("max_lead").value)
        self.consumed = 0
        self.i = 0
        self.pub = self.create_publisher(Image, self.get_parameter("topic").value, RELIABLE_ALL)
        self.create_subscription(
            UInt64, self.get_parameter("consumed_topic").value, self._on_consumed, 10)

        self.exit_when_done = bool(self.get_parameter("exit_when_done").value)
        self.waiting_logged = False
        fps = float(self.get_parameter("fps").value)
        self.create_timer(1.0 / max(fps, 0.01), self.tick)
        self.get_logger().info(
            f"Publishing {len(self.files)} images from {search} at {fps} fps "
            f"(max_lead={self.max_lead})")

    def _on_consumed(self, msg):
        self.consumed = int(msg.data)

    def tick(self):
        # RELIABLE QoS is not TRANSIENT_LOCAL: anything published before the
        # node finishes loading its backbone and subscribes is simply lost, and
        # the flow control below would then wait forever. So wait for a peer.
        if self.pub.get_subscription_count() == 0:
            if not self.waiting_logged:
                self.get_logger().info("Waiting for a subscriber before publishing...")
                self.waiting_logged = True
            return

        if self.i >= len(self.files):
            if self.exit_when_done and self.consumed >= len(self.files):
                self.get_logger().info(
                    f"All {len(self.files)} images published and consumed. Shutting down.")
                raise SystemExit(0)
            return
        if self.max_lead > 0 and (self.i - self.consumed) >= self.max_lead:
            return  # node is behind; let it catch up rather than buffering

        path = self.files[self.i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f"unreadable: {path}")
            self.i += 1
            return

        msg = Image()
        msg.header.stamp = sec_to_stamp(_natural_key(path))
        msg.header.frame_id = "camera"
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        self.pub.publish(msg)
        self.i += 1
        if self.i % 50 == 0:
            self.get_logger().info(f"published {self.i}/{len(self.files)} (consumed {self.consumed})")


def main():
    rclpy.init()
    node = ImageFolderPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
