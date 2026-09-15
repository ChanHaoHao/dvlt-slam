"""ROS <-> numpy conversions for the DVLT-SLAM node.

Deliberately does NOT use cv_bridge. Jazzy ships cv_bridge compiled against
numpy 1.26, and this stack runs numpy 2.x, so importing it aborts with an ABI
error. sensor_msgs/Image is a header plus a byte array, and the encodings we
care about need no colour-space machinery, so the conversion is a reshape.
"""

import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, PoseStamped
from builtin_interfaces.msg import Time as TimeMsg

# Channel count per supported encoding. bgr8 is what the solver wants, because
# FrameTracker and cv2.imwrite both assume OpenCV channel order.
_CHANNELS = {"bgr8": 3, "rgb8": 3, "8UC3": 3, "mono8": 1, "8UC1": 1}


def stamp_to_sec(stamp: TimeMsg) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(seconds: float) -> TimeMsg:
    sec = int(seconds)
    return TimeMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 BGR."""
    enc = msg.encoding
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {enc!r}; expected one of {sorted(_CHANNELS)}")
    channels = _CHANNELS[enc]

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # step is the row stride in bytes and may exceed width*channels (padding).
    expected = msg.height * msg.step
    if buf.size < expected:
        raise ValueError(f"image buffer too small: {buf.size} < {expected}")
    rows = buf[:expected].reshape(msg.height, msg.step)
    img = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    elif enc == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def se3_to_transform(pose: np.ndarray, stamp: TimeMsg, parent: str, child: str) -> TransformStamped:
    """4x4 SE(3) -> TransformStamped. Rotation via quaternion, no scipy needed."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(pose[0, 3])
    t.transform.translation.y = float(pose[1, 3])
    t.transform.translation.z = float(pose[2, 3])
    qx, qy, qz, qw = _mat_to_quat(pose[:3, :3])
    t.transform.rotation.x, t.transform.rotation.y = qx, qy
    t.transform.rotation.z, t.transform.rotation.w = qz, qw
    return t


def se3_to_pose_stamped(pose: np.ndarray, stamp: TimeMsg, frame: str) -> PoseStamped:
    p = PoseStamped()
    p.header.stamp = stamp
    p.header.frame_id = frame
    p.pose.position.x = float(pose[0, 3])
    p.pose.position.y = float(pose[1, 3])
    p.pose.position.z = float(pose[2, 3])
    qx, qy, qz, qw = _mat_to_quat(pose[:3, :3])
    p.pose.orientation.x, p.pose.orientation.y = qx, qy
    p.pose.orientation.z, p.pose.orientation.w = qz, qw
    return p


def _mat_to_quat(r: np.ndarray):
    """Rotation matrix -> (x, y, z, w), Shepperd's method for numerical stability."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


_CLOUD_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def make_pointcloud2(points: np.ndarray, colors: np.ndarray, stamp: TimeMsg, frame: str) -> PointCloud2:
    """(N,3) float + (N,3) uint8 RGB -> PointCloud2 with a packed rgb field.

    Built by hand rather than via point_cloud2.create_cloud: the packed-rgb
    layout RViz expects needs a uint32 bit-pattern reinterpreted as float32,
    which the generic helper will not produce from a float array.
    """
    n = points.shape[0]
    arr = np.zeros((n, 4), dtype=np.float32)
    arr[:, :3] = points.astype(np.float32, copy=False)
    if colors is not None and len(colors):
        c = colors.astype(np.uint32)
        packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        arr[:, 3] = packed.view(np.float32) if packed.dtype == np.float32 else \
            packed.astype(np.uint32).view(np.float32)

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = n
    msg.fields = _CLOUD_FIELDS
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = arr.tobytes()
    return msg
