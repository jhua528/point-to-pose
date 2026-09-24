"""ROS2 node (runs on the camera machine) that offloads pose tracking to a remote GPU server.

Only needs rclpy + cv_bridge + pyzmq -- point2pose itself is not imported here.
"""

import argparse
import json
import time
import zlib

import cv2
import message_filters
import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, Pose
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image


class PoseClient(Node):
    def __init__(self, args):
        super().__init__("point2pose_client")
        self.bridge = CvBridge()
        self.args = args
        self.K = None
        self.initialized = False
        self.last_sent = 0.0

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(args.server)
        self.get_logger().info(f"connected to {args.server}")

        self.create_subscription(CameraInfo, args.info_topic, self.on_info, 10)
        rgb_sub = message_filters.Subscriber(self, Image, args.rgb_topic)
        depth_sub = message_filters.Subscriber(self, Image, args.depth_topic)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.05)
        sync.registerCallback(self.on_frame)

        self.pub = self.create_publisher(PoseArray, "/point2pose/poses", 10)

    def on_info(self, msg):
        if self.K is None:
            self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"got intrinsics fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}")

    def request(self, meta, rgb, depth):
        ok, rgb_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            raise RuntimeError("rgb encode failed")
        if depth.dtype == np.float32:
            meta["depth_enc"] = "raw_f32"
            meta["depth_shape"] = list(depth.shape)
            depth_buf = zlib.compress(np.ascontiguousarray(depth).tobytes(), 1)
        else:
            meta["depth_enc"] = "png"
            depth_buf = cv2.imencode(".png", depth)[1].tobytes()
        self.sock.send_multipart([json.dumps(meta).encode(), rgb_buf.tobytes(), depth_buf])
        return self.sock.recv_json()

    def on_frame(self, rgb_msg, depth_msg):
        if self.K is None:
            return
        now = time.time()
        if now - self.last_sent < 1.0 / self.args.rate:
            return
        self.last_sent = now

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_factor = self.args.depth_factor
        if depth.dtype == np.float32:
            # float32 metres: send raw bytes so no precision is lost to quantisation.
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_factor = 1.0

        meta = {
            "cmd": "track" if self.initialized else "init",
            "K": self.K.flatten().tolist(),
            "depth_factor": depth_factor,
            "timestamp": rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9,
        }
        if not self.initialized:
            meta["points"] = json.loads(self.args.points)
            meta["labels"] = json.loads(self.args.labels)

        try:
            reply = self.request(meta, rgb, depth)
        except zmq.Again:
            self.get_logger().error("server timeout; resetting socket")
            self.sock.close()
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, self.args.timeout_ms)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(self.args.server)
            return

        if not reply.get("ok"):
            self.get_logger().error(f"server error: {reply.get('error')}")
            return
        if not self.initialized:
            self.initialized = True
            self.get_logger().info("pipeline initialized")

        poses = np.asarray(reply["poses"], dtype=np.float64)
        out = PoseArray()
        out.header = rgb_msg.header
        for T in poses:
            p = Pose()
            p.position.x, p.position.y, p.position.z = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            out.poses.append(p)
        self.pub.publish(out)
        self.get_logger().info(
            f"frame {reply['frame_id']} latency={reply['latency_ms']}ms lost={reply['lost']}",
            throttle_duration_sec=2.0,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="tcp://172.26.177.251:5555")
    ap.add_argument("--rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/depth/image_rect_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--points", default="[[[320,240]]]", help="JSON: one [[x,y],...] list per object")
    ap.add_argument("--labels", default="[[1]]", help="JSON: one [1/0,...] list per object")
    ap.add_argument("--depth-factor", type=float, default=1000.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--timeout-ms", type=int, default=15000)
    args = ap.parse_args()

    rclpy.init()
    node = PoseClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
