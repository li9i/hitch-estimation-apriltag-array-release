import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import Buffer
from tf2_ros import TransformListener
from tf2_ros import LookupException
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
import tf_transformations


class AprilTagArrayPoseEstimation(Node):
    """
    ROS2 node for estimating the pose of an AprilTag array relative to a camera.

    Summary:
        Computes the transform:
            camera_frame
                -> individual apriltag frames (from apriltag_ros)
                -> averaged apriltag array optical frame

    The node listens to AprilTag detections, looks up each detected tag's TF
    relative to the camera frame, averages their poses, and publishes the
    resulting transform both to TF and optionally to a topic.

    Frames involved:
        - tag36h11:{id}:
            Individual AprilTag frames published by apriltag_ros.
        - apriltag_array_optical_frame:
            Virtual frame representing the averaged pose of all detected tags.
    """

    def __init__(self):
        """
        Initialize the AprilTagArrayPoseEstimation node.

        Declares and reads parameters, sets up TF listeners/broadcasters,
        subscriptions, publishers, and a periodic timer for processing and
        publishing the averaged transform.
        """
        super().__init__('apriltag_array_pose_estimation')

        # Uncomment ONLY when debugging
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.declare_parameter('detection_topic', '/robot/rear_rgb_camera/apriltag_ros/detections')
        self.declare_parameter('camera_frame', 'robot_rear_color_optical_frame')
        self.declare_parameter('apriltag_array_optical_frame', 'robot_cart_apriltag_array_optical_link')
        self.declare_parameter('camera_to_array_pose_topic', '/robot/rear_rgb_camera/apriltag_plane/transform')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('publish_to_topic', True)
        self.declare_parameter('publish_to_tf', True)

        self.publish_rate = self.get_parameter('publish_rate').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.array_optical_frame = self.get_parameter('apriltag_array_optical_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.camera_to_array_pose_topic = self.get_parameter('camera_to_array_pose_topic').value
        self.b_publish_to_topic = self.get_parameter('publish_to_topic').value
        self.b_publish_to_tf = self.get_parameter('publish_to_tf').value

        self.br = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for camera -> array transform as a topic
        self.pose_publisher = self.create_publisher(
            TransformStamped,
            self.camera_to_array_pose_topic,
            10
        )

        self.create_subscription(
            AprilTagDetectionArray,
            self.detection_topic,
            self.on_detections,
            10
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_and_publish
        )

        self.detected_ids = []

    def average_quaternions(self, quats):
        """
        Compute the average quaternion from a list of quaternions.

        Uses the eigenvector method to compute a statistically meaningful
        average rotation.

        Args:
            quats (list[np.ndarray]):
                List of quaternions in (x, y, z, w) format.

        Returns:
            np.ndarray:
                Averaged quaternion (x, y, z, w).
        """
        A = np.zeros((4, 4))
        for q in quats:
            A += np.outer(q, q)
        A /= len(quats)

        eigvals, eigvecs = np.linalg.eigh(A)
        return eigvecs[:, np.argmax(eigvals)]

    def calculate_camera_to_array_optical_frame_tf(self):
        """
        Calculate the averaged transform from the camera frame to the array frame.

        Looks up TF transforms from the camera frame to each detected AprilTag
        frame, then averages their positions and orientations.

        Returns:
            tuple[np.ndarray, np.ndarray] or None:
                (avg_position, avg_quaternion) if successful,
                None if no valid transforms are available.
        """
        if not self.detected_ids:
            return None

        poses = []
        for tag_frame in self.detected_ids:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    rclpy.time.Time()
                )
                t3 = t.transform
                poses.append((
                    np.array([t3.translation.x, t3.translation.y, t3.translation.z]),
                    np.array([t3.rotation.x, t3.rotation.y, t3.rotation.z, t3.rotation.w])
                ))
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().warn(
                    f"Could not calculate transform from '{self.camera_frame}' "
                    f"to april tag frame '{tag_frame}'; skipping"
                )
                continue

        if not poses:
            return None

        pos_arr, quat_arr = zip(*poses)
        avg_p = np.mean(pos_arr, axis=0)
        avg_q = self.average_quaternions(list(quat_arr))

        return avg_p, avg_q

    def on_detections(self, msg):
        """
        Callback for AprilTag detection messages.

        Extracts detected tag frame names from the message and stores them
        for later TF lookup.

        Args:
            msg (AprilTagDetectionArray):
                Incoming detection message from apriltag_ros.
        """
        self.detected_ids = [f"{d.family}:{d.id}" for d in msg.detections]

    def process_and_publish(self):
        """
        Periodic processing callback.

        Computes the camera-to-array transform, stores it internally,
        and publishes it to TF and/or a topic depending on configuration.
        """
        result = self.calculate_camera_to_array_optical_frame_tf()

        if result:
            avg_p, avg_q = result
        else:
            self.get_logger().warn(
                f"Could not calculate transform from '{self.camera_frame}' "
                f"to '{self.array_optical_frame}'; skipping"
            )
            return

        self.set_camera_to_optical_tf(avg_p, avg_q)

        if self.b_publish_to_tf:
            self.publish_to_tf()

        if self.b_publish_to_topic:
            self.publish_to_topic()

    def publish_to_tf(self):
        """
        Publish the camera-to-array transform to the TF tree.

        Uses the internally stored transformation matrix to populate
        a TransformStamped message.
        """
        trans = self.T_camera_to_optical[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.T_camera_to_optical)

        t = self.transform_stamped_msg(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.camera_frame,
            child_frame_id=self.array_optical_frame,
            translation=trans,
            rotation=quat
        )
        self.br.sendTransform(t)

    def publish_to_topic(self):
        """
        Publish the camera-to-array transform to a ROS topic.

        The message content is identical to what is published to TF,
        but sent as a TransformStamped message on a dedicated topic.
        """
        trans = self.T_camera_to_optical[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.T_camera_to_optical)

        t = self.transform_stamped_msg(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.camera_frame,
            child_frame_id=self.array_optical_frame,
            translation=trans,
            rotation=quat
        )
        self.pose_publisher.publish(t)

    def set_camera_to_optical_tf(self, avg_p, avg_q):
        """
        Store the camera-to-array transform internally as a 4x4 matrix.

        Args:
            avg_p (np.ndarray):
                Averaged translation vector (x, y, z).
            avg_q (np.ndarray):
                Averaged orientation quaternion (x, y, z, w).
        """
        self.T_camera_to_optical = tf_transformations.quaternion_matrix(avg_q)
        self.T_camera_to_optical[0:3, 3] = avg_p

        r, p, y = tf_transformations.euler_from_quaternion(avg_q)
        self.get_logger().debug(
            f"camera_to_optical: pos=({avg_p[0]:.3f},{avg_p[1]:.3f},{avg_p[2]:.3f}), "
            f"rpy=({math.degrees(r):.1f}°, {math.degrees(p):.1f}°, {math.degrees(y):.1f}°)"
        )

    def transform_stamped_msg(self, stamp, frame_id, child_frame_id, translation, rotation):
        """
        Helper function to construct a TransformStamped message.

        Args:
            stamp (builtin_interfaces.msg.Time):
                Timestamp for the transform.
            frame_id (str):
                Parent frame ID.
            child_frame_id (str):
                Child frame ID.
            translation (array-like):
                Translation vector (x, y, z).
            rotation (array-like):
                Quaternion (x, y, z, w).

        Returns:
            TransformStamped:
                Populated transform message.
        """
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        return t


def main():
    """
    Entry point for the AprilTagArrayPoseEstimation node.
    """
    rclpy.init()
    node = AprilTagArrayPoseEstimation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
