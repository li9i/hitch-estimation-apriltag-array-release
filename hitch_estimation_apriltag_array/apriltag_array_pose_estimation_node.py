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
    Summary:
    Compute the following transform:
    camera_frame -->
        (apriltag_ros detection)--> tag_frames -->
            (average tag frames)--> apriltag_array_optical_frame

    Details:
    The array of apriltags has these frames attached to it:
    - tag36h11:{id}:                The unique frame attached to tag whose id is {id},
                                    as detected by the apriltag_ros algorithm.
    - apriltag_array_optical_frame: The apriltag array's optical frame, i.e.
                                    the frame of the ensemble of tags, as an
                                    average of the frames of tags detected by the
                                    apriltag_ros algorithm.
    """
    def __init__(self):
        super().__init__('apriltag_array_pose_estimation')

        # Uncomment ONLY when debugging
        #self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

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

        # ... and also to a separate topic
        self.pose_publisher = self.create_publisher(TransformStamped, self.camera_to_array_pose_topic, 10)

        self.create_subscription(
            AprilTagDetectionArray,
            self.detection_topic,
            self.on_detections,
            10
        )
        self.timer = self.create_timer(1.0/self.publish_rate, self.process_and_publish)

        self.detected_ids = []

    def average_quaternions(self, quats):
        A = np.zeros((4, 4))
        for q in quats:
            A += np.outer(q, q)
        A /= len(quats)
        eigvals, eigvecs = np.linalg.eigh(A)
        return eigvecs[:, np.argmax(eigvals)]

    def calculate_camera_to_array_optical_frame_tf(self):
        if not self.detected_ids:
            return None

        poses = []
        for tag_frame in self.detected_ids:
            try:
                t = self.tf_buffer.lookup_transform(self.camera_frame, tag_frame, rclpy.time.Time())
                t3 = t.transform
                poses.append((np.array([t3.translation.x, t3.translation.y, t3.translation.z]),
                              np.array([t3.rotation.x, t3.rotation.y, t3.rotation.z, t3.rotation.w])))
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().warn(f"Could not calculate transform from '{self.camera_frame}' to april tag frame '{tag_frame}'; skipping")
                continue

        if not poses:
            return None

        # Average pose of all detected april tags
        pos_arr, quat_arr = zip(*poses)
        avg_p = np.mean(pos_arr, axis=0)
        avg_q = self.average_quaternions(list(quat_arr))
        return avg_p, avg_q

    def on_detections(self, msg):
        self.detected_ids = [f"{d.family}:{d.id}" for d in msg.detections]

    def process_and_publish(self):

        # Calculate camera frame -> apriltag array optical frame from the
        # frames of the individually detected april tags
        result = self.calculate_camera_to_array_optical_frame_tf()

        # Guard against result=None
        if result:
            avg_p, avg_q = result
        else:
            self.get_logger().warn(f"Could not calculate transform from '{self.camera_frame}' to '{self.array_optical_frame}'; skipping")
            return

        # Store camera frame -> apriltag array optical frame
        self.set_camera_to_optical_tf(avg_p, avg_q)

        # Publish camera_frame -> array_optical_frame in /tf
        if self.b_publish_to_tf:
            self.publish_to_tf()

        # Publish camera_frame -> array_optical_frame to separate topic
        if self.b_publish_to_topic:
            self.publish_to_topic()

    def publish_to_tf(self):

        # Extract translation and quaternion
        trans = self.T_camera_to_optical[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.T_camera_to_optical)

        # Create and publish the transform
        t = self.transform_stamped_msg(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.camera_frame,
                child_frame_id=self.array_optical_frame,
                translation=trans,
                rotation=quat
        )
        self.br.sendTransform(t)

    def publish_to_topic(self):

        # Extract translation and quaternion
        trans = self.T_camera_to_optical[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.T_camera_to_optical)

        # Create and publish the transform
        t = self.transform_stamped_msg(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.camera_frame,
            child_frame_id=self.array_optical_frame,
            translation=trans,
            rotation=quat
        )
        self.pose_publisher.publish(t)

    def set_camera_to_optical_tf(self, avg_p, avg_q):
        self.T_camera_to_optical = tf_transformations.quaternion_matrix(avg_q)
        self.T_camera_to_optical[0:3, 3] = avg_p

        r, p, y = tf_transformations.euler_from_quaternion(avg_q)
        self.get_logger().debug(
            f"camera_to_optical: pos=({avg_p[0]:.3f},{avg_p[1]:.3f},{avg_p[2]:.3f}), "
            f"rpy=({math.degrees(r):.1f}°, {math.degrees(p):.1f}°, {math.degrees(y):.1f}°)"
        )

    def transform_stamped_msg(self, stamp, frame_id, child_frame_id, translation, rotation):
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
    rclpy.init()
    node = AprilTagArrayPoseEstimation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
