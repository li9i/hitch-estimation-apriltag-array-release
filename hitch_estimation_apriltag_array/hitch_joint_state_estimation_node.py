#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from tf2_ros import TransformListener
from tf2_ros import Buffer
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_matrix
from tf_transformations import quaternion_from_euler
from tf_transformations import quaternion_matrix


class MobileBaseTrailerHitchJointStatePublisher(Node):
    """
    ROS2 node for estimating and publishing the trailer hitch joint state.

    Summary:
        Computes the yaw angle of a trailer hitch joint using:
        - IMU data from the mobile base
        - The pose of an AprilTag array attached to the trailer

    The node publishes:
        - JointState messages describing the hitch joint angle
        - (Optionally) corresponding TF transforms for visualization/debugging

    Intended use:
        Estimating the relative yaw angle between a mobile base and a towed
        trailer using visual fiducials (AprilTags).
    """

    def __init__(self):
        """
        Initialize the hitch joint state estimation node.

        Sets up parameters, subscriptions, publishers, TF interfaces,
        and periodic timers for publishing joint states and diagnostics.
        """
        super().__init__('hitch_joint_state_estimation')

        # Uncomment ONLY when debugging
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.ns = self.get_namespace()
        self.init_params()

        # Subscribe to camera -> apriltag array transform
        self.tags_sub = self.create_subscription(
            TransformStamped,
            self.camera_to_array_pose_topic,
            self.transform_cb,
            1
        )

        # Subscribe to mobile base IMU
        self.mobile_base_imu_sub = self.create_subscription(
            Imu,
            self.mobile_base_imu_topic,
            self.mobile_base_imu_callback,
            1
        )

        # Publish hitch joint state
        self.joint_state_pub = self.create_publisher(
            JointState,
            self.joint_states_topic,
            1
        )

        self.mobile_base_imu_data = Imu()
        self.transform = TransformStamped()
        self.hitch_joint_state = JointState()
        self.prev_hitch_joint_state = JointState()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Periodic joint state publication
        self.pub_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.hitch_joint_state_pub_callback
        )

        # Periodic console output of hitch angle
        self.disp_timer = self.create_timer(
            2.0,
            self.display_hitch_joint_state_callback
        )

    def calculate_joint_transform(self):
        """
        Compute the hitch joint angles using IMU and AprilTag data.

        Workflow:
            1. Extract roll, pitch, yaw from the mobile base IMU.
            2. Lookup the transform from the mobile base frame to the
               AprilTag array optical frame.
            3. Extract roll, pitch, yaw from the AprilTag orientation.
            4. Apply a fixed yaw adjustment to align frames.

        Returns:
            tuple:
                (imu_roll, imu_pitch, imu_yaw,
                 at_roll, at_pitch, at_yaw)

            If TF lookup fails, returns zeros and publishes the last known
            joint state.
        """
        # Extract roll/pitch/yaw from IMU
        q1 = self.mobile_base_imu_data.orientation
        euler1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
        imu_roll, imu_pitch, imu_yaw = euler1

        # Lookup pose of AprilTag array relative to the mobile base
        try:
            transform = self.tf_buffer.lookup_transform(
                self.mobile_base_link,
                self.apriltag_array_optical_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            p0 = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            q0 = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
        except Exception as e:
            self.get_logger().error(
                f"Failed to lookup apriltags transform: {str(e)}"
            )
            self.hitch_joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_pub.publish(self.hitch_joint_state)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        r, p, y = euler_from_quaternion(q0)
        self.get_logger().debug(
            "g_base2apriltag: "
            f"rpy=({math.degrees(r):.1f}°, "
            f"{math.degrees(p):.1f}°, "
            f"{math.degrees(y):.1f}°)"
        )

        at_roll, at_pitch, at_yaw = euler_from_quaternion(q0)

        # Empirical yaw offset to align coordinate conventions
        at_yaw = at_yaw - math.pi / 2

        return imu_roll, imu_pitch, imu_yaw, at_roll, at_pitch, at_yaw

    def display_hitch_joint_state_callback(self):
        """
        Periodically log the trailer hitch yaw angle to the console.
        """
        if self.hitch_joint_state.position:
            yaw_deg = self.hitch_joint_state.position[0] * 180 / math.pi
            self.get_logger().info(
                f"Trailer hitch angle (deg): {yaw_deg:+.2f}"
            )

    def from_translation_rotation(self, p, q):
        """
        Construct a homogeneous transform from translation and rotation.

        Args:
            p (list[float]):
                Translation [x, y, z].
            q (list[float]):
                Quaternion [x, y, z, w].

        Returns:
            np.ndarray:
                4x4 homogeneous transformation matrix.
        """
        matrix = quaternion_matrix(q)
        matrix[0:3, 3] = p
        return matrix

    def hitch_joint_state_pub_callback(self):
        """
        Periodic callback to compute and publish the hitch joint state.

        Uses the latest IMU and AprilTag transform data to estimate the
        hitch yaw angle, then publishes it as a JointState message and
        optionally as TF transforms.
        """
        if not hasattr(self.transform, 'header'):
            return

        fall_back = False

        imu_roll, imu_pitch, imu_yaw, at_roll, at_pitch, at_yaw = \
            self.calculate_joint_transform()

        if not isinstance(at_yaw, float):
            self.get_logger().error(
                "Value of 'at_yaw' is not a float; skipping"
            )
            fall_back = True
            return

        self.hitch_joint_state.header.frame_id = self.mobile_base_hitch_joint
        self.hitch_joint_state.header.stamp = self.get_clock().now().to_msg()
        self.hitch_joint_state.name = [self.trailer_hitch_joint]
        self.hitch_joint_state.position = [at_yaw]
        self.hitch_joint_state.velocity = [0.0]
        self.hitch_joint_state.effort = [0.0]

        self.prev_joint_state = self.hitch_joint_state

        if self.publish_joint_states:
            self.publish_joint_transform_to_topic(fall_back)

        if self.publish_tf:
            self.publish_joint_transform_to_tf(fall_back)

    def init_params(self):
        """
        Declare and retrieve ROS parameters used by this node.
        """
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mobile_base_link', 'robot_base_footprint'),
                ('camera_frame', 'robot_rear_rgbd_camera_link'),
                ('apriltag_array_optical_frame', 'robot_cart_apriltag_array_optical_frame'),
                ('camera_to_array_pose_topic', '/robot/rear_rgb_camera/apriltag_plane/transform'),
                ('mobile_base_imu_topic', '/mobile_base/imu/data'),
                ('joint_states_topic', '/mobile_base/trailer/joint_states'),
                ('publish_rate', 1.0),
                ('publish_joint_states', True),
                ('publish_tf', False),
                ('mobile_base_hitch_joint', 'robot_hitch_joint'),
                ('trailer_hitch_joint', 'robot_cart_hitch_joint')
            ]
        )

        self.publish_rate = self.get_parameter('publish_rate').value
        self.mobile_base_link = self.get_parameter('mobile_base_link').value
        self.camera_to_array_pose_topic = self.get_parameter('camera_to_array_pose_topic').value
        self.mobile_base_imu_topic = self.get_parameter('mobile_base_imu_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.apriltag_array_optical_frame = self.get_parameter('apriltag_array_optical_frame').value
        self.mobile_base_hitch_joint = self.get_parameter('mobile_base_hitch_joint').value
        self.trailer_hitch_joint = self.get_parameter('trailer_hitch_joint').value
        self.publish_joint_states = self.get_parameter('publish_joint_states').value
        self.publish_tf = self.get_parameter('publish_tf').value

    def mobile_base_imu_callback(self, data):
        """
        Store the most recent IMU message from the mobile base.

        Args:
            data (sensor_msgs.msg.Imu):
                Incoming IMU message.
        """
        self.mobile_base_imu_data = data

    def publish_joint_transform_to_tf(self, fall_back):
        """
        Publish the hitch joint as a TF transform.

        Args:
            fall_back (bool):
                If True, reuse the previously published joint state.
        """
        joint_names = self.hitch_joint_state.name

        if fall_back and self.prev_hitch_joint_state:
            joint_positions = self.prev_hitch_joint_state.position
        else:
            joint_positions = self.hitch_joint_state.position

        for name, position in zip(joint_names, joint_positions):
            parent_frame = self.mobile_base_hitch_joint
            child_frame = name

            # Adjust axis of rotation depending on the joint
            if 'roll_joint' in name:
                quat = quaternion_from_euler(position, 0, 0)
            elif 'pitch_joint' in name:
                quat = quaternion_from_euler(0, position, 0)
            elif 'yaw_joint' in name:
                quat = quaternion_from_euler(0, 0, position)
            else:
                quat = quaternion_from_euler(0, 0, 0)

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent_frame
            transform.child_frame_id = child_frame
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(transform)

    def publish_joint_transform_to_topic(self, fall_back):
        """
        Publish the hitch joint state to a JointState topic.

        Args:
            fall_back (bool):
                If True, publish the previously known joint state.
        """
        if fall_back and self.prev_hitch_joint_state:
            self.joint_state_pub.publish(self.prev_hitch_joint_state)
        else:
            self.joint_state_pub.publish(self.hitch_joint_state)

    def transform_cb(self, data):
        """
        Callback for incoming camera-to-AprilTag transforms.

        Args:
            data (TransformStamped):
                Transform from camera frame to AprilTag array frame.
        """
        self.transform = data
        self.pose0 = self.transformStamped_to_poseStamped(self.transform)

    def transformStamped_to_poseStamped(self, msg_in):
        """
        Convert a TransformStamped message into a PoseStamped message.

        Args:
            msg_in (TransformStamped):
                Input transform message.

        Returns:
            PoseStamped:
                Equivalent pose representation.
        """
        msg_out = PoseStamped()
        msg_out.header = msg_in.header
        msg_out.pose.position = Point(
            x=msg_in.transform.translation.x,
            y=msg_in.transform.translation.y,
            z=msg_in.transform.translation.z
        )
        msg_out.pose.orientation = msg_in.transform.rotation
        return msg_out


def main(args=None):
    """
    Entry point for the MobileBaseTrailerHitchJointStatePublisher node.
    """
    rclpy.init(args=args)
    node = MobileBaseTrailerHitchJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
