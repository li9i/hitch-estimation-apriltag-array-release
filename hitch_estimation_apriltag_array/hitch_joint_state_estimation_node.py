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

    def __init__(self):
        super().__init__('hitch_joint_state_estimation')

        # Uncomment ONLY when debugging
        #self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.ns = self.get_namespace()
        self.init_params()

        self.tags_sub = self.create_subscription(
            TransformStamped,
            self.camera_to_array_pose_topic,
            self.transform_cb,
            1)

        self.mobile_base_imu_sub = self.create_subscription(
            Imu,
            self.mobile_base_imu_topic,
            self.mobile_base_imu_callback,
            1)

        self.joint_state_pub = self.create_publisher(
            JointState,
            self.joint_states_topic,
            1)

        self.mobile_base_imu_data = Imu()
        self.transform = TransformStamped()
        self.hitch_joint_state = JointState()
        self.prev_hitch_joint_state = JointState()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pub_timer = self.create_timer(1.0/self.publish_rate,
                self.hitch_joint_state_pub_callback)

        # Display the hitch state once in a while
        self.disp_timer = self.create_timer(2.0,
                self.display_hitch_joint_state_callback)

    def calculate_joint_transform(self):
        # calculate roll and pitch from IMUs
        q1 = self.mobile_base_imu_data.orientation
        euler1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
        imu_roll = euler1[0]
        imu_pitch = euler1[1]
        imu_yaw = euler1[2]

        # get pose of apriltag plane with regards to its optical frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.mobile_base_link,
                self.apriltag_array_optical_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
            p0 = [transform.transform.translation.x,
                  transform.transform.translation.y,
                  transform.transform.translation.z]
            q0 = [transform.transform.rotation.x,
                  transform.transform.rotation.y,
                  transform.transform.rotation.z,
                  transform.transform.rotation.w]
        except Exception as e:
            self.get_logger().error(f"Failed to lookup apriltags transform: {str(e)}")
            self.hitch_joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_pub.publish(self.hitch_joint_state)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        r, p, y = euler_from_quaternion(q0)
        self.get_logger().debug("g_base2apriltag: "
            f"rpy=({math.degrees(r):.1f}°, {math.degrees(p):.1f}°, {math.degrees(y):.1f}°)"
        )

        at_roll, at_pitch, at_yaw = euler_from_quaternion(q0)

        # Needed adjustment (why?)
        at_yaw = at_yaw - math.pi/2

        return imu_roll, imu_pitch, imu_yaw, at_roll, at_pitch, at_yaw

    def display_hitch_joint_state_callback(self):
        if self.hitch_joint_state.position:
            yaw_deg = self.hitch_joint_state.position[0]*180/math.pi
            self.get_logger().info(f"Trailer hitch angle (deg): {yaw_deg:+.2f}")

    def from_translation_rotation(self, p, q):
        """
        Constructs a 4x4 homogeneous transformation matrix from a translation and quaternion.

        Parameters:
            p (list of floats): [x, y, z] translation
            q (list of floats): [x, y, z, w] quaternion

        Returns:
            numpy.ndarray: 4x4 transformation matrix
        """
        matrix = quaternion_matrix(q)
        matrix[0:3, 3] = p
        return matrix

    def hitch_joint_state_pub_callback(self):
        if not hasattr(self.transform, 'header'):
            return

        # Publish previously calculated yaw if there is none
        fall_back = False

        # calculate joint transform
        imu_roll, imu_pitch, imu_yaw, at_roll, at_pitch, at_yaw = self.calculate_joint_transform()

        if not isinstance(at_yaw, float):
            self.get_logger().error("Value of 'at_yaw' is not a float; skipping")
            fall_back = True
            return

        # Setup msg/tf fields
        self.hitch_joint_state.header.frame_id = self.mobile_base_hitch_joint
        self.hitch_joint_state.header.stamp = self.get_clock().now().to_msg()
        self.hitch_joint_state.name = [self.trailer_hitch_joint]
        self.hitch_joint_state.position = [at_yaw]
        self.hitch_joint_state.velocity = [0.0]
        self.hitch_joint_state.effort = [0.0]

        self.prev_joint_state = self.hitch_joint_state

        # Publish to separate JointState topic and/or to /tf
        if self.publish_joint_states:
            self.publish_joint_transform_to_topic(fall_back)
        if self.publish_tf:
            self.publish_joint_transform_to_tf(fall_back)

    def init_params(self):
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
            ])

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
        self.mobile_base_imu_data = data

    def publish_joint_transform_to_tf(self, fall_back):
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

        if fall_back and self.prev_hitch_joint_state:
            self.joint_state_pub.publish(self.prev_hitch_joint_state.position)
        else:
            self.joint_state_pub.publish(self.hitch_joint_state)

    def transform_cb(self, data):
        self.transform = data
        self.pose0 = self.transformStamped_to_poseStamped(self.transform)

    def transformStamped_to_poseStamped(self, msg_in):
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
    rclpy.init(args=args)
    node = MobileBaseTrailerHitchJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
