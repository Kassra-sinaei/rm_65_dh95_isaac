#!/usr/bin/env python3.10

import numpy as np
from array import array
import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage

import pinocchio
from pinocchio.visualize import MeshcatVisualizer


class RealmanControlNode(Node):
    def __init__(self):
        super().__init__('RealmanControlNode')

        # URDF & Pinocchio setup
        self.URDFPATH = "./urdf/overseas_65_corrected.urdf"
        self.MESH_DIR = "./urdf"
        self.INIT_ARM = [0, -1.75, -0.6, 1.5, 0, 0, 0, 1.75, 0.6, -1.5, 0, 0]   # left arm and right arm
        self.JOINT_MSG_NAME = [f"l_joint{i}" for i in range(1, 7)] + [f"r_joint{i}" for i in range(1, 7)] + ["l_finger_joint", "r_finger_joint", "platform_joint"]
        self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
            self.URDFPATH, self.MESH_DIR, pinocchio.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
         
        # initial config & visualizer
        self.q = pinocchio.neutral(self.model)
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1.0])

        print(f"q: {self.q.T}")
        pinocchio.forwardKinematics(self.model, self.data, self.q)


        # state buffers x, y, z, rx, ry, rz, rw, platform, head 2, l 6, r 6.
        self.pin_q = np.zeros(22) 
        self.door_handle_pose = np.zeros(7)  # [x, y, z, rx, ry, rz, rw]

        # subscriptions
        self.create_subscription(
            JointState, '/isaac_sim/joint_states',
            self.jointStateCallback, 10
        )
        self.create_subscription(
            TFMessage, '/tf',
            self.basePoseCallback, 10
        )

        self.create_subscription(
            TFMessage, '/isaac_sim/door_handle',
            self.doorHandleCallback, 10
        )

        # publishers
        self.joint_state_pub = self.create_publisher(
            JointState, '/isaac_sim/joint_command', 10
        )
        self.base_pub = self.create_publisher(
            Twist, '/isaac_sim/cmd_vel', 10
        )

    def jointStateCallback(self, msg):
        # update platform joint
        self.pin_q[7] = msg.position[0]
        pos_map = dict(zip(msg.name, msg.position))
        for side, base in (('l', 10), ('r', 16)):
            self.pin_q[base:base + 6] = [pos_map[f"{side}_joint{i}"] for i in range(1, 7)]

    def basePoseCallback(self, msg):
        for t in msg.transforms:
            trans = t.transform.translation
            rot   = t.transform.rotation
            self.pin_q[0:7] = np.array([
                trans.x, trans.y, trans.z,
                rot.x, rot.y, rot.z, rot.w
            ])
    def doorHandleCallback(self, msg):
        for t in msg.transforms:
            trans = t.transform.translation
            rot   = t.transform.rotation
            self.door_handle_pose = np.array([
                trans.x, trans.y, trans.z,
                rot.x, rot.y, rot.z, rot.w
            ])


    def initPose(self, joint_state_msg, velocity_msgs):
        joint_state_msg.position[0:12] = array('d', self.INIT_ARM)
        joint_state_msg.position[12] = 0
        joint_state_msg.position[13] = 0
        joint_state_msg.position[14] = 0.4

        velocity_msgs.linear.x   = 0.0
        velocity_msgs.angular.z  = 0.0


    def main(self):
        # Create a JointState message
        joint_state_msg = JointState()
        velocity_msgs   = Twist()
        joint_state_msg.name  = self.JOINT_MSG_NAME

        joint_state_msg.position = [0.0]*15
        self.initPose(joint_state_msg, velocity_msgs)


        try:
            while rclpy.ok():
                joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                self.joint_state_pub.publish(joint_state_msg)
                self.base_pub.publish(velocity_msgs)
                time.sleep(0.05)
                rclpy.spin_once(self)
                self.viz.display(self.pin_q)
                print(self.door_handle_pose)
        except KeyboardInterrupt:
            print("\nShutting down publisher...")
        finally:
            self.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    RealmanControlNode().main()
