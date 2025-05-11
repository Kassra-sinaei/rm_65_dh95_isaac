#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import norm, solve
from array import array


from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage

from config import Config
from realmanState import RealmanState
from controller import Controller

class RealmanControlNode(Node):
    def __init__(self):
        super().__init__('RealmanControlNode')

        self.config = Config()
        self.rm_state = RealmanState(self.config)
        self.rm_controller = Controller(self.config)

        self.door_handle_pose = np.zeros(7)  # [x, y, z, rx, ry, rz, rw]
        self.create_timer(0.01, self._control_loop)
        self.counter = 0
        self.pregrasped = False


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
        self.rm_state.update_joint_state(msg)

    def basePoseCallback(self, msg):
        self.rm_state.update_base_pose(msg)

    def doorHandleCallback(self, msg):
        for t in msg.transforms:
            trans = t.transform.translation
            rot   = t.transform.rotation
            self.door_handle_pose = np.array([
                trans.x, trans.y, trans.z,
                rot.x, rot.y, rot.z, rot.w
            ])


    def initPose(self):
        self.sendRosCommand(self.config.INIT_JCOMMAND)

    def _control_loop(self):

        if self.counter < 200:
            self.initPose()
            
        elif not self.pregrasped:
            self.pregrasp(True)
            self.pregrasped = True

        self.counter += 1
        self.rm_controller.viz.display(self.rm_state.state)


    def pregrasp(self, left_arm):

        # we use left arm to pregrasp the door handle
        arm_idx = 0
        jCommand = self.rm_controller.find_arm_inverse_kinematics(self.rm_state.state, self.door_handle_pose[:3] + self.config.HANDEL_GRIP_OFFSET, np.eye(3), arm_idx)
        self.sendRosCommand(jCommand)


    def sendRosCommand(self, joint_command = None, base_command = None):
        if joint_command is not None:
            joint_state_msg = JointState()
            joint_state_msg.name = self.config.JOINT_MSG_NAME
            joint_state_msg.position = array('d', joint_command)
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_pub.publish(joint_state_msg)
        if base_command is not None:
            velocity_msgs = Twist()
            velocity_msgs.linear.x   = base_command[0]
            velocity_msgs.angular.z  = base_command[1]
            velocity_msgs.header.stamp = self.get_clock().now().to_msg()
            self.base_pub.publish(velocity_msgs)

                

def main():
    rclpy.init()
    node = RealmanControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()