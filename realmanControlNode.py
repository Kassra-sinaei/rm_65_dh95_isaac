#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
import numpy as np
from array import array


from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage

from config import Config
from realmanState import RealmanState
from controller import Controller
from enum import Enum, auto

class RobotState(Enum):
    INIT_POSE    = auto()
    PREGRASP     = auto()
    GRASP        = auto()
    TURN         = auto()
    OPENING      = auto()
    # add more states here as you go…
    # LIFT   = auto()
    # DONE   = auto()


class RealmanControlNode(Node):
    def __init__(self):
        super().__init__('RealmanControlNode')

        self.config = Config()
        self.rm_state = RealmanState(self.config)
        self.rm_controller = Controller(self.config)

        self.door_handle_pose = np.zeros(7)  # [x, y, z, rx, ry, rz, rw]
        self.create_timer(0.01, self._control_loop)

        self.state = RobotState.INIT_POSE
        self.state_start_time = self.get_clock().now()
        self.pregrasp_jcmd = None
        self.grasp_jcmd = None
        self.turn_jcmd = None
        self.open_jcmd = None

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
        # dispatch based on current state
        if self.state == RobotState.INIT_POSE:
            self._handle_init_pose()
        elif self.state == RobotState.PREGRASP:
            self._handle_pregrasp()
        elif self.state == RobotState.GRASP:
            self._handle_grasp()
        elif self.state == RobotState.TURN:
            self._handle_turn()
        elif self.state == RobotState.OPENING:
            self._handle_opening()
        
        # visualize 
        self.rm_controller.viz.display(self.rm_state.state)



    def _handle_init_pose(self):
        # keep sending init-pose until 200 ticks have elapsed
        self.sendRosCommand(self.config.INIT_JCOMMAND)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 300 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            self._transition_to(RobotState.PREGRASP)

    def _handle_pregrasp(self):
        # on first entry, compute and cache IK
        if self.pregrasp_jcmd is None:
            self.pregrasp_jcmd = self.rm_controller.find_arm_inverse_kinematics(
                self.rm_state.state,
                self.door_handle_pose[:3] + self.config.HANDEL_PREGRIP_OFFSET,
                np.eye(3),
                arm_idx=0
            )

            self.pregrasp_count = 0
            self.get_logger().info("Computed pregrasp IK once")

        # every loop just send the _cached_ command
        self.sendRosCommand(self.pregrasp_jcmd)
        if self.pregrasp_count > 100:
            # 300 * 0.01s == 3 seconds
            self._transition_to(RobotState.GRASP)
        self.pregrasp_count += 1

    def _handle_grasp(self):
        # on first entry, compute and cache IK
        if self.grasp_jcmd is None:
            self.grasp_jcmd = self.rm_controller.find_arm_inverse_kinematics(
                self.rm_state.state,
                self.door_handle_pose[:3] + self.config.HANDEL_GRIP_OFFSET,
                np.eye(3),
                arm_idx=0
            )
            self.grasp_count = 0
            self.get_logger().info("Computed grasp IK once")

        # every loop just send the _cached_ command
        if self.grasp_count > 100:
            self.grasp_jcmd[13] = 1.0
        if self.grasp_count > 150:
            self._transition_to(RobotState.TURN)
        self.sendRosCommand(self.grasp_jcmd)
        self.grasp_count += 1
    
    def _handle_turn(self):
        if self.turn_jcmd is None:
            self.turn_jcmd = self.rm_controller.find_arm_inverse_kinematics(
                self.rm_state.state,
                self.door_handle_pose[:3] + self.config.HANDEL_TURN_OFFSET,
                self.config.HANDEL_TURN_ROTATION,
                arm_idx=0
            )
            self.grasp_count = 0
            self.turn_jcmd[13] = 1.0
            self.get_logger().info("Computed turn IK once")
        self.sendRosCommand(self.turn_jcmd)
        if (self.get_clock().now() - self.state_start_time).nanoseconds > 300 * 10_000_000:
            # 3 seconds
            self._transition_to(RobotState.OPENING)
    
    def _handle_opening(self):
        if self.open_jcmd is None:
            self.open_jcmd = self.rm_controller.find_arm_inverse_kinematics(
                self.rm_state.state,
                self.door_handle_pose[:3] + self.config.RIGHRT_ARM_PUSH_POSITION,
                np.eye(3),
                arm_idx=1
            )
        base_command = np.array([1.0, 0.0])
        self.sendRosCommand(base_command=base_command)
        self.get_logger().info("Opening door")
        
        if (self.get_clock().now() - self.state_start_time).nanoseconds > 10 * 10_000_000:
            self.open_jcmd[13] = 0.0

        self.sendRosCommand(self.open_jcmd)


    def _transition_to(self, new_state: RobotState):
        self.get_logger().info(f"→ Transition: {self.state.name} → {new_state.name}")
        self.state = new_state
        self.state_start_time = self.get_clock().now()

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