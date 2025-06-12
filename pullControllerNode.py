#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
import math
from array import array


from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Point
from tf2_msgs.msg import TFMessage

from config import Config
from realmanState import RealmanState
from controller import Controller
from enum import Enum, auto

class RobotState(Enum):
    INIT_POSE    = auto()
    APPROACH_DOOR = auto()
    DETECT_HANDLE = auto()
    PREGRASP     = auto()
    GRASP        = auto()
    PULL         = auto()
    CONTACTSWITCH = auto()
    TRAVERSE     = auto()
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
        self.grip_Handle_pose = None

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

        self.create_subscription(
            TFMessage, '/isaac_sim/door_handle',
            self.doorHandleCallback, 10
        )
        # self.create_subscription(
        #     Point, '/grip_point',
        #     self.gripHandleCallback, 10
        # )

        # publishers
        self.joint_state_pub = self.create_publisher(
            JointState, '/isaac_sim/joint_command', 10
        )
        self.base_pub = self.create_publisher(
            Twist, '/isaac_sim/cmd_vel', 10
        )
        self.active_arm_id = 1


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
    def gripHandleCallback(self, msg):
        pose_in_camera = np.array([msg.x, msg.y, msg.z])
        # convert to world frame
        if self.grip_Handle_pose is None:
            self.grip_Handle_pose = self.rm_controller.convert_pose_from_camera_to_world(
                self.rm_state.state,
                pose_in_camera
            )
    def doorHandleCallback(self, msg):
        if self.grip_Handle_pose is None:
            for tf in msg.transforms:
                t, r = tf.transform.translation, tf.transform.rotation
                self.grip_Handle_pose = np.array([
                    t.x, t.y, t.z,
                    r.x, r.y, r.z, r.w
                ])

    def _control_loop(self):
        # dispatch based on current state
        if self.state == RobotState.INIT_POSE:
            self._handle_init_pose()         
        elif self.state == RobotState.DETECT_HANDLE:
            self._handle_detect_handle()
        elif self.state == RobotState.APPROACH_DOOR:
            # self._approach_door()   
            self._transition_to(RobotState.PREGRASP)
        elif self.state == RobotState.PREGRASP:
            self._handle_pregrasp()
        elif self.state == RobotState.GRASP:
            self._handle_grasp()
        # elif self.state == RobotState.PULL:
        #     self._door_pull()
        # elif self.state == RobotState.CONTACTSWITCH:
        #     self._switch_contact()
        # elif self.state == RobotState.TRAVERSE:
        #     self._traverse()
        
        # visualize 
        self.rm_controller.viz.display(self.rm_state.state)



    def _handle_init_pose(self):
        # keep sending init-pose until 200 ticks have elapsed
        rot = pin.Quaternion(np.array([[np.cos(np.pi), 0.0, np.sin(np.pi)], 
                                    [0,1,0],
                                    [-np.sin(np.pi), 0.0, np.cos(np.pi)]]))
        base_rot = pin.Quaternion(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                                            [np.sin(np.pi/2), np.cos(np.pi/2), 0],
                                            [0, 0, 1]]))
        self.init_pose_command = self.rm_controller.pink_ik(base_rot, np.array([0,0,0]),rot, np.array([0.2, -0.4, 0.5]), rot, np.array([0.2, 0.4,0.5]))
        self.sendRosCommand(self.init_pose_command)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 600 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            self._transition_to(RobotState.DETECT_HANDLE)
    
    def _approach_door(self):
        grip_pose = np.array([self.grip_Handle_pose[0], self.grip_Handle_pose[1], 
                              0.0])
        rotation_matrix = pin.Quaternion(self.rm_state.state[3], self.rm_state.state[4], self.rm_state.state[5], self.rm_state.state[6]).toRotationMatrix()
        current_state = np.array([self.rm_state.state[0], self.rm_state.state[1], 
                                  np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])])
        goal = grip_pose - self.config.PULL_BASE_OFFSET
        error = goal - current_state
        # error[2] = (error[2]) % (2 * np.pi)
        print(f"Error: {error.T}, d: {np.linalg.norm(error[0:2])}, theta: {abs(error[2])}")
        if np.linalg.norm(error[0:2]) > 0.1 or abs(error[2]) > 0.05:
            base_command = self.rm_controller.compute_base_twist_pd(error, T = 5)
            self.sendRosCommand(base_command=base_command)
        else:
            self._transition_to(RobotState.PREGRASP)
            self.sendRosCommand(base_command=[0.0, 0.0])
            self.grip_Handle_pose = None

    def _handle_detect_handle(self):
        if self.grip_Handle_pose is not None:
            self.get_logger().info("Handle detected")
            self._transition_to(RobotState.APPROACH_DOOR)

    def _handle_pregrasp(self):
        # on first entry, compute and cache IK
        if self.grip_Handle_pose is None:
            return
        if self.pregrasp_jcmd is None:
            rot_active = pin.Quaternion(np.array([[1, 0, 0], 
                                                  [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                                                  [0, np.sin(np.pi/2), np.cos(np.pi/2)]]))
            rot_idle = pin.Quaternion(np.eye(3))
            self.pregrasp_jcmd = self.rm_controller.pink_ik(
                pin.Quaternion(self.rm_state.state[3:7]), self.rm_state.state[0:3],
                rot_active, self.grip_Handle_pose[:3] + self.config.HANDEL_PREGRIP_OFFSET,
                rot_idle, self.rm_state.state[0:3] + np.array([0.6, -0.4, self.grip_Handle_pose[2]-0.3])
            )

            self.pregrasp_count = 0

        # every loop just send the _cached_ command
        self.sendRosCommand(self.pregrasp_jcmd)
        if self.pregrasp_count > 600:  # 6 seconds``
            self._transition_to(RobotState.GRASP)
        self.pregrasp_count += 1

    def _handle_grasp(self):
        # on first entry, compute and cache IK
        if self.grasp_jcmd is None:
            rot_active = pin.Quaternion(np.array([[1, 0, 0], 
                                                  [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                                                  [0, np.sin(np.pi/2), np.cos(np.pi/2)]]))
            rot_idle = pin.Quaternion(np.eye(3))
            self.grasp_jcmd = self.rm_controller.pink_ik(
                pin.Quaternion(self.rm_state.state[3:7]), self.rm_state.state[0:3],
                rot_active, self.grip_Handle_pose[:3],
                rot_idle, self.rm_state.state[0:3] + np.array([0.6, -0.4, self.grip_Handle_pose[2]-0.3])
            )
            self.grasp_count = 0

        # every loop just send the _cached_ command
        if self.grasp_count > 300:
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
                arm_idx=self.active_arm_id
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
                arm_idx=self.active_arm_id
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