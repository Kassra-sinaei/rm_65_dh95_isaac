#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
import numpy as np
from array import array
import math
import pinocchio as pin
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Point
from tf2_msgs.msg import TFMessage

from config import Config
from realmanState import RealmanState
from controller import Controller
from enum import Enum, auto

class RobotState(Enum):
    INIT_POSE    = auto()
    DETECT_HANDLE = auto()
    PREGRASP     = auto()
    GRASP        = auto()
    PULL         = auto()
    TURN         = auto()
    OPENING      = auto()
    IKTEST       = auto()
    HOLD_DOOR_LEFT  = auto()
    HOLD_DOOR_RIGHT  = auto()
    PUSH_DOOR    = auto()
    IDLE         = auto()
    # add more states here as you go…
    # LIFT   = auto()
    # DONE   = auto()


class RealmanControlNode(Node):
    def __init__(self):
        super().__init__('PullDoorController')

        self.config = Config()
        self.rm_state = RealmanState(self.config)
        self.rm_controller = Controller(self.config)

        self.door_handle_pose = np.zeros(7)  # [x, y, z, rx, ry, rz, rw]

        self.state = RobotState.INIT_POSE
        self.state_start_time = self.get_clock().now()
        self.pregrasp_jcmd = None
        self.grasp_jcmd = None
        self.pull_jcmd = None
        self.turn_jcmd = None
        self.open_jcmd = None
        self.pull_base_cmd = None
        self.grip_Handle_pose = None
        self.first_entry = True

        # subscriptions
        self.create_subscription(
            JointState, '/isaac_sim/joint_states',
            self.jointStateCallback, 10
        )
        self.create_subscription(
            TFMessage, '/isaac_sim/odom',
            self.basePoseCallback, 10
        )

        self.create_subscription(
            TFMessage, '/isaac_sim/pull_door_handle',
            self.doorHandleCallback, 10
        )

        self.create_subscription(
            Point, '/grip_point',
            self.gripHandleCallback, 10
        )

        # publishers
        self.joint_state_pub = self.create_publisher(
            JointState, '/isaac_sim/joint_command', 10
        )
        self.base_pub = self.create_publisher(
            Twist, '/isaac_sim/cmd_vel', 10
        )

        self.pregrasp_pose = None
        self.grasp_pose = None
        self.hold_door_left_pose = None
        self.hold_door_left_reach_pose = None
        self.hold_door_right_pose = None
        self.hold_door_right_backward_pose = None

        # wait for subscribers to connect
        self.create_timer(self.config.PIN_DT, self._control_loop)

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
        self.grip_Handle_pose = self.rm_controller.convert_pose_from_camera_to_world(
            self.rm_state.state,
            pose_in_camera
        )



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
        elif self.state == RobotState.PULL:
            self._handle_pull()
        elif self.state == RobotState.HOLD_DOOR_LEFT:
            self._handle_hold_door_left()
        elif self.state == RobotState.HOLD_DOOR_RIGHT:
            self._handle_hold_door_right()
        # elif self.state == RobotState.DETECT_HANDLE:
        #     self._handle_detect_handle()
        # elif self.state == RobotState.PREGRASP:
        #     self._handle_pregrasp()
        # elif self.state == RobotState.GRASP:
        #     self._handle_grasp()
        # elif self.state == RobotState.TURN:
        #     self._handle_turn()
        # elif self.state == RobotState.OPENING:
        #     self._handle_opening()
        
        # visualize 
        if self.config.FLOATING_BASE:
            self.rm_controller.viz.display(self.rm_state.state)
        else:
            self.rm_controller.viz.display(self.rm_state.state[7:])
    
    def _handle_init_pose(self):
        # keep sending init-pose until 200 ticks have elapsed
        self.sendRosCommand(self.config.INIT_JCOMMAND)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 500 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            # self._transition_to(RobotState.IKTEST)
            self._transition_to(RobotState.PREGRASP)

    def _handle_hold_door_left(self):
        if self.first_entry is True:
            self.first_entry = False
            self.get_logger().info("Entering HOLD_DOOR_LEFT state and sending initial pose command")
            self.get_logger().info("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            self.get_logger().info(f"Left hand pose: {l_hand}")
            self.get_logger().info(f"Right hand pose: {r_hand}")
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.state_start_time = self.get_clock().now()

        base_cmd = np.array([0.0, 0.0])
        
        # Approaching the door
        if (self.get_clock().now() - self.state_start_time).nanoseconds < 400 * 10_000_000:
            self.hold_door_left_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            self.initial_r_hand_pose.rotation, 
                                                            self.initial_r_hand_pose.translation)
            base_cmd = np.array([0.2, 0.0])
            # The right arm gripper keeps closing
            self.hold_door_left_jcmd[14] = 1.0

        # Preholding the door (angle is hardcoded)
        if 400 * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds < 700 * 10_000_000:
            if self.hold_door_left_pose is None:
                l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
                r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
                self.initial_l_hand_pose = l_hand
                self.initial_r_hand_pose = r_hand
                base_world_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
                base_pose_world = pin.SE3(base_world_rot, self.rm_state.state[0:3])
                r_hand_local = base_pose_world.actInv(r_hand)
                l_hand_local = base_pose_world.actInv(l_hand)
                l_hand_local.translation = r_hand_local.translation + self.config.HOLD_DOOR_LEFT_TRANSLATION_OFFSET_FROM_R_HAND
                self.hold_door_left_pose = base_pose_world.act(l_hand_local)
                self.hold_door_left_pose.rotation = self.hold_door_left_pose.rotation @ self.config.HOLD_DOOR_LEFT_ROTATION_OFFSET
                # self.rm_controller.update_pink_ik_configuration(self.rm_state.state)

            self.hold_door_left_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            self.hold_door_left_pose.rotation, 
                                                            self.hold_door_left_pose.translation,
                                                            None, None)
            base_cmd = np.array([0.0, 0.0])
            self.hold_door_left_jcmd[14] = 1.0

        # Reach the door to hold it with an offset from the left hand in the left hand frame
        if  700 * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds < 1000 * 10_000_000:
            if self.hold_door_left_reach_pose is None:
                l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
                self.hold_door_left_reach_pose = l_hand
                self.hold_door_left_reach_pose.translation += l_hand.rotation @ self.config.HOLD_DOOR_LEFT_REACH_TRANSLATION_OFFSET
            
            self.hold_door_left_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            self.hold_door_left_reach_pose.rotation, 
                                                            self.hold_door_left_reach_pose.translation,
                                                            None, None)
            base_cmd = np.array([0.0, 0.0])
            self.hold_door_left_jcmd[14] = 1.0
            # self._transition_to(RobotState.HOLD_DOOR_RIGHT)

            
        self.sendRosCommand(self.hold_door_left_jcmd, base_cmd)

        if (self.get_clock().now() - self.state_start_time).nanoseconds >= 1000 * 10_000_000:
            self.first_entry = True
            self._transition_to(RobotState.HOLD_DOOR_RIGHT)

    def _handle_hold_door_right(self):
        if self.first_entry is True:
            self.first_entry = False
            self.get_logger().info("Entering HOLD_DOOR_RIGHT state and sending initial pose command")
            self.get_logger().info("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            self.get_logger().info(f"Left hand pose: {l_hand}")
            self.get_logger().info(f"Right hand pose: {r_hand}")
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.state_start_time = self.get_clock().now()

        # Fix the joint angles and keep the right gripper open
        if (self.get_clock().now() - self.state_start_time).nanoseconds < 400 * 10_000_000:
            self.hold_door_right_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                                pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                                self.rm_state.state[0:3],
                                                                None, None,
                                                                None, None)
            self.hold_door_right_jcmd[14] = 0.0

        # Backward the right arm
        if 400 * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds < 100000 * 10_000_000:
            if self.hold_door_right_backward_pose is None:
                r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
                self.hold_door_right_backward_pose = r_hand
                self.hold_door_right_backward_pose.translation += r_hand.rotation @ self.config.HOLD_DOOR_RIGHT_BACKWARD_TRANSLATION_OFFSET
            
            self.hold_door_right_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            self.hold_door_right_backward_pose.rotation, 
                                                            self.hold_door_right_backward_pose.translation)
            self.hold_door_right_jcmd[14] = 0.0

        
        
        self.sendRosCommand(self.hold_door_right_jcmd)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 100000 * 10_000_000:
            self.first_entry = True
            self._transition_to(RobotState.DETECT_HANDLE)

    def _handle_detect_handle(self):
        if self.grip_Handle_pose is not None:
            self.get_logger().info("Handle detected")
            self._transition_to(RobotState.PREGRASP)

    def _handle_pregrasp(self):
        if self.door_handle_pose is not None and self.pregrasp_pose is None:
            base_world_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
            # Rotate along x for 90 degrees and then 180 degrees along z
            door_handle_rot = base_world_rot @ self.config.HANDLE_PREGRASP_ROTATION_OFFSET
            door_handle_pose_des = pin.SE3(door_handle_rot, self.door_handle_pose[:3])
            # Transform the handle pose to the local frame with the offset
            base_pose_world = pin.SE3(base_world_rot, self.rm_state.state[0:3])
            door_handle_pose_des_local = base_pose_world.actInv(door_handle_pose_des)
            door_handle_pose_des_local.translation += self.config.HANDLE_PREGRASP_TRANSLATION_OFFSET_LOCAL
            door_handle_pose_des = base_pose_world.act(door_handle_pose_des_local)
            self.pregrasp_pose = door_handle_pose_des
            self.initial_l_hand_pose = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            self.initial_r_hand_pose = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.state_start_time = self.get_clock().now()

        if self.pregrasp_pose is not None:
            self.pregrasp_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            self.initial_l_hand_pose.rotation, 
                                                            self.initial_l_hand_pose.translation,
                                                            self.pregrasp_pose.rotation, 
                                                            self.pregrasp_pose.translation)
            self.sendRosCommand(self.pregrasp_jcmd)
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 300 * 10_000_000:
                self._transition_to(RobotState.GRASP)


        
        # # on first entry, compute and cache IK
        # if self.pregrasp_jcmd is None:
        #     self.pregrasp_jcmd = self.rm_controller.find_arm_inverse_kinematics(
        #         self.rm_state.state,
        #         self.grip_Handle_pose  + self.config.HANDEL_PREGRIP_OFFSET,
        #         np.eye(3),
        #         arm_idx=0
        #     )

        #     self.pregrasp_count = 0
        #     self.get_logger().info("Computed pregrasp IK once")

        # # every loop just send the _cached_ command
        # self.sendRosCommand(self.pregrasp_jcmd)
        # if self.pregrasp_count > 100:
        #     # 300 * 0.01s == 3 seconds
        #     self._transition_to(RobotState.GRASP)
        # self.pregrasp_count += 1

    def _handle_grasp(self):
        if self.door_handle_pose is not None and self.grasp_pose is None:
            base_world_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
            # Rotate along x for 90 degrees and then 180 degrees along z
            door_handle_rot = base_world_rot @ self.config.HANDLE_PREGRASP_ROTATION_OFFSET
            door_handle_pose_des = pin.SE3(door_handle_rot, self.door_handle_pose[:3])
            # Transform the handle pose to the local frame with the offset
            base_pose_world = pin.SE3(base_world_rot, self.rm_state.state[0:3])
            door_handle_pose_des_local = base_pose_world.actInv(door_handle_pose_des)
            door_handle_pose_des_local.translation += self.config.HANDLE_GRASP_TRANSLATION_OFFSET_LOCAL
            door_handle_pose_des = base_pose_world.act(door_handle_pose_des_local)
            self.grasp_pose = door_handle_pose_des
            self.initial_l_hand_pose = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            self.initial_r_hand_pose = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.state_start_time = self.get_clock().now()

        if self.grasp_pose is not None:
            self.grasp_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            self.initial_l_hand_pose.rotation, 
                                                            self.initial_l_hand_pose.translation,
                                                            self.grasp_pose.rotation, 
                                                            self.grasp_pose.translation)
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 200 * 10_000_000:
                self.grasp_jcmd[14] = 1.0
                
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 400 * 10_000_000:
                self.pull_jcmd = self.grasp_jcmd
                self.sendRosCommand(self.grasp_jcmd)
                self._transition_to(RobotState.PULL)

            self.sendRosCommand(self.grasp_jcmd)
    
    def _handle_pull(self):
        # Keep sending the same joint command as the previous state
        if self.pull_base_cmd is None:
            backward_speed = -0.2
            self.pull_travel_time = 1.5 / abs(backward_speed)
            self.pull_turn_time = 1.0
            # swing_time = 1
            self.pull_base_cmd = np.array([backward_speed, 0.0])
            self.state_start_time = self.get_clock().now()
            

        if self.pull_base_cmd is not None:
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.sendRosCommand(base_command=self.pull_base_cmd)
        
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * self.pull_travel_time * 10_000_000:
                self.pull_base_cmd = np.array([0.0, 0])
                # self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * (self.pull_travel_time + self.pull_turn_time) * 10_000_000:
                base_command = np.array([0.0, 0.0])
                self.sendRosCommand(base_command=base_command)
                self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
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