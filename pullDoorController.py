#!/usr/bin/env python3.10
from turtle import back
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

import crocoddyl

from utils.trajectory import create_waypoint_trajectory, generate_waypoint_trajectory, create_arc_trajectory, create_circular_trajectory

class RobotState(Enum):
    INIT_POSE    = auto()
    DETECT_HANDLE = auto()
    APPROACH     = auto()
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

        # Set true if the door is not aligned with simulation frames
        self.rotated_door = True

        self.config = Config()
        self.rm_state = RealmanState(self.config, self.rotated_door)
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
        self.push_base_cmd = None
        self.grip_Handle_pose = None
        self.first_entry = True
        self.base_traj = None

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
        self.pull_swing_pose = None
        self.push_swing_pose = None

        # wait for subscribers to connect
        self.create_timer(self.config.PIN_DT, self._control_loop)

    def jointStateCallback(self, msg):
        # update platform joint
        self.rm_state.update_joint_state(msg)

    def basePoseCallback(self, msg):
        self.rm_state.update_base_pose(msg)

    def doorHandleCallback(self, msg):
        for t in msg.transforms:
            if self.rotated_door:
                trans = t.transform.translation
                rot   = t.transform.rotation
                self.door_handle_pose = np.array([
                    -trans.y, trans.x, trans.z,
                    rot.x, rot.y, rot.z, rot.w
                ])
            else:
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
        # elif self.state == RobotState.DETECT_HANDLE:
        #     self._handle_detect_handle()
        elif self.state == RobotState.APPROACH:
            self._approach_door()
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
        elif self.state == RobotState.PUSH_DOOR:
            self._handle_push_door()
        # elif self.state == RobotState.PREGRASP:
        #     self._handle_pregrasp()
        # elif self.state == RobotState.GRASP:
        #     self._handle_grasp()
        # elif self.state == RobotState.TURN:
        #     self._handle_turn()
        # elif self.state == RobotState.OPENING:
        #     self._handle_opening()
        
        # visualize 
        # if self.config.FLOATING_BASE:
        #     self.rm_controller.viz.display(self.rm_state.state)
        # else:
        #     self.rm_controller.viz.display(self.rm_state.state[7:])
    
    def _handle_init_pose(self):
        # keep sending init-pose until 200 ticks have elapsed
        self.sendRosCommand(self.config.INIT_JCOMMAND)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 500 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            # self._transition_to(RobotState.IKTEST)
            self._transition_to(RobotState.APPROACH)

    def _handle_detect_handle(self):
        if self.grip_Handle_pose is not None:
            self.get_logger().info("Handle detected")
            self._transition_to(RobotState.PREGRASP)

    def _approach_door(self):
        grip_pose = np.array([self.door_handle_pose[0], self.door_handle_pose[1], 0.0])
        goal = grip_pose - self.config.PULL_BASE_OFFSET
        goal = np.array([-1, 0, 0])
        rotation_matrix = pin.Quaternion( self.rm_state.state[6], self.rm_state.state[3], self.rm_state.state[4], self.rm_state.state[5]).toRotationMatrix() 
        goal_transform = pin.SE3(rotation_matrix, np.array([goal[0], goal[1], self.rm_state.state[2]]))
        # self.rm_controller.viz.viewer["base_goal"].set_transform(goal_transform.homogeneous)
        current_state = np.array([self.rm_state.state[0], self.rm_state.state[1],
                                    np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])])
        if self.base_traj is None:
            # self.reference_traj = self.rm_controller.base_controller.generate_reference_trajectory(current_state, goal)
            self.reference_traj = goal * np.ones((self.rm_controller.base_controller.N + 1, 3))
            # self.rm_controller.base_controller.plot_ref(goal, current_state, self.reference_traj)

        if np.linalg.norm((goal - current_state)[0:2]) > 0.1 or (goal[2] -current_state[2]) > 0.05:
            control_input, success = self.rm_controller.base_controller.control(current_state, self.reference_traj)
            print(f"Current Pose: {current_state.T}, goal: {goal.T}")
            print(f"Control Input: {control_input.T}")
            self.sendRosCommand(base_command=control_input)
        else:
            self.sendRosCommand(base_command=[0.0, 0.0])
            self._transition_to(RobotState.PREGRASP)
            breakpoint()

    def _handle_pregrasp(self):
        if self.door_handle_pose is not None and self.pregrasp_pose is None:
            base_world_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
            if self.rotated_door:
                base_world_rot = base_world_rot @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
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
                                                            #np.array([self.rm_state.state[1], -self.rm_state.state[0], self.rm_state.state[2]]),
                                                            self.rm_state.state[0:3],
                                                            self.initial_l_hand_pose.rotation, 
                                                            self.initial_l_hand_pose.translation,
                                                            self.pregrasp_pose.rotation, 
                                                            self.pregrasp_pose.translation)
            self.sendRosCommand(self.pregrasp_jcmd)
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 300 * 10_000_000:
                self._transition_to(RobotState.GRASP)
                breakpoint()

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
            if backward_speed == 0.0:
                self.pull_travel_time = 0.1
            else:
                self.pull_travel_time = 1.5 / abs(backward_speed)
            self.pull_swing_time = 3.0
            # swing_time = 1
            self.pull_base_cmd = np.array([backward_speed, 0.0])
            self.state_start_time = self.get_clock().now()
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)

            # Initialize trajectory parameters
            self.trajectory_start_time = None
            

        if self.pull_base_cmd is not None:
            # Swing the arm to the right to open the door
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * self.pull_travel_time * 10_000_000:
                if self.pull_swing_pose is None:
                    r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
                    self.pull_swing_pose = r_hand.copy()
                    self.pull_swing_pose.translation += r_hand.rotation @ np.array([-0.15, 0.0, -0.05])
                    self.waypoints = create_waypoint_trajectory(
                        r_hand,
                        self.pull_swing_pose
                    )
                    self.trajectory_duration = self.pull_swing_time
                    self.trajectory_start_time = self.get_clock().now()

                elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9
                current_target_pose = generate_waypoint_trajectory(
                    self.waypoints,
                    self.trajectory_duration,
                    elapsed_time
                )
                self.pull_swing_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            current_target_pose.rotation, 
                                                            current_target_pose.translation)
                self.pull_swing_jcmd[14] = 1.0
                self.pull_base_cmd = np.array([0.0, 0.0])
                self.sendRosCommand(self.pull_swing_jcmd)
                # self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
            if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * (self.pull_travel_time + self.pull_swing_time) * 10_000_000:
                self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
            self.sendRosCommand(base_command=self.pull_base_cmd)

    def _handle_hold_door_left(self):
        if self.first_entry is True:
            self.first_entry = False
            self.get_logger().info("Entering HOLD_DOOR_LEFT state and fetching initial pose command")
            # self.get_logger().info("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            self.initial_r_hand_pose_local = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1], world_frame=False)
            # self.get_logger().info(f"Left hand pose: {l_hand}")
            # self.get_logger().info(f"Right hand pose: {r_hand}")
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.hold_door_left_approach_time = 0.0
            self.hold_door_left_prehold_time = 3.0
            self.hold_door_left_reach_time = 3.0
            self.state_start_time = self.get_clock().now()

        base_cmd = np.array([0.0, 0.0])
        
        # Approaching the door
        if (self.get_clock().now() - self.state_start_time).nanoseconds < 100 * self.hold_door_left_approach_time * 10_000_000:
            self.hold_door_left_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            self.initial_r_hand_pose.rotation, 
                                                            self.initial_r_hand_pose.translation)
            # base_cmd = np.array([0.2, 0.0])
            base_cmd = np.array([0.0, 0.0])
            # The right arm gripper keeps closing
            self.hold_door_left_jcmd[14] = 1.0

        # Preholding the door (position and orientation are hardcoded relative to the right hand)
        if 100 * (self.hold_door_left_approach_time) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
            < 100 * (self.hold_door_left_approach_time + self.hold_door_left_prehold_time) * 10_000_000:
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
        if  100 * (self.hold_door_left_approach_time + self.hold_door_left_prehold_time) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
            < 100 * (self.hold_door_left_approach_time + self.hold_door_left_prehold_time + self.hold_door_left_reach_time) * 10_000_000:
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

        if (self.get_clock().now() - self.state_start_time).nanoseconds >= 100 * (self.hold_door_left_approach_time + self.hold_door_left_prehold_time + self.hold_door_left_reach_time) * 10_000_000:
            self.first_entry = True
            self._transition_to(RobotState.HOLD_DOOR_RIGHT)

    def _handle_hold_door_right(self):
        if self.first_entry is True:
            self.first_entry = False
            self.get_logger().info("Entering HOLD_DOOR_RIGHT state and fetching initial pose command")
            # self.get_logger().info("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            # self.get_logger().info(f"Left hand pose: {l_hand}")
            # self.get_logger().info(f"Right hand pose: {r_hand}")
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)
            self.hold_door_right_release_gripper_time = 3.0
            self.hold_door_right_backward_arm_time = 2.5
            self.hold_door_right_contact_switch = 8.0
            self.hold_door_right_put_down_left_arm = 4.0
            self.state_start_time = self.get_clock().now()
            
            # Initialize trajectory parameters
            self.trajectory_start_time = None
            self.trajectory_start_time_homing = None
            self.initial_r_hand_pose_for_trajectory = None
            self.final_r_hand_pose_for_trajectory = None

        base_cmd = np.array([0.0, 0.0])

        # Fix the joint angles and keep the right gripper open
        if (self.get_clock().now() - self.state_start_time).nanoseconds < 100 * (self.hold_door_right_release_gripper_time) * 10_000_000:
            self.hold_door_right_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                                pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                                self.rm_state.state[0:3],
                                                                None, None,
                                                                None, None)
            self.hold_door_right_jcmd[14] = 0.0

        # Backward the right arm
        if 100 * (self.hold_door_right_release_gripper_time) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
            < 100 * (self.hold_door_right_release_gripper_time + self.hold_door_right_backward_arm_time) * 10_000_000:
            if self.hold_door_right_backward_pose is None:
                r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
                self.initial_r_hand_pose = r_hand.copy()
                self.hold_door_right_backward_pose = r_hand.copy()
                self.hold_door_right_backward_pose.translation += r_hand.rotation @ self.config.HOLD_DOOR_RIGHT_BACKWARD_TRANSLATION_OFFSET
                self.backward_trajectory_start_time = self.get_clock().now()
                self.backward_waypoints = create_waypoint_trajectory(
                    self.initial_r_hand_pose, self.hold_door_right_backward_pose
                )
            
            # Calculate trajectory progress
            elapsed_time = (self.get_clock().now() - self.backward_trajectory_start_time).nanoseconds / 1e9

            # Generate smooth trajectory through waypoints
            current_target_pose = generate_waypoint_trajectory(
                self.backward_waypoints,
                self.hold_door_right_backward_arm_time,
                elapsed_time
            )

            self.hold_door_right_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            current_target_pose.rotation, 
                                                            current_target_pose.translation)
            self.hold_door_right_jcmd[14] = 0.0

        # Generate and track trajectory for the right arm
        if 100 * (self.hold_door_right_release_gripper_time + self.hold_door_right_backward_arm_time) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
            < 100 * (self.hold_door_right_release_gripper_time + self.hold_door_right_backward_arm_time + self.hold_door_right_contact_switch) * 10_000_000:
            # Initialize trajectory on first entry to this phase
            if self.trajectory_start_time is None:
                self.trajectory_start_time = self.get_clock().now()
                l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
                r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
                self.initial_r_hand_pose_for_trajectory = r_hand.copy()
                
                # Create final pose with offset
                self.mid_r_hand_pose_for_trajectory = r_hand.copy()
                self.mid_r_hand_pose_for_trajectory.rotation = r_hand.rotation @ pin.utils.rpyToMatrix(np.array([0.0, 0.0, -np.pi/2.0]))
                self.mid_r_hand_pose_for_trajectory.translation += r_hand.rotation @ np.array([0.2, -0.15, 0.0])
                self.final_r_hand_pose_for_trajectory = r_hand.copy()
                self.final_r_hand_pose_for_trajectory.rotation = r_hand.rotation @ pin.utils.rpyToMatrix(np.array([0.0, 0.0, -np.pi/2.0])) \
                    @ pin.utils.rpyToMatrix(np.array([np.pi/8.0, 0.0, 0.0]))
                self.final_r_hand_pose_for_trajectory.translation += r_hand.rotation @ np.array([0.22, -0.15, 0.12])
                # self.final_r_hand_pose_for_trajectory = r_hand.copy()
                # self.final_r_hand_pose_for_trajectory.rotation = r_hand.rotation @ pin.utils.rpyToMatrix(np.array([0.0, 0.0, -np.pi/2.0]))
                # self.final_r_hand_pose_for_trajectory.translation += r_hand.rotation @ np.array([0.16, -0.15, 0.15])
                
                # For now, use simple direct trajectory
                self.waypoints = create_waypoint_trajectory(
                    self.initial_r_hand_pose_for_trajectory,
                    self.final_r_hand_pose_for_trajectory,
                    intermediate_waypoints=[self.mid_r_hand_pose_for_trajectory]
                )
                # self.trajectory_duration = 3.0 * len(self.waypoints)
                self.trajectory_duration = self.hold_door_right_contact_switch
                
                self.get_logger().info(f"Starting waypoint trajectory with {len(self.waypoints)} waypoints")
                for i, wp in enumerate(self.waypoints):
                    self.get_logger().info(f"Waypoint {i}: {wp.translation}")
            
            # Calculate trajectory progress
            elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9
            
            # Generate smooth trajectory through waypoints
            current_target_pose = generate_waypoint_trajectory(
                self.waypoints,
                self.trajectory_duration,
                elapsed_time
            )
            
            self.hold_door_right_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                    pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                    self.rm_state.state[0:3],
                                                    None, None,
                                                    current_target_pose.rotation, 
                                                    current_target_pose.translation)
            self.hold_door_right_jcmd[14] = 0.0
            base_cmd = np.array([0.0, 0.0])

        if 100 * (self.hold_door_right_release_gripper_time + self.hold_door_right_backward_arm_time \
            + self.hold_door_right_contact_switch) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
            < 100 * (self.hold_door_right_release_gripper_time + self.hold_door_right_backward_arm_time \
                + self.hold_door_right_contact_switch + self.hold_door_right_put_down_left_arm) * 10_000_000:
            if self.trajectory_start_time_homing is None:
                self.trajectory_start_time_homing = self.get_clock().now()
                self.init_current_state = np.array(self.rm_state.state[7:13])

            elapsed_time = (self.get_clock().now() - self.trajectory_start_time_homing).nanoseconds / 1e9
            phase = elapsed_time / self.hold_door_right_put_down_left_arm            
            # Convert lists to numpy arrays for arithmetic operations
            
            l_arm_homing = np.array(self.config.INIT_ARM[0:6])
            result = (1 - phase) * self.init_current_state + phase * l_arm_homing
            # Convert back to list for assignment
            self.hold_door_right_jcmd[0:6] = result.tolist()

        self.sendRosCommand(self.hold_door_right_jcmd, base_cmd)

        if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * (self.hold_door_right_release_gripper_time \
            + self.hold_door_right_backward_arm_time \
            + self.hold_door_right_contact_switch + self.hold_door_right_put_down_left_arm) * 10_000_000:
            self.first_entry = True
            self._transition_to(RobotState.PUSH_DOOR)

    def _handle_push_door(self):
        if self.first_entry == True:
            self.first_entry = False
            forward_speed = 0.2
            if forward_speed == 0.0:
                self.push_travel_time = 0.0
            else:
                self.push_travel_time = 2.0 / abs(forward_speed)
            self.push_swing_time = 5.0
            # swing_time = 1
            self.push_base_cmd = np.array([forward_speed, 0.0])
            self.state_start_time = self.get_clock().now()
            self.rm_controller.update_pink_ik_configuration(self.rm_state.state)

            # Initialize trajectory parameters
            self.trajectory_start_time = None
            

        if self.push_base_cmd is not None:
            # Swing the arm to the right to open the door
            if (self.get_clock().now() - self.state_start_time).nanoseconds < 100 * self.push_swing_time * 10_000_000:
                if self.push_swing_pose is None:
                    base_world_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
                    base_pose_world = pin.SE3(base_world_rot, self.rm_state.state[0:3])
                    r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, 
                        self.config.PIN_GIRPPER_FRAME_NAME[1]).copy()
                    r_hand_local = self.rm_controller.compute_frame_pose(self.rm_state.state, 
                        self.config.PIN_GIRPPER_FRAME_NAME[1], world_frame=False).copy()
                    r_hand_local.translation += np.array([-0.4, 0.1, 0.0])
                    self.push_swing_pose = base_pose_world.act(r_hand_local)
                    self.push_swing_pose.rotation = r_hand.rotation @ pin.utils.rpyToMatrix(np.array([np.pi/4.0, 0.0, 0.0]))
                    self.waypoints = create_waypoint_trajectory(
                        r_hand,
                        self.push_swing_pose
                    )
                    self.trajectory_duration = self.push_swing_time
                    self.trajectory_start_time = self.get_clock().now()

                elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9
                current_target_pose = generate_waypoint_trajectory(
                    self.waypoints,
                    self.trajectory_duration,
                    elapsed_time
                )
                self.push_swing_jcmd = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix(), 
                                                            self.rm_state.state[0:3],
                                                            None, None,
                                                            current_target_pose.rotation, 
                                                            current_target_pose.translation)
                self.push_swing_jcmd[14] = 0.0
                self.push_base_cmd = np.array([0.0, 0.0])
                self.sendRosCommand(self.push_swing_jcmd)
                # self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
            if 100 * (self.push_swing_time) * 10_000_000 <= (self.get_clock().now() - self.state_start_time).nanoseconds \
                < 100 * (self.push_swing_time + self.push_travel_time) * 10_000_000:
                self.push_swing_jcmd[0:12] = self.config.INIT_ARM
                self.push_base_cmd = np.array([0.2, 0.0])
                self.sendRosCommand(self.push_swing_jcmd)

            if (self.get_clock().now() - self.state_start_time).nanoseconds > 100 * (self.push_swing_time + self.push_travel_time) * 10_000_000:
                self.push_base_cmd = np.array([0.0, 0.0])
                # self._transition_to(RobotState.HOLD_DOOR_LEFT)
            
            self.sendRosCommand(base_command=self.push_base_cmd)


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