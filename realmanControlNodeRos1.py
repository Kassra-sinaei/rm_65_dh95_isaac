#!/usr/bin/env python3.8
import rospy
import numpy as np
from array import array
import pinocchio as pin

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
    TURN         = auto()
    OPENING      = auto()
    IKTEST       = auto()
    HOLD_DOOR_1  = auto()
    HOLD_DOOR_2  = auto()
    PUSH_DOOR    = auto()
    # add more states here as you go…
    # LIFT   = auto()
    # DONE   = auto()


class RealmanControlNode:
    def __init__(self):
        rospy.init_node('RealmanControlNode', anonymous=True)
        self.config = Config()
        self.rm_state = RealmanState(self.config)
        self.rm_controller = Controller(self.config)

        self.door_handle_pose = np.zeros(7)  # [x, y, z, rx, ry, rz, rw]

        self.state = RobotState.INIT_POSE
        self.state_start_time = rospy.get_rostime().now()
        self.pregrasp_jcmd = None
        self.grasp_jcmd = None
        self.turn_jcmd = None
        self.open_jcmd = None
        self.grip_Handle_pose = None
        self.first_entry = False

        # subscriptions
        rospy.Subscriber(
            '/isaac_sim/joint_states',
            JointState,
            self.jointStateCallback, queue_size=10
        )
        rospy.Subscriber(
            '/isaac_sim/base_link',
            TFMessage,
            self.basePoseCallback, queue_size=10
        )
        rospy.Subscriber(
            '/isaac_sim/door_handle',
            TFMessage,
            self.doorHandleCallback, queue_size=10
        )
        rospy.Subscriber(
            '/grip_point',
            Point,
            self.gripHandleCallback, queue_size=10
        )
        # publishers
        self.joint_state_pub = rospy.Publisher(
            '/isaac_sim/joint_command', JointState, queue_size=10
        )
        self.base_pub = rospy.Publisher(
            '/isaac_sim/cmd_vel', Twist, queue_size=10
        )
        self.hold_door_1_pose = None
        self.hold_door_2_pose = None
        
        rospy.sleep(1)  # wait for subscribers to connect

        self.control_timer = rospy.Timer(rospy.Duration(self.config.PIN_DT), self._control_loop)
            
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

    def _control_loop(self, event):
        # dispatch based on current state
        if self.state == RobotState.INIT_POSE:
            self._handle_init_pose()
        elif self.state == RobotState.HOLD_DOOR_1:
            self._handle_hold_door_1()
        elif self.state == RobotState.HOLD_DOOR_2:
            self._handle_hold_door_2()
        # elif self.state == RobotState.IKTEST:
        #     self._handle_pink_ik_test()
        # elif self.state == RobotState.PREGRASP:
        #     self._handle_pregrasp()
        # elif self.state == RobotState.GRASP:
        #     self._handle_grasp()
        # elif self.state == RobotState.TURN:
        #     self._handle_turn()
        # elif self.state == RobotState.OPENING:
        #     self._handle_opening()
        
        
        # visualize 
        self.rm_controller.viz.display(self.rm_state.state)



    def _handle_init_pose(self):
        # keep sending init-pose until 200 ticks have elapsed
        self.sendRosCommand(self.config.INIT_JCOMMAND)

        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 500 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            # self._transition_to(RobotState.IKTEST)
            self._transition_to(RobotState.HOLD_DOOR_1)
    
    def _handle_pink_ik_test(self):
        if self.first_entry is False:
            rospy.loginfo("Entering INIT_POSE state and sending initial pose command")
            self.first_entry = True
            rospy.loginfo("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.rm_controller.configuration.update(self.rm_state.state)
            # self.initial_r_hand_pose.translation[0]  # set initial right hand pose translation
            # self.initial_r_hand_pose.translation[1]
            self.state_start_time = rospy.get_rostime().now()
            self.initial_l_hand_height = self.initial_l_hand_pose.translation[2]

            
        # keep sending init-pose until 200 ticks have elapsed
        # self.sendRosCommand(self.config.INIT_JCOMMAND)
        # keep sending init-pose until 200 ticks have elapsed

        # hand_rot = pin.Quaternion(np.array([[np.cos(np.pi), 0.0, np.sin(np.pi)], 
        #                             [0,1,0],
        #                             [-np.sin(np.pi), 0.0, np.cos(np.pi)]]))
        base_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()
        # Add a sinousoidal offset to the translation of the left hand
        # Stop at a certain time
        # l_hand_desired_trans = self.rm_controller.compute_frame_pose(
        #     self.rm_state.state, 
        #     self.config.PIN_GIRPPER_FRAME_NAME[0]).translation
        # l_hand_desired_trans[2] = self.initial_l_hand_height
        # if (rospy.get_rostime().now() - self.state_start_time).to_nsec() < 500 * 10_000_000:        
        l_hand_desired_trans = self.initial_l_hand_pose.translation + np.array([
            np.sin(rospy.get_rostime().now().to_sec() * 1.0) * 0.15,
            0.0, 
            0.0            
        ])


        self.init_pose_command = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized(), 
                                                            self.rm_state.state[0:3],
                                                            self.initial_l_hand_pose.rotation, 
                                                            l_hand_desired_trans, 
                                                            self.initial_r_hand_pose.rotation, 
                                                            self.initial_r_hand_pose.translation)

        self.sendRosCommand(self.init_pose_command)

        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 10000 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            rospy.loginfo("Current EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.first_entry = False
            self._transition_to(RobotState.DETECT_HANDLE)

    def _handle_hold_door_1(self):
        if self.first_entry is False:
            rospy.loginfo("Entering INIT_POSE state and sending initial pose command")
            self.first_entry = True
            rospy.loginfo("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.rm_controller.configuration.update(self.rm_state.state)
            # self.initial_r_hand_pose.translation[0]  # set initial right hand pose translation
            # self.initial_r_hand_pose.translation[1]
            self.state_start_time = rospy.get_rostime().now()
            self.initial_l_hand_height = self.initial_l_hand_pose.translation[2]
            self.hold_door_1_pose = l_hand
            self.hold_door_1_pose.translation[0] += 0.5
            self.hold_door_1_pose.translation[1] -= 0.3
            self.hold_door_1_pose.translation[2] += 0.4
            # Set the rotation to be 90 degrees around the y-axis
            self.hold_door_1_pose.rotation = self.hold_door_1_pose.rotation @ pin.Quaternion(np.array([[np.cos(np.pi/2), 0.0, np.sin(np.pi/2)],
                                                                                                  [0,1,0],
                                                                                                  [-np.sin(np.pi/2), 0.0, np.cos(np.pi/2)]])).normalized().toRotationMatrix()   

            
        base_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()   
        # l_hand_desired_trans = self.initial_l_hand_pose.translation + np.array([
        #     np.sin(rospy.get_rostime().now().to_sec() * 1.0) * 0.15,
        #     0.0, 
        #     0.0            
        # ])


        self.init_pose_command = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized(), 
                                                            self.rm_state.state[0:3],
                                                            self.hold_door_1_pose.rotation, 
                                                            self.hold_door_1_pose.translation, 
                                                            self.initial_r_hand_pose.rotation, 
                                                            self.initial_r_hand_pose.translation)

        self.sendRosCommand(self.init_pose_command)

        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 500 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            rospy.loginfo("Current EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.first_entry = False
            self._transition_to(RobotState.HOLD_DOOR_2)
    
    def _handle_hold_door_2(self):
        if self.first_entry is False:
            rospy.loginfo("Entering INIT_POSE state and sending initial pose command")
            self.first_entry = True
            rospy.loginfo("Initial EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            self.initial_l_hand_pose = l_hand
            self.initial_r_hand_pose = r_hand
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.rm_controller.configuration.update(self.rm_state.state)
            # self.initial_r_hand_pose.translation[0]  # set initial right hand pose translation
            # self.initial_r_hand_pose.translation[1]
            self.state_start_time = rospy.get_rostime().now()
            self.initial_l_hand_height = self.initial_l_hand_pose.translation[2]
            self.hold_door_2_pose = self.hold_door_1_pose
            self.hold_door_2_pose.translation[0] -= 0.2
            # self.hold_door_2_pose.translation[1] -= 0.1
            self.hold_door_2_pose.translation[2] -= 0.2

            
        base_rot = pin.Quaternion(self.rm_state.state[3:7]).normalized().toRotationMatrix()   
        # l_hand_desired_trans = self.initial_l_hand_pose.translation + np.array([
        #     np.sin(rospy.get_rostime().now().to_sec() * 1.0) * 0.15,
        #     0.0, 
        #     0.0            
        # ])


        self.init_pose_command = self.rm_controller.pink_ik_incremental(self.rm_state.state,
                                                            pin.Quaternion(self.rm_state.state[3:7]).normalized(), 
                                                            self.rm_state.state[0:3],
                                                            self.initial_l_hand_pose.rotation, 
                                                            self.initial_l_hand_pose.translation, 
                                                            self.hold_door_2_pose.rotation, 
                                                            self.hold_door_2_pose.translation)

        self.sendRosCommand(self.init_pose_command)

        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 500 * 10_000_000:
            # 300 * 0.01s == 3 seconds
            rospy.loginfo("Current EE pose: ")
            l_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[0])
            r_hand = self.rm_controller.compute_frame_pose(self.rm_state.state, self.config.PIN_GIRPPER_FRAME_NAME[1])
            rospy.loginfo(f"Left hand pose: {l_hand}")
            rospy.loginfo(f"Right hand pose: {r_hand}")
            self.first_entry = False
            self._transition_to(RobotState.DETECT_HANDLE)

        
    def _handle_detect_handle(self):
        if self.grip_Handle_pose is not None:
            rospy.loginfo("Handle detected")
            self._transition_to(RobotState.PREGRASP)

    def _handle_pregrasp(self):
        # on first entry, compute and cache IK
        if self.pregrasp_jcmd is None:
            self.pregrasp_jcmd = self.rm_controller.find_arm_inverse_kinematics(
                self.rm_state.state,
                self.grip_Handle_pose  + self.config.HANDEL_PREGRIP_OFFSET,
                np.eye(3),
                arm_idx=0
            )

            self.pregrasp_count = 0
            rospy.loginfo("Computed pregrasp IK once")

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
            rospy.loginfo("Computed grasp IK once")

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
            rospy.loginfo("Computed turn IK once")
        self.sendRosCommand(self.turn_jcmd)
        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 300 * 10_000_000:
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
        rospy.loginfo("Opening door with right arm")
        
        if (rospy.get_rostime().now() - self.state_start_time).to_nsec() > 10 * 10_000_000:
            self.open_jcmd[13] = 0.0

        self.sendRosCommand(self.open_jcmd)


    def _transition_to(self, new_state: RobotState):
        rospy.loginfo(f"Transitioning from {self.state.name} to {new_state.name}")
        self.state = new_state
        self.state_start_time = rospy.get_rostime().now()

    def sendRosCommand(self, joint_command = None, base_command = None):
        if joint_command is not None:
            joint_state_msg = JointState()
            joint_state_msg.name = self.config.JOINT_MSG_NAME
            joint_state_msg.position = array('d', joint_command)
            joint_state_msg.header.stamp = rospy.get_rostime().now()
            self.joint_state_pub.publish(joint_state_msg)
        if base_command is not None:
            velocity_msgs = Twist()
            velocity_msgs.linear.x   = base_command[0]
            velocity_msgs.angular.z  = base_command[1]
            self.base_pub.publish(velocity_msgs)               

def main():
    # Initialize the RealmanControlNode
    node = RealmanControlNode()
    try:
        rospy.spin()  # Keep the node running
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("Shutting down RealmanControlNode")
        rospy.signal_shutdown("Node shutdown requested")


if __name__ == '__main__':
    main()