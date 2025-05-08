#!/usr/bin/env python3.10

# import threading
import numpy as np
from array import array
import time
import math

import rclpy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
import pinocchio
from pathlib import Path
from sys import argv

# urdf_filename = "./a1/urdf/a1.urdf"
urdf_filename = "./urdf/overseas_65_corrected.urdf"
model = pinocchio.buildModelFromUrdf(urdf_filename)
print("model name: " + model.name)
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print(f"q: {q.T}")
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    
breakpoint()
r_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
l_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

base_pose = np.array([0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
base_yaw = 0.0

class RM_Kinematics:
    def __init__(self, left=False):
        a = 0.39
        b = 0.25
        self.isLeft = left
    
    def Rpitch(self, theta):
        # around y
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def Ryaw(self, theta):
        # around z
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    def geo_fk(self, theta, A = None, B = None):
        if A is None:
            A = 0.39
        if B is None:
            B = 0.25
        if self.isLeft:
            theta1 = -theta1
            theta4 = -theta4
            theta6 = -theta6
        theta1, theta2, theta3, theta4, theta5, theta6 = theta
        rot = self.Ryaw(theta1) @ self.Rpitch(-theta2) @ self.Rpitch(-theta3) @ self.Ryaw(theta4) @ self.Rpitch(-theta5) @ self.Ryaw(theta6)
        pos = np.array(
            [-A*np.sin(theta2)*np.cos(theta1), -A*np.sin(theta2)*np.sin(theta1), A*np.cos(theta2)]) + np.array(
                [-B*np.sin(theta2+theta3)*np.cos(theta1), -B*np.sin(theta2+theta3)*np.sin(theta1), B*np.cos(theta2+theta3)])
        return pos, rot


    def geo_inverse_kinematics(self,goal_pos, goal_rot ,A= None, B = None):
        if A is None:
            A = 0.39
        if B is None:
            B = 0.25
        theta1 = np.arctan2(goal_pos[1], goal_pos[0])
        r = np.linalg.norm(goal_pos, 2)
        theta3 = np.arccos(max(min((r**2 - A**2 -B**2)/(2*A*B), 1), -1))
        phi = np.arccos(max(min((r**2 + A**2 -B**2)/(2*A*r), 1), -1))
        theta2 = -phi - np.arctan2(np.sqrt(goal_pos[0]**2 + goal_pos[1]**2), goal_pos[2])
        R = self.Rpitch(-theta3).T @ self.Rpitch(-theta2).T @ self.Ryaw(theta1).T @ goal_rot
        theta6 = np.arctan2(R[2,1],-R[2,0])
        theta4 = np.arctan2(R[1,2],R[0,2])
        theta5 = -np.arccos(R[2,2])
        if self.isLeft:
            return -1*np.array([theta1, theta2, theta3, theta4, theta5, theta6])
        return np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    

def jointStateCallback(msg):
    global l_theta, r_theta
    l_theta[0] = msg.position[9]
    r_theta[0] = msg.position[10]
    l_theta[1] = msg.position[11]
    r_theta[1] = msg.position[12]
    l_theta[2] = msg.position[13]
    r_theta[2] = msg.position[14]
    l_theta[3] = msg.position[15]
    r_theta[3] = msg.position[16]
    l_theta[4] = msg.position[17]
    r_theta[4] = msg.position[18]
    l_theta[5] = msg.position[19]
    r_theta[5] = msg.position[20]
    print(f"Left Arm Angles: {l_theta} \n Right Arm Angles: {r_theta}")
    pass

def basePoseCallback(msg):
    global base_pose, base_yaw
    base_pose[0] = msg.position.x
    base_pose[1] = msg.position.y
    base_pose[2] = msg.position.z
    base_yaw = math.atan2(2.0 * (msg.orientation.w * msg.orientation.z + msg.orientation.x * msg.orientation.y), 1.0 - 2.0 * (msg.orientation.y**2 + msg.orientation.z**2))
    print(f"Base Pose: {base_pose}, Base Yaw: {base_yaw}")
    pass

def main():
    rclpy.init()
    node = rclpy.create_node('kinematics_node')

    joint_state_sub = node.create_subscription(JointState, '/isaac_sim/joint_states', jointStateCallback, 10)
    base_pose_sub = node.create_subscription(Pose, '/isaac_sim/base_pose', basePoseCallback, 10)
    joint_state_pub = node.create_publisher(JointState, '/isaac_sim/joint_command', 10)
    base_pub = node.create_publisher(Twist, '/isaac_sim/cmd_vel', 10)

    l_arm_solver = RM_Kinematics(True)
    r_arm_solver = RM_Kinematics()

    # Create a JointState message
    joint_state_msg = JointState()
    velocity_msgs = Twist()
    joint_state_msg.name = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6',
                            'l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_finger_joint', 'r_finger_joint', 'platform_joint']
    goal_rot = np.eye(3)
    goal_pos = np.array([0, 0, 0.4])
    r_angles = r_arm_solver.geo_inverse_kinematics(goal_pos, goal_rot)
    l_angles = l_arm_solver.geo_inverse_kinematics(goal_pos, goal_rot)
    # joint_state_msg.position = r_angles.tolist() + l_angles.tolist()
    # joint_state_msg.velocity = [0.0 for _ in range(12)] 
    # joint_state_msg.effort = [0.0 for _ in range(12)]
    joint_state_msg.position = [0.0 for _ in range(15)] 
    joint_state_msg.position[0:6]  = array('d', [0, 1.75, 0.6, -1.5, 0, 0])
    joint_state_msg.position[6:12] = array('d', [0, -1.75, -0.6, 1.5, 0, 0])
    joint_state_msg.position[12] = 0.785398
    joint_state_msg.position[13] = 0.785398
    joint_state_msg.position[14] = 0.4
    velocity_msgs.linear.x = 0.0
    velocity_msgs.angular.z = 0.0

    # Publish the JointState message
    try:
        while rclpy.ok():
            joint_state_msg.header.stamp = node.get_clock().now().to_msg()
            joint_state_pub.publish(joint_state_msg)
            base_pub.publish(velocity_msgs)
            time.sleep(0.05) 
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        print("\nShutting down publisher...")

if __name__ == '__main__':
    main()