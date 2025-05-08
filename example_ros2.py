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
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from pathlib import Path
from sys import argv
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

urdf_model_path = "./urdf/overseas_65_corrected.urdf"
mesh_dir = "./urdf"
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pinocchio.JointModelFreeFlyer()
)
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration
q = pinocchio.neutral(model)
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)

viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1.0])

print(f"q: {q.T}")
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    
r_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
l_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
q_isaac = np.zeros(22)

base_pose = np.zeros(7) # x, y, z, rx, ry, rz, rw
base_yaw = 0.0

    

def jointStateCallback(msg):
    global l_theta, r_theta, q_isaac
    #update the platform joint
    q_isaac[7] = msg.position[0]
    pos_map = dict(zip(msg.name, msg.position))
    left_joints  = [f"l_joint{i}" for i in range(1,7)]
    right_joints = [f"r_joint{i}" for i in range(1,7)]
    left_positions  = [pos_map[j] for j in left_joints ]
    right_positions = [pos_map[j] for j in right_joints]
    q_isaac[10:16] = left_positions
    q_isaac[16:22] = right_positions
    print("Left:",  left_positions)
    print("Right:", right_positions)

    pass

def basePoseCallback(msg):
    global base_pose, base_yaw, q_isaac
    for t in msg.transforms: 
        trans = t.transform.translation
        rot   = t.transform.rotation

        q_isaac[0:7] = np.array([trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w])
        # print(f"Base Pose: {trans}, Base rot: {rot}")
    pass

def main():
    global q_isaac, base_pose
    rclpy.init()
    node = rclpy.create_node('kinematics_node')

    joint_state_sub = node.create_subscription(JointState, '/isaac_sim/joint_states', jointStateCallback, 10)
    base_pose_sub = node.create_subscription(TFMessage, '/tf', basePoseCallback, 10)
    joint_state_pub = node.create_publisher(JointState, '/isaac_sim/joint_command', 10)
    base_pub = node.create_publisher(Twist, '/isaac_sim/cmd_vel', 10)

    # Create a JointState message
    joint_state_msg = JointState()
    velocity_msgs = Twist()
    joint_state_msg.name = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6',
                            'l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_finger_joint', 'r_finger_joint', 'platform_joint']

    joint_state_msg.position = [0.0 for _ in range(15)] 
    joint_state_msg.position[0:6]  = array('d', [0, 1.75, 0.6, -1.5, 0, 0])
    joint_state_msg.position[6:12] = array('d', [0, -1.75, -0.6, 1.5, 0, 0])
    joint_state_msg.position[12] = 0.785398
    joint_state_msg.position[13] = 0.785398
    joint_state_msg.position[14] = 0.4
    velocity_msgs.linear.x = 1.0
    velocity_msgs.angular.z = 0.0

    # Publish the JointState message
    try:
        while rclpy.ok():
            
            joint_state_msg.header.stamp = node.get_clock().now().to_msg()
            joint_state_pub.publish(joint_state_msg)
            base_pub.publish(velocity_msgs)
            time.sleep(0.05) 
            rclpy.spin_once(node)
            viz.display(q_isaac)
    except KeyboardInterrupt:
        print("\nShutting down publisher...")

if __name__ == '__main__':
    main()