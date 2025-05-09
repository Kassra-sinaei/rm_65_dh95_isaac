#!/usr/bin/env python3.10

import numpy as np
from numpy.linalg import norm, solve
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
        self.viz.displayFrames(True)
        print(f"q: {self.q.T}")
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        print(f"model: {self.model}")
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
        print("getting joint state")
        for side, base in (('l', 10), ('r', 16)):
            self.pin_q[base:base + 6] = [pos_map[f"{side}_joint{i}"] for i in range(1, 7)]
        print("self.pin_q:", self.pin_q)

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
    
    def approach(self, joint_state_msg, left_arm):
        success = False
        des_pose = pinocchio.SE3(np.eye(3), np.array([3.40, 0.43, 1.2]))
        q_init_app = np.array([ 3.06160593e+00, -6.69872388e-05,  2.42999896e-01,  5.70327359e-07,
        2.86647435e-07,  7.07373738e-01,  7.06839621e-01,  4.09100000e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.75070000e+00,
        -5.99800000e-01,  1.49990000e+00, -2.00000000e-04,  1.00000000e-04,
        0.00000000e+00,  1.75070000e+00,  5.99800000e-01, -1.50010000e+00,
        -1.00000000e-04,  1.00000000e-04]
        )
        print(f"q_init_app: {q_init_app}")
        JOINT_ID = 10 if left_arm else 16
        eps = 1e-3
        IT_MAX = 1000
        DT = 0.05
        damp = 1e-12
        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q_init_app)
            iMd = self.data.oMi[JOINT_ID].actInv(des_pose)
            err = pinocchio.log(iMd).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            # J = pinocchio.computeJointJacobian(self.model, self.data, q_init_app, JOINT_ID)
            J = pinocchio.computeFrameJacobian(self.model, self.data, q_init_app, self.model.getFrameId("l_gripper_base_link"))
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            J_select = J[:,9:15]
            # print(f"J: {J}")
            # print(J.shape)
            v_select = -J_select.T.dot(solve(J_select.dot(J_select.T) + damp * np.eye(6), err))
            v = np.zeros(21)
            v[9:15] = v_select
            q_init_app = pinocchio.integrate(self.model, q_init_app, v * DT)
            self.viz.display(q_init_app)
            if not i % 10:
                print(f"{i}: error = {err.T}")
            i += 1
        if success:
            print("Convergence achieved!")
        else:
            print(
                "\n"
                "Warning: the iterative algorithm has not reached convergence "
                "to the desired precision"
            )
        print(f"\nresult: {q_init_app.flatten().tolist()}")
                



    def main(self):
        # Create a JointState message
        joint_state_msg = JointState()
        velocity_msgs   = Twist()
        joint_state_msg.name  = self.JOINT_MSG_NAME

        joint_state_msg.position = [0.0]*15
        self.initPose(joint_state_msg, velocity_msgs)

        need_to_approach = True
        try:
            while rclpy.ok():
                joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                self.joint_state_pub.publish(joint_state_msg)
                self.base_pub.publish(velocity_msgs)
                rclpy.spin_once(self)
                self.viz.display(self.pin_q)
                time.sleep(2)
                if need_to_approach:
                    self.approach(joint_state_msg, True)
                    need_to_approach = False  
                # print(self.door_handle_pose)
        except KeyboardInterrupt:
            print("\nShutting down publisher...")
        finally:
            self.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    RealmanControlNode().main()
