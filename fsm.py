#!/usr/bin/env python3
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import time, threading
from math import atan2

import matplotlib.pylab as plt

import rclpy.logging; plt.ion()


from pink import solve_ik, Configuration
from pink.tasks import FrameTask
import crocoddyl

from ament_index_python import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage

from planner import Planner

class FSM(Node):
    def __init__(self, type, side, width):
        # ROS2 Node Setup
        super().__init__('fsm')
        self.rate = self.create_rate(20)

        self.joint_state_sub = self.create_subscription(JointState, '/isaac_sim/joint_states', self.jointStateCallback, 2)
        self.base_pose_sub = self.create_subscription(TFMessage, '/tf', self.basePoseCallback, 2)
        self.handle_sub = self.create_subscription(TFMessage, '/isaac_sim/door_handle', self.handleCallback, 2)
        self.joint_state_pub = self.create_publisher(JointState, '/isaac_sim/joint_command', 2)
        self.base_pub = self.create_publisher(Twist, '/isaac_sim/cmd_vel', 2)
        # Set robot State
        self.state = "initial"
        self.base_pose = np.array([0.0, 0.0, 0.0])      # x y psi
        self.handle_pose = np.zeros(7)    # [x y z] [x y z w]
        self.dual_arm_config = np.zeros(13)     # platform[1] r_arm[6] l_arm[6]
        
        # urdf_path = os.path.join(get_package_share_directory('door_fsm'), '../../../..','src', 'door_fsm','description', 'rm_model.urdf')
        # mesh_dir = os.path.join(get_package_share_directory('door_fsm'),  '../../../..','src', 'door_fsm','description', 'meshes')

        urdf_path = "/home/kasra/alphaz/rm_ws/src/door_fsm/description/rm_model.urdf"
        mesh_dir = "/home/kasra/alphaz/rm_ws/src/door_fsm/description/meshes"

        # Setup Pinocchio Model and RobotWrapper
        self.robot = RobotWrapper.BuildFromURDF(
            urdf_path,
            [mesh_dir]
        )
        self.model = pin.buildModelFromUrdf(
            urdf_path,
        )
        self.nq = self.robot.nq
        self.display_count = 0
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)


        self.r_gripper = self.model.getFrameId('r_gripper_frame')
        self.l_gripper = self.model.getFrameId('l_gripper_frame')
        self.base = self.model.getFrameId('base_link_underpan')
        # joint_names = [self.robot.model.names[i] for i in range(self.robot.model.njoints)]
        # print("Joint names:", joint_names)

        # Setup Pink
        self.dt = 1e-1
        self.configuration = Configuration(self.robot.model, self.data, self.q)
        self.tasks = {
            'r_gripper':FrameTask('r_gripper_frame', position_cost=1.0, orientation_cost=1.0),
            'l_gripper':FrameTask('l_gripper_frame', position_cost=1.0, orientation_cost=1.0)
        }
        self.joint_names = ['platform_joint',
                            'r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6',
                            'l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6']
        # Setup Crocoddyl
        self.base_model = crocoddyl.ActionModelUnicycle()
        self.base_model.dt = 0.1
        self.base_model.costWeights = np.matrix([10, 9]).T
        self.base_model.stateWeights = np.matrix([1, 1, 5]).T
        self.data  = self.base_model.createData()

        # Door Parameters
        self.door_type = type     # push or pull
        self.handle_side = side   # left or right
        self.door_width = width   # width of the door
        if self.handle_side == "left":
            self.handle_offset = np.array([1.15, 0.20])
        else:
            self.handle_offset = np.array([1.15, -0.20])

    def unicycleController(self, start, goal, T):
        e = (start - goal)
        e.reshape(3, 1)
        T = int(T / self.base_model.dt)
        problem = crocoddyl.ShootingProblem(e, [ self.base_model ] * T, self.base_model)
        ddp = crocoddyl.SolverDDP(problem)
        if ddp.solve():
            # print(e)
            return ddp.us[0]
        else:
            return None

    def pinkIK(self, r_goal_p, r_goal_q, l_goal_p, l_goal_q, max_iter=20000, eps=5e-2):
        self.configuration.q = self.q

        self.tasks['r_gripper'].set_target(pin.SE3(r_goal_q, r_goal_p))
        self.tasks['l_gripper'].set_target(pin.SE3(l_goal_q, l_goal_p))

        for t in np.arange(0.0, 2.0, self.dt):
            velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver="quadprog")
            self.configuration.integrate_inplace(velocity, self.dt)

        return self.configuration.q
    
    def basePoseCallback(self, msg):
        w = msg.transforms[0].transform.rotation.w
        x = msg.transforms[0].transform.rotation.x
        y = msg.transforms[0].transform.rotation.y
        z = msg.transforms[0].transform.rotation.z
        psi = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) 
        self.base_pose = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   psi])
        # print(psi)
        if self.state == "initial" and np.linalg.norm(self.base_pose[0:2] - (self.handle_pose[0:2] - self.handle_offset)) < 0.15 and abs(psi) < 0.1:
            self.state = "approach"
        else:
            self.state = "initial"
        pass

    def handleCallback(self, msg):
        self.handle_pose = np.array([msg.transforms[0].transform.translation.x,
                                      msg.transforms[0].transform.translation.y,
                                      msg.transforms[0].transform.translation.z,
                                      msg.transforms[0].transform.rotation.x,
                                      msg.transforms[0].transform.rotation.y,
                                      msg.transforms[0].transform.rotation.z,
                                      msg.transforms[0].transform.rotation.w])
        # print("Handle Pose updated ...")
        pass

    def jointStateCallback(self, msg):
        self.dual_arm_config[0] = msg.position[0]
        self.dual_arm_config[1] = msg.position[4]
        self.dual_arm_config[2] = msg.position[3]
        self.dual_arm_config[3] = msg.position[10]
        self.dual_arm_config[4] = msg.position[9]
        self.dual_arm_config[5] = msg.position[14]
        self.dual_arm_config[6] = msg.position[13]
        self.dual_arm_config[7] = msg.position[16]
        self.dual_arm_config[8] = msg.position[15]
        self.dual_arm_config[9] = msg.position[18]
        self.dual_arm_config[10] = msg.position[17]
        self.dual_arm_config[11] = msg.position[20]
        self.dual_arm_config[12] = msg.position[19]
        # print("Joint State updated ...")
        pass

    def approach(self):
        goal = np.zeros(3)
        goal[0:2] = self.handle_pose[0:2] - self.handle_offset
        # goal[2] = atan2(2.0 * (self.handle_pose[6] * self.handle_pose[5] + self.handle_pose[3] * self.handle_pose[4]), 
        #                 1.0 - 2.0 * (self.handle_pose[4] * self.handle_pose[4] + self.handle_pose[5] * self.handle_pose[5]))
        goal[2] = 0
        if self.state == "initial":
            try:
                res = self.unicycleController(self.base_pose, goal, 10)
                msg = Twist()
                msg.linear.x = res[0]
                msg.angular.z = res[1]
                self.base_pub.publish(msg)
                # rclpy.logging.get_logger('fsm').info("Publishing base velocity ...")
            except Exception as e:
                rclpy.logging.get_logger('fsm').error("Error in unicycleController")


    def grasp(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.base_pub.publish(msg)
        if self.door_type == "push":
            local_handle = np.array([self.handle_pose[0] - self.base_pose[0], 
                                        self.handle_pose[1] - self.base_pose[1], 
                                        self.handle_pose[2] - 0.243])
            if self.handle_side == "left":
                r_goal_p = local_handle
                r_goal_q = pin.Quaternion(self.handle_pose[6], self.handle_pose[3], self.handle_pose[4], self.handle_pose[5]).normalized().matrix()
                l_goal_p = np.array([0.4, 0.2, 1.5])
                l_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()
            else:
                l_goal_p = local_handle
                l_goal_q = pin.Quaternion(self.handle_pose[6], self.handle_pose[3], self.handle_pose[4], self.handle_pose[5]).normalized().matrix()
                r_goal_p = np.array([0.2, -0.2, 1.5])
                r_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()
            q = self.pinkIK(r_goal_p, r_goal_q, l_goal_p, l_goal_q)
            msg = JointState()
            msg.name = self.joint_names
            msg.position = [q[0], 
                            q[9], q[10], q[11], q[12], q[13], q[14],
                            q[3], q[4], q[5], q[6], q[7], q[8]]
            self.joint_state_pub.publish(msg)
            print("publishing joint state ...")
        else:
            # TODO: implement pull type door grasping
            pass
                                

def test(fsm):
    r_goal_p = np.array([-0.4, -0.4, 1.0])
    r_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()
    l_goal_p = np.array([0.4, -0.4, 1.0])
    l_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()

    start_time = time.time()
    q = fsm.pinkIK(r_goal_p, r_goal_q, l_goal_p, l_goal_q)
    end_time = time.time()
    print(f"pinkIK execution time: {end_time - start_time:.4f} seconds")
    print(q)
    # fsm.viz.display(q)
    # fsm.viz.addBox('r_goal', [.1, .1, .1], [.1, .1, .5, .6])
    # fsm.viz.applyConfiguration('r_goal', pin.SE3(r_goal_q, r_goal_p))
    # fsm.viz.addBox('l_goal', [.1, .1, .1], [.1, .1, .5, .6])
    # fsm.viz.applyConfiguration('l_goal', pin.SE3(l_goal_q, l_goal_p))
    time.sleep(5)

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([2.0, 1.0, 0.0])
    start_time = time.time()
    u = fsm.unicycleController(start, goal, 10)
    end_time = time.time()
    print(f"unicycleController execution time: {end_time - start_time:.4f} seconds")
    print(u)


if __name__ == "__main__":
    rclpy.init()
    fsm = FSM('push', 'left', 0.9)
    # test(fsm)

    thread = threading.Thread(target=rclpy.spin, args=(fsm, ), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            if fsm.state == "initial":
                fsm.approach()
            elif fsm.state == "approach":
                fsm.grasp()
            else:
                break
            fsm.rate.sleep()

    except KeyboardInterrupt:
        pass
    
    rclpy.logging.get_logger('fsm').info("Keyboard Interrupt, shutting down ...")
    fsm.destroy_node()
    rclpy.shutdown()
    thread.join()
