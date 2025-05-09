#!/usr/bin/env python3
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import time

import matplotlib.pylab as plt; plt.ion()


from pink import solve_ik, Configuration
from pink.tasks import FrameTask
import crocoddyl

from ament_index_python import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage


class FSM(Node):
    def __init__(self):
        # ROS2 Node Setup
        super().__init__('fsm')
        self.joint_state_sub = self.create_subscription(JointState, '/isaac_sim/joint_states', self.jointStateCallback, 10)
        self.base_pose_sub = self.create_subscription(TFMessage, '/tf', self.basePoseCallback, 10)
        self.handle_sub = self.create_subscription(TFMessage, '/isaac_sim/door_handle', self.basePoseCallback, 10)
        self.joint_state_pub = self.create_publisher(JointState, '/isaac_sim/joint_command', 10)
        self.base_pub = self.create_publisher(Twist, '/isaac_sim/cmd_vel', 10)
        # Set robot State
        self.state = "initial"
        self.base_pose = np.array([0.0, 0.0, 0.0])      # x y psi
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
        joint_names = [self.robot.model.names[i] for i in range(self.robot.model.njoints)]
        print("Joint names:", joint_names)

        # Setup Pink
        self.dt = 1e-1
        self.configuration = Configuration(self.robot.model, self.data, self.q)
        self.tasks = {
            'r_gripper':FrameTask('r_gripper_frame', position_cost=1.0, orientation_cost=1.0),
            'l_gripper':FrameTask('l_gripper_frame', position_cost=1.0, orientation_cost=1.0)
        }

        # Setup Crocoddyl
        self.model = crocoddyl.ActionModelUnicycle()
        self.model.dt = 1e-2
        self.model.costWeights = np.array([1, 1 ])
        self.data  = self.model.createData()

    def approach(self):


    def unicycleController(self, start, goal, T):
        e = start - goal
        e.reshape(3, 1)
        T = int(T / self.model.dt)
        problem = crocoddyl.ShootingProblem(e, [ self.model ] * T, self.model)
        ddp = crocoddyl.SolverDDP(problem)
        if ddp.solve():
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
        self.base_pose = np.array([msg.transforms[0].transform.translation.x,
                                   msg.transforms[0].transform.translation.y,
                                   msg.transforms[0].transform.rotation.z])
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
    fsm = FSM()
    # test(fsm)
    
