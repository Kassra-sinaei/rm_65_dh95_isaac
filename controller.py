import numpy as np
from numpy.linalg import norm, solve
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import crocoddyl
from pink.tasks import FrameTask
from pink import solve_ik, Configuration
import meshcat_shapes



class Controller:
    def __init__(self, config):
        self.config = config

        # URDF & Pinocchio setup
        self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
            self.config.URDFPATH, self.config.MESH_DIR, pinocchio.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
        self.tasks = {
            'base': FrameTask(self.config.PIN_BASE_FRAME_NAME, position_cost=10.0, orientation_cost=1.0),
            'r_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[1], position_cost=10.0, orientation_cost=1.0),
            'l_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[0], position_cost=10.0, orientation_cost=1.0)
        }
        self.joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6',
                            'r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6',
                            'platform_joint', 'head_joint1', 'head_joint2']

        # initial config & visualizer
        self.q = pinocchio.neutral(self.model)
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1.0])
        # self.viz.displayFrames(True)
        self.viewer = self.viz.viewer
        meshcat_shapes.frame(self.viewer["l_gripper_target"], opacity=0.5)
        meshcat_shapes.frame(self.viewer["l_gripper"], opacity=1.0)
        meshcat_shapes.frame(self.viewer["r_gripper_target"], opacity=0.5)
        meshcat_shapes.frame(self.viewer["r_gripper"], opacity=1.0)
        meshcat_shapes.frame(self.viewer["base_target"], opacity=0.5)
        meshcat_shapes.frame(self.viewer["base"], opacity=1.0)
        
        print(f"model: {self.model}")

        # Crocoddyl MPC
        self.base_model = crocoddyl.ActionModelUnicycle()
        self.base_model.dt = 0.01
        self.base_model.costWeights = np.matrix([100, 80]).T
        self.base_model.stateWeights = np.matrix([1, 1, 1]).T
        self.base_data  = self.base_model.createData()

    def pink_ik(self, base_q, base_p, r_goal_q, r_goal_p, l_goal_q, l_goal_p, current_config):
        configuration = Configuration(self.model, self.data, current_config)

    def pink_ik(self, cur_q, base_rot, base_p, l_goal_rot, l_goal_p, r_goal_rot, r_goal_p):
        # Update current configuration (The base keeps going down when using current configuration?)
        # self.configuration.update(cur_q)
            
        # Update tasks with desired poses
        self.tasks['base'].set_target(pinocchio.SE3(base_rot, base_p))
        # self.tasks['base'].set_target_from_configuration(self.configuration)
        self.tasks['r_gripper'].set_target(pinocchio.SE3(r_goal_rot, r_goal_p))
        self.tasks['l_gripper'].set_target(pinocchio.SE3(l_goal_rot, l_goal_p))
        
        self.viewer["l_gripper_target"].set_transform(self.tasks['l_gripper'].transform_target_to_world.np)
        self.viewer["l_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["l_gripper"].frame).np
        )

        self.viewer["r_gripper_target"].set_transform(self.tasks['r_gripper'].transform_target_to_world.np)
        self.viewer["r_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["r_gripper"].frame).np
        )


        for t in np.arange(0.0, 10, self.config.PIN_DT):
            velocity = solve_ik(configuration, self.tasks.values(), self.config.PIN_DT, solver="quadprog")
            configuration.integrate_inplace(velocity, self.config.PIN_DT)
        joint_command = [configuration.q[i] for i in self.config.PIN_Q_TO_JCOMMAND]

        r_gripper_frame_id = self.model.getFrameId(self.config.PIN_GIRPPER_FRAME_NAME[1])
        l_gripper_frame_id = self.model.getFrameId(self.config.PIN_GIRPPER_FRAME_NAME[0])
        
        pinocchio.forwardKinematics(self.model, self.data, configuration.q)
        r_gripper_error = pinocchio.log(pinocchio.updateFramePlacement(self.model, self.data, r_gripper_frame_id).actInv(pinocchio.SE3(r_goal_q, r_goal_p))).vector
        l_gripper_error = pinocchio.log(pinocchio.updateFramePlacement(self.model, self.data, l_gripper_frame_id).actInv(pinocchio.SE3(l_goal_q, l_goal_p))).vector
        
        print(f"Right gripper error: {r_gripper_error}")
        print(f"Left gripper error: {l_gripper_error}")
        return joint_command
      
    def find_arm_inverse_kinematics(self, curr_state, des_position, des_rot, arm_idx):

        des_rot =  des_rot @ self.config.PIN_ARM_ROTATION_OFFSET[arm_idx]
        frame_id = self.model.getFrameId(self.config.PIN_GIRPPER_FRAME_NAME[arm_idx])
        des_pose = pinocchio.SE3(des_rot, des_position)
        print("finding ik for arm", arm_idx, "with des_pose", des_pose)
        pin_q = curr_state.copy()
        sol_viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        sol_viz.initViewer(self.viz.viewer)
        sol_viz.loadViewerModel(rootNodeName="ik_sol_viz" , color=[1.0, 1.0, 1.0, 0.5])
        SUCCESS = False
        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, pin_q)
            oMf = pinocchio.updateFramePlacement(self.model, self.data, frame_id)
            fMd = oMf.actInv(des_pose)
            err = pinocchio.log(fMd).vector
            if norm(err) < self.config.PIN_EPS:
                SUCCESS = True                                                      
                break
            if i >= self.config.PIN_IT_MAX:
                break
            J = pinocchio.computeFrameJacobian(self.model, self.data, pin_q, frame_id)
            J = -np.dot(pinocchio.Jlog6(fMd.inverse()), J)
            J_select = J[:,self.config.PIN_JACOB_JOINT_ID[arm_idx]]
            v_select = -J_select.T.dot(solve(J_select.dot(J_select.T) + self.config.PIN_DAMP * np.eye(6), err))
            v = np.zeros(21)
            v[self.config.PIN_JACOB_JOINT_ID[arm_idx]] = v_select
            pin_q = pinocchio.integrate(self.model, pin_q, v * self.config.PIN_DT)
            sol_viz.display(pin_q)
            if not i % 100:
                print(f"{i}: error = {err.T}")
                print(f"v: {v}")
                print(f"\nresult: {pin_q.flatten().tolist()}")
            i += 1
        if SUCCESS:
            print("IK success")
        else:
            print("IK failed")

        # convert pinocchio q to joint command
        joint_command = [pin_q[i] for i in self.config.PIN_Q_TO_JCOMMAND]
        
        return joint_command
    
    def convert_pose_from_camera_to_world(self, curr_state, pose):
        pin_q = curr_state.copy()
        pinocchio.forwardKinematics(self.model, self.data, pin_q)
        cam_frame_id = self.model.getFrameId("camera_link")
        oMf = pinocchio.updateFramePlacement(self.model, self.data, cam_frame_id)
        # the camera baselink is rotated by 90 degrees around the z axis
        offset = pinocchio.SE3(self.config.CAMERA_ROTATION_OFFSET, np.array([0,0,0]))
        cam_in_world = oMf.act(offset.act(pose))
        return cam_in_world
    
    def base_yaw_control(self, yaw_error):
        return np.array([0.0, 20*yaw_error])
    
    def compute_base_twist(self, e, T = 10):
        """
        Computes the base twist to move towards the desired position.
        """
        e.reshape(3, 1)
        T = int(T / self.base_model.dt)
        problem = crocoddyl.ShootingProblem(e, [ self.base_model ] * T, self.base_model)
        ddp = crocoddyl.SolverDDP(problem)
        if ddp.solve():
            # print(e)
            return ddp.us[0]
        else:
            print("DDP solve failed: ")
            return None

    def compute_frame_pose(self, q, frame_name):
        """
        Computes the end-effector pose for a given joint configuration.
        """
        pinocchio.forwardKinematics(self.model, self.data, q)
        frame_id = self.model.getFrameId(frame_name)
        oMf = pinocchio.updateFramePlacement(self.model, self.data, frame_id)
        return oMf