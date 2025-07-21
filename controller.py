import numpy as np
from numpy.linalg import norm, solve
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import crocoddyl
from pink.tasks import FrameTask
from pink import solve_ik, Configuration
import meshcat_shapes

from uni_mpc import MPCController

class Controller:
    def __init__(self, config):
        self.config = config

        # URDF & Pinocchio setup
        if self.config.FLOATING_BASE:
            self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
                self.config.URDFPATH, self.config.MESH_DIR, pinocchio.JointModelFreeFlyer()
            )
            self.tasks = {
                'base': FrameTask(self.config.PIN_BASE_FRAME_NAME, position_cost=10.0, orientation_cost=1.0),
                'r_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[1], position_cost=10.0, orientation_cost=1.0),
                'l_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[0], position_cost=10.0, orientation_cost=1.0)
            }
        else:
            self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
                self.config.URDFPATH, self.config.MESH_DIR
            )
            self.tasks = {
                'r_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[1], position_cost=10.0, orientation_cost=1.0),
                'l_gripper':FrameTask(self.config.PIN_GIRPPER_FRAME_NAME[0], position_cost=10.0, orientation_cost=1.0)
            }
        self.data = self.model.createData()
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
        meshcat_shapes.frame(self.viewer["base_goal"], opacity=1.0)
        
        print(f"model: {self.model}")

        # Crocoddyl MPC
        self.base_controller = MPCController(
            horizon = 50, dt = self.config.BASE_DT,
            Q = self.config.Q, R = self.config.R, Qf = self.config.Qf,
            v_max = self.config.v_max, omega_max = self.config.omega_max
        )

        # Pink setup
        self.configuration = Configuration(self.model, self.data, np.array(pinocchio.neutral(self.model)))
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

    def update_pink_ik_configuration(self, cur_q):
        if self.config.FLOATING_BASE:
            self.configuration.update(cur_q)
        else:
            self.configuration.update(cur_q[7:])

    def pink_ik_incremental(self, cur_q, base_rot, base_p, l_goal_rot, l_goal_p, r_goal_rot, r_goal_p):
        """Run pink ik solver in an incremental manner.
        The desired poses are in the world frame. If the base is fixed, the desired poses are transformed
        to the local base frame.
        Args:
            cur_q: current joint configuration
            base_rot: base rotation
            base_p: base position
            l_goal_rot: left gripper rotation
            l_goal_p: left gripper position
            r_goal_rot: right gripper rotation
            r_goal_p: right gripper position
        Returns:
            joint_command: joint command
        """
        base_world = pinocchio.SE3(base_rot, base_p)
        if l_goal_rot is not None:
            l_goal_world = pinocchio.SE3(l_goal_rot, l_goal_p)
            l_goal_local = base_world.actInv(l_goal_world)
        if r_goal_rot is not None:
            r_goal_world = pinocchio.SE3(r_goal_rot, r_goal_p)
            r_goal_local = base_world.actInv(r_goal_world)
        
        # Update tasks with current poses
        if self.config.FLOATING_BASE:
            self.tasks['base'].set_target(base_world)
        
        # Convert goal poses to local base frame if using fixed base
        if self.config.FLOATING_BASE:
            if l_goal_rot is not None:
                self.tasks['l_gripper'].set_target(l_goal_world)
            else:
                self.tasks['l_gripper'].set_target_from_configuration(self.configuration)
            if r_goal_rot is not None:
                self.tasks['r_gripper'].set_target(r_goal_world)
            else:
                self.tasks['r_gripper'].set_target_from_configuration(self.configuration)
        else:
            if l_goal_rot is not None:
                self.tasks['l_gripper'].set_target(l_goal_local)
            else:
                self.tasks['l_gripper'].set_target_from_configuration(self.configuration)
            if r_goal_rot is not None:
                self.tasks['r_gripper'].set_target(r_goal_local)
            else:
                self.tasks['r_gripper'].set_target_from_configuration(self.configuration)

        
        self.viewer["l_gripper_target"].set_transform(self.tasks['l_gripper'].transform_target_to_world.np)
        self.viewer["l_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["l_gripper"].frame, 
            world_frame=self.config.FLOATING_BASE).np
        )

        self.viewer["r_gripper_target"].set_transform(self.tasks['r_gripper'].transform_target_to_world.np)
        self.viewer["r_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["r_gripper"].frame,
            world_frame=self.config.FLOATING_BASE).np
        )
        self.viewer["base_target"].set_transform(self.tasks['base'].transform_target_to_world.np)


        velocity = solve_ik(self.configuration, self.tasks.values(), self.config.PIN_DT, solver="quadprog")
        self.configuration.integrate_inplace(velocity, self.config.PIN_DT)
            
        joint_command = [self.configuration.q[i] for i in self.config.PIN_Q_TO_JCOMMAND]
        # joint_command[-3] = 0.3  # platform joint
        return joint_command

    def pink_ik_incremental_local(self, cur_q,l_goal_rot, l_goal_p, r_goal_rot, r_goal_p):
        """Run pink ik solver in an incremental manner.
        The desired poses are in the local base frame.
        Args:
            cur_q: current joint configuration
            l_goal_rot: left gripper rotation
            l_goal_p: left gripper position
            r_goal_rot: right gripper rotation
            r_goal_p: right gripper position
        Returns:
            joint_command: joint command
        """
        # Don't allow running this function if the base is floating
        if self.config.FLOATING_BASE:
            raise ValueError("pink_ik_incremental_local is not supported for floating base")
        
        # Update tasks with current poses
        if l_goal_rot is not None:
            l_goal_local = pinocchio.SE3(l_goal_rot, l_goal_p)
            self.tasks['l_gripper'].set_target(l_goal_local)
        if r_goal_rot is not None:
            r_goal_local = pinocchio.SE3(r_goal_rot, r_goal_p)
            self.tasks['r_gripper'].set_target(r_goal_local)
        
        self.viewer["l_gripper_target"].set_transform(self.tasks['l_gripper'].transform_target_to_world.np)
        self.viewer["l_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["l_gripper"].frame, 
            world_frame=self.config.FLOATING_BASE).np
        )

        self.viewer["r_gripper_target"].set_transform(self.tasks['r_gripper'].transform_target_to_world.np)
        self.viewer["r_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["r_gripper"].frame,
            world_frame=self.config.FLOATING_BASE).np
        )

        velocity = solve_ik(self.configuration, self.tasks.values(), self.config.PIN_DT, solver="quadprog")
        self.configuration.integrate_inplace(velocity, self.config.PIN_DT)
        joint_command = [self.configuration.q[i] for i in self.config.PIN_Q_TO_JCOMMAND]
        return joint_command

    def pink_ik(self, cur_q, base_rot, base_p, l_goal_rot, l_goal_p, r_goal_rot, r_goal_p):
        self.update_pink_ik_configuration(cur_q)
        base_world = pinocchio.SE3(base_rot, base_p)
        l_goal_world = pinocchio.SE3(l_goal_rot, l_goal_p)
        r_goal_world = pinocchio.SE3(r_goal_rot, r_goal_p)
        
        # Update tasks with desired poses
        if self.config.FLOATING_BASE:
            self.tasks['base'].set_target(base_world)
        
        # Convert goal poses to local base frame if using fixed base
        if self.config.FLOATING_BASE:
            self.tasks['l_gripper'].set_target(l_goal_world)
            self.tasks['r_gripper'].set_target(r_goal_world)
        else:
            l_goal_local = base_world.actInv(l_goal_world)
            r_goal_local = base_world.actInv(r_goal_world)
            self.tasks['l_gripper'].set_target(l_goal_local)
            self.tasks['r_gripper'].set_target(r_goal_local)

        
        self.viewer["l_gripper_target"].set_transform(self.tasks['l_gripper'].transform_target_to_world.np)
        self.viewer["l_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["l_gripper"].frame, 
            world_frame=self.config.FLOATING_BASE).np
        )

        self.viewer["r_gripper_target"].set_transform(self.tasks['r_gripper'].transform_target_to_world.np)
        self.viewer["r_gripper"].set_transform(
            self.compute_frame_pose(cur_q, self.tasks["r_gripper"].frame,
            world_frame=self.config.FLOATING_BASE).np
        )

        # Solve until it converges
        for _ in np.arange(0.0, 5.0, self.config.PIN_DT):
            velocity = solve_ik(self.configuration, self.tasks.values(), 
                                self.config.PIN_DT, solver="quadprog")
            self.configuration.integrate_inplace(velocity, self.config.PIN_DT)
        joint_command = [self.configuration.q[i] for i in self.config.PIN_Q_TO_JCOMMAND]
        return joint_command

    def find_arm_inverse_kinematics(self, curr_state, des_position, des_rot, arm_idx):

        des_rot =  des_rot @ self.config.PIN_ARM_ROTATION_OFFSET[arm_idx]
        frame_id = self.model.getFrameId(self.config.PIN_GIRPPER_FRAME_NAME[arm_idx])
        des_pose = pinocchio.SE3(des_rot, des_position)
        print("finding ik for arm", arm_idx, "with des_pose", des_pose)
        if self.config.FLOATING_BASE:
            pin_q = curr_state.copy()
        else:
            # Convert the desired pose to the local base frame
            pin_q = curr_state[7:].copy()
            base_rot = pinocchio.Quaternion(pin_q[3:7]).normalized().toRotationMatrix()
            base_world = pinocchio.SE3(base_rot, pin_q[0:3])
            des_pose = base_world.actInv(des_pose)

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
        oMf = self.compute_frame_pose(curr_state, "camera_link")
        # the camera baselink is rotated by 90 degrees around the z axis
        offset = pinocchio.SE3(self.config.CAMERA_ROTATION_OFFSET, np.array([0,0,0]))
        cam_in_world = oMf.act(offset.act(pose))
        return cam_in_world


    def compute_base_twist_pd(self, x_i, x_g, T = None):
        error =(x_g - x_i).reshape(3, 1)
        d = np.linalg.norm(error[:2])
        if d > 0.1:
            return np.array([0.2 * d, -0.5 * (np.sin(x_i[2]) - error[1,0]/d)])
        else:
            return np.array([0.0, 1.5 * (error[2,0])])
    
    def compute_base_twist(self, x_s, d, T = 10):
        """
        Computes the base twist to move towards the desired position.
        """
        if d > 0.15:
            x_s.reshape(3, 1)
            T = int(T / self.config.BASE_DT)
            problem = crocoddyl.ShootingProblem(x_s, [ self.base_model ] * T, self.base_model)
            ddp = crocoddyl.SolverDDP(problem)
            us_init = [np.zeros(2) for _ in range(T)]
            xs_init = ddp.problem.rollout(us_init)
            if ddp.solve(xs_init, us_init, maxiter=25):
                return ddp.us[0]
            else:
                raise RuntimeError("DDP failed to solve the problem")
        else:
            return np.array([0.0, -1.5 * (x_s[2])])

    def compute_frame_pose(self, q, frame_name, world_frame = True):
        """
        Computes the end-effector pose in the world frame for a given joint configuration.
        """
        base_rot = pinocchio.Quaternion(q[3:7]).normalized().toRotationMatrix()
        base_world = pinocchio.SE3(base_rot, q[0:3])
        if self.config.FLOATING_BASE:
            pin_q = q.copy()
        else:
            pin_q = q[7:].copy()
        pinocchio.forwardKinematics(self.model, self.data, pin_q)
        frame_id = self.model.getFrameId(frame_name)
        oMf = pinocchio.updateFramePlacement(self.model, self.data, frame_id)
        if self.config.FLOATING_BASE and world_frame:
            return oMf
        elif self.config.FLOATING_BASE and not world_frame:
            return base_world.actInv(oMf)
        elif not self.config.FLOATING_BASE and world_frame:
            return base_world.act(oMf)
        else:
            return oMf