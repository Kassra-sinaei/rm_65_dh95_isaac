import numpy as np
from numpy.linalg import norm, solve
import pinocchio
from pinocchio.visualize import MeshcatVisualizer




class Controller:
    def __init__(self, config):
        self.config = config

        # URDF & Pinocchio setup
        self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
            self.config.URDFPATH, self.config.MESH_DIR, pinocchio.JointModelFreeFlyer()
        )
        self.data = self.model.createData()

        # initial config & visualizer
        self.q = pinocchio.neutral(self.model)
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1.0])
        self.viz.displayFrames(True)
        
        print(f"model: {self.model}")


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
        breakpoint()
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
            if not i % 10:
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
        cam_in_world = oMf.act(pose)


        
