#This is the config file for the project
import numpy as np

class Config:
    def __init__(self):
        # URDF & Pinocchio setup
        self.URDFPATH = "./urdf/overseas_65_corrected.urdf"
        self.MESH_DIR = "./urdf"
        self.INIT_ARM = [0, -1.75, -0.6, 1.5, 0, 0, 0, 1.75, 0.6, -1.5, 0, 0]   # left arm and right arm
        self.INIT_JCOMMAND = self.INIT_ARM + [0.4, 0, 0]  # arm + platform + finger 2
        self.JOINT_MSG_NAME = [f"l_joint{i}" for i in range(1, 7)] + [f"r_joint{i}" for i in range(1, 7)] + ["platform_joint", "l_finger_joint", "r_finger_joint"]

        self.HANDEL_PREGRIP_OFFSET = np.array([-0.3, -0.1, 0.0])
        self.HANDEL_GRIP_OFFSET    = np.array([-0.17, -0.1, 0.0])
        self.HANDEL_TURN_OFFSET    = self.HANDEL_GRIP_OFFSET + np.array([0.0, +0.02, -0.04])
        # 45 degree turn on the x axis
        self.HANDEL_TURN_ROTATION = np.array([[1, 0, 0],
                                              [0, np.cos(np.pi/4), -np.sin(np.pi/4)],
                                                [0, np.sin(np.pi/4), np.cos(np.pi/4)]])

        self.RIGHRT_ARM_PUSH_POSITION = np.array([-0.3, -0.55, 0.0])
        # pinocchio parameters
        self.PIN_L_ENDFECCTOR_JOINT_ID = 10
        self.PIN_R_ENDFECCTOR_JOINT_ID = 16
        self.PIN_EPS = 1e-3
        self.PIN_DT = 0.05
        self.PIN_IT_MAX = 1000
        self.PIN_DAMP = 1e-6
        self.PIN_LARM_ROTATION_OFFSET = np.array([
                                            [0,  0,  1],
                                            [0, -1,  0],
                                            [1,  0,  0]])
        self.PIN_RARM_ROTATION_OFFSET = np.array([
                                            [0,  0, 1],
                                            [0, 1,  0],
                                            [-1, 0,  0]])
        self.PIN_ARM_ROTATION_OFFSET = [self.PIN_LARM_ROTATION_OFFSET, self.PIN_RARM_ROTATION_OFFSET]
        self.PIN_GIRPPER_FRAME_NAME = ["l_gripper_base_link", "r_gripper_base_link"]
        self.PIN_JACOB_JOINT_ID = [[9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20]] # left and right arm
        self.PIN_Q_TO_JCOMMAND = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 7, 8, 9]   # arm + platform + finger 2