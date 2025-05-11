import numpy as np

from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
class RealmanState:
    def __init__(self, config):
        # state buffers x, y, z, rx, ry, rz, rw, platform, head 2, l 6, r 6. Order is made to match the pinocchio model
        self.state = np.zeros(22) 



    def update_state(self, state):
        self.state = state

    def update_joint_state(self, jstate_msg):
        self.state[7] = jstate_msg.position[0]
        pos_map = dict(zip(jstate_msg.name, jstate_msg.position))
        for side, base in (('l', 10), ('r', 16)):
            self.state[base:base + 6] = [pos_map[f"{side}_joint{i}"] for i in range(1, 7)]


    def update_base_pose(self, tf_msg):
        for tf in tf_msg.transforms:
            t, r = tf.transform.translation, tf.transform.rotation
            self.state[:7] = np.array([
                t.x, t.y, t.z,
                r.x, r.y, r.z, r.w
            ])
