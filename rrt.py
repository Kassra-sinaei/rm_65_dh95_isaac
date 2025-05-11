from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors
from utils.datastructures.storage import Storage
from utils.datastructures.pathtree import PathTree
from utils.datastructures.mtree import MTree
from tp4.collision_wrapper import CollisionWrapper

import numpy as np
import pinocchio as pin
import time
from tqdm import tqdm
from pinocchio.robot_wrapper import RobotWrapper
import hppfcl

from pink import solve_ik, Configuration
from pink.tasks import FrameTask

class System():

    def __init__(self, robot):
        self.robot = robot
        robot.gmodel = robot.collision_model
        self.display_edge_count = 0
        self.colwrap = CollisionWrapper(robot)  # For collision checking
        self.nq = self.robot.nq
        self.display_count = 0
        self.viz = MeshcatVisualizer(robot, "robot", colors.red)
        
    def distance(self, q1, q2):
        """
        Must return a distance between q1 and q2 which can be a batch of config.
        """
        if len(q2.shape) > len(q1.shape):
            q1 = q1[None, ...]
        e = np.mod(np.abs(q1 - q2), 2 * np.pi)
        e[e > np.pi] = 2 * np.pi - e[e > np.pi]
        return np.linalg.norm(e, axis=-1)

    def random_config(self, free=True):
        """
        Must return a random configuration which is not in collision if free=True
        """
        q = 2 * np.pi * np.random.rand(15) - np.pi
        if not free:
            return q
        while self.is_colliding(q):
            q = 2 * np.pi * np.random.rand(15) - np.pi
        return q

    def is_colliding(self, q):
        """
        Use CollisionWrapper to decide if a configuration is in collision
        """
        self.colwrap.computeCollisions(q)
        collisions = self.colwrap.getCollisionList()
        return (len(collisions) > 0)

    def get_path(self, q1, q2, l_min=None, l_max=None, eps=0.2):
        """
        generate a continuous path with precision eps between q1 and q2
        If l_min of l_max is mention, extrapolate or cut the path such
        that 
        """
        q1 = np.mod(q1 + np.pi, 2 * np.pi) - np.pi
        q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi

        diff = q2 - q1
        query = np.abs(diff) > np.pi
        q2[query] = q2[query] - np.sign(diff[query]) * 2 * np.pi

        d = self.distance(q1, q2)
        if d < eps:
            return np.stack([q1, q2], axis=0)
        
        if l_min is not None or l_max is not None:
            new_d = np.clip(d, l_min, l_max)
        else:
            new_d = d
            
        N = int(new_d / eps + 2)

        return np.linspace(q1, q1 + (q2 - q1) * new_d / d, N)
        
    def is_free_path(self, q1, q2, l_min=0.2, l_max=1., eps=0.2):
        """
        Create a path and check collision to return the last
         non-colliding configuration. Return X, q where X is a boolean
        who state is the steering has work.
        We require at least l_min must be cover without collision to validate the path.
        """
        q_path = self.get_path(q1, q2, l_min, l_max, eps)
        N = len(q_path)
        N_min = N - 1 if l_min is None else min(N - 1, int(l_min / eps))
        for i in range(N):
            if self.is_colliding(q_path[i]):
                break
        if i < N_min:
            return False, None
        if i == N - 1:
            return True, q_path[-1]
        return True, q_path[i - 1]

    def reset(self):
        """
        Reset the system visualization
        """
        for i in range(self.display_count):
            self.viz.delete(f"world/sph{i}")
            self.viz.delete(f"world/cil{i}")
        self.display_count = 0
    
    def display_edge(self, q1, q2, radius=0.01, color=[1.,0.,0.,1]):
        M1 = self.robot.framePlacement(q1, 22)  # Placement of the end effector tip.
        M2 = self.robot.framePlacement(q2, 22)  # Placement of the end effector tip.
        middle = .5 * (M1.translation + M2.translation)
        direction = M2.translation - M1.translation
        length = np.linalg.norm(direction)
        dire = direction / length
        orth = np.cross(dire, np.array([0, 0, 1]))
        orth /= np.linalg.norm(orth)
        orth2 = np.cross(dire, orth)
        orth2 /= np.linalg.norm(orth2)
        Mcyl = pin.SE3(np.stack([orth2, dire, orth], axis=1), middle)
        name = f"world/sph{self.display_count}"
        self.viz.addSphere(name, radius, [1.,0.,0.,1])
        self.viz.applyConfiguration(name,M2)
        name = f"world/cil{self.display_count}"
        self.viz.addCylinder(name, length, radius / 4, [0., 1., 0., 1])
        self.viz.applyConfiguration(name,Mcyl)
        self.display_count +=1
        
    def display_motion(self, qs, step=1e-1):
        # Given a point path display the smooth movement
        for i in range(len(qs) - 1):
            for q in self.get_path(qs[i], qs[i+1])[:-1]:
                self.viz.display(q)
                time.sleep(step)
        self.viz.display(qs[-1])

    def draw_init_goal(self, q1, q2):
        """
        Draw the initial and goal configuration
        """

        # Add a small blue sphere at the start configuration
        # Add a small blue box at the start configuration
        start_name = "world/start_marker"
        self.viz.addBox(start_name, [0.1, 0.1, 0.1], [0., 0., 1., 1.])  # Blue color
        self.viz.applyConfiguration(start_name, q1)

        # Add a small green box at the goal configuration
        goal_name = "world/goal_marker"
        self.viz.addBox(goal_name, [0.1, 0.1, 0.1], [0., 1., 0., 1.])  # Green color
        self.viz.applyConfiguration(goal_name, q2)


class RRT():
    """
    Can be splited into RRT base because different rrt
    have factorisable logic
    """
    def __init__(
        self,
        system,
        node_max=500000,
        iter_max=1000000,
        N_bias=10,
        l_min=.2,
        l_max=.5,
        steer_delta=.1,
    ):
        """
        [Here, in proper code, we would document the parameters of our function. Do that below,
        using the Google style for docstrings.]
        https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

        Args:
            node_max: ...
            iter_max: ...
            ...
        """
        self.system = system
        # params
        self.l_max = l_max
        self.l_min = l_min
        self.N_bias = N_bias
        self.node_max = node_max
        self.iter_max = iter_max
        self.steer_delta = steer_delta
        # intern
        self.NNtree = None
        self.storage = None
        self.pathtree = None
        # The distance function will be called on N, dim object
        self.real_distance = self.system.distance
        # Internal for computational_opti in calculating distance
        self._candidate = None
        self._goal = None
        self._cached_dist_to_candidate = {}
        self._cached_dist_to_goal = {}

    def distance(self, q1_idx, q2_idx):
        if isinstance(q2_idx, int):
            if q1_idx == q2_idx:
                return 0.
            if q1_idx == -1 or q2_idx == -1:
                if q2_idx == -1:
                    q1_idx, q2_idx = q2_idx, q1_idx
                if q2_idx not in self._cached_dist_to_candidate:
                    self._cached_dist_to_candidate[q2_idx] = self.real_distance(
                        self._candidate, self.storage[q2_idx]
                    )
                return self._cached_dist_to_candidate[q2_idx]
            if q1_idx == -2 or q2_idx == -2:
                if q2_idx == -2:
                    q1_idx, q2_idx = q2_idx, q1_idx
                if q2_idx not in self._cached_dist_to_goal:
                    self._cached_dist_to_goal[q2_idx] = self.real_distance(
                        self._goal, self.storage[q2_idx]
                    )
                return self._cached_dist_to_goal[q2_idx]
            return self.real_distance(self.storage[q1_idx], self.storage[q2_idx])
        if q1_idx == -1:
            q = self._candidate
        elif q1_idx == -2:
            q = self._goal
        else:
            q = self.storage[q1_idx]
        return self.real_distance(q, self.storage[q2_idx])

    def new_candidate(self):
        q = self.system.random_config(free=True)
        self._candidate = q
        self._cached_dist_to_candidate = {}
        return q

    def solve(self, qi, validate, qg=None):
        self.system.reset()
        self._goal = qg
        
        # Reset internal datastructures
        self.storage = Storage(self.node_max, self.system.nq)
        self.pathtree = PathTree(self.storage)
        self.NNtree = MTree(self.distance)
        qi_idx = self.storage.add_point(qi)
        self.NNtree.add_point(qi_idx)
        self.it_trace = []

        found = False
        iterator = range(self.iter_max)
        for i in tqdm(iterator):
            # New candidate
            if i % self.N_bias == 0:
                q_new = self._goal
                q_new_idx = -2
            else:
                q_new = self.new_candidate()
                q_new_idx = -1

            # Find closest neighboor to q_new
            q_near_idx, d = self.NNtree.nearest_neighbour(q_new_idx)
            
            # Steer from it toward the new checking for colision
            success, q_prox = self.system.is_free_path(
                self.storage.data[q_near_idx],
                q_new,
                l_min=self.l_min,
                l_max=self.l_max,
                eps=self.steer_delta
            )

            if not success:
                self.it_trace.append(0)
                continue
            self.it_trace.append(1)
            
            # Add the points in data structures
            q_prox_idx = self.storage.add_point(q_prox)
            self.NNtree.add_point(q_prox_idx)
            self.pathtree.update_link(q_prox_idx, q_near_idx)
            self.system.display_edge(self.storage[q_near_idx], self.storage[q_prox_idx])

            # Test if it reach the goal
            if validate(q_prox, qg):
                q_g_idx = self.storage.add_point(q_prox)
                self.NNtree.add_point(q_g_idx)
                self.pathtree.update_link(q_g_idx, q_prox_idx)
                found = True
                break
        self.iter_done = i + 1
        self.found = found
        return found

    def get_path(self, q_g):
        assert self.found
        path = self.pathtree.get_path()
        return np.concatenate([path, q_g[None, :]])

def addCylinderToUniverse(name, radius, length, placement, color=colors.red):
    geom = pin.GeometryObject(
        name,
        0,
        hppfcl.Cylinder(radius, length),
        placement
    )
    new_id = collision_model.addGeometryObject(geom)
    geom.meshColor = np.array(color)
    visual_model.addGeometryObject(geom)
    
    for link_id in range(robot.model.nq):
        collision_model.addCollisionPair(
            pin.CollisionPair(link_id, new_id)
        )
    return geom


def addBoxToUniverse(name, size, placement, color=colors.red):
    geom = pin.GeometryObject(
        name,
        0,
        hppfcl.Box(*size),
        placement
    )
    new_id = collision_model.addGeometryObject(geom)
    geom.meshColor = np.array(color)
    visual_model.addGeometryObject(geom)
    
    for link_id in range(robot.model.nq):
        collision_model.addCollisionPair(
            pin.CollisionPair(link_id, new_id)
        )
    return geom

def validation(key, q_g):
        vec = robot.framePlacement(key, 22).translation - robot.framePlacement(q_g, 22).translation
        return (float(np.linalg.norm(vec)) < eps_final)


if __name__ == "__main__":
    urdf_path = "/home/kasra/alphaz/rm_ws/src/door_fsm/description/rm_model.urdf"
    mesh_dir = "/home/kasra/alphaz/rm_ws/src/door_fsm/description/meshes"

    robot = RobotWrapper.BuildFromURDF(
            urdf_path,
            [mesh_dir]
        )
    model = pin.buildModelFromUrdf(
            urdf_path,
        )
    data = model.createData()
    # Print all robot joint names in order
    print("Robot joint names in order:")
    for i in range(1, model.njoints):
        joint_name = model.names[i]
        print(f"{i}: {joint_name}")
    collision_model = robot.collision_model
    visual_model = robot.visual_model

    system = System(robot)
    rrt = RRT(
        system,
        N_bias=20,
        l_min=0.2,
        l_max=0.5,
        steer_delta=0.1,
    )
    eps_final = .1
    
    r_goal_p = np.array([0.4, -0.3, 1.5])
    r_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()
    l_goal_p = np.array([0.4, 0.3, 1.5])
    l_goal_q = pin.Quaternion(1, 0, 0, 0).normalized().matrix()
    max_iter=20000
    eps=5e-2
    q_i = pin.neutral(model)
    dt = 1e-1
    configuration = Configuration(robot.model, data, q_i)
    tasks = {
        'r_gripper':FrameTask('r_gripper_frame', position_cost=1.0, orientation_cost=1.0),
        'l_gripper':FrameTask('l_gripper_frame', position_cost=1.0, orientation_cost=1.0)
    }
    configuration.q = q_i

    tasks['r_gripper'].set_target(pin.SE3(r_goal_q, r_goal_p))
    tasks['l_gripper'].set_target(pin.SE3(l_goal_q, l_goal_p))

    for t in np.arange(0.0, 2.0, dt):
        velocity = solve_ik(configuration, tasks.values(), dt, solver="quadprog")
        configuration.integrate_inplace(velocity, dt)

    rrt.solve(q_i, validation, qg=configuration.q)
    system.display_motion(rrt.get_path(configuration.q))
    system.draw_init_goal(r_goal_p, l_goal_p)