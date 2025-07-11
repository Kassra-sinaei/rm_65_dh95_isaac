import numpy as np
import qpsolvers

import meshcat_shapes
import pink
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer
from config import Config
from utils.trajectory import create_waypoint_trajectory, generate_waypoint_trajectory, create_arc_trajectory, create_circular_trajectory

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc


if __name__ == "__main__":
    config = Config()
    # URDF & Pinocchio setup
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        config.URDFPATH, config.MESH_DIR
    )
    data = model.createData()
    tasks = {
        # 'base': FrameTask(config.PIN_BASE_FRAME_NAME, position_cost=10.0, orientation_cost=10.0),
        # 'platform': FrameTask(config.PIN_PLATFORM_FRAME_NAME, position_cost=10.0, orientation_cost=10.0),
        # 'r_gripper':FrameTask(config.PIN_GIRPPER_FRAME_NAME[1], position_cost=1.0, orientation_cost=1.0),
        'l_gripper':FrameTask(config.PIN_GIRPPER_FRAME_NAME[0], position_cost=1.0, orientation_cost=1.0)
    }

    # initial config & visualizer
    q = pin.neutral(model)
    # q[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # base position and orientation

    pin.forwardKinematics(model, data, q)
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1.0])
    # self.viz.displayFrames(True)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["l_gripper_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["l_gripper"], opacity=1.0)
    meshcat_shapes.frame(viewer["r_gripper_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["r_gripper"], opacity=1.0)
    meshcat_shapes.frame(viewer["base_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["base"], opacity=1.0)
    
    print(f"model: {model}")
    
    configuration = pink.Configuration(model, data, np.array(q))
    for task in tasks.values():
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    initial_pose = configuration.get_transform_frame_to_world(
                tasks["l_gripper"].frame
            )
    final_pose = pin.SE3(pin.utils.rpyToMatrix(1.57, 0.0, 0.0), np.array([0.2, -0.5, 1.4]))
    mid_pose = pin.SE3(pin.utils.rpyToMatrix(1.57, 0.0, 0.0), np.array([0.4, -0.5, 1.0]))
    mid_pose_2 = pin.SE3(pin.utils.rpyToMatrix(1.57, 0.0, 0.0), np.array([0.4, -0.5, 1.4]))
    # Create a trajectory
    waypoints = create_waypoint_trajectory(
        initial_pose,
        final_pose,
        [mid_pose, mid_pose_2]
    )
    trajectory_duration = 5.0 * len(waypoints)
    while True:
        # Update task targets
        current_target_pose = generate_waypoint_trajectory(
            waypoints,
            trajectory_duration,
            t
        )
        tasks["l_gripper"].transform_target_to_world = current_target_pose

        # Update visualization frames
        viewer["l_gripper_target"].set_transform(current_target_pose.np)
        viewer["l_gripper"].set_transform(
            configuration.get_transform_frame_to_world(
                tasks["l_gripper"].frame
            ).np
        )
        # viewer["r_gripper_target"].set_transform(end_effector_target.np)
        # viewer["r_gripper"].set_transform(
        #     configuration.get_transform_frame_to_world(
        #         tasks["r_gripper"].frame
        #     ).np
        # )
        # viewer["base_target"].set_transform(
        #     tasks["base"].transform_target_to_world.np
        # )
        # viewer["base"].set_transform(
        #     configuration.get_transform_frame_to_world(
        #         tasks["base"].frame
        #     ).np
        # )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
        # print(f"state: {configuration.q}")