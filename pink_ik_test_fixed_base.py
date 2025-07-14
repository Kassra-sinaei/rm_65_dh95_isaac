import numpy as np
import qpsolvers

import meshcat_shapes
import pink
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer
from config import Config

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
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
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
    q = pinocchio.neutral(model)
    # q[:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # base position and orientation

    pinocchio.forwardKinematics(model, data, q)
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
    while True:
        # Update task targets
        end_effector_target = tasks["l_gripper"].transform_target_to_world
        end_effector_target.translation[0] = 0.4
        # if t < 3.14 * 1.25:
        end_effector_target.translation[1] = -0.5 - 0.2 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 1.0
        end_effector_target.rotation = pinocchio.utils.rpyToMatrix(
            1.57, 0.0, 0.0)

        # Update visualization frames
        viewer["l_gripper_target"].set_transform(end_effector_target.np)
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