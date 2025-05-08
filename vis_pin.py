# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat
 
import sys
from pathlib import Path
 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time
# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
urdf_model_path = "./urdf/overseas_65_corrected.urdf"
mesh_dir = "./urdf"
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)

viz.loadViewerModel(color=[1.0, 1.0, 1.0, 1])
rq = pin.neutral(model)
rq[2] = 0.27
print("default position:", rq)
print("lenght of q:", len(rq))
viz.display(rq)
breakpoint()
# changeing positon and display 
for i in range(100):
    rq[2] += 0.01
    viz.display(rq)
    time.sleep(0.02)


# viz.displayVisuals(True)
 
# rm_model, rm_collision_model, rm_visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
# )
