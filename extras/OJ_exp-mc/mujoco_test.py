import sys
sys.path.append('/Users/ojasjoshi/.mujoco/mjpro150/bin')
import mujoco_py
from os.path import dirname
model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)