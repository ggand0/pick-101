"""View the SO-101 arm in MuJoCo's interactive viewer."""
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/scene.xml")
data = mujoco.MjData(model)

print("Controls:")
print("  - Click and drag to rotate view")
print("  - Scroll to zoom")
print("  - Double-click on joints to apply forces")
print("  - Press ESC to quit")

mujoco.viewer.launch(model, data)
