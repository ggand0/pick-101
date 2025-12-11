"""Apply random actions to the SO-101 arm and render."""
import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/scene.xml")
data = mujoco.MjData(model)

print(f"Actuators: {model.nu}")
print(f"Joint ranges:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    low, high = model.jnt_range[i]
    print(f"  {name}: [{low:.2f}, {high:.2f}]")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        # Random control signal within actuator range
        data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)

        # Step simulation
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        # Real-time pacing
        time.sleep(model.opt.timestep)
