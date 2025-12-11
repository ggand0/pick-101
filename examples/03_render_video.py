"""Render SO-101 arm to a video file (no display needed)."""
import numpy as np
import mujoco
import imageio

model = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/scene.xml")
data = mujoco.MjData(model)

# Create renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation parameters
duration = 5.0  # seconds
fps = 30
frames = []

print(f"Recording {duration}s at {fps}fps...")

# Simple sinusoidal motion for each joint
t = 0
while t < duration:
    # Sinusoidal control signal (each joint at different frequency)
    for i in range(model.nu):
        data.ctrl[i] = 0.5 * np.sin(2 * np.pi * (0.5 + 0.1 * i) * t)

    # Step simulation
    mujoco.mj_step(model, data)

    # Render frame at fps rate
    if int(t * fps) > len(frames):
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)

    t += model.opt.timestep

# Save video
output_path = "so101_motion.mp4"
imageio.mimsave(output_path, frames, fps=fps)
print(f"Saved to {output_path}")
