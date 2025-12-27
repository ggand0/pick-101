"""Mesh vertex inspection for SO-101 gripper fingertips.

Uses MuJoCo's mesh vertex API to find actual fingertip positions.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def get_mesh_vertices_world(model, data, geom_id):
    """Get all mesh vertices transformed to world coordinates."""
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        raise ValueError(f"Geom {geom_id} is not a mesh geom")

    vert_start = model.mesh_vertadr[mesh_id]
    vert_count = model.mesh_vertnum[mesh_id]
    local_verts = model.mesh_vert[vert_start:vert_start + vert_count].copy()

    # Transform: world_pos = geom_xpos + local_verts @ geom_xmat.T
    geom_xpos = data.geom_xpos[geom_id]
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
    world_verts = geom_xpos + local_verts @ geom_xmat.T

    return world_verts, local_verts


def find_extreme_vertex(world_verts, local_verts, axis, is_max=True):
    """Find the extreme vertex along an axis.

    Returns (world_pos, local_pos, index).
    """
    if is_max:
        idx = np.argmax(world_verts[:, axis])
    else:
        idx = np.argmin(world_verts[:, axis])
    return world_verts[idx], local_verts[idx], idx


def add_marker_sphere(scn, pos, rgba, size=0.005):
    """Add a colored sphere marker at position."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0, 0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    scn.ngeom += 1


def main():
    scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Initialize to a neutral pose with open gripper
    data.ctrl[5] = model.actuator_ctrlrange[5][1]  # Open gripper
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Finger geom IDs from XML
    # Geom 27, 28: static finger (on gripper body / wrist_roll_follower)
    # Geom 29, 30: moving finger (on moving_jaw_so101_v1)
    static_geom_ids = [27, 28]
    moving_geom_ids = [29, 30]

    print("=== Finger Geom Info ===")
    for gid in static_geom_ids + moving_geom_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        mesh_id = model.geom_dataid[gid]
        vert_count = model.mesh_vertnum[mesh_id] if mesh_id >= 0 else 0
        print(f"  Geom {gid}: {name}, mesh_id={mesh_id}, vertices={vert_count}")

    print("\n=== Finding Fingertip Vertices ===")

    # Find extreme vertices for each finger geom
    fingertip_candidates = []

    for gid in static_geom_ids:
        world_verts, local_verts = get_mesh_vertices_world(model, data, gid)

        # Static finger: try all axes
        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            for is_max, dir_name in [(True, 'max'), (False, 'min')]:
                world_pos, local_pos, idx = find_extreme_vertex(world_verts, local_verts, axis, is_max)
                fingertip_candidates.append({
                    'geom_id': gid,
                    'axis': axis_name,
                    'direction': dir_name,
                    'world_pos': world_pos,
                    'local_pos': local_pos,
                    'type': 'static'
                })
                print(f"  Geom {gid} {axis_name}-{dir_name}: world={world_pos}, local={local_pos}")

    for gid in moving_geom_ids:
        world_verts, local_verts = get_mesh_vertices_world(model, data, gid)

        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            for is_max, dir_name in [(True, 'max'), (False, 'min')]:
                world_pos, local_pos, idx = find_extreme_vertex(world_verts, local_verts, axis, is_max)
                fingertip_candidates.append({
                    'geom_id': gid,
                    'axis': axis_name,
                    'direction': dir_name,
                    'world_pos': world_pos,
                    'local_pos': local_pos,
                    'type': 'moving'
                })
                print(f"  Geom {gid} {axis_name}-{dir_name}: world={world_pos}, local={local_pos}")

    # For fingertips, we typically want the points that are:
    # - Lowest Z (closest to table)
    # - Closest together (opposing tips)

    print("\n=== Potential Fingertip Pairs (lowest Z) ===")
    static_lowest = min([c for c in fingertip_candidates if c['type'] == 'static'],
                        key=lambda x: x['world_pos'][2])
    moving_lowest = min([c for c in fingertip_candidates if c['type'] == 'moving'],
                        key=lambda x: x['world_pos'][2])

    print(f"Static lowest Z: geom {static_lowest['geom_id']} {static_lowest['axis']}-{static_lowest['direction']}")
    print(f"  World: {static_lowest['world_pos']}")
    print(f"  Local: {static_lowest['local_pos']}")

    print(f"Moving lowest Z: geom {moving_lowest['geom_id']} {moving_lowest['axis']}-{moving_lowest['direction']}")
    print(f"  World: {moving_lowest['world_pos']}")
    print(f"  Local: {moving_lowest['local_pos']}")

    midpoint = (static_lowest['world_pos'] + moving_lowest['world_pos']) / 2
    print(f"Midpoint: {midpoint}")

    # Launch viewer to visualize
    print("\n=== Launching Viewer ===")
    print("Red = static finger extremes, Blue = moving finger extremes")
    print("Green = midpoint of lowest-Z vertices")
    print("Yellow = graspframe site for comparison")

    graspframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            viewer.user_scn.ngeom = 0

            # Show all static finger extremes (red)
            for c in fingertip_candidates:
                if c['type'] == 'static':
                    add_marker_sphere(viewer.user_scn, c['world_pos'], [1, 0, 0, 0.5], size=0.003)

            # Show all moving finger extremes (blue)
            for c in fingertip_candidates:
                if c['type'] == 'moving':
                    add_marker_sphere(viewer.user_scn, c['world_pos'], [0, 0, 1, 0.5], size=0.003)

            # Show lowest-Z vertices as larger spheres
            add_marker_sphere(viewer.user_scn, static_lowest['world_pos'], [1, 0, 0, 1], size=0.006)
            add_marker_sphere(viewer.user_scn, moving_lowest['world_pos'], [0, 0, 1, 1], size=0.006)

            # Update midpoint dynamically
            world_static, local_static = get_mesh_vertices_world(model, data, static_lowest['geom_id'])
            world_moving, local_moving = get_mesh_vertices_world(model, data, moving_lowest['geom_id'])

            static_tip = world_static[np.argmin(world_static[:, 2])]
            moving_tip = world_moving[np.argmin(world_moving[:, 2])]
            current_midpoint = (static_tip + moving_tip) / 2

            add_marker_sphere(viewer.user_scn, current_midpoint, [0, 1, 0, 1], size=0.008)

            # Show graspframe for comparison
            graspframe_pos = data.site_xpos[graspframe_id]
            add_marker_sphere(viewer.user_scn, graspframe_pos, [1, 1, 0, 1], size=0.005)

            viewer.sync()
            time.sleep(0.02)


def find_closest_opposing_vertices(model, data, static_gid, moving_gid):
    """Find vertices on each finger that are closest to each other (opposing tips)."""
    static_world, static_local = get_mesh_vertices_world(model, data, static_gid)
    moving_world, moving_local = get_mesh_vertices_world(model, data, moving_gid)

    # For efficiency, sample every 10th vertex
    step = 10
    static_sample = static_world[::step]
    moving_sample = moving_world[::step]

    # Find pair with minimum distance
    min_dist = float('inf')
    best_static_idx = 0
    best_moving_idx = 0

    for i, sv in enumerate(static_sample):
        dists = np.linalg.norm(moving_sample - sv, axis=1)
        j = np.argmin(dists)
        if dists[j] < min_dist:
            min_dist = dists[j]
            best_static_idx = i * step
            best_moving_idx = j * step

    return (static_world[best_static_idx], static_local[best_static_idx],
            moving_world[best_moving_idx], moving_local[best_moving_idx], min_dist)


if __name__ == "__main__":
    import sys
    if "--no-viewer" in sys.argv:
        scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)

        # Set arm to top-down grasp pose with closed gripper
        print("=== Testing Top-Down Grasp Pose (gripper closed) ===")
        data.qpos[1] = 0.5   # shoulder_lift
        data.qpos[2] = 1.0   # elbow
        data.qpos[3] = 1.65  # wrist_flex pointing down
        data.qpos[4] = np.pi / 2  # wrist_roll horizontal fingers
        data.ctrl[1] = 0.5
        data.ctrl[2] = 1.0
        data.ctrl[3] = 1.65
        data.ctrl[4] = np.pi / 2
        data.ctrl[5] = -0.5  # gripper partially closed
        for _ in range(200):
            mujoco.mj_step(model, data)

        static_geom_ids = [27, 28]
        moving_geom_ids = [29, 30]

        print("\n=== Geom Info ===")
        for gid in static_geom_ids + moving_geom_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            mesh_id = model.geom_dataid[gid]
            body_id = model.geom_bodyid[gid]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  Geom {gid}: mesh_id={mesh_id}, body={body_name}")

        print("\n=== Finding Opposing Fingertip Vertices ===")
        static_w, static_l, moving_w, moving_l, dist = find_closest_opposing_vertices(
            model, data, 27, 29)
        print(f"Static finger tip:")
        print(f"  World: {static_w}")
        print(f"  Local (mesh): {static_l}")
        print(f"Moving finger tip:")
        print(f"  World: {moving_w}")
        print(f"  Local (mesh): {moving_l}")
        print(f"Distance: {dist:.4f}m")
        print(f"Midpoint (world): {(static_w + moving_w) / 2}")

        # Find fingertips using MESH-LOCAL coordinates (orientation-independent)
        # From STL analysis:
        # - Static finger (wrist_roll_follower): tip at max local Z (+0.105m)
        # - Moving jaw: tip at min local Y (-0.082m)
        print("\n=== Fingertip Vertices (mesh-local extremes) ===")
        fingertip_positions = {}

        # Static finger: max Z in mesh-local
        gid = 27
        world_verts, local_verts = get_mesh_vertices_world(model, data, gid)
        # Find top 1% by local Z
        z_local = local_verts[:, 2]
        threshold = np.percentile(z_local, 99)
        tip_mask = z_local >= threshold
        tip_world = world_verts[tip_mask]
        tip_local = local_verts[tip_mask]
        centroid_world = np.mean(tip_world, axis=0)
        centroid_local = np.mean(tip_local, axis=0)
        print(f"Geom {gid} (static) - max local Z:")
        print(f"  Num tip vertices: {np.sum(tip_mask)}")
        print(f"  Mesh-local centroid: {centroid_local}")
        print(f"  World centroid: {centroid_world}")
        fingertip_positions['static'] = {'world': centroid_world, 'local': centroid_local, 'geom_id': gid}

        # Moving jaw: find vertices closest to static fingertip (opposing surface)
        gid = 29
        world_verts, local_verts = get_mesh_vertices_world(model, data, gid)
        static_tip = fingertip_positions['static']['world']

        # Find vertices within 5mm of the closest point to static tip
        dists = np.linalg.norm(world_verts - static_tip, axis=1)
        min_dist = np.min(dists)
        tip_mask = dists <= min_dist + 0.005  # within 5mm of closest
        tip_world = world_verts[tip_mask]
        tip_local = local_verts[tip_mask]
        centroid_world = np.mean(tip_world, axis=0)
        centroid_local = np.mean(tip_local, axis=0)
        print(f"Geom {gid} (moving) - closest to static tip:")
        print(f"  Min distance to static: {min_dist:.4f}m")
        print(f"  Num tip vertices: {np.sum(tip_mask)}")
        print(f"  Mesh-local centroid: {centroid_local}")
        print(f"  World centroid: {centroid_world}")
        fingertip_positions['moving'] = {'world': centroid_world, 'local': centroid_local, 'geom_id': gid}

        # Get graspframe for comparison
        graspframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")
        print(f"\nGraspframe site (for comparison): {data.site_xpos[graspframe_id]}")

        # Now compute the LOCAL coordinates for sites using fingertip centroids
        print("\n=== Computing Site Positions (body-local) from Centroids ===")

        # Get body transforms
        gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        moving_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

        gripper_xpos = data.xpos[gripper_body_id]
        gripper_xmat = data.xmat[gripper_body_id].reshape(3, 3)
        moving_jaw_xpos = data.xpos[moving_jaw_id]
        moving_jaw_xmat = data.xmat[moving_jaw_id].reshape(3, 3)

        # Transform world to body-local: local = (world - body_xpos) @ inv(body_xmat.T)
        static_tip_world = fingertip_positions['static']['world']
        moving_tip_world = fingertip_positions['moving']['world']

        static_tip_body_local = (static_tip_world - gripper_xpos) @ gripper_xmat
        moving_tip_body_local = (moving_tip_world - moving_jaw_xpos) @ moving_jaw_xmat

        print(f"Static fingertip site pos (gripper body frame): {static_tip_body_local}")
        print(f"Moving fingertip site pos (moving_jaw body frame): {moving_tip_body_local}")

        # Show midpoint
        mid_world = (static_tip_world + moving_tip_world) / 2
        print(f"\nFingertip midpoint (world): {mid_world}")
        print(f"Graspframe site (world): {data.site_xpos[graspframe_id]}")
        print(f"Difference: {mid_world - data.site_xpos[graspframe_id]}")

        # Verify the actual site positions
        print("\n=== Verifying Fingertip Site Positions ===")
        static_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")
        moving_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "moving_fingertip")

        static_site_world = data.site_xpos[static_tip_site_id]
        moving_site_world = data.site_xpos[moving_tip_site_id]
        site_mid = (static_site_world + moving_site_world) / 2

        print(f"Static site (world): {static_site_world}")
        print(f"Moving site (world): {moving_site_world}")
        print(f"Site midpoint (world): {site_mid}")
        print(f"Expected (vertex centroid) static: {static_tip_world}")
        print(f"Expected (vertex centroid) moving: {moving_tip_world}")
        print(f"Static site error: {np.linalg.norm(static_site_world - static_tip_world):.4f}m")
        print(f"Moving site error: {np.linalg.norm(moving_site_world - moving_tip_world):.4f}m")

        # Test midpoint at different gripper openings
        print("\n=== Midpoint at Different Gripper Openings ===")
        gripper_range = model.actuator_ctrlrange[5]
        print(f"Gripper ctrl range: [{gripper_range[0]:.2f}, {gripper_range[1]:.2f}]")

        for gripper_ctrl in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            # Reset to grasp pose
            data.qpos[1] = 0.5
            data.qpos[2] = 1.0
            data.qpos[3] = 1.65
            data.qpos[4] = np.pi / 2
            data.ctrl[1] = 0.5
            data.ctrl[2] = 1.0
            data.ctrl[3] = 1.65
            data.ctrl[4] = np.pi / 2
            data.ctrl[5] = gripper_ctrl
            for _ in range(200):
                mujoco.mj_step(model, data)

            static_site = data.site_xpos[static_tip_site_id].copy()
            moving_site = data.site_xpos[moving_tip_site_id].copy()
            mid = (static_site + moving_site) / 2
            gap = np.linalg.norm(static_site - moving_site)
            gripper_qpos = data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]]

            print(f"ctrl={gripper_ctrl:+.1f}: gap={gap:.3f}m, mid_Z={mid[2]:.4f}, gripper_qpos={gripper_qpos:.3f}")

    else:
        main()
