"""Generate a hollow cylinder bowl mesh as STL."""
import numpy as np
import struct
from pathlib import Path


def generate_hollow_cylinder(
    outer_radius: float,
    inner_radius: float,
    height: float,
    num_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces for a hollow cylinder.

    Returns vertices (N, 3) and faces (M, 3) arrays.
    """
    vertices = []
    faces = []

    # Generate points around the circles
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # Bottom outer ring (z=0)
    for a in angles:
        vertices.append([outer_radius * np.cos(a), outer_radius * np.sin(a), 0])
    # Bottom inner ring (z=0)
    for a in angles:
        vertices.append([inner_radius * np.cos(a), inner_radius * np.sin(a), 0])
    # Top outer ring (z=height)
    for a in angles:
        vertices.append([outer_radius * np.cos(a), outer_radius * np.sin(a), height])
    # Top inner ring (z=height)
    for a in angles:
        vertices.append([inner_radius * np.cos(a), inner_radius * np.sin(a), height])

    vertices = np.array(vertices)
    n = num_segments

    # Indices for each ring
    bo = np.arange(0, n)           # bottom outer
    bi = np.arange(n, 2*n)         # bottom inner
    to = np.arange(2*n, 3*n)       # top outer
    ti = np.arange(3*n, 4*n)       # top inner

    # Generate faces (triangles)
    for i in range(n):
        j = (i + 1) % n

        # Outer wall (2 triangles per segment)
        faces.append([bo[i], bo[j], to[j]])
        faces.append([bo[i], to[j], to[i]])

        # Inner wall (2 triangles per segment, reversed winding)
        faces.append([bi[i], ti[i], ti[j]])
        faces.append([bi[i], ti[j], bi[j]])

        # Bottom ring (between outer and inner)
        faces.append([bo[i], bi[i], bi[j]])
        faces.append([bo[i], bi[j], bo[j]])

        # Top ring (between outer and inner)
        faces.append([to[i], to[j], ti[j]])
        faces.append([to[i], ti[j], ti[i]])

    # Bottom cap (fill inner circle)
    center_bottom = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])
    for i in range(n):
        j = (i + 1) % n
        faces.append([center_bottom, bi[j], bi[i]])

    faces = np.array(faces)
    return vertices, faces


def write_stl_binary(filepath: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write mesh to binary STL file."""
    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        f.write(b'\0' * 80)
        # Number of triangles
        f.write(struct.pack('<I', len(faces)))

        for face in faces:
            v0, v1, v2 = vertices[face]
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])

            # Write normal
            f.write(struct.pack('<3f', *normal))
            # Write vertices
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            # Attribute byte count
            f.write(struct.pack('<H', 0))


def main():
    # Bowl dimensions (matching your real bowl)
    # 15cm diameter opening = 7.5cm outer radius
    # Wall thickness ~0.5cm
    # Height 6cm
    outer_radius = 0.075  # 7.5cm
    inner_radius = 0.070  # 7.0cm (0.5cm wall)
    height = 0.06         # 6cm

    vertices, faces = generate_hollow_cylinder(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        height=height,
        num_segments=32,
    )

    # Output path
    output_dir = Path(__file__).parent.parent / "SO-ARM100/Simulation/SO101/assets"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "bowl.stl"

    write_stl_binary(output_path, vertices, faces)
    print(f"Generated bowl mesh: {output_path}")
    print(f"  Outer radius: {outer_radius*100:.1f}cm")
    print(f"  Inner radius: {inner_radius*100:.1f}cm")
    print(f"  Height: {height*100:.1f}cm")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")


if __name__ == "__main__":
    main()
