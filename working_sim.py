import pickle
import numpy as np
import torch
import open3d as o3d
from manual_sim.config import cfg
from manual_sim.state import SpringMassSystemWarp
import warp as wp
import os
from pxr import Usd, UsdGeom, Gf

# Save numpy trajectory array
def save_trajectory(trajectory, output_path='output/trajectory.npy'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, trajectory)
    print(f"Trajectory saved to {output_path}")
    print(f"Trajectory shape: {trajectory.shape}")


def generate_usd_scene(trajectory, springs, output_path='output/simulation.usda', fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(trajectory) - 1)
    
    root_xform = UsdGeom.Xform.Define(stage, '/World')
    
    # Create point cloud
    points_geom = UsdGeom.Points.Define(stage, '/World/ObjectPoints')
    points_geom.CreateDisplayColorAttr([(0.2, 0.6, 1.0)])
    widths = [0.01] * trajectory.shape[1]
    points_geom.CreateWidthsAttr(widths) # Give points a width
    
    # Provide the data (in the expected format (points, index)) to the points_geom object
    for frame_idx in range(len(trajectory)):
        points = trajectory[frame_idx]
        points_list = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]
        points_geom.GetPointsAttr().Set(points_list, frame_idx)
        
    # Create springs (creates connections between points, doesn't actually do any sprin simulation)
    if springs is not None and len(springs) > 0:
        curves_geom = UsdGeom.BasisCurves.Define(stage, '/World/Springs')
        curves_geom.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves_geom.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
        curves_geom.CreateWidthsAttr([0.002] * len(springs))
        
        curve_vertex_counts = [2] * len(springs)
        curves_geom.GetCurveVertexCountsAttr().Set(curve_vertex_counts)
        
        for frame_idx in range(len(trajectory)):
            spring_points = []
            for spring in springs:
                p1 = trajectory[frame_idx][spring[0]]
                p2 = trajectory[frame_idx][spring[1]]
                spring_points.extend([
                    Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2])),
                    Gf.Vec3f(float(p2[0]), float(p2[1]), float(p2[2]))
                ])
            curves_geom.GetPointsAttr().Set(spring_points, frame_idx)
    
    # Create round plane at Z=0 (the actual collision plane)
    plane = UsdGeom.Mesh.Define(stage, '/World/GroundPlane')
    plane_size = 2.0
    plane.CreatePointsAttr([
        Gf.Vec3f(-plane_size, -plane_size, 0),
        Gf.Vec3f(plane_size, -plane_size, 0),
        Gf.Vec3f(plane_size, plane_size, 0),
        Gf.Vec3f(-plane_size, plane_size, 0)
    ])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])  # Bright red for visibility
    plane.CreateDoubleSidedAttr(True)  # Make visible from both sides
    
    # Set USD up axis to Z
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    
    stage.Save()
    print(f"USD scene saved to {output_path}")
    print(f"Total frames: {len(trajectory)}, Total points: {trajectory.shape[1]}")

# Builds spring mesh which we iterate through to simulation the object motion
def build_spring_mesh(object_points, object_radius=0.024, object_max_neighbours=29):
    object_points = object_points.cpu().detach().numpy()
    
    # Initialize Point cloud object
    pcd = o3d.geometry.PointCloud()

    # Populate point cloud object with points
    pcd.points = o3d.utility.Vector3dVector(object_points)

    # Create tree representation of point-cloud (for efficient nearest neighbours search later I believe)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Initialize springs
    spring_flags = np.zeros((len(object_points), len(object_points)))
    springs = []
    rest_lengths = []
    
    # Iterate through point cloud
    for i in range(len(object_points)):
        # Determine nearest neighbours within a radius
        [k, idx, _] = tree.search_hybrid_vector_3d(
            object_points[i], object_radius, object_max_neighbours
        )

        # Remove self from list of neighbours
        idx = idx[1:] 
        
        # Determine length of springs at rest
        for j in idx:
            rest_length = np.linalg.norm(object_points[i] - object_points[j])

            # Checks whether spring link exists between spring i and j (and that it's not degenerate), if so, adds to list of springs
            if (spring_flags[i, j] == 0 and 
                spring_flags[j, i] == 0 and 
                rest_length > 1e-4):
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])
                rest_lengths.append(rest_length)
    
    print(f"Created {len(springs)} springs")
    
    # Convert to numpy arrays
    springs = np.array(springs, dtype=np.int32)
    rest_lengths = np.array(rest_lengths, dtype=np.float32)
    masses = np.ones(len(object_points), dtype=np.float32)
    
    return (
        torch.tensor(object_points, dtype=torch.float32),
        torch.tensor(springs, dtype=torch.int32),
        torch.tensor(rest_lengths, dtype=torch.float32),
        torch.tensor(masses, dtype=torch.float32)
    )

# Load PhysTwin data
with open('data/data/different_types/single_lift_sloth/final_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract structure points (surface + interior)
surface_points = torch.tensor(data['surface_points'], dtype=torch.float32)
interior_points = torch.tensor(data['interior_points'], dtype=torch.float32)
structure_points = torch.cat([surface_points, interior_points], dim=0)

# Lift sloth above ground for falling
structure_points[:, 2] -= 0.25  # Add 0.5m height in Y axis
structure_points[:, 1] -= 0.25  # Add 0.5m height in Y axis

# Load optimized parameters
with open('data/experiments_optimization/single_lift_sloth/optimal_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Build mesh (no controllers)
vertices, springs, rest_lengths, masses = build_spring_mesh(
    structure_points,
    object_radius=params.get('object_radius', 0.024),
    object_max_neighbours=int(params.get('object_max_neighbours', 29))
)

# Create dummy ground truth data (required but unused for falling sim)
num_frames = 30
num_gt_points = len(data['object_points'][0])
dummy_gt_points = torch.zeros((num_frames, num_gt_points, 3), dtype=torch.float32)
dummy_visibilities = torch.zeros((num_frames, num_gt_points), dtype=torch.bool)
dummy_motions_valid = torch.zeros((num_frames, num_gt_points), dtype=torch.bool)

# Create dummy controller trajectory (empty but with correct shape)
dummy_controllers = torch.zeros((num_frames, 0, 3), dtype=torch.float32)

# Create simulator from PhysTwin simulation
simulator = SpringMassSystemWarp(
    init_vertices=vertices,
    init_springs=springs,
    init_rest_lengths=rest_lengths,
    init_masses=masses,
    dt=cfg.dt,
    num_substeps=cfg.num_substeps,
    spring_Y=params.get('global_spring_Y', 7899.0),
    collide_elas=params.get('collide_elas', 0.078),
    collide_fric=params.get('collide_fric', 1.66),
    dashpot_damping=params.get('dashpot_damping', 79.3),
    drag_damping=params.get('drag_damping', 19.0),
    collide_object_elas=params.get('collide_object_elas', 0.736),
    collide_object_fric=params.get('collide_object_fric', 0.736),
    collision_dist=params.get('collision_dist', 0.019),
    num_object_points=len(structure_points),
    num_surface_points=len(surface_points),
    num_original_points=len(surface_points),
    controller_points=dummy_controllers,
    reverse_z=True,  # Normal gravity pointing down in -Z
    spring_Y_min=cfg.spring_Y_min,
    spring_Y_max=cfg.spring_Y_max,
    gt_object_points=dummy_gt_points,
    gt_object_visibilities=dummy_visibilities,
    gt_object_motions_valid=dummy_motions_valid,
    self_collision=False
)

# Set initial state
simulator.set_init_state(
    simulator.wp_init_vertices,
    simulator.wp_init_velocities
)

# Simulate falling
num_frames = 30  # ~1 seconds at 30fps
trajectory = []

# Get initial state in numpy form for comparison
initial_x = wp.to_torch(simulator.wp_states[0].wp_x).cpu().detach().numpy()
print(f"Initial position - first point: {initial_x[0]}")

# 1-1 frame to simulated step ratio, perhaps not ideal
for frame_idx in range(num_frames):
    print(f"Frame {frame_idx}/{num_frames}")
    
    # Simulator tracks collision with ground
    if simulator.object_collision_flag:
        simulator.update_collision_graph()
    
    # Step through simulator
    simulator.step()
    
    # Save trajectory
    trajectory.append(wp.to_torch(simulator.wp_states[-1].wp_x).cpu().detach().numpy().copy())  # COPY the array!
    
    # Update state for start of next frame
    simulator.set_init_state(
        simulator.wp_states[-1].wp_x,
        simulator.wp_states[-1].wp_v,
        pure_inference=True
    )

trajectory = np.array(trajectory)

# Save results
save_trajectory(trajectory, 'output/falling_trajectory.npy')
generate_usd_scene(trajectory, springs.cpu().detach().numpy(), 'output/falling_simulation.usda')

print("\nSimulation complete!")
print(f"Trajectory: output/falling_trajectory.npy")
print(f"USD Scene: output/falling_simulation.usda")