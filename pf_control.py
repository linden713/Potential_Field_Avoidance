#!/usr/bin/env python3
"""
This script demonstrates a potential field based controller for collision-free motion 
using Isaac Lab libraries. In this modified version, only repulsive forces are computed 
for obstacle avoidance. For each joint, the repulsive force is calculated based on its 
workspace (world) position relative to the obstacle, and then mapped to joint space using 
the joint’s Jacobian matrix.

Usage:
    ./isaaclab.sh -p pf_control.py 
"""


import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher
import roboticstoolbox as rtb

# Add argparse parameters
parser = argparse.ArgumentParser(description="Potential Field Controller Demo in Isaac Lab (Repulsive Only).")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

#  Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip
panda = rtb.models.DH.Panda()
from isaacsim.core.api.objects import sphere
from isaacsim.util.debug_draw import _debug_draw

def draw_line(start, gradient, color=(1, 0, 0, 0.8), size=5.0, clear=False):
    draw = _debug_draw.acquire_debug_draw_interface()
    if clear:
        draw.clear_lines()
    start_list = [start]
    end_list = [start + gradient]
    colors = [color]
    sizes = [size]
    draw.draw_lines(start_list, end_list, colors, sizes)

# ========================= Scene =========================
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )
    
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")

# ========================= Potential Field Single-Step Update Function (Repulsive Only) =========================
def potential_field_step_repulsive(robot, current_joint_angles, target_joint_angles,
                                   obstacles, config_step=0.05, rep_strength=1000, rep_threshold=0.2, q_criteria=0.1):
    """
    Compute the repulsive force for each joint, and map it to joint space via the joint's Jacobian matrix to obtain joint updates.
    
    Parameters:
      current_joint_angles: Joint angle configuration, a numpy array of shape (n,)
      target_joint_angles: Target joint angle configuration, a numpy array of shape (n,)
      obstacles: List of obstacles; each obstacle is a dictionary that must contain at least the "center" key (a (3,) numpy array)
      config_step: Step size for configuration update
      rep_strength: Repulsive force coefficient
      rep_threshold: Distance threshold (in the same units as workspace coordinates) within which the repulsive force is active
      q_criteria: Convergence criterion (in joint space distance)
      
    Returns:
      q_next: Updated joint configuration
      is_done: Boolean flag indicating if the target is reached (True means the target has been reached)
    """
    num_joints = current_joint_angles.shape[0]
    all_poses = panda.fkine_all(current_joint_angles)
    joint_positions = all_poses.t[1:,:]  #  (7,3)
    F_rep_total = np.zeros((num_joints, 3))
    
    for i, position in enumerate(joint_positions):
        for obs in obstacles:
            # Obstacle center; ensure it is a (3,) numpy array
            obs_center = np.array(obs["center"])
            d_vec = obs_center - position 
            d = np.linalg.norm(d_vec)
            # Compute repulsive force when the joint is closer to the obstacle than the threshold
            if d < rep_threshold + obs["radius"] and d > 1e-6:
                F_rep = rep_strength * (1.0/d - 1.0/rep_threshold) * (1.0/(d**2)) * (d_vec/d)
                F_rep_total[i] += F_rep
    
    # Visualize the repulsive forces
    draw_interface = _debug_draw.acquire_debug_draw_interface()
    draw_interface.clear_lines()
    for i, position in enumerate(joint_positions):
        # Adjust the line thickness or color based on the magnitude of the force (here we simply draw it directly)
        draw_line(np.array(position), np.array(F_rep_total[i]/(rep_strength*10)), clear=False)
    
    
    # Map the repulsive forces to joint space using each joint's Jacobian matrix, and accumulate joint updates
    dq_total = np.zeros_like(current_joint_angles)
    # Retrieve the Jacobian matrices for all robot links; assume the shape is (num_links, 6, num_joints)
    # where the first 3 rows are the position Jacobian and the last 3 rows correspond to angular velocity
    jacobians_tensor = robot.root_physx_view.get_jacobians()[0]  # Get the Jacobians for the first environment/robot instance
    jacobians = jacobians_tensor.cpu().numpy()  # (num_links, 6, num_joints)
    
    for i in range(num_joints):
        J_i = jacobians[i, :3, :7] # Take the position part; shape (3, num_joints)
        # Map the repulsive force of the joint to joint space (resulting in influence on all joints)     
        tau_i = J_i.T.dot(F_rep_total[i])  #  (num_joints,)
        dq_total += tau_i 

    norm_dq = np.linalg.norm(dq_total)
    if norm_dq < 1e-6:
        dq = config_step * np.random.randn(num_joints) * 0.01
    else:
        dq = config_step * (dq_total / norm_dq)
        
    q_next = current_joint_angles + dq
    
    # Check for convergence 
    if np.linalg.norm(current_joint_angles - target_joint_angles) < q_criteria:
        is_done = True
    else:
        is_done = False
    return q_next, is_done

# ========================= Main Control Loop =========================
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
            
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    
    active_joint_ids = robot_entity_cfg.joint_ids  #  [0,1,...,6]
    
    # Goal joint position(Not used)
    target_joint_position = np.array([-0.0176, -0.1834,  0.0164, -2.0978,  0.0031,  1.9144,  0.7828])
    
    step_count = 0
    sim_dt = sim.get_physics_dt()

    obstacle = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.2, 0.2, 0.2]), #+ pose.position[0].cpu().numpy(),
        orientation=np.array([1, 0, 0, 0]),
        radius= 0.05,
        # visual_material=target_material,
    )
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    while simulation_app.is_running():
      
        obs_center , _ = obstacle.get_local_pose()
        obs_center = obs_center.detach().cpu().numpy().reshape(3,)
        
        obstacle_def = {"center": obs_center, "radius": 0.05, "k_rep": 1000}
        
        # Get current joint angles
        q_current = robot.data.joint_pos.clone().cpu().numpy().flatten() #(num_instances, num_joints).
        
        # Compute the updated joint configuration
        q_next, is_done = potential_field_step_repulsive(robot, q_current[:7], target_joint_position,
                                                           [obstacle_def])

        q_next_tensor = torch.tensor(q_next, device=sim.device, dtype=torch.float32)
        robot.set_joint_position_target(q_next_tensor, joint_ids=active_joint_ids)


        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        # step_count += 1
        if is_done:
            print(f"Converged in {step_count} steps")
            break
    else:
        print("Did not reach the goal within the allotted steps.")

# ========================= Main =========================
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete. Starting repulsive potential field control demo...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
