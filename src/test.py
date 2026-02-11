import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import A2C
from rich.console import Console
from rich.panel import Panel
import os

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position

console = Console()

def create_test_environment(scenario, origin, config):
    """Create environment for visualization"""
    box_size = scenario["box_size"]

    # Create UAVs
    uavs = []

    for id, params in config["test"]["missions"].items():
        uav = FixedWingAircraft(
            id_tag = id,
            initial_position = Position(
                params["initial_position"][0], 
                params["initial_position"][1]
            ),
            initial_heading = params["initial_heading"],
            cruise_speed = params["cruise_speed"],
            turning_radius = params["turning_radius"],
            mission = list(params["waypoints"]),
        )
        uavs.append(uav)
    
    # Create bounding box
    lat_off = (box_size / 2.0) / 111_320.0
    lon_off = (box_size / 2.0) / (111_320.0 * np.cos(np.radians(origin[0])))
    
    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)
    
    env = MultiUAVEnv(
        uavs, 
        tl=tl, 
        br=br, 
        dt=0.05, 
        mode=config["test"]["mode"]
    )
    return env, tl, br


def run_and_record_episode(model, env, transformer, max_steps):
    """Run episode and record all positions and waypoints"""
    obs, _ = env.reset()
    
    # Record data for each UAV
    uav_data = []
    for i, ac in enumerate(env.aircraft_list):
        uav_data.append({
            'id': ac.id_tag,
            'positions': [],
            'waypoints_visited': [],
            'waypoints_positions': [],
            'all_waypoints': []  # All waypoints that existed during episode
        })
    
    done = False
    truncated = False
    step = 0
    total_reward = 0
    
    while not (done or truncated) and step < max_steps:
        # Record current positions
        for i, ac in enumerate(env.aircraft_list):
            pos = ac.position.to_tuple()
            x, y = transformer.geo_to_local(pos[0], pos[1])
            uav_data[i]['positions'].append((x, y, ac.heading))
            
            # Record current waypoint if it exists
            if ac.waypoint_manager.current_waypoint:
                wp = ac.waypoint_manager.current_waypoint.to_tuple()
                wp_x, wp_y = transformer.geo_to_local(wp[0], wp[1])
                if (wp_x, wp_y) not in uav_data[i]['all_waypoints']:
                    uav_data[i]['all_waypoints'].append((wp_x, wp_y))
        
        # Take action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
    
    return uav_data, step, total_reward, info["waypoints_hit"]


def visualize_episode(uav_data, tl, br, transformer, scenario_name, 
                      steps, total_reward, arrivals, save_path):
    """Create visualization of flight paths"""
    
    # Get bounds in local coordinates
    tl_x, tl_y = transformer.geo_to_local(tl[0], tl[1])
    br_x, br_y = transformer.geo_to_local(br[0], br[1])
    
    # Determine number of UAVs
    num_uavs = len(uav_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw boundary box
    width = br_x - tl_x
    height = tl_y - br_y
    rect = Rectangle((tl_x, br_y), width, height, 
                     linewidth=2, edgecolor='red', facecolor='none', 
                     linestyle='--', label='Geofence')
    ax.add_patch(rect)
    
    # Colors for different UAVs
    colors = plt.cm.tab10(np.linspace(0, 1, num_uavs))
    
    # Plot each UAV's data
    for i, data in enumerate(uav_data):
        positions = np.array(data['positions'])
        
        if len(positions) == 0:
            continue
        
        # Plot flight path
        ax.plot(positions[:, 0], positions[:, 1], 
               color=colors[i], alpha=0.6, linewidth=1.5,
               label=f"{data['id']} Path")
        
        # Plot start position
        ax.scatter(positions[0, 0], positions[0, 1], 
                  color=colors[i], marker='o', s=200, 
                  edgecolors='black', linewidths=2, zorder=5,
                  label=f"{data['id']} Start")
        
        # Plot end position
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  color=colors[i], marker='s', s=200, 
                  edgecolors='black', linewidths=2, zorder=5,
                  label=f"{data['id']} End")
        
        # Plot all waypoints that existed during episode
        all_wps = np.array(data['all_waypoints'])
        if len(all_wps) > 0:
            ax.scatter(all_wps[:, 0], all_wps[:, 1], 
                      color=colors[i], marker='x', s=100, alpha=0.3,
                      linewidths=2, label=f"{data['id']} Waypoints")
        
        # Plot visited waypoints (successful arrivals)
        visited_wps = np.array(data['waypoints_visited'])
        if len(visited_wps) > 0:
            ax.scatter(visited_wps[:, 0], visited_wps[:, 1], 
                      color=colors[i], marker='*', s=300, 
                      edgecolors='yellow', linewidths=2, zorder=6,
                      label=f"{data['id']} Arrivals")
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(
        f'{scenario_name} - Test Run\n'
        f'Steps: {steps} | Reward: {total_reward:.0f} | Waypoint Arrivals: {arrivals}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved visualization to {save_path}")
    plt.close()

import matplotlib.animation as animation
import numpy as np

def animate_episode(uav_data, tl, br, transformer, scenario_name, save_path):
    """
    Generates a high-fidelity MP4/GIF animation of UAV trajectories.
    Converts heading from radians to degrees and adjusts for marker orientation.
    """
    # 1. Coordinate Space Setup
    tl_x, tl_y = transformer.geo_to_local(tl[0], tl[1])
    br_x, br_y = transformer.geo_to_local(br[0], br[1])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw Static Geofence
    width, height = br_x - tl_x, tl_y - br_y
    rect = Rectangle((tl_x, br_y), width, height, 
                     linewidth=2, edgecolor='red', facecolor='none', 
                     linestyle='--', label='Geofence boundary')
    ax.add_patch(rect)
    
    # 2. Initialization of Dynamic Elements
    num_uavs = len(uav_data)
    colors = plt.cm.get_cmap('tab10', num_uavs)
    
    lines = []        # To store trajectory trails
    uav_heads = []    # To store rotating markers
    
    # Find the longest trajectory to define animation length
    max_frames = max(len(d['positions']) for d in uav_data)

    for i, data in enumerate(uav_data):
        # Trail line: alpha starts low to emphasize the 'head'
        ln, = ax.plot([], [], color=colors(i), alpha=0.5, linewidth=1.5,
                     label=f"UAV {data['id']}")
        lines.append(ln)
        
        # Aircraft Head: Triangle marker
        # (3, 0, 0) is an equilateral triangle pointing 'Up'
        head, = ax.plot([], [], color=colors(i), marker=(3, 0, 0), 
                       markersize=15, markeredgecolor='black', linestyle='None')
        uav_heads.append(head)

        # Plot static waypoints for this specific UAV
        all_wps = np.array(data['all_waypoints'])
        if len(all_wps) > 0:
            ax.scatter(all_wps[:, 0], all_wps[:, 1], color=colors(i), 
                      marker='x', s=100, alpha=0.2)

    def init():
        """Initialize axes and background elements"""
        ax.set_xlim(tl_x - 100, br_x + 100)
        ax.set_ylim(br_y - 100, tl_y + 100)
        ax.set_xlabel('Local X (m)')
        ax.set_ylabel('Local Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        return lines + uav_heads

    def update(frame):
        """Update loop for each frame of the animation"""
        for i, data in enumerate(uav_data):
            pos_history = data['positions']
            
            # If the UAV has data for this frame
            if frame < len(pos_history):
                x, y, heading_rad = pos_history[frame]
                
                # Update Trail
                path_segments = np.array(pos_history[:frame+1])
                lines[i].set_data(path_segments[:, 0], path_segments[:, 1])
                
                # Update UAV Head Position
                uav_heads[i].set_data([x], [y])
                
                # Heading Transformation Logic:
                # 1. Convert Rad -> Deg
                heading_deg = np.degrees(heading_rad)
                
                # 2. Adjust Orientation
                # Matplotlib's (3,0,0) triangle points at 90 deg (North).
                # If your 0 rad is East (0 deg), we subtract 90 from the heading
                # to align the triangle's tip with the trajectory.
                marker_rotation = -heading_deg
                uav_heads[i].set_marker((3, 0, marker_rotation))
                
        # Update title with progress
        ax.set_title(f"{scenario_name}\nFrame: {frame}/{max_frames}")
        return lines + uav_heads

    # 3. Animation Construction
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init, 
        blit=True, interval=40, repeat=False
    )

    # 4. Save Execution
    console.print(f"[bold yellow]Rendering animation: {save_path}[/bold yellow]")
    
    # Determine writer based on extension
    if save_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Gemini Flight Engine'), bitrate=1800)
    else:
        writer = 'pillow' # For .gif

    ani.save(save_path, writer=writer)
    plt.close()
    console.print(f"[bold green]✓[/bold green] Animation successfully exported.")

# In your test() function, replace the call to visualize_episode with:
# save_path = os.path.join(config["test"]["save_dir"], f"{scenario['name']}.mp4")
# animate_episode(uav_data, tl, br, transformer, scenario['name'], save_path)


def test(config):
    console.print(
        Panel.fit(
            "[bold white]Flight Path Visualizer[/bold white]",
            subtitle="Multi-UAV Trajectory Analysis",
        )
    )
    
    # Create output directory
    os.makedirs(config["test"]["save_dir"], exist_ok=True)
    
    # Load model
    model_path = config["test"]["model_path"]
    if not os.path.exists(model_path):
        console.print(f"[red]Error:[/red] Model not found at {model_path}")
        return
    
    console.print(f"[green]✓[/green] Loading model from {model_path}")
    model = A2C.load(model_path, device='cpu')
    console.print(f"[green]✓[/green] Model loaded successfully\n")
    
    # Fixed origin for consistency
    origin = [
        float(config["test"]["env"]["origin"][0]), 
        float(config["test"]["env"]["origin"][1])
    ]
    console.print(f"Test origin: ({origin[0]:.4f}, {origin[1]:.4f})\n")
    
    # Run visualizations
    scenario = {
        "name": config["test"]["test_name"],
        "box_size": config["test"]["env"]["box_size"],
        "num_uavs": len(config["test"]["missions"]),
    }
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Test: {scenario['name']}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    # Create environment
    env, tl, br = create_test_environment(scenario, origin, config)
    transformer = env.transformer
    
    # Run episode and record data
    uav_data, _, total_reward, arrivals = run_and_record_episode(
        model, env, transformer, 
        config["test"]["env"]["max_steps"]
    )
    
    console.print(
        f"Total Reward: {total_reward:.2f} | "
        f"Waypoint Arrivals: {arrivals}"
    )
    
    # Create visualization
    save_path = os.path.join(
        config["test"]["save_dir"], 
        f"{scenario['name'].replace(' ', '_')}.png"
    )
    
    visualize_episode(
        uav_data, tl, br, transformer,
        scenario['name'],
        config["test"]["env"]["max_steps"], 
        total_reward, arrivals,
        save_path
    )

    save_path = os.path.join(config["test"]["save_dir"], f"{scenario['name']}.mp4")
    animate_episode(uav_data, tl, br, transformer, scenario['name'], save_path)
    
    env.close()
    
    console.print(f"\n[bold green]All visualizations complete![/bold green]")
    console.print(f"Saved to: {config['test']['save_dir']}/")


if __name__ == "__main__":
    test()