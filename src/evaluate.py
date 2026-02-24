import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from itertools import combinations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from stable_baselines3 import SAC, PPO, A2C

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position

console = Console()

ALGO_MAP = {"SAC": SAC, "PPO": PPO, "A2C": A2C}

def build_env(eval_config):
    """Build a fresh MultiUAVEnv from eval config for each episode."""
    env_cfg = eval_config["env"]
    origin = env_cfg["origin"]
    box_size = eval_config["env"]["box_size"]

    lat_off = (box_size / 2.0) / 111_320.0
    lon_off = (box_size / 2.0) / (111_320.0 * np.cos(np.radians(origin[0])))

    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)

    # Build UAV list from config
    uavs = []
    for uav_id, params in eval_config["missions"].items():
        uav = FixedWingAircraft(
            id_tag=uav_id,
            initial_position=Position(
                params["initial_position"][0],
                params["initial_position"][1]
            ),
            initial_heading=params["initial_heading"],
            cruise_speed=params["cruise_speed"],
            turning_radius=params["turning_radius"],
        )
        uavs.append(uav)

    env = MultiUAVEnv(
        uavs,
        tl=tl,
        br=br,
        dt=env_cfg.get("dt", 0.3),
        max_steps=env_cfg["max_steps"],
        boundary_margin=eval_config.get("boundary_margin", 0.15),
        mission_waypoint_count=eval_config.get("mission_waypoint_count", 3),
        mode=eval_config.get("mode", "gen_mission"),
        caution_dist=eval_config.get("caution_dist", 100),
        critical_dist=eval_config.get("critical_dist", 35),
    )
    return env

def run_episode(model, env, max_steps):
    """
    Run one episode and return a dict of raw metrics.

    Metrics collected:
        waypoints_hit - total waypoints reached
        total_reward - sum of all step rewards
        collision_events - number of steps any pair was inside critical_dist
        near_miss_events - steps any pair was inside caution_dist but outside critical_dist
        geofence_violations - steps any UAV was outside the boundary
        steps_taken - how many steps the episode ran
        distance_traveled - total meters flown across all UAVs
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    collision_events = 0
    near_miss_events = 0
    geofence_violations = 0
    distance_per_uav = [0.0] * len(env.aircraft_list)
    prev_positions = [ac.position.to_tuple() for ac in env.aircraft_list]

    while not (done or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Collision / near-miss tracking
        for i1, i2 in combinations(range(len(env.aircraft_list)), 2):
            ac1, ac2 = env.aircraft_list[i1], env.aircraft_list[i2]
            sep = env._compute_local_distance(
                ac1.position.to_tuple(),
                ac2.position.to_tuple()
            )
            if sep < env.critical_dist:
                collision_events += 1
            elif sep < env.caution_dist:
                near_miss_events += 1

        # Geofence tracking
        for ac in env.aircraft_list:
            pos = ac.position.to_tuple()
            if not (env.min_lat < pos[0] < env.max_lat) or \
               not (env.min_lon < pos[1] < env.max_lon):
                geofence_violations += 1

        # Distance traveled
        for i, ac in enumerate(env.aircraft_list):
            curr = ac.position.to_tuple()
            dist = env._compute_local_distance(prev_positions[i], curr)
            distance_per_uav[i] += dist
            prev_positions[i] = curr

    return {
        "waypoints_hit": info["waypoints_hit"],
        "total_reward": total_reward,
        "collision_events": collision_events,
        "near_miss_events": near_miss_events,
        "geofence_violations": geofence_violations,
        "steps_taken": step,
        "distance_traveled": sum(distance_per_uav),
    }

def evaluate_model(model_cfg, eval_config, n_episodes):
    """Run N episodes for one model and return aggregated stats."""
    algo_name = model_cfg["algorithm"].upper()
    model_path = model_cfg["model_path"]
    label = model_cfg.get("label", f"{algo_name}")
    device = model_cfg.get("device", "cpu")

    console.print(Panel(
        f"[bold cyan]Evaluating: {label}[/bold cyan]\n"
        f"Algorithm: {algo_name}\n"
        f"Model: {model_path}\n"
        f"Episodes: {n_episodes}",
        expand=False
    ))

    if algo_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from {list(ALGO_MAP.keys())}")

    model = ALGO_MAP[algo_name].load(model_path, device=device)

    all_metrics = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Running {n_episodes} episodes...", total=n_episodes)

        for ep in range(n_episodes):
            env = build_env(eval_config)
            metrics = run_episode(model, env, eval_config["env"]["max_steps"])
            all_metrics.append(metrics)
            env.close()
            progress.update(task, advance=1)

    # Aggregate
    keys = all_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"]  = float(np.std(vals))
        agg[f"{k}_min"]  = float(np.min(vals))
        agg[f"{k}_max"]  = float(np.max(vals))

    # Collision rate = fraction of episodes with at least one collision
    agg["collision_rate"] = float(
        np.mean([1.0 if m["collision_events"] > 0 else 0.0 for m in all_metrics])
    )

    # Waypoints per step (efficiency)
    agg["wp_efficiency_mean"] = float(
        np.mean([m["waypoints_hit"] / max(m["steps_taken"], 1) for m in all_metrics])
    )

    return {
        "label": label,
        "algorithm": algo_name,
        "model_path": model_path,
        "n_episodes": n_episodes,
        "aggregated": agg,
        "raw": all_metrics,
    }

def print_summary_table(results):
    table = Table(title="Model Evaluation Summary", show_lines=True)

    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Algorithm", style="bold")
    table.add_column("WP Hits\n(mean ± std)", justify="right")
    table.add_column("Reward\n(mean ± std)", justify="right")
    table.add_column("Collisions\n(rate)", justify="right")
    table.add_column("Near Misses\n(mean)", justify="right")
    table.add_column("Geofence\nViolations (mean)", justify="right")
    table.add_column("WP Efficiency\n(wp/step)", justify="right")

    for r in results:
        agg = r["aggregated"]
        table.add_row(
            r["label"],
            r["algorithm"],
            f"{agg['waypoints_hit_mean']:.1f} ± {agg['waypoints_hit_std']:.1f}",
            f"{agg['total_reward_mean']:.0f} ± {agg['total_reward_std']:.0f}",
            f"{agg['collision_rate']*100:.1f}%",
            f"{agg['near_miss_events_mean']:.1f}",
            f"{agg['geofence_violations_mean']:.1f}",
            f"{agg['wp_efficiency_mean']*1000:.2f} wp/1k steps",
        )

    console.print(table)

def save_csv(results, save_dir):
    path = os.path.join(save_dir, "evaluation_results.csv")
    rows = []
    for r in results:
        row = {"label": r["label"], "algorithm": r["algorithm"], "n_episodes": r["n_episodes"]}
        row.update(r["aggregated"])
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]✓[/green] CSV saved to {path}")
    return path

def save_comparison_chart(results, save_dir):
    """
    Side-by-side bar chart comparing key metrics across all models.
    """
    metrics = [
        ("waypoints_hit_mean",      "Waypoints Hit (mean)"),
        ("total_reward_mean",       "Total Reward (mean)"),
        ("collision_rate",          "Collision Rate (%)"),
        ("near_miss_events_mean",   "Near Misses (mean)"),
        ("geofence_violations_mean","Geofence Violations (mean)"),
        ("wp_efficiency_mean",      "WP Efficiency (wp/step x1000)"),
    ]

    n_metrics = len(metrics)
    n_models = len(results)
    labels = [r["label"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Model Comparison — DON'T Evaluation", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for ax_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        values = []
        errors = []

        for r in results:
            agg = r["aggregated"]
            val = agg[metric_key]
            # Scale collision rate to percentage, efficiency to per 1k steps
            if metric_key == "collision_rate":
                val *= 100
            elif metric_key == "wp_efficiency_mean":
                val *= 1000
            values.append(val)

            # Add error bars where std is available
            std_key = metric_key.replace("_mean", "_std")
            if std_key in agg:
                std = agg[std_key]
                if metric_key == "wp_efficiency_mean":
                    std *= 1000
                errors.append(std)
            else:
                errors.append(0)

        x = np.arange(n_models)
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.8)

        # Error bars (skip for rates which have no std)
        if any(e > 0 for e in errors):
            ax.errorbar(x, values, yerr=errors, fmt='none',
                       color='black', capsize=5, linewidth=1.5)

        ax.set_title(metric_label, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(metric_label)
        ax.grid(axis='y', alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    # Legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n_models)]
    fig.legend(handles=patches, loc='lower center', ncol=n_models,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(save_dir, "evaluation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓[/green] Chart saved to {path}")
    return path

def evaluate(config):
    eval_cfg = config["evaluate"]
    save_dir = eval_cfg["save_dir"]
    n_episodes = eval_cfg["n_episodes"]
    os.makedirs(save_dir, exist_ok=True)

    console.print(Panel.fit(
        "[bold white]DON'T Model Evaluator[/bold white]",
        subtitle=f"Comparing {len(eval_cfg['models'])} models over {n_episodes} episodes each"
    ))

    results = []
    for model_cfg in eval_cfg["models"]:
        result = evaluate_model(model_cfg, eval_cfg, n_episodes)
        results.append(result)

    console.print("\n")
    print_summary_table(results)

    save_csv(results, save_dir)
    save_comparison_chart(results, save_dir)

    console.print(Panel(
        f"[bold green]Evaluation Complete[/bold green]\n"
        f"Results saved to: {save_dir}/",
        expand=False
    ))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/eval_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    evaluate(config)