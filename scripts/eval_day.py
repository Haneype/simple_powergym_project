"""
训练后评估与可视化:
  加载 best_model,用确定性策略跑一个完整24小时,绘制:
  1. 全天电压对比 (MASAC vs ESS不动作)
  2. 各 ESS 的 SOC 曲线
  3. 各 ESS 的有功/无功出力曲线
支持 --config 指定系统,结果按系统分目录存。
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from env.grid_env import GridEnv
from algo.masac import MASAC


def run_day(env, algo=None):
    obs = env.reset()
    vmin_hist, vmax_hist = [], []
    soc_hist = [[] for _ in range(env.n_agents)]
    p_hist = [[] for _ in range(env.n_agents)]
    q_hist = [[] for _ in range(env.n_agents)]

    done = False
    while not done:
        if algo is None:
            actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
        else:
            actions = algo.select_actions(obs, deterministic=True)
        obs, rewards, done, info = env.step(actions)

        vmin_hist.append(info["v_min"])
        vmax_hist.append(info["v_max"])
        for i in range(env.n_agents):
            soc_hist[i].append(env.ess_models[i].soc)
            p_hist[i].append(env.net.sgen.at[env.ess_sgen_idx[i], "p_mw"])
            q_hist[i].append(env.net.sgen.at[env.ess_sgen_idx[i], "q_mvar"])

    return {"vmin": vmin_hist, "vmax": vmax_hist,
            "soc": soc_hist, "p": p_hist, "q": q_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["train"].get("results_dir", "results")
    plots_dir = cfg["train"].get("plots_dir", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    env = GridEnv(cfg)
    algo = MASAC(env.n_agents, env.obs_dim, 2, env.agents, cfg)
    model_path = os.path.join(results_dir, "best_model.pth")
    algo.load(model_path)
    print(f"已加载 {model_path} (系统: {cfg['env']['system']}, "
          f"{env.n_agents}个ESS)")

    base = run_day(env, algo=None)
    ctrl = run_day(env, algo=algo)

    hours = list(range(env.episode_steps))
    v_min, v_max = env.v_min, env.v_max

    # ===== 图1: 全天电压对比 =====
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hours, base["vmin"], color="tab:blue", ls="--",
            label="Vmin (no action)")
    ax.plot(hours, base["vmax"], color="tab:red", ls="--",
            label="Vmax (no action)")
    ax.plot(hours, ctrl["vmin"], color="tab:blue", ls="-", marker="o",
            markersize=3, label="Vmin (MASAC)")
    ax.plot(hours, ctrl["vmax"], color="tab:red", ls="-", marker="o",
            markersize=3, label="Vmax (MASAC)")
    ax.axhline(v_max, color="gray", ls="-", lw=0.8)
    ax.axhline(v_min, color="gray", ls="-", lw=0.8)
    ax.fill_between(hours, v_min, v_max, color="green", alpha=0.06,
                    label="Safe band")
    ax.set_xlabel("Hour"); ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage over a day ({cfg['env']['system']}): MASAC vs No-action")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eval_voltage.png"), dpi=120)
    print(f"已保存 {plots_dir}/eval_voltage.png")

    # ===== 图2: SOC 曲线 =====
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(env.n_agents):
        ax.plot(hours, ctrl["soc"][i], "-o", markersize=3,
                label=f"{env.agents[i]} (bus {env.ess_buses[i]})")
    ax.set_xlabel("Hour"); ax.set_ylabel("SOC")
    ax.set_title(f"ESS SOC over a day ({cfg['env']['system']})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eval_soc.png"), dpi=120)
    print(f"已保存 {plots_dir}/eval_soc.png")

    # ===== 图3: ESS 出力 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(env.n_agents):
        axes[0].plot(hours, ctrl["p"][i], "-o", markersize=3,
                     label=f"{env.agents[i]}")
        axes[1].plot(hours, ctrl["q"][i], "-o", markersize=3,
                     label=f"{env.agents[i]}")
    axes[0].set_title("ESS Active Power Injection (MW)\n(+ = discharge, - = charge)")
    axes[0].set_xlabel("Hour"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].axhline(0, color="k", lw=0.5)
    axes[1].set_title("ESS Reactive Power Injection (MVar)")
    axes[1].set_xlabel("Hour"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[1].axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eval_power.png"), dpi=120)
    print(f"已保存 {plots_dir}/eval_power.png")

    # ===== 数值总结 =====
    base_under = sum(max(v_min - x, 0) for x in base["vmin"])
    base_over = sum(max(x - v_max, 0) for x in base["vmax"])
    ctrl_under = sum(max(v_min - x, 0) for x in ctrl["vmin"])
    ctrl_over = sum(max(x - v_max, 0) for x in ctrl["vmax"])
    print("\n=== 全天电压越限量对比 ===")
    print(f"不动作: 欠压累计 {base_under:.4f}, 过压累计 {base_over:.4f}")
    print(f"MASAC : 欠压累计 {ctrl_under:.4f}, 过压累计 {ctrl_over:.4f}")


if __name__ == "__main__":
    main()
