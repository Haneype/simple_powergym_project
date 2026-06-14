"""
训练主循环 (对应论文 Algorithm 1)
  - 每 episode 重置环境,逐时隙交互、存经验、更新 MASAC
  - 定期用确定性策略评估,记录电压越限、奖励、网损
  - 保存最优模型和训练曲线
  - 支持 --config 指定配置文件,结果按系统分目录存
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from env.grid_env import GridEnv
from algo.masac import MASAC


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(env, algo, n_episodes=1):
    """用确定性策略评估,返回平均指标"""
    total_reward, total_viol, total_loss = 0.0, 0.0, 0.0
    steps = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            actions = algo.select_actions(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            total_reward += list(rewards.values())[0]
            v_over = max(info["v_max"] - env.v_max, 0)
            v_under = max(env.v_min - info["v_min"], 0)
            total_viol += (v_over + v_under)
            total_loss += info["p_loss"]
            steps += 1
    return {
        "reward": total_reward / n_episodes,
        "violation": total_viol / steps,
        "p_loss": total_loss / steps,
    }


def evaluate_baseline(env):
    """ESS 全程不动作的基线,用于对比"""
    obs = env.reset()
    done = False
    total_viol, steps = 0.0, 0
    zero = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    while not done:
        obs, rewards, done, info = env.step(zero)
        v_over = max(info["v_max"] - env.v_max, 0)
        v_under = max(env.v_min - info["v_min"], 0)
        total_viol += (v_over + v_under)
        steps += 1
    return total_viol / steps


def plot_curves(hist, baseline_viol, plots_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(hist["episode"], hist["eval_reward"], "b-o", markersize=3)
    axes[0].set_title("Eval Reward")
    axes[0].set_xlabel("Episode")
    axes[0].grid(True)

    axes[1].plot(hist["episode"], hist["eval_viol"], "r-o", markersize=3,
                 label="MASAC")
    axes[1].axhline(baseline_viol, color="gray", ls="--",
                    label="No-action baseline")
    axes[1].set_title("Voltage Violation")
    axes[1].set_xlabel("Episode")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(hist["episode"], hist["eval_loss"], "g-o", markersize=3)
    axes[2].set_title("Power Loss (MW)")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=120)
    print("曲线已保存。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 结果目录(配置里没有就用默认)
    results_dir = cfg["train"].get("results_dir", "results")
    plots_dir = cfg["train"].get("plots_dir", "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    set_seed(cfg["train"]["seed"])

    env = GridEnv(cfg)
    algo = MASAC(env.n_agents, env.obs_dim, 2, env.agents, cfg)
    print(f"系统: {cfg['env']['system']}, 智能体数: {env.n_agents}, "
          f"ESS节点: {env.ess_buses}")

    baseline_viol = evaluate_baseline(env)
    print(f"基线(ESS不动作)平均电压越限: {baseline_viol:.5f}\n")

    episodes = cfg["train"]["episodes"]
    eval_interval = cfg["train"]["eval_interval"]
    save_interval = cfg["train"]["save_interval"]

    hist = {"episode": [], "train_reward": [],
            "eval_reward": [], "eval_viol": [], "eval_loss": []}
    best_eval_reward = -1e9

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            actions = algo.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)
            algo.store(obs, actions, rewards, next_obs, done)
            algo.update()
            obs = next_obs
            ep_reward += list(rewards.values())[0]

        if ep % eval_interval == 0 or ep == 1:
            ev = evaluate(env, algo)
            hist["episode"].append(ep)
            hist["train_reward"].append(ep_reward)
            hist["eval_reward"].append(ev["reward"])
            hist["eval_viol"].append(ev["violation"])
            hist["eval_loss"].append(ev["p_loss"])
            print(f"Ep {ep:4d} | 训练奖励 {ep_reward:8.2f} | "
                  f"评估奖励 {ev['reward']:8.2f} | "
                  f"越限 {ev['violation']:.5f} (基线 {baseline_viol:.5f}) | "
                  f"网损 {ev['p_loss']:.4f} | alpha {algo.alpha:.3f}")

            if ev["reward"] > best_eval_reward:
                best_eval_reward = ev["reward"]
                algo.save(os.path.join(results_dir, "best_model.pth"))

        if ep % save_interval == 0:
            algo.save(os.path.join(results_dir, f"model_ep{ep}.pth"))

    plot_curves(hist, baseline_viol, plots_dir)
    print(f"\n训练完成。最优评估奖励: {best_eval_reward:.2f}")
    print(f"模型: {results_dir}/best_model.pth")
    print(f"曲线: {plots_dir}/training_curves.png")


if __name__ == "__main__":
    main()
