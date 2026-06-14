"""
用随机动作验证环境:
  - 能正常 reset/step 跑完一个 episode
  - 电压、网损、奖励数值是否合理
  - 对比"无ESS动作"(全0) vs "随机动作"的电压情况
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from env.grid_env import GridEnv

with open("config/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

env = GridEnv(cfg)
print(f"智能体数量: {env.n_agents}, 名称: {env.agents}")
print(f"动作空间: {env.action_space}, 观测维度: {env.obs_dim}")

print("\n=== 场景A: ESS全程不动作(全0) ===")
obs = env.reset()
zero_action = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
for t in range(env.episode_steps):
    obs, rewards, done, info = env.step(zero_action)
    if t % 4 == 0:
        print(f"t={t:2d}: Vmin={info['v_min']:.4f} Vmax={info['v_max']:.4f} "
              f"网损={info['p_loss']:.4f} 奖励={list(rewards.values())[0]:.4f}")
    if done:
        break

print("\n=== 场景B: ESS随机动作 ===")
obs = env.reset()
np.random.seed(0)
total_r = 0
for t in range(env.episode_steps):
    actions = {a: np.random.uniform(-1, 1, 2).astype(np.float32) for a in env.agents}
    obs, rewards, done, info = env.step(actions)
    total_r += list(rewards.values())[0]
    if t % 4 == 0:
        print(f"t={t:2d}: Vmin={info['v_min']:.4f} Vmax={info['v_max']:.4f} "
              f"网损={info['p_loss']:.4f} 奖励={list(rewards.values())[0]:.4f}")
    if done:
        break
print(f"\n随机动作累计奖励: {total_r:.4f}")

print("\n检查要点:")
print("- 两个场景都应正常跑完24步,不报错")
print("- 负荷高峰时段(t=8,19附近)Vmin应较低,可能<0.95")
print("- 光伏正午(t=12附近)Vmax可能升高")
print("- 观测样例:", obs[env.agents[0]])
