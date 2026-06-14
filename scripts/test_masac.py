import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import numpy as np
from algo.masac import MASAC

with open("config/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

n_agents, obs_dim, act_dim = 3, 6, 2
agents = ["ess_0", "ess_1", "ess_2"]
# 把 start_steps 调小,方便快速测更新
cfg["algo"]["start_steps"] = 50
cfg["algo"]["batch_size"] = 32

algo = MASAC(n_agents, obs_dim, act_dim, agents, cfg)

print("=== 测试选动作 ===")
obs = {a: np.random.randn(obs_dim).astype(np.float32) for a in agents}
act = algo.select_actions(obs)
for a in agents:
    print(f"{a}: {act[a]}, 范围在-1~1: {np.all(np.abs(act[a])<=1)}")

print("\n=== 填充 buffer 并更新 ===")
for step in range(200):
    obs = {a: np.random.randn(obs_dim).astype(np.float32) for a in agents}
    act = algo.select_actions(obs)
    rew = {a: float(np.random.randn()) for a in agents}
    nobs = {a: np.random.randn(obs_dim).astype(np.float32) for a in agents}
    algo.store(obs, act, rew, nobs, done=(step % 24 == 23))
    metrics = algo.update()
    if step % 50 == 0 and metrics:
        print(f"step {step}: critic_loss={metrics['critic_loss']:.4f} "
              f"actor_loss={metrics['actor_loss']:.4f} "
              f"alpha={metrics['alpha']:.4f}")

print("\n=== 测试存取 ===")
algo.save("results/test_masac.pth")
algo.load("results/test_masac.pth")
print("保存和加载成功。")
os.remove("results/test_masac.pth")

print("\n全部通过:MASAC 选动作/存储/更新/存取均正常。")
print("注意:这里用随机数据,loss不会下降,只验证流程不报错。")
