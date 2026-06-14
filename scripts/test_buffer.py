import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from algo.replay_buffer import MultiAgentReplayBuffer

n_agents, obs_dim, act_dim = 3, 6, 2
buf = MultiAgentReplayBuffer(capacity=100, n_agents=n_agents,
                             obs_dim=obs_dim, act_dim=act_dim)
agents = ["ess_0", "ess_1", "ess_2"]

# 存 10 条假数据
for step in range(10):
    obs = {a: np.random.randn(obs_dim).astype(np.float32) for a in agents}
    act = {a: np.random.uniform(-1, 1, act_dim).astype(np.float32) for a in agents}
    rew = {a: float(step) for a in agents}
    nobs = {a: np.random.randn(obs_dim).astype(np.float32) for a in agents}
    buf.store(obs, act, rew, nobs, done=(step == 9))

print(f"缓冲区大小: {len(buf)} (应为10)")
batch = buf.sample(4)
print("采样batch各字段形状:")
for k, v in batch.items():
    print(f"  {k}: {v.shape}")
print("\n预期: obs [4,3,6], actions [4,3,2], rewards [4,3], "
      "next_obs [4,3,6], dones [4,1]")
