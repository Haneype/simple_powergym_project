"""
多智能体经验回放缓冲区 (off-policy)
存储整个多智能体系统的转移:
  - 每个 agent 的局部观测 obs
  - 每个 agent 的动作 action
  - 全局共享奖励 reward
  - 每个 agent 的下一观测 next_obs
  - done 标志
CTDE 训练时,critic 需要拼接所有 agent 的观测和动作作为全局信息,
所以这里按 agent 分别存储,采样时再组装。
"""

import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, act_dim):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 预分配数组,形状 [capacity, n_agents, dim]
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0      # 当前写入位置
        self.size = 0     # 当前已存数量

    def store(self, obs, actions, rewards, next_obs, done):
        """
        obs/next_obs: dict {agent: array(obs_dim)}
        actions:      dict {agent: array(act_dim)}
        rewards:      dict {agent: float}
        done:         bool
        """
        agents = sorted(obs.keys())  # 固定顺序,保证拼接一致
        for i, ag in enumerate(agents):
            self.obs[self.ptr, i] = obs[ag]
            self.actions[self.ptr, i] = actions[ag]
            self.rewards[self.ptr, i] = rewards[ag]
            self.next_obs[self.ptr, i] = next_obs[ag]
        self.dones[self.ptr, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机采样一个 batch,返回 numpy 数组(由算法转成 tensor)"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],            # [B, n_agents, obs_dim]
            "actions": self.actions[idx],    # [B, n_agents, act_dim]
            "rewards": self.rewards[idx],    # [B, n_agents]
            "next_obs": self.next_obs[idx],  # [B, n_agents, obs_dim]
            "dones": self.dones[idx],        # [B, 1]
        }

    def __len__(self):
        return self.size
