"""
SAC 神经网络:
  - GaussianActor: squashed 高斯策略 (每个 agent 一个)
      输入局部观测,输出动作及其对数概率
  - Critic: 集中式 Q 网络 (CTDE)
      输入全局观测(所有agent观测拼接)+ 全局动作,输出 Q 值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """Squashed 高斯策略网络"""
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """
        重参数化采样 + tanh 压缩。
        返回:
          action: 压缩到 (-1,1) 的动作
          log_prob: 该动作的对数概率(已做 tanh 雅可比修正)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # 重参数化采样 (rsample 支持梯度回传)
        x = normal.rsample()
        action = torch.tanh(x)

        # 对数概率,修正 tanh 变换的雅可比项
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, obs):
        """评估时用:直接取均值的 tanh,不采样"""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    """
    集中式双 Q 网络 (CTDE)
    输入: 全局观测 (n_agents*obs_dim) + 全局动作 (n_agents*act_dim)
    输出: 两个 Q 值
    """
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim):
        super().__init__()
        in_dim = n_agents * obs_dim + n_agents * act_dim

        # Q1
        self.q1_fc1 = nn.Linear(in_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_fc1 = nn.Linear(in_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, global_obs, global_act):
        """
        global_obs: [B, n_agents*obs_dim]
        global_act: [B, n_agents*act_dim]
        """
        x = torch.cat([global_obs, global_act], dim=-1)

        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2
