"""
MASAC: 多智能体软演员-评论家 (CTDE)
  - 每个 agent 独立 actor (分散执行)
  - 共享集中式 critic (集中训练, 看全局信息)
  - 自适应温度系数 alpha
对应论文 Algorithm 1 及式 20-26
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from algo.base import BaseMultiAgentAlgorithm
from algo.networks import GaussianActor, Critic
from algo.replay_buffer import MultiAgentReplayBuffer


class MASAC(BaseMultiAgentAlgorithm):
    def __init__(self, n_agents, obs_dim, act_dim, agent_names, cfg):
        acfg = cfg["algo"]
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_names = sorted(agent_names)  # 固定顺序,与 buffer 一致
        self.device = torch.device(cfg["train"]["device"])

        self.gamma = acfg["gamma"]
        self.tau = acfg["tau"]
        self.batch_size = acfg["batch_size"]
        self.start_steps = acfg["start_steps"]
        hidden = acfg["hidden_dim"]

        # ---- 每个 agent 一个 actor ----
        self.actors = [GaussianActor(obs_dim, act_dim, hidden).to(self.device)
                       for _ in range(n_agents)]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=acfg["lr_actor"])
                           for a in self.actors]

        # ---- 共享集中式 critic + 目标网络 ----
        self.critic = Critic(n_agents, obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=acfg["lr_critic"])

        # ---- 自适应温度 alpha ----
        self.auto_entropy = acfg["auto_entropy"]
        if self.auto_entropy:
            self.target_entropy = -float(act_dim)  # 每个 agent 的目标熵
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=acfg["lr_alpha"])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        # ---- 经验回放 ----
        self.buffer = MultiAgentReplayBuffer(acfg["buffer_size"], n_agents,
                                             obs_dim, act_dim)
        self.total_steps = 0

    # ---------------- 执行 ----------------
    def select_actions(self, obs, deterministic=False):
        """分散执行:每个 agent 用自己的 actor + 局部观测"""
        actions = {}
        # 探索初期用随机动作填充 buffer
        if (not deterministic) and self.total_steps < self.start_steps:
            for ag in self.agent_names:
                actions[ag] = np.random.uniform(-1, 1, self.act_dim).astype(np.float32)
            return actions

        for i, ag in enumerate(self.agent_names):
            o = torch.as_tensor(obs[ag], dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    a = self.actors[i].deterministic(o)
                else:
                    a, _ = self.actors[i].sample(o)
            actions[ag] = a.squeeze(0).cpu().numpy()
        return actions

    def store(self, obs, actions, rewards, next_obs, done):
        self.buffer.store(obs, actions, rewards, next_obs, done)
        self.total_steps += 1

    # ---------------- 训练 ----------------
    def update(self):
        if len(self.buffer) < self.batch_size or self.total_steps < self.start_steps:
            return {}

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], device=self.device)           # [B,N,O]
        actions = torch.as_tensor(batch["actions"], device=self.device)   # [B,N,A]
        rewards = torch.as_tensor(batch["rewards"], device=self.device)   # [B,N]
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device) # [B,N,O]
        dones = torch.as_tensor(batch["dones"], device=self.device)       # [B,1]

        B = obs.shape[0]
        # 协作式:用所有 agent 平均奖励作为全局奖励
        global_reward = rewards.mean(dim=1, keepdim=True)  # [B,1]

        global_obs = obs.reshape(B, -1)            # [B, N*O]
        global_act = actions.reshape(B, -1)        # [B, N*A]
        global_next_obs = next_obs.reshape(B, -1)  # [B, N*O]

        # ===== 1) Critic 更新 (式20-22) =====
        with torch.no_grad():
            # 下一状态:每个 agent 用各自 actor 采样动作 + 对数概率
            next_acts, next_logps = [], []
            for i in range(self.n_agents):
                a, lp = self.actors[i].sample(next_obs[:, i, :])
                next_acts.append(a)
                next_logps.append(lp)
            next_global_act = torch.cat(next_acts, dim=-1)        # [B, N*A]
            # 熵项:所有 agent 对数概率求和
            next_logp_sum = torch.stack(next_logps, dim=0).sum(dim=0)  # [B,1]

            q1_t, q2_t = self.critic_target(global_next_obs, next_global_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp_sum
            target_q = global_reward + self.gamma * (1 - dones) * q_t

        q1, q2 = self.critic(global_obs, global_act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ===== 2) Actor 更新 (式23-24) =====
        # 逐个 agent 更新:只换当前 agent 的动作为新采样,其余用 buffer 动作
        actor_losses = []
        logp_total = 0.0
        for i in range(self.n_agents):
            new_a, logp = self.actors[i].sample(obs[:, i, :])
            # 拼出全局动作:当前 agent 用新动作,其余 detach(用 buffer 里的)
            act_list = []
            for j in range(self.n_agents):
                if j == i:
                    act_list.append(new_a)
                else:
                    act_list.append(actions[:, j, :])
            cur_global_act = torch.cat(act_list, dim=-1)

            q1_pi, q2_pi = self.critic(global_obs, cur_global_act)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * logp - q_pi).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

            actor_losses.append(actor_loss.item())
            logp_total += logp.mean().item()

        # ===== 3) 温度 alpha 更新 (式25-26) =====
        if self.auto_entropy:
            # 重新采样算当前总熵(detach actor)
            with torch.no_grad():
                logp_sum = 0.0
                for i in range(self.n_agents):
                    _, lp = self.actors[i].sample(obs[:, i, :])
                    logp_sum = logp_sum + lp
            alpha_loss = -(self.log_alpha *
                           (logp_sum + self.n_agents * self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # ===== 4) 目标网络软更新 =====
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": float(np.mean(actor_losses)),
            "alpha": self.alpha,
            "avg_logp": logp_total / self.n_agents,
        }

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ---------------- 存取 ----------------
    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.auto_entropy and ckpt["log_alpha"] is not None:
            self.log_alpha.data.copy_(ckpt["log_alpha"].data)
            self.alpha = self.log_alpha.exp().item()
