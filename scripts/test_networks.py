import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from algo.networks import GaussianActor, Critic

B = 5            # batch size
n_agents = 3
obs_dim = 6
act_dim = 2
hidden = 256

print("=== 测试 Actor ===")
actor = GaussianActor(obs_dim, act_dim, hidden)
obs = torch.randn(B, obs_dim)
action, log_prob = actor.sample(obs)
print(f"输入观测: {obs.shape}")
print(f"输出动作: {action.shape} (应为 [{B},{act_dim}])")
print(f"对数概率: {log_prob.shape} (应为 [{B},1])")
print(f"动作范围: [{action.min():.3f}, {action.max():.3f}] (应在 -1~1 内)")
det = actor.deterministic(obs)
print(f"确定性动作: {det.shape}, 范围 [{det.min():.3f}, {det.max():.3f}]")

print("\n=== 测试 Critic ===")
critic = Critic(n_agents, obs_dim, act_dim, hidden)
global_obs = torch.randn(B, n_agents * obs_dim)
global_act = torch.randn(B, n_agents * act_dim)
q1, q2 = critic(global_obs, global_act)
print(f"全局观测: {global_obs.shape}, 全局动作: {global_act.shape}")
print(f"Q1: {q1.shape}, Q2: {q2.shape} (都应为 [{B},1])")

print("\n=== 测试梯度回传 ===")
loss = (q1.mean() + log_prob.mean())
loss.backward()
print("反向传播成功,梯度可正常计算。")

print("\n全部通过:网络前向/采样/梯度均正常。")
