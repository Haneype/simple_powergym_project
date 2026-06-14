import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, numpy as np
from env.grid_env import GridEnv

with open("config/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

env = GridEnv(cfg)
obs = env.reset()
zero = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
print("ESS不动作,逐步看全天电压:")
for t in range(env.episode_steps):
    obs, r, done, info = env.step(zero)
    flag = ""
    if info["v_max"] > 1.05: flag += " <过压!>"
    if info["v_min"] < 0.95: flag += " <欠压!>"
    print(f"t={t:2d}: Vmin={info['v_min']:.4f} Vmax={info['v_max']:.4f}{flag}")
    if done: break
