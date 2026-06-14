"""
独立测试 ESS 模型:
  1. 正常充放电时 SOC 是否正确变化
  2. SOC 接近上下限时功率是否被正确限制
  3. 无功是否受 PCS 容量约束
  4. 寿命损耗成本是否非负且符号正确
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from env.ess_model import ESSModel

# 读配置
with open("config/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ess = ESSModel(cfg["ess"], dt=cfg["env"]["dt"])

print("=== 测试1: 持续充电,观察 SOC 上升并在上限处饱和 ===")
ess.reset()
for t in range(8):
    p, q, soc, cost = ess.step(p_e=0.5, q_e=0.0)
    print(f"t={t}: 请求充电0.5MW -> 实际P={p:.3f}MW, SOC={soc:.3f}, 损耗={cost:.4f}")

print("\n=== 测试2: 持续放电,观察 SOC 下降并在下限处饱和 ===")
ess.reset()
for t in range(8):
    p, q, soc, cost = ess.step(p_e=-0.5, q_e=0.0)
    print(f"t={t}: 请求放电-0.5MW -> 实际P={p:.3f}MW, SOC={soc:.3f}, 损耗={cost:.4f}")

print("\n=== 测试3: 无功受 PCS 容量约束 (S_max=0.5) ===")
ess.reset()
# 有功0.3时,最大无功应为 sqrt(0.5^2-0.3^2)=0.4
p, q, soc, cost = ess.step(p_e=0.3, q_e=10.0)  # 请求过大的无功
print(f"请求Q=10MVar -> 实际Q={q:.3f}MVar (理论上限 sqrt(0.5^2-0.3^2)={ (0.5**2-0.3**2)**0.5:.3f})")

print("\n=== 测试4: 有功为0时无功上限应为 S_max ===")
ess.reset()
p, q, soc, cost = ess.step(p_e=0.0, q_e=10.0)
print(f"P=0时请求Q=10MVar -> 实际Q={q:.3f}MVar (应为 S_max=0.5)")

print("\n测试完成。检查:SOC应在[0.1,0.9]内,损耗非负,无功不超容量。")
