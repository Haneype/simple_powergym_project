"""
验证自建 69 节点系统:数据是否录对、电压分布、薄弱节点
预期:总负荷约 3802 kW,潮流收敛,最低电压约 0.909 (节点65附近)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
from env.networks_custom import build_case69

net = build_case69()
print("=== 69节点网络信息 ===")
print(f"母线数量: {len(net.bus)}")
print(f"支路数量: {len(net.line)}")
print(f"负荷数量: {len(net.load)}")
print(f"总有功负荷: {net.load.p_mw.sum()*1000:.2f} kW (预期约 3802)")
print(f"总无功负荷: {net.load.q_mvar.sum()*1000:.2f} kVAr (预期约 2695)")

pp.runpp(net)
print(f"\n潮流收敛: {net.converged}")
vm = net.res_bus.vm_pu
print(f"电压范围: {vm.min():.4f} ~ {vm.max():.4f} p.u.")
print(f"最低电压母线: {vm.idxmin()} (节点{vm.idxmin()+1}), 电压 {vm.min():.4f}")
print(f"  预期最低电压约 0.909,在节点 65 附近")
print(f"总有功网损: {net.res_line.pl_mw.sum()*1000:.2f} kW")

print("\n=== 电压最低的 12 个母线(节点号 = 索引+1)===")
for idx, v in vm.sort_values().head(12).items():
    load = net.load[net.load.bus == idx].p_mw.sum() * 1000
    print(f"母线索引 {idx} (节点{idx+1}): {v:.4f} p.u., 负荷 {load:.1f} kW")
