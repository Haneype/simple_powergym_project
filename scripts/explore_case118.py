"""
探查 IEEE 118 节点系统:电压分布、薄弱节点,为 ESS 布点做准备
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn

net = pn.case118()
print("=== 网络基本信息 ===")
print(f"母线数量: {len(net.bus)}")
print(f"线路数量: {len(net.line)}")
print(f"负荷数量: {len(net.load)}")
print(f"发电机数量: {len(net.gen) if hasattr(net,'gen') else 0}")
print(f"外部电网数量: {len(net.ext_grid)}")

pp.runpp(net)
print(f"\n潮流收敛: {net.converged}")
vm = net.res_bus.vm_pu
print(f"电压范围: {vm.min():.4f} ~ {vm.max():.4f} p.u.")

print("\n=== 电压最低的 12 个母线 ===")
for bus_idx, v in vm.sort_values().head(12).items():
    print(f"母线 {bus_idx}: {v:.4f} p.u.")

print("\n=== 有负荷的母线(ESS适合放这些地方)===")
load_buses = net.load.bus.values
print(f"负荷母线数: {len(load_buses)}")
# 在有负荷且电压偏低的母线里挑候选
low_load_buses = [(b, vm.at[b]) for b in load_buses]
low_load_buses.sort(key=lambda x: x[1])
print("有负荷且电压最低的 12 个母线:")
for b, v in low_load_buses[:12]:
    print(f"母线 {b}: {v:.4f} p.u., 负荷 {net.load[net.load.bus==b].p_mw.sum():.2f} MW")
