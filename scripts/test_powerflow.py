"""
验证 pandapower 能建 IEEE 33 节点系统并跑潮流
重点:找出电压最低的节点,决定 ESS 该放在哪里
"""
import pandapower as pp
import pandapower.networks as pn

# 加载 IEEE 33 节点配电系统
net = pn.case33bw()

print("=== 网络基本信息 ===")
print(f"母线数量: {len(net.bus)}")
print(f"线路数量: {len(net.line)}")
print(f"负荷数量: {len(net.load)}")
print(f"外部电网(slack)数量: {len(net.ext_grid)}")

# 跑潮流
pp.runpp(net)

print("\n=== 潮流计算结果 ===")
print(f"潮流是否收敛: {net.converged}")
vm = net.res_bus.vm_pu
print(f"电压幅值范围: {vm.min():.4f} ~ {vm.max():.4f} p.u.")
print(f"最低电压在母线: {vm.idxmin()}  (电压={vm.min():.4f})")
print(f"总有功负荷: {net.load.p_mw.sum():.4f} MW")
print(f"总网损(有功): {net.res_line.pl_mw.sum():.4f} MW")

print("\n=== 电压最低的 8 个母线 ===")
low_buses = vm.sort_values().head(8)
for bus_idx, v in low_buses.items():
    print(f"母线 {bus_idx}: {v:.4f} p.u.")

print("\n=== 所有母线电压 ===")
print(vm.to_string())
