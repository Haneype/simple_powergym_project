"""
探查 CIGRE 中压配电网:看是否适合做多ESS电压调节
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn

# 带分布式能源(光伏+风机)的版本
net = pn.create_cigre_network_mv(with_der="pv_wind")
print("=== CIGRE MV 网络信息 ===")
print(f"母线数量: {len(net.bus)}")
print(f"线路数量: {len(net.line)}")
print(f"负荷数量: {len(net.load)}")
print(f"分布式电源(sgen)数量: {len(net.sgen)}")
if len(net.sgen) > 0:
    print("\n分布式电源明细:")
    print(net.sgen[["name", "bus", "p_mw"]].to_string())

pp.runpp(net)
print(f"\n潮流收敛: {net.converged}")
vm = net.res_bus.vm_pu
print(f"电压范围: {vm.min():.4f} ~ {vm.max():.4f} p.u.")

print("\n=== 各母线电压 ===")
print(vm.to_string())

print("\n=== 负荷明细 ===")
print(net.load[["bus", "p_mw", "q_mvar"]].to_string())
