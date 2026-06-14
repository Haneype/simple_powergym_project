"""
储能 (ESS) 模型
实现:
  - SOC 动态 (论文式12离散化)
  - 功率/SOC 约束修正 (论文式14-15)
  - PCS 无功容量约束 (论文式9-10)
  - 寿命损耗成本 (论文式7)
约定:
  P_e > 0 表示充电, P_e < 0 表示放电 (与论文 SOC 递推式一致)
  功率单位 MW / MVA, 容量单位 MWh, 时间单位 小时
"""

import numpy as np


class ESSModel:
    def __init__(self, cfg, dt):
        """
        cfg: 配置字典中的 ess 部分
        dt:  时间步长(小时)
        """
        self.E_max = cfg["E_max"]          # 额定容量 MWh
        self.S_max = cfg["S_max"]          # PCS 视在功率 MVA
        self.P_max = cfg["P_max"]          # 最大充放电功率 MW
        self.soc_min = cfg["soc_min"]
        self.soc_max = cfg["soc_max"]
        self.soc_init = cfg["soc_init"]
        self.eta_c = cfg["eta_c"]          # 充电效率
        self.eta_dc = cfg["eta_dc"]        # 放电效率
        self.c = cfg["life_loss_coef"]     # 寿命损耗系数
        self.dt = dt

        self.soc = self.soc_init

    def reset(self):
        """回到初始 SOC,episode 开始时调用"""
        self.soc = self.soc_init
        return self.soc

    # ---------- 约束 ----------
    def clip_power(self, p_e):
        """
        将请求的充放电功率修正到可行范围。
        同时考虑:
          1) PCS 最大功率 P_max
          2) SOC 上下限(防止过充/过放)
        返回修正后的功率 p_e (MW)
        """
        # 1) 受 PCS 功率上限约束
        p_e = float(np.clip(p_e, -self.P_max, self.P_max))

        # 2) 受 SOC 约束:计算本时隙允许的最大充/放电功率
        if p_e > 0:  # 充电,SOC 上升,受 soc_max 限制
            # soc + eta_c * p * dt / E_max <= soc_max
            p_allow = (self.soc_max - self.soc) * self.E_max / (self.eta_c * self.dt)
            p_e = min(p_e, max(p_allow, 0.0))
        elif p_e < 0:  # 放电,SOC 下降,受 soc_min 限制
            # soc + (1/eta_dc) * p * dt / E_max >= soc_min  (p<0)
            p_allow = (self.soc_min - self.soc) * self.E_max * self.eta_dc / self.dt
            p_e = max(p_e, min(p_allow, 0.0))
        return p_e

    def max_reactive(self, p_e):
        """
        给定有功 p_e,PCS 可提供的最大无功幅值 (论文式9)
        Q <= sqrt(S_max^2 - P_e^2)
        """
        val = self.S_max ** 2 - p_e ** 2
        return np.sqrt(val) if val > 0 else 0.0

    def clip_reactive(self, q_e, p_e):
        """将请求的无功修正到 PCS 容量允许范围内"""
        q_lim = self.max_reactive(p_e)
        return float(np.clip(q_e, -q_lim, q_lim))

    # ---------- 动态 ----------
    def step(self, p_e, q_e):
        """
        执行一个时隙:
          - 修正功率到可行域
          - 更新 SOC
          - 计算寿命损耗成本
        返回 (p_e_actual, q_e_actual, soc, life_cost)
        """
        # 先修正有功,再据此修正无功(无功容量依赖有功)
        p_e = self.clip_power(p_e)
        q_e = self.clip_reactive(q_e, p_e)

        # 更新 SOC (论文式12)
        if p_e >= 0:   # 充电
            self.soc += self.eta_c * p_e * self.dt / self.E_max
        else:          # 放电
            self.soc += (1.0 / self.eta_dc) * p_e * self.dt / self.E_max

        # 数值保护,防止浮点误差越界
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))

        # 寿命损耗成本 (论文式7)
        life_cost = self.life_loss(p_e)

        return p_e, q_e, self.soc, life_cost

    def life_loss(self, p_e):
        """
        寿命损耗成本 (论文式7)
        充电: c * P_e * (1 - eta_c)
        放电: c * |P_e| * (1/eta_dc - 1)
        """
        if p_e > 0:    # 充电
            return self.c * p_e * (1.0 - self.eta_c) * self.dt
        elif p_e < 0:  # 放电
            return self.c * abs(p_e) * (1.0 / self.eta_dc - 1.0) * self.dt
        return 0.0
