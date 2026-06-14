"""
配电网电压调节环境 (IEEE 33节点 + 分布式储能)
遵循 Gym 风格的多智能体接口:
  reset() -> obs_dict
  step(actions_dict) -> (obs_dict, rewards_dict, done, info)

每个 ESS 是一个 agent。
动作:每个 agent 输出 [P, Q],归一化到 [-1, 1]
观测:本地电压、SOC、时刻、本地负荷/光伏 (部分可观测,论文式13)
奖励:分段设计 (论文式16/17),全局共享(协作式)
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from gymnasium import spaces

from env.ess_model import ESSModel
from env.networks_custom import build_case69



class GridEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        ecfg = cfg["env"]
        self.v_min = ecfg["v_min"]
        self.v_max = ecfg["v_max"]
        self.episode_steps = ecfg["episode_steps"]
        self.dt = ecfg["dt"]
        self.rho1 = cfg["reward"]["rho1"]

        # ---- 加载网络(按配置切换系统) ----
        system = ecfg["system"]
        net_loaders = {
            "case33bw": pn.case33bw,
            "case118": pn.case118,
            "case69": build_case69,  # 自建69节点
            "cigre_mv": lambda: pn.create_cigre_network_mv(with_der=False),
        }
        if system not in net_loaders:
            raise ValueError(f"不支持的系统: {system}")
        self.net = net_loaders[system]()
        pp.runpp(self.net)  # 先跑一次确认可用

        # 记录每条负荷的基准值,后面乘以时变曲线
        self.base_load_p = self.net.load.p_mw.values.copy()
        self.base_load_q = self.net.load.q_mvar.values.copy()

        # ---- 创建 ESS ----
        self.ess_buses = cfg["ess"]["buses"]
        self.n_agents = len(self.ess_buses)
        self.agents = [f"ess_{i}" for i in range(self.n_agents)]

        # 每个 ESS 在网络里建一个 sgen,初始 P=Q=0
        self.ess_sgen_idx = []
        for bus in self.ess_buses:
            idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0,
                                 name=f"ESS_at_bus{bus}")
            self.ess_sgen_idx.append(idx)

        # 每个 ESS 一个模型实例
        self.ess_models = [ESSModel(cfg["ess"], self.dt)
                           for _ in range(self.n_agents)]
        self.P_max = cfg["ess"]["P_max"]
        self.S_max = cfg["ess"]["S_max"]

        # ---- 在 ESS 节点也加光伏 (sgen),制造过压场景 ----
        # 光伏容量设为略大于 ESS,白天注入会抬高电压
        self.pv_sgen_idx = []
        self.pv_cap = 0.9  # MW 光伏峰值容量
        for bus in self.ess_buses:
            idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0,
                                 name=f"PV_at_bus{bus}")
            self.pv_sgen_idx.append(idx)

        # ---- 生成 24 小时曲线 ----
        self._build_profiles()

        # ---- 定义空间 ----
        # 动作: [P_norm, Q_norm] in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)
        # 观测维度: [本地电压, SOC, 时刻sin, 时刻cos, 本地负荷, 本地光伏]
        self.obs_dim = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(self.obs_dim,), dtype=np.float32)

        self.t = 0

    def _build_profiles(self):
        """生成 24 小时的负荷系数曲线和光伏出力系数曲线 (归一化)"""
        hours = np.arange(self.episode_steps)
        # 负荷:双峰(早8点、晚19点),夜间低
        load = (0.6
                + 0.25 * np.exp(-((hours - 8) ** 2) / 8)
                + 0.35 * np.exp(-((hours - 19) ** 2) / 6))
        self.load_profile = load / load.max()  # 归一到峰值1

        # 光伏:正午鼓包,夜间0
        pv = np.maximum(0, np.sin((hours - 6) / 12 * np.pi))
        pv[(hours < 6) | (hours > 18)] = 0.0
        self.pv_profile = pv  # 已在 [0,1]

    def reset(self):
        self.t = 0
        for m in self.ess_models:
            m.reset()
        self._apply_profiles()
        # ESS 出力清零
        for idx in self.ess_sgen_idx:
            self.net.sgen.at[idx, "p_mw"] = 0.0
            self.net.sgen.at[idx, "q_mvar"] = 0.0
        pp.runpp(self.net)
        return self._get_obs()

    def _apply_profiles(self):
        """把当前时刻 t 的负荷/光伏系数应用到网络"""
        k_load = self.load_profile[self.t]
        k_pv = self.pv_profile[self.t]
        # 负荷
        self.net.load.p_mw = self.base_load_p * k_load
        self.net.load.q_mvar = self.base_load_q * k_load
        # 光伏注入
        for idx in self.pv_sgen_idx:
            self.net.sgen.at[idx, "p_mw"] = self.pv_cap * k_pv

    def step(self, actions):
        """
        actions: dict {agent_id: np.array([P_norm, Q_norm])}
        """
        life_costs = []
        # 1) 把每个 agent 的归一化动作转成实际功率,经 ESS 模型修正后写入网络
        for i, agent in enumerate(self.agents):
            a = np.asarray(actions[agent], dtype=np.float32)
            p_req = float(a[0]) * self.P_max      # 反归一化到 [-P_max, P_max]
            q_req = float(a[1]) * self.S_max      # 反归一化到 [-S_max, S_max]

            # 注意符号约定:
            # ESS模型里 P>0=充电(吸收), 但 sgen 注入 P>0=向电网供电(放电)
            # 所以传给 ESS 模型时取动作原值(P_req>0 视为充电),
            # 写入 sgen 时取负:充电=从电网取电=sgen注入为负
            p_act, q_act, soc, cost = self.ess_models[i].step(p_req, q_req)
            life_costs.append(cost)

            # 写入网络: sgen 注入 = -p_act (充电为负注入), 无功直接注入
            self.net.sgen.at[self.ess_sgen_idx[i], "p_mw"] = -p_act
            self.net.sgen.at[self.ess_sgen_idx[i], "q_mvar"] = q_act

        # 2) 跑潮流
        try:
            pp.runpp(self.net)
            converged = self.net.converged
        except Exception:
            converged = False

        # 3) 计算奖励
        reward = self._compute_reward(life_costs, converged)
        rewards = {agent: reward for agent in self.agents}  # 协作:共享奖励

        # 4) 推进时间
        self.t += 1
        done = self.t >= self.episode_steps

        # 5) 下一时刻的负荷/光伏(若未结束)
        if not done:
            self._apply_profiles()
            try:
                pp.runpp(self.net)
            except Exception:
                pass

        obs = self._get_obs()
        info = {
            "converged": converged,
            "v_min": float(self.net.res_bus.vm_pu.min()),
            "v_max": float(self.net.res_bus.vm_pu.max()),
            "p_loss": float(self.net.res_line.pl_mw.sum()),
            "life_cost": float(sum(life_costs)),
        }
        return obs, rewards, done, info

    def _compute_reward(self, life_costs, converged):
        """论文式16/17 分段奖励"""
        if not converged:
            return -100.0  # 潮流不收敛,重罚

        vm = self.net.res_bus.vm_pu.values
        # 越限量
        over = np.maximum(vm - self.v_max, 0.0).sum()
        under = np.maximum(self.v_min - vm, 0.0).sum()
        violation = over + under

        if violation > 1e-6:
            # 有越限:惩罚越限量 (式16)
            return -self.rho1 * violation
        else:
            # 无越限:负的 (网损 + 寿命损耗) (式17)
            p_loss = self.net.res_line.pl_mw.sum()
            return -(p_loss + sum(life_costs))

    def _get_obs(self):
        """每个 agent 的局部观测 (部分可观测, 论文式13)"""
        obs = {}
        vm = self.net.res_bus.vm_pu
        # 时刻编码 (周期性)
        ang = 2 * np.pi * self.t / self.episode_steps
        t_sin, t_cos = np.sin(ang), np.cos(ang)
        k_load = self.load_profile[min(self.t, self.episode_steps - 1)]
        k_pv = self.pv_profile[min(self.t, self.episode_steps - 1)]

        for i, agent in enumerate(self.agents):
            bus = self.ess_buses[i]
            local_v = float(vm.at[bus])
            soc = self.ess_models[i].soc
            obs[agent] = np.array([local_v, soc, t_sin, t_cos, k_load, k_pv],
                                  dtype=np.float32)
        return obs
