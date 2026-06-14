"""
多智能体算法基类:定义统一接口。
以后实现 MADDPG / MAPPO 等,只需继承并实现这些方法,
训练脚本和环境都不用改。
"""
from abc import ABC, abstractmethod


class BaseMultiAgentAlgorithm(ABC):
    @abstractmethod
    def select_actions(self, obs, deterministic=False):
        """输入 obs dict,返回 actions dict"""
        ...

    @abstractmethod
    def store(self, obs, actions, rewards, next_obs, done):
        """存一条转移"""
        ...

    @abstractmethod
    def update(self):
        """更新参数,返回指标 dict(可为空)"""
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def load(self, path):
        ...
