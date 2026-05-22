import sys

sys.path.append(r"C:\PycharmProjects\powergym-master")

from powergym.env_register import make_env

env_name = "13Bus"
env = make_env("13Bus")


print("=" * 40)
print("Environment:", env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Horizon:", env.horizon)

print("Capacitors:", env.cap_num)
print("Regulators:", env.reg_num)
print("Batteries:", env.bat_num)