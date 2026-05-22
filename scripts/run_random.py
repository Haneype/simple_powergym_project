import sys

sys.path.append(r"C:\PycharmProjects\powergym-master")

from powergym.env_register import make_env

env = make_env("13Bus")
obs = env.reset(load_profile_idx=0)

done = False
total_reward = 0
step_count = 0

while not done:
    action = env.random_action()
    obs, reward, done, info = env.step(action)

    total_reward += reward
    step_count += 1

    print("Step:", step_count)
    print("Action:", action)
    print("Reward:", reward)
    print("Done:", done)
    print("-" * 40)

print("Total reward:", total_reward)
