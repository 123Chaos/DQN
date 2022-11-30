import base64
import time
from collections import deque
import os
import pathlib
import shutil

from IPython import display as ipydisplay
import torch
import numpy as np
from utils_env import MyEnv
from utils_drl import Agent
import matplotlib.pyplot as plt
from utils_drl_dueling import Agent as duelingAgent

end_target = 500

avg_rewards1 = []
for target in range(end_target):
    model_name = f"model_{target:03d}"
    model_path = f"./Duelings/models/{model_name}"
    # model_path = f"./models/model_004"
    device = torch.device("cuda")
    env = MyEnv(device)
    agent = duelingAgent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path)

    obs_queue = deque(maxlen=5)
    avg_reward, frames = env.evaluate(obs_queue, agent, render=False)
    avg_rewards1.append(avg_reward)
    print(f"Duel {model_name} Avg. Reward: {avg_reward:.1f}.Time:{time.time()}")

# target_dir = "render086"
# os.mkdir(target_dir)
# for ind, frame in enumerate(frames):
#     frame.save(os.path.join(target_dir, f"{ind:06d}.png"), format="png")

# avg_rewards2 = []
# for target in range(end_target):
#     model_name = f"model_{target:03d}"
#     model_path = f"./Origins/models/{model_name}"
#     # model_path = f"./models/model_004"
#     device = torch.device("cuda:0")
#     env = MyEnv(device)
#     agent = Agent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path)
#
#     obs_queue = deque(maxlen=5)
#     avg_reward, frames = env.evaluate(obs_queue, agent, render=False)
#     avg_rewards2.append(avg_reward)
#     print(f"Origin {model_name} Avg. Reward: {avg_reward:.1f}.Time:{time.time()}")

x = [i for i in range(end_target)]
plt.figure()
plt.plot(x, avg_rewards1, color='red')
# plt.plot(x, avg_rewards2, color='blue')
avgDuel = np.array(avg_rewards1)
np.save("avgDuel.npy", avgDuel)
# avgOri = np.array(avg_rewards2)
# np.save("avgOri.npy", avgOri)

plt.show()
