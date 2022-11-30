from collections import deque
import os
import shutil
import time
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 0
MAX_STEPS = 500_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
if os.path.exists(SAVE_PREFIX):
    shutil.rmtree(SAVE_PREFIX)
os.mkdir(SAVE_PREFIX)
localtime = time.asctime(time.localtime(time.time()))
with open("rewards.txt", "a") as fp:
    fp.write(f"{localtime}\n")

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    restore="./pretrain/model_weights_b"
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
# observation: 对当前情况的观测
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)  # 根据当前状态选择策略；根据training选择是否更新epsilon
    obs, reward, done = env.step(action)  # 实行这个策略后，获得状态的改变以及reward信息。
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)  # 经验池

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()  # 将权重从策略网络同步到目标网络

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
