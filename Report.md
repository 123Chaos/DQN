[toc]

# *Reinforcement Learning and Game Theory, Fall, 2022*

> Based on the Atari game breakout.
>
> All Source Code AND Parameters are based on: https://gitee.com/goluke/dqn-breakout

### Team Information

- 任铭
- 20337231

<hr>

- 范云骢
- 20337191

### Introduction

  In some cases, Q-learning algorithm cannot solve problems effectively. For example, Q-table can be used to store the Q value of each state action pair when the state and action space is discrete and the dimension is not high, while Q-table is unrealistic when the state and action space is high-dimensional continuous. Therefore, in order to deal with problems effectively, people found the DQN algorithm.

  DQN (Deep Q-Network) is the groundbreaking work of Deep Reinforcement Learning. It introduces deep learning into reinforcement learning and builds the End-to-end architecture of Perception to Decision. DQN was originally published by DeepMind in NIPS 2013, and an improved version was published in Nature 2015.

##### NIPS DQN

- The following figure shows the pseudocode of DQN in NIP 2013:

![1](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\1.png)



##### Nature DQN

- In Nature 2015, DQN was improved, and a target network was proposed. The pseudocode is shown below:

![2](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\2.png)

In fact, when we calculate the value y~i~ , we use the target network, and then we are actually trained to be main network, and then every C steps, the target network parameter will be "assigned" to the main network parameter. So it's going to be updated once.

<hr>

### Source Code

###### How DQN trains

![3](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\3.png)

###### main.py

> Required Parameters.

```python
GAMMA = 0.99  # discount factor
GLOBAL_SEED = 0  
MEM_SIZE = 100_000  # total memory size 
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4  # the frequency of updating policy network
TARGET_UPDATE = 10_000  # the frequency of updating target network
WARM_STEPS = 50_000  # the number of warming steps
MAX_STEPS = 50_000_000  # total learning steps  
EVALUATE_FREQ = 100_000
```

> Check and modify the game

```python
if done:
    observations, _, _ = env.reset()
    for obs in observations:
        obs_queue.append(obs)
```

> Observe the current state

```pyth
state = env.make_state(obs_queue).to(device).float()
```

> Select an action according to the current state using $\epsilon-greedy$ algorithm

```python
action = agent.run(state, training)
```

> Execute the action and get the information of observation, reward and done

```python
obs, reward, done = env.step(action)	
```

> Add observation to the observation queue and push the current state, action, reward and done into memory replay

```python
obs_queue.append(obs)
memory.push(env.make_folded_state(obs_queue), action, reward, done)
```

> Check if it is time to update the policy network and target network

```python
if step % POLICY_UPDATE == 0 and training:
	agent.learn(memory, BATCH_SIZE)

if step % TARGET_UPDATE == 0:
	agent.sync()   
```

> Check if it is time to record the reward information and the performance of the network, and save them to local disk

```python
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
```

###### utils_drl.py

- The purpose of this file is to define an agent, which has four behaviors.

1. `run()`: Decide the next action based on the current state
2. `learn()`: Information obtained from the experience pool by random sampling is used to update parameters in the policy grid
3. `sync()`: Example Synchronize the weight from the policy network to the target network
4. `save()`: Save the structure and parameters of the policy network to the local disk

###### utils_env.py

- The purpose of this file is to create an execution environment for the breakout and various definitions to get and manipulate the current state. The main functions are as follows.

1. `reset()`: Initialize the game, set the agent to the initial state, and keep the same 5 steps, observe the initial environment
2. `step()`: Receive the action sequence number and execute it, returning the next state reward and information about whether the game is complete
3. `evaluate()`: The performance is evaluated by running the game set and the average reward is returned

###### utils_memory.py

- This file gives the definition of the replay memory pool, which structure is a simple deque.

![4](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\4.png)

> it can store the last $n$ experiences at most where $n$ is the capacity of the pool. As for the sampling process, it use the random sampling to avoid the data dependencies. Like above.

###### utils_model.py

- The definition of this file is to create a neural network.

1. `__init__()`: It consists of three convolution layers and two fully connected layers, the output of the last layer corresponds to the Q value of each action.
2. `forward()`: Method function of network computation.Which is followed by function `relu()`.
3. `relu()`: Activation function to increase the nonlinearity of the network model.

<hr>

### Dueling DQN

> Here is how Dueling DQN works. We can clearly see that the normal DQN above has only one output, which is the Q value of each action; The Dueling DQN breaks down the Value of the state and the Advantage of each action.

![8](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\8.png)
$$
Q(s, a;\theta,\alpha,\beta) = V(s;\theta,\beta) + (A(s,a;\theta,\alpha)-\frac{1}{\mathcal{A}}\sum_{a'\in|\mathcal{A}|} A(s,a';\theta,\alpha))
$$

​	This formula mainly centralizes the advantage function, aiming to embody the respective influence of value function and advantage function. The other part is the same as the nature DQN. The main changes of codes are shown below. It breaks down the value of the state, plus the advantage of each action on that state. Because sometimes, no matter what you do in one state, it doesn't have much of an impact on the next state.

> DQN

![4](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\4.png)

> Dueling-DQN

![5](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\5.png)

##### Results

> Rewards During Training

We can find out that DuelingDQN works better than OriginDQN while training

![6](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\6.png)

> Rewards Without $\epsilon-greedy$



![7](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\7.png)

<hr>

### Summary

​	We use `.Ipynb` for the model performance. Similar to Q-leanring, DQN is an algorithm based on value iteration. However, in ordinary Q-learning, Q-table can be used to store the Q value of each state action pair when the state and action space is discrete and the dimension is not high. However, when the state and action space is high-dimensional continuous, It is difficult to use Q-Table without too much action space and state. Therefore, Q-table can be updated into a function fitting problem here, and Q-Table can be replaced by a function fitting to generate Q values, so that similar states can get similar output actions. Therefore, it can be thought that deep neural network has a good effect on the extraction of complex features, so Deep Learning can be combined with Reinforcement Learning. This becomes DQN.

<hr>

### Distribution

![9](C:\Users\Chaos\Desktop\{20337231}+{任铭}+{20337191}+{范云骢}+{Mid-term Assignments}\assets\9.png)

### References

[【强化学习】Deep Q-Network (DQN) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/108286901)

[DQN从入门到放弃5 深度解读DQN算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/21421729)

[深度强化学习中深度Q网络（Q-Learning+CNN）的讲解以及在Atari游戏中的实战（超详细 附源码）_showswoller的博客-CSDN博客](https://blog.csdn.net/jiebaoshayebuhui/article/details/128045201)

•[https://](https://arxiv.org/pdf/1511.06581.pdf)[arxiv.org/pdf/1511.06581.pdf](https://arxiv.org/pdf/1511.06581.pdf)