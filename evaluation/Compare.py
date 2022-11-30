import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


duel = []
t = open(f"./Duelings/rewards.txt", "r")
for line in t:
    tmp = re.findall(r"\d+\.?\d*", line)
    tmp = [float(i) for i in tmp]
    duel.append(tmp)
duel = np.array(duel, dtype=float)
duel = duel.transpose()
# duel[1, : ] = duel[1, :].astype(int)
print(duel)

ori = []
t = open(f"./Origins/rewards.txt", "r")
for line in t:
    tmp = re.findall(r"\d+\.?\d*", line)
    tmp = [float(i) for i in tmp]
    ori.append(tmp)
ori = np.array(ori, dtype=float)
ori = ori.transpose()
# ori[1, : ] = ori[1, :].astype(int)
print(ori)


END = 300
cut = 5
plt.figure()
plt.plot([np.mean(duel[1, i*cut:(i+1)*cut])/100000 for i in range(END//cut)], duel[2,0:END//cut], color='red', label="DuelingDQN")
plt.plot([np.mean(ori[1, i*cut:(i+1)*cut])/100000 for i in range(END//cut)], ori[2,0:END//cut], color='blue', label="Origin")
plt.legend(loc='best')
plt.title("Rewards During Training")
plt.savefig("Rewards During Training.png")
plt.show()

Ori = np.load(f"avgOri.npy")
Duel = np.load(f"avgDuel.npy")
plt.figure()
plt.plot([i*cut for i in range(END//cut)], [np.mean(Duel[i*cut:(i+1)*cut]) for i in range(END//cut)], color='red', label="DuelingDQN")
plt.plot([i*cut for i in range(END//cut)], [np.mean(Ori[i*cut:(i+1)*cut]) for i in range(END//cut)], color='blue', label="Origin")
plt.legend(loc='best')
plt.title("Rewards Without ε-greedy")
plt.savefig("Rewards Without ε-greedy.png")
plt.show()
