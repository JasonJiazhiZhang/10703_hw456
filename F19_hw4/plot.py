mport numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  
eval_stats = np.load("data/test_avg_rewards.npy")
eval_interval = 100
y = eval_stats[:, 0]
err = eval_stats[:, 1]
plt.errorbar(range(0, len(y)*eval_interval, eval_interval), y, err)
plt.ylabel("Average reward")
plt.xlabel("Num of episodes")
plt.save('avg_reward.png')
