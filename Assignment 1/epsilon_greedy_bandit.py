import numpy as np

np.random.seed(7)
true_rewards = [1.0, 1.5, 2.0, 1.2]  # illustrative reward means
n_arms = len(true_rewards)
epsilon = 0.1
steps = 1000

counts = np.zeros(n_arms, dtype=int)
estimates = np.zeros(n_arms, dtype=float)
total_reward = 0.0

for step in range(steps):
    if np.random.rand() < epsilon:
        action = np.random.randint(n_arms)
    else:
        action = int(np.argmax(estimates))

    reward = np.random.normal(true_rewards[action], 1.0)
    counts[action] += 1
    estimates[action] += (reward - estimates[action]) / counts[action]
    total_reward += reward

print("Arm selections:", counts)
print("Estimated rewards:", np.round(estimates, 3))
print("Average reward:", round(total_reward / steps, 3))
