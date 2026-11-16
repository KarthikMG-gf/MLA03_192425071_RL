"""Q4: Proximal Policy Optimization (PPO) Clipped Surrogate Objective."""
import numpy as np

# --- PART A: MANDATORY COMPLETE THE CODE ---
old_probability = 0.4
new_probability = 0.5
advantage = 2.0
epsilon = 0.2

# TODO 1: Calculate probability ratio
ratio = new_probability / old_probability
# TODO 2: Calculate the unclipped objective
objective1 = ratio * advantage
# TODO 3: Calculate the clipped probability ratio
clipped_ratio = max(1 - epsilon, min(1 + epsilon, ratio))
# TODO 4: Calculate the clipped objective
objective2 = clipped_ratio * advantage
# TODO 5: Select the minimum objective
ppo_objective = min(objective1, objective2)

print("Probability Ratio:", ratio)
print("Clipped Ratio:", clipped_ratio)
print("PPO Objective:", ppo_objective)

# --- PART B: STUDENT B CLIPPING SENSITIVITY TEST ---
if __name__ == "__main__":
    for eps in (0.05, 0.2, 0.5):
        c_rat = np.clip(ratio, 1 - eps, 1 + eps)
        obj = min(ratio * advantage, c_rat * advantage)
        print(f"Epsilon={eps:.2f} -> Clipped Ratio={c_rat:.2f}, PPO Objective={obj:.2f}")
