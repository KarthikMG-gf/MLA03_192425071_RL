"""Q5: DDPG Target Network Soft Update & 2D Continuous Control Step."""
import numpy as np

# --- PART A: MANDATORY COMPLETE THE CODE ---
main_weights = np.array([0.8, 0.6, 0.4])
target_weights = np.array([0.2, 0.3, 0.1])
tau = 0.1

# TODO 1: Calculate the contribution from main network weights
main_contrib = tau * main_weights
# TODO 2: Calculate the contribution from target network weights
target_contrib = (1.0 - tau) * target_weights
# TODO 3: Calculate updated target network weights
target_weights = main_contrib + target_contrib

print("Updated Target Weights:", target_weights)

# --- PART B: STUDENT B 2D CONTINUOUS NAVIGATION DEMO ---
class Continuous2DMazeEnv:
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.goal = np.array([4.0, 4.0])
        self.enemy = np.array([2.0, 2.2])

    def step(self, action_vel):
        # action_vel in [-1.0, 1.0]^2
        vel = np.clip(action_vel, -1.0, 1.0)
        self.pos = np.clip(self.pos + 0.5 * vel, 0.0, 4.0)
        dist_goal = np.linalg.norm(self.goal - self.pos)
        dist_enemy = np.linalg.norm(self.enemy - self.pos)
        reward = -0.5 * dist_goal
        done = False
        if dist_enemy < 0.4:
            reward -= 15.0
            done = True
        elif dist_goal < 0.3:
            reward += 25.0
            done = True
        return self.pos.copy(), reward, done

if __name__ == "__main__":
    env = Continuous2DMazeEnv()
    state, r, d = env.step(np.array([0.8, 0.6]))
    print(f"2D Continuous Step -> New Pos: {state}, Reward: {r:.2f}, Done: {d}")
