"""Q3: Actor-Critic TD Error, Advantage & A2C Agent."""
import numpy as np

# --- PART A: MANDATORY COMPLETE THE CODE ---
gamma = 0.9
reward = 5
value_current = 8
value_next = 10

# TODO 1: Calculate the TD target
td_target = reward + gamma * value_next
# TODO 2: Calculate the TD error
td_error = td_target - value_current
# TODO 3: Calculate the advantage
advantage = td_error

print("TD Target:", td_target)
print("TD Error:", td_error)
print("Advantage:", advantage)

# --- PART B: STUDENT B ACTOR-CRITIC TRAINING ---
def run_a2c(episodes=150, lr_actor=0.05, lr_critic=0.1, gamma=0.9):
    np.random.seed(7)
    actor_theta = np.zeros((25, 4))
    critic_v = np.zeros(25)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    walls = {(0, 2), (1, 1), (3, 3)}
    wins = 0

    for _ in range(episodes):
        pos = (0, 0)
        treasures = {(0, 3), (2, 3)}
        for _ in range(30):
            s = pos[0] * 5 + pos[1]
            e = np.exp(actor_theta[s] - np.max(actor_theta[s]))
            probs = e / np.sum(e)
            a = np.random.choice(4, p=probs)
            nr, nc = pos[0] + actions[a][0], pos[1] + actions[a][1]
            if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in walls:
                pos = (nr, nc)
            r = -1
            if pos in treasures:
                r += 10
                treasures.remove(pos)
            done = (pos == (4, 4)) or (pos == (3, 1))
            if pos == (4, 4):
                r += 25
                wins += 1
            elif pos == (3, 1):
                r -= 15

            s_next = pos[0] * 5 + pos[1]
            target = r + (0.0 if done else gamma * critic_v[s_next])
            delta = target - critic_v[s]
            critic_v[s] += lr_critic * delta
            grad_log = -probs
            grad_log[a] += 1.0
            actor_theta[s] += lr_actor * delta * grad_log
            if done:
                break
    print(f"A2C ({episodes} eps) -> Win Rate: {wins/episodes*100:.1f}%")

if __name__ == "__main__":
    run_a2c()
