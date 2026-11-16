"""Q2: Vanilla Policy Gradient / REINFORCE on Adaptive Maze."""
import numpy as np

# --- PART A: MANDATORY COMPLETE THE CODE ---
gamma = 0.9
def calculate_returns(rewards, gamma):
    returns = []
    G = 0
    for reward in reversed(rewards):
        # TODO 1: Calculate the discounted return
        G = reward + gamma * G
        # TODO 2: Insert the return at the beginning of the returns list
        returns.insert(0, G)
    # TODO 3: Return the calculated returns
    return returns

# --- PART B: STUDENT B REINFORCE TRAINING (150 EPISODES) ---
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

def train_reinforce(episodes=150, alpha=0.05, gamma=0.9):
    np.random.seed(42)
    theta = np.zeros((25, 4))  # 25 grid cells x 4 actions
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    walls = {(0, 2), (1, 1), (3, 3)}
    ep_rewards, wins = [], 0

    for ep in range(episodes):
        pos = (0, 0)
        treasures = {(0, 3), (2, 3)}
        enemy = (3, 1)
        traj = []
        for _ in range(30):
            s_idx = pos[0] * 5 + pos[1]
            probs = softmax(theta[s_idx])
            a = np.random.choice(4, p=probs)
            nr, nc = pos[0] + actions[a][0], pos[1] + actions[a][1]
            if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in walls:
                pos = (nr, nc)
            r = -1
            if pos in treasures:
                r += 10
                treasures.remove(pos)
            if pos == enemy:
                r -= 15
                traj.append((s_idx, a, r))
                break
            if pos == (4, 4):
                r += 25
                wins += 1
                traj.append((s_idx, a, r))
                break
            traj.append((s_idx, a, r))

        returns = calculate_returns([x[2] for x in traj], gamma)
        ep_rewards.append(sum(x[2] for x in traj))
        for (s_idx, a, _), G in zip(traj, returns):
            probs = softmax(theta[s_idx])
            grad_log = -probs
            grad_log[a] += 1.0
            theta[s_idx] += alpha * G * grad_log

    print(f"REINFORCE ({episodes} eps) -> Avg Last-20 Reward: {np.mean(ep_rewards[-20:]):.2f}, Win Rate: {wins/episodes*100:.1f}%")

if __name__ == "__main__":
    sample_rewards = [1, 0, 2, 10]
    returns = calculate_returns(sample_rewards, gamma)
    print("Discounted Returns:", [round(x, 3) for x in returns])
    train_reinforce()
