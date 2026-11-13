import numpy as np
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Warehouse environment (same map and rewards as the assignment)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE, START, GOAL = 5, (0, 0), (4, 4)
OBSTACLES = {(1, 1), (1, 3), (2, 3), (3, 1)}
GAMMA, ALPHA, EPSILON = 0.95, 0.15, 0.15
CHECKPOINTS = [10, 25, 50, 100, 200, 500]
EVAL_EPISODES, MAX_STEPS = 100, 100

class Warehouse:
    def reset(self):
        self.position = START
        return self.position

    def step(self, action):
        dr, dc = ACTIONS[int(action)]
        r, c = self.position
        nxt = (r + dr, c + dc)
        if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or nxt in OBSTACLES:
            nxt, reward = self.position, -10.0
        else:
            reward = -1.0
        done = nxt == GOAL
        if done:
            reward = 20.0
        self.position = nxt
        return nxt, reward, done

def transition(state, action):
    dr, dc = ACTIONS[action]
    nxt = (state[0] + dr, state[1] + dc)
    if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or nxt in OBSTACLES:
        return state, -10.0
    return nxt, 20.0 if nxt == GOAL else -1.0

STATES = [(r, c) for r in range(SIZE) for c in range(SIZE)
          if (r, c) not in OBSTACLES]
INDEX = {s: i for i, s in enumerate(STATES)}

def value_iteration():
    values = {s: 0.0 for s in STATES}
    for _ in range(2000):
        delta = 0.0
        for s in STATES:
            if s == GOAL:
                continue
            new_v = max(
                transition(s, a)[1] + GAMMA * values[transition(s, a)[0]]
                for a in ACTIONS
            )
            delta = max(delta, abs(new_v - values[s]))
            values[s] = new_v
        if delta < 1e-10:
            break
    return {
        s: int(np.argmax([
            transition(s, a)[1] + GAMMA * values[transition(s, a)[0]]
            for a in ACTIONS
        ]))
        for s in STATES if s != GOAL
    }

def choose_action(q, state, rng):
    if rng.random() < EPSILON:
        return int(rng.integers(4))
    row = q[INDEX[state]]
    best = np.flatnonzero(np.isclose(row, row.max()))
    return int(rng.choice(best))

def train_td(method, seed):
    rng = np.random.default_rng(seed)
    env = Warehouse()
    q = np.zeros((len(STATES), 4))
    history = {}
    episode_returns, episode_success = [], []

    for episode in range(1, max(CHECKPOINTS) + 1):
        state = env.reset()
        action = choose_action(q, state, rng) if method == "SARSA" else None
        total, done = 0.0, False

        for _ in range(MAX_STEPS):
            if method == "Q-Learning":
                action = choose_action(q, state, rng)
            next_state, reward, done = env.step(action)
            total += reward

            if done:
                target = reward
                next_action = None
            elif method == "SARSA":
                next_action = choose_action(q, next_state, rng)
                target = reward + GAMMA * q[INDEX[next_state], next_action]
            else:
                next_action = None
                target = reward + GAMMA * np.max(q[INDEX[next_state]])

            q[INDEX[state], action] += ALPHA * (
                target - q[INDEX[state], action]
            )
            state = next_state
            action = next_action
            if done:
                break

        episode_returns.append(total)
        episode_success.append(int(done))

        if episode in CHECKPOINTS:
            history[episode] = {
                "training_mean_return_last_10": float(np.mean(episode_returns[-10:])),
                "training_success_rate_last_10": float(np.mean(episode_success[-10:])),
                "greedy_evaluation": evaluate_q(q, seed=seed + 10000 + episode)
            }

    return history

def evaluate_policy(policy, seed=123):
    env = Warehouse()
    successes, returns, steps_list = 0, [], []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        total, done = 0.0, False
        steps = 0
        while steps < MAX_STEPS and not done:
            action = policy[state]
            state, reward, done = env.step(action)
            total += reward
            steps += 1
        successes += int(done)
        returns.append(total)
        steps_list.append(steps)
    return {
        "success_rate": successes / EVAL_EPISODES,
        "mean_reward": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps_list))
    }

def evaluate_q(q, seed=123):
    policy = {
        s: int(np.argmax(q[INDEX[s]]))
        for s in STATES if s != GOAL
    }
    return evaluate_policy(policy, seed)

def main():
    vi = evaluate_policy(value_iteration())
    all_results = {
        "Value Iteration": {
            "training_episodes": 0,
            "training_mean_return_last_10": None,
            "training_success_rate_last_10": None,
            "greedy_evaluation": vi
        }
    }

    for method, seed in [("SARSA", 42), ("Q-Learning", 43)]:
        history = train_td(method, seed)
        for ep, result in history.items():
            all_results.setdefault(method, {})[str(ep)] = result

    out = Path(__file__).resolve().parents[1] / "Outputs"
    figdir = Path(__file__).resolve().parents[1] / "Figures"
    out.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)
    (out / "comparison_checkpoints.json").write_text(
        json.dumps(all_results, indent=2)
    )

    rows = []
    for method, data in all_results.items():
        if method == "Value Iteration":
            ev = data["greedy_evaluation"]
            rows.append([method, 0, "", "", ev["success_rate"],
                         ev["mean_reward"], ev["mean_steps"]])
        else:
            for ep in CHECKPOINTS:
                item = data[str(ep)]
                ev = item["greedy_evaluation"]
                rows.append([method, ep,
                             item["training_mean_return_last_10"],
                             item["training_success_rate_last_10"],
                             ev["success_rate"], ev["mean_reward"],
                             ev["mean_steps"]])
    with (out / "comparison_checkpoints.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Training episodes",
                         "Mean training return (last 10)",
                         "Training success rate (last 10)",
                         "Greedy evaluation success rate",
                         "Greedy evaluation mean reward",
                         "Greedy evaluation mean steps"])
        writer.writerows(rows)

    # A learning-curve comparison is more informative than a single
    # final success-rate bar chart when all methods reach 100%.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, marker in [("SARSA", "o"), ("Q-Learning", "s")]:
        xs = CHECKPOINTS
        ys_success = [all_results[method][str(ep)]["greedy_evaluation"]["success_rate"] * 100
                      for ep in xs]
        ys_reward = [all_results[method][str(ep)]["training_mean_return_last_10"]
                     for ep in xs]
        axes[0].plot(xs, ys_success, marker=marker, linewidth=2, label=method)
        axes[1].plot(xs, ys_reward, marker=marker, linewidth=2, label=method)

    axes[0].axhline(vi["success_rate"] * 100, linestyle="--",
                    label="Value Iteration (planning baseline)")
    axes[0].set_title("Greedy evaluation success rate")
    axes[0].set_xlabel("TD training episodes")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Recent training performance")
    axes[1].set_xlabel("TD training episodes")
    axes[1].set_ylabel("Mean return (last 10 episodes)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Warehouse navigation: learning progress comparison")
    fig.tight_layout()
    fig.savefig(figdir / "learning_progress_comparison.png", dpi=180)
    plt.close(fig)

    print("Value Iteration greedy evaluation:", vi)
    for method in ["SARSA", "Q-Learning"]:
        print("\n" + method)
        for ep in CHECKPOINTS:
            item = all_results[method][str(ep)]
            print(f"{ep:>3} episodes | recent train return="
                  f"{item['training_mean_return_last_10']:.2f} | "
                  f"recent train success="
                  f"{item['training_success_rate_last_10']:.2f} | "
                  f"greedy eval success="
                  f"{item['greedy_evaluation']['success_rate']:.2f} | "
                  f"eval reward={item['greedy_evaluation']['mean_reward']:.2f} | "
                  f"eval steps={item['greedy_evaluation']['mean_steps']:.2f}")
    print("\nSaved CSV, JSON and learning_progress_comparison.png")

if __name__ == "__main__":
    main()
