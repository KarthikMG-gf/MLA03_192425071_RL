import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset" / "traffic_arrivals.csv"
OUT = ROOT / "Outputs"
OUT.mkdir(exist_ok=True)

# Four candidate green-split timing options (service capacities per approach/step).
# They are illustrative assumptions, not real-world calibrated signal timings.
CAPACITIES = np.array([5.0, 6.0, 7.0, 8.0])
N_EPISODES = 80
STEPS_PER_EPISODE = 12
EPSILON = 0.10
SEED = 23

def load_arrivals():
    rows = []
    with DATA.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append([int(row[k]) for k in
                         ("north_arrivals", "east_arrivals",
                          "south_arrivals", "west_arrivals")])
    return np.asarray(rows, dtype=float)

def one_step(queue, arrivals, arm, rng):
    # The selected timing option sets a shared simplified service capacity.
    service = np.full(4, CAPACITIES[arm] / 4.0)
    available = queue + arrivals
    departed = np.minimum(available, service)
    next_queue = available - departed
    reward = -(next_queue.sum()) + 0.25 * departed.sum()
    return next_queue, float(reward), float(next_queue.mean())

def run_controller(method, arrivals):
    rng = np.random.default_rng(SEED)
    counts = np.zeros(4, dtype=int)
    estimates = np.zeros(4, dtype=float)
    total_reward = 0.0
    queue_sum = 0.0
    decisions = 0

    for ep in range(N_EPISODES):
        queue = np.zeros(4, dtype=float)
        for step in range(STEPS_PER_EPISODE):
            arrival = arrivals[(ep * STEPS_PER_EPISODE + step) % len(arrivals)]
            if method == "fixed_time":
                # Fixed schedule cycles through timing options in a preset order.
                arm = step % 4
            elif method == "greedy":
                arm = int(np.argmax(estimates))
            elif method == "epsilon_greedy":
                if rng.random() < EPSILON:
                    arm = int(rng.integers(0, 4))
                else:
                    arm = int(np.argmax(estimates))
            else:
                raise ValueError(method)

            queue, reward, mean_queue = one_step(queue, arrival, arm, rng)
            counts[arm] += 1
            estimates[arm] += (reward - estimates[arm]) / counts[arm]
            total_reward += reward
            queue_sum += mean_queue
            decisions += 1

    return {
        "controller": method,
        "average_reward": total_reward / decisions,
        "average_mean_queue": queue_sum / decisions,
        "action_counts": counts.tolist(),
        "estimated_arm_rewards": estimates.round(4).tolist(),
        "decisions": decisions
    }

def main():
    arrivals = load_arrivals()
    results = [run_controller(name, arrivals) for name in
               ("fixed_time", "greedy", "epsilon_greedy")]
    out_json = OUT / "traffic_experiment_results.json"
    out_csv = OUT / "traffic_experiment_summary.csv"
    out_json.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["controller","average_reward","average_mean_queue",
                         "arm_0_selections","arm_1_selections",
                         "arm_2_selections","arm_3_selections","decisions"])
        for r in results:
            writer.writerow([r["controller"], round(r["average_reward"],4),
                             round(r["average_mean_queue"],4),
                             *r["action_counts"],r["decisions"]])
    for r in results:
        print(f'\nController: {r["controller"]}')
        print(f'Average reward: {r["average_reward"]:.4f}')
        print(f'Average mean queue: {r["average_mean_queue"]:.4f}')
        print(f'Arm selections: {r["action_counts"]}')
        print(f'Estimated arm rewards: {r["estimated_arm_rewards"]}')
    print(f"\nSaved results to {out_csv} and {out_json}")

if __name__ == "__main__":
    main()
