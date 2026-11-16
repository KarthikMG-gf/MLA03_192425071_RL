"""Q1: Adaptive Maze Environment (3 Scenarios) & Reward Calculation."""
import numpy as np

# --- PART A: MANDATORY COMPLETE THE CODE ---
def calculate_reward(action, next_position, treasure, enemy, exit_point):
    reward = -1
    # TODO 1: Add a positive reward when treasure is collected
    if treasure is not None and next_position == treasure:
        reward += 10
    # TODO 2: Add a negative reward when the agent encounters an enemy
    if enemy is not None and next_position == enemy:
        reward -= 15
    # TODO 3: Add a positive reward when the agent reaches the exit
    if next_position == exit_point:
        reward += 25
    # TODO 4: Return the final reward
    return reward

# --- PART B: STUDENT B 3-SCENARIO MAZE INVESTIGATION ---
class AdaptiveMazeEnv:
    def __init__(self, moving_enemy=False, dynamic_obstacle=False):
        self.size = 5
        self.moving_enemy = moving_enemy
        self.dynamic_obstacle = dynamic_obstacle
        self.walls = {(0, 2), (1, 1), (3, 3)}
        self.reset()

    def reset(self):
        self.agent = (0, 0)
        self.enemy = (3, 1)
        self.treasures = {(0, 3), (2, 3)}
        self.exit = (4, 4)
        self.steps = 0
        return self.agent

    def step(self, action):
        self.steps += 1
        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        dr, dc = moves[action]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc

        active_walls = set(self.walls)
        if self.dynamic_obstacle and (self.steps // 4) % 2 == 1:
            active_walls.add((2, 2))

        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in active_walls:
            self.agent = (nr, nc)

        if self.moving_enemy:
            er, ec = self.enemy
            cand = [(er + r, ec + c) for r, c in moves.values()
                    if 0 <= er + r < self.size and 0 <= ec + c < self.size and (er + r, ec + c) not in active_walls]
            if cand:
                self.enemy = cand[np.random.randint(len(cand))]

        t_hit = self.agent if self.agent in self.treasures else None
        if t_hit:
            self.treasures.remove(t_hit)
        e_hit = self.enemy if self.agent == self.enemy else None
        r = calculate_reward(action, self.agent, t_hit, e_hit, self.exit)
        done = (self.agent == self.exit) or (e_hit is not None) or (self.steps >= 30)
        return self.agent, r, done, (t_hit is not None), (e_hit is not None)

def run_scenarios(episodes=50):
    np.random.seed(42)
    scenarios = [
        ("Scenario 1 (Static Maze)", False, False),
        ("Scenario 2 (Moving Enemy)", True, False),
        ("Scenario 3 (Dynamic Obstacles)", True, True)
    ]
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    print("\n--- Student B: 3-Scenario Experimental Observations ---")
    for name, m_enemy, d_obs in scenarios:
        env = AdaptiveMazeEnv(moving_enemy=m_enemy, dynamic_obstacle=d_obs)
        total_steps, total_treasures, total_collisions, wins = 0, 0, 0, 0
        for _ in range(episodes):
            env.reset()
            done = False
            while not done:
                act = actions[np.random.randint(4)]
                pos, _, done, got_t, hit_e = env.step(act)
                if got_t:
                    total_treasures += 1
                if hit_e:
                    total_collisions += 1
            total_steps += env.steps
            if env.agent == env.exit:
                wins += 1
        print(f"{name:<32} | Avg Steps: {total_steps/episodes:5.1f} | Treasures: {total_treasures:2d} | Collisions: {total_collisions:2d} | Win Rate: {wins/episodes*100:5.1f}%")

if __name__ == "__main__":
    print("--- Part A: Complete the Code (Reward Calculation) ---")
    reward = calculate_reward("RIGHT", (2, 3), (2, 3), None, (4, 4))
    print("Test Case 1 (Treasure Collected) -> Game Reward:", reward)
    print("Test Case 2 (Enemy Encountered)  -> Game Reward:", calculate_reward("DOWN", (3, 1), None, (3, 1), (4, 4)))
    print("Test Case 3 (Reached Exit)       -> Game Reward:", calculate_reward("RIGHT", (4, 4), None, None, (4, 4)))
    
    run_scenarios(episodes=50)
