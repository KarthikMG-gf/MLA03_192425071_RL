
from collections import deque

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

NAMES = ["Up", "Down", "Left", "Right"]


class Warehouse:
    size = 5
    start = (0, 0)
    goal = (4, 4)

    obstacles = {
        (1, 1),
        (1, 3),
        (2, 3),
        (3, 1)
    }

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        dr, dc = ACTIONS[int(action)]
        r, c = self.position

        next_state = (r + dr, c + dc)

        if (
            not (0 <= next_state[0] < self.size)
            or not (0 <= next_state[1] < self.size)
            or next_state in self.obstacles
        ):
            next_state = self.position
            reward = -10.0
        else:
            reward = -1.0

        done = next_state == self.goal

        if done:
            reward = 20.0

        self.position = next_state

        return next_state, reward, done

    def render(self, path=None):
        grid = [
            ["." for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for r, c in self.obstacles:
            grid[r][c] = "#"

        grid[0][0] = "S"
        grid[4][4] = "G"

        if path:
            for r, c in path[1:-1]:
                if grid[r][c] == ".":
                    grid[r][c] = "*"

        return "\n".join(" ".join(row) for row in grid)


def find_path(env):
    """Find a valid path using Breadth-First Search."""

    queue = deque([(env.start, [])])
    visited = {env.start}

    while queue:
        state, actions = queue.popleft()

        if state == env.goal:
            return actions

        for action, (dr, dc) in ACTIONS.items():
            r, c = state
            next_state = (r + dr, c + dc)

            if (
                0 <= next_state[0] < env.size
                and 0 <= next_state[1] < env.size
                and next_state not in env.obstacles
                and next_state not in visited
            ):
                visited.add(next_state)
                queue.append(
                    (next_state, actions + [action])
                )

    return None


def main():
    env = Warehouse()
    env.reset()

    print("WAREHOUSE NAVIGATION")
    print("====================")
    print("\nInitial warehouse:")
    print(env.render())

    actions = find_path(env)

    if actions is None:
        print("\nNo valid path found!")
        return

    print("\nExecuting navigation...\n")

    path = [env.start]
    total_reward = 0

    for step_number, action in enumerate(actions, start=1):
        state, reward, done = env.step(action)

        total_reward += reward
        path.append(state)

        print(
            f"Step {step_number}: "
            f"{NAMES[action]} -> {state}, "
            f"Reward: {reward}"
        )

        if done:
            print("\nGoal reached!")
            break

    print("\nFinal warehouse path:")
    print(env.render(path))

    print("\nRESULTS")
    print("=======")
    print("Total steps:", len(actions))
    print("Total reward:", total_reward)
    print("Start:", env.start)
    print("Goal:", env.goal)
    print("Final position:", env.position)


if __name__ == "__main__":
    main()
