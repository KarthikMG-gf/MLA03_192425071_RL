"""B4 - Cooperative multi-agent disaster rescue simulation."""

AGENTS = {
    "Surveillance Drone": {
        "state": ["camera observations", "zone status", "detected victims"],
        "actions": ["scan", "move", "report"],
        "objective": "identify victims and hazardous locations",
    },
    "Ground Rescue Robot": {
        "state": ["position", "obstacles", "victim location"],
        "actions": ["move", "avoid", "rescue"],
        "objective": "reach victims safely",
    },
    "Medical Supply Robot": {
        "state": ["supply availability", "location", "demand"],
        "actions": ["deliver", "collect", "return"],
        "objective": "distribute medical supplies to priority locations",
    },
    "Communication Robot": {
        "state": ["agent status", "network condition", "alerts"],
        "actions": ["send", "receive", "coordinate"],
        "objective": "maintain reliable information exchange",
    },
}


def team_reward(agent_rewards, penalty):
    """Shared reward = sum of individual rewards - total penalty."""
    return sum(agent_rewards) - penalty


def cooperative_simulation():
    """Simple role-assigned scenario illustrating cooperative task allocation."""
    assignments = {
        "Surveillance Drone": "scan zones and report victim coordinates",
        "Ground Rescue Robot": "navigate to reported victim and perform rescue",
        "Medical Supply Robot": "deliver supplies to the confirmed priority zone",
        "Communication Robot": "relay status and prevent duplicate assignments",
    }
    print("Cooperative task assignments:")
    for agent, task in assignments.items():
        print(f"{agent}: {task}")

    # Illustrative outcome values for comparing coordination modes.
    return {
        "Cooperative CTDE": {"victims_rescued": 4, "completion_time": 18,
                             "collisions": 0, "resource_utilization": 0.90,
                             "team_reward": 50},
        "Independent Q-learning": {"victims_rescued": 3, "completion_time": 25,
                                   "collisions": 2, "resource_utilization": 0.72,
                                   "team_reward": 39},
    }


def main():
    print("Agent definitions:")
    for name, details in AGENTS.items():
        print(f"\n{name}")
        print("  State:", ", ".join(details["state"]))
        print("  Actions:", ", ".join(details["actions"]))
        print("  Objective:", details["objective"])

    rewards = [10, 15, 20, 12]
    penalty = 7
    result = team_reward(rewards, penalty)
    print("\nTeam reward:", result)

    results = cooperative_simulation()
    print("\nCoordination evaluation (illustrative scenario):")
    for method, metrics in results.items():
        print(f"\n{method}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
