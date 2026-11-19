"""B3 - Simplified meta-learning adaptation across disaster environments."""

DISASTER_ENVIRONMENTS = {
    "Earthquake": {
        "hazards": ["collapsed buildings", "blocked pathways"],
        "learning_requirements": ["obstacle avoidance", "victim identification"],
    },
    "Flood": {
        "hazards": ["waterlogged roads", "restricted movement"],
        "learning_requirements": ["safe route selection", "water-level assessment"],
    },
    "Forest Fire": {
        "hazards": ["fire zones", "rapidly changing hazards"],
        "learning_requirements": ["hazard avoidance", "emergency navigation"],
    },
}


def adapt_parameter(theta, target, alpha):
    """Assignment update: theta' = theta - alpha * (theta - target)."""
    return theta - alpha * (theta - target)


def run_adaptation_comparison():
    """Toy numerical comparison to illustrate adaptation speed, not a trained RL benchmark."""
    target = 5.0
    alpha = 0.5
    methods = {
        "Training from scratch": 0.0,
        "Transfer learning": 2.0,
        "Meta-learning": 3.5,
    }
    print("\nIllustrative adaptation comparison (distance to target parameter):")
    for name, initial in methods.items():
        parameter = initial
        episodes = 0
        while abs(parameter - target) > 0.25 and episodes < 20:
            parameter = adapt_parameter(parameter, target, alpha)
            episodes += 1
        print(f"{name}: adaptation episodes={episodes}, final parameter={parameter:.3f}, error={abs(parameter-target):.3f}")


def main():
    print("Disaster environment configurations:")
    for name, config in DISASTER_ENVIRONMENTS.items():
        print(f"\n{name}")
        print("  Hazards:", ", ".join(config["hazards"]))
        print("  Learning requirements:", ", ".join(config["learning_requirements"]))

    theta = 2.0
    target = 5.0
    alpha = 0.5
    adapted = adapt_parameter(theta, target, alpha)
    print("\nMAML-style single-task adaptation example")
    print("Initial parameter:", theta)
    print("Adapted parameter:", adapted)

    run_adaptation_comparison()


if __name__ == "__main__":
    main()
