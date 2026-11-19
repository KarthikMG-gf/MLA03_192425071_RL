"""B5 - POMDP belief update and ethical rescue decision support."""


def update_belief(prior, likelihood):
    """Bayesian belief update: normalize prior[s] * P(observation | s)."""
    if len(prior) != len(likelihood):
        raise ValueError("Prior and likelihood must have the same length")
    if any(p < 0 for p in prior) or any(z < 0 for z in likelihood):
        raise ValueError("Probabilities must be non-negative")
    unnormalized = [p * z for p, z in zip(prior, likelihood)]
    normalization = sum(unnormalized)
    if normalization == 0:
        raise ValueError("Observation has zero probability under the supplied belief")
    return [value / normalization for value in unnormalized]


def ethical_rescue_decision(victim_belief, route_risk, privacy_safe,
                            resource_priority, human_review_required=False):
    """Apply transparent safety, fairness, privacy and oversight checks."""
    if not privacy_safe:
        return "PAUSE: restrict/delete unnecessary personal data and use privacy-safe observations"
    if route_risk >= 0.7:
        return "DO NOT USE FAST HAZARDOUS ROUTE: request a safer route or human authorization"
    if victim_belief < 0.5:
        return "VERIFY: gather another observation before dispatch"
    if human_review_required:
        return "ESCALATE: obtain human supervisor approval"
    return f"DISPATCH: allocate resources according to documented priority ({resource_priority})"


def main():
    prior = [0.6, 0.4]
    likelihood = [0.8, 0.3]
    updated = update_belief(prior, likelihood)
    print("Updated belief:", [round(x, 1) for x in updated])

    print("\nEthical decision examples:")
    cases = [
        (0.8, 0.2, True, "highest medical urgency", False),
        (0.9, 0.9, True, "high medical urgency", True),
        (0.7, 0.2, False, "high medical urgency", False),
        (0.3, 0.2, True, "unconfirmed location", False),
    ]
    for i, case in enumerate(cases, start=1):
        print(f"Case {i}: {ethical_rescue_decision(*case)}")

    print("\nMDP/POMDP note:")
    print("An MDP selects actions from a fully observed state. A POMDP maintains a probability")
    print("distribution (belief state) over possible states and updates it after each observation.")


if __name__ == "__main__":
    main()
