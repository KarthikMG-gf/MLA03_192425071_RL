"""B2 - MAXQ task hierarchy and value calculation for flood rescue."""

# Composite tasks contain smaller composite tasks or primitive actions.
MAXQ_HIERARCHY = {
    "FLOOD_RESCUE": ["NAVIGATION", "VICTIM_DETECTION", "MEDICAL_ASSISTANCE", "RETURN_TO_BASE"],
    "NAVIGATION": ["MOVE", "AVOID_WATER"],
    "VICTIM_DETECTION": ["SCAN", "IDENTIFY"],
    "MEDICAL_ASSISTANCE": ["DELIVER_SUPPLY", "CONFIRM_RESCUE"],
    "RETURN_TO_BASE": ["MOVE", "AVOID_WATER"],
}
PRIMITIVE_TASKS = {"MOVE", "AVOID_WATER", "SCAN", "IDENTIFY", "DELIVER_SUPPLY", "CONFIRM_RESCUE"}


def maxq_value(subtask_value, completion_value):
    """MAXQ value: Q(i,s,a) = V(a,s) + C(i,s,a)."""
    return subtask_value + completion_value


def execute_flood_rescue():
    """Illustrative dependency-aware execution order for the given hierarchy."""
    sequence = [
        "MOVE to affected area", "AVOID_WATER on unsafe streets",
        "SCAN surroundings", "IDENTIFY victim",
        "DELIVER_SUPPLY", "CONFIRM_RESCUE", "RETURN_TO_BASE",
    ]
    for step, action in enumerate(sequence, start=1):
        print(f"{step}. {action}")


def main():
    print("MAXQ task hierarchy:")
    for parent, children in MAXQ_HIERARCHY.items():
        print(f"{parent} -> {', '.join(children)}")
    print("Primitive tasks:", ", ".join(sorted(PRIMITIVE_TASKS)))

    subtasks = {
        "Navigation": (5, 3),
        "Victim Detection": (8, 4),
        "Medical Assistance": (10, 6),
    }
    print("\nMAXQ values:")
    values = {}
    for task, (v, c) in subtasks.items():
        values[task] = maxq_value(v, c)
        print(f"{task} : {values[task]}")
    preferred = max(values, key=values.get)
    print("Preferred subtask based on the supplied values:", preferred)

    print("\nDependency-aware rescue execution:")
    execute_flood_rescue()


if __name__ == "__main__":
    main()
