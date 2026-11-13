gamma = 0.8
probabilities = [0.6, 0.4]
rewards = [5, -2]
next_state_values = [10, 4]

def bellman_return(probabilities, rewards, next_state_values, gamma):
    if not (len(probabilities) == len(rewards) == len(next_state_values)):
        raise ValueError("Input lists must have equal lengths.")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("Transition probabilities must sum to 1.")
    total = 0.0
    for i in range(len(probabilities)):
        total += probabilities[i] * (
            rewards[i] + gamma * next_state_values[i]
        )
    return total

if __name__ == "__main__":
    print("Bellman return:", round(bellman_return(
        probabilities, rewards, next_state_values, gamma), 2))
