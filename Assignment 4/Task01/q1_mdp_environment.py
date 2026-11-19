"""Q1: MDP design and one-step predictive agricultural environment model."""
import numpy as np

TARGET_LOW, TARGET_HIGH = 0.45, 0.65

def transition(state, action, rain=0.0):
    """state=[moisture,temp,humidity,nutrients,crop_health]; action=0 off, 1 controlled, 2 full."""
    moisture, temp, humidity, nutrients, crop = state
    irrigation = {0: 0.0, 1: 0.5, 2: 1.0}[int(action)]
    next_m = np.clip(moisture + 0.18*irrigation + 0.04*rain
                     - 0.015*max(temp-25, 0)/10, 0, 1)
    in_band = TARGET_LOW <= next_m <= TARGET_HIGH
    next_crop = np.clip(crop + (0.025 if in_band else -0.015)
                        - 0.01*max(temp-32, 0)/10, 0, 1)
    next_nutrients = np.clip(nutrients - 0.002, 0, 1)
    next_state = np.array([next_m, temp, humidity, next_nutrients, next_crop])
    water = 10*irrigation
    reward = 4*next_crop - 0.08*water - 2*abs(next_m-0.55)
    return next_state, float(reward)

def main():
    state = np.array([0.32, 31.0, 0.55, 0.70, 0.72])
    print("Initial state [moisture,temp,humidity,nutrients,crop]:", state)
    for action, name in enumerate(["OFF", "CONTROLLED", "FULL"]):
        ns, r = transition(state, action, rain=0.1)
        print(f"{name:10s} next={np.round(ns,3)} reward={r:.3f}")
    print("Best one-step action:", max(range(3), key=lambda a: transition(state,a,0.1)[1]))
if __name__ == "__main__": main()
