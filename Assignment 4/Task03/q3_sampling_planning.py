"""Q3: Sampling-based planning; deterministic forecast or sampled rain."""
import itertools, random

ACTIONS = [0, 1]  # 0=wait, 1=irrigate
def simulate_irrigation(initial_moisture, actions, rainfall=None):
    moisture=float(initial_moisture); total_reward=0.0; water=0.0
    rainfall = rainfall or [0.0]*len(actions)
    for t, action in enumerate(actions):
        rain=rainfall[t] if t<len(rainfall) else 0.0
        moisture=max(0.0,min(1.0,moisture + 0.18*action + 0.04*rain - 0.01))
        reward=4.0 - 20.0*abs(moisture-0.55) - 1.2*action
        total_reward += (0.95**t)*reward
        water += 10*action
    return total_reward, moisture, water

def plan(initial_moisture=0.30, horizon=3, samples=100, seed=7):
    random.seed(seed)
    candidates=[]
    for _ in range(samples):
        seq=[random.choice(ACTIONS) for _ in range(horizon)]
        # Sample uncertain rainfall per step: 0 (no rain), 1 (rain)
        rain=[random.choice([0.0,1.0]) for _ in range(horizon)]
        ret, final_m, water=simulate_irrigation(initial_moisture,seq,rain)
        candidates.append((ret,seq,final_m,water,rain))
    candidates.sort(reverse=True,key=lambda x:x[0])
    return candidates

if __name__=="__main__":
    results=plan()
    for row in results[:5]: print(f"return={row[0]:.3f} actions={row[1]} final_m={row[2]:.3f} water={row[3]:.1f} rain={row[4]}")
    print("Selected first action:", results[0][1][0])
