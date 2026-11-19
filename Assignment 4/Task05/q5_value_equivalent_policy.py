"""Q5: Integrated model-based agricultural policy optimization and evaluation."""
import numpy as np, csv, time
from pathlib import Path

def step(state, action, rain):
    # state=[moisture,crop_health]; action=[irrigation 0..1, fertilizer 0..1]
    m,c=state; irrigation,fertilizer=action
    m2=np.clip(m+.18*irrigation+.04*rain-.012,0,1)
    band=1-abs(m2-.55)/.55
    c2=np.clip(c+.035*max(0,band)+.018*fertilizer-.012*max(0,.35-m2),0,1)
    reward=5*c2-0.8*(10*irrigation)-0.35*fertilizer-2*abs(m2-.55)
    return np.array([m2,c2]),float(reward)

def value_rollout(state, action_seq, rain_seq):
    total=0.; s=np.array(state,dtype=float)
    for t,a in enumerate(action_seq):
        s,r=step(s,a,rain_seq[t]); total+=(.96**t)*r
    return total,s

def choose_action(state, rain_forecast=0.0):
    # Enumerate short candidate sequences; return first action of highest-return sequence.
    candidates=[(i,f) for i in (0.,.5,1.) for f in (0.,.5,1.)]
    best=None
    for a0 in candidates:
      for a1 in candidates:
        seq=[a0,a1]; rains=[rain_forecast,rain_forecast]
        ret,_=value_rollout(state,seq,rains)
        if best is None or ret>best[0]: best=(ret,a0)
    return best[1],best[0]

def evaluate_scenarios():
    scenarios={
      "S1_low_moisture_hot":([.22,.62],0.0),
      "S2_moderate_expected_rain":([.52,.78],1.0),
      "S3_dry_hot_limited_water":([.18,.58],0.0)}
    rows=[]
    for name,(initial,rain) in scenarios.items():
        s=np.array(initial,dtype=float); total=0.; water=0.; start=time.perf_counter()
        for _ in range(8):
            action,_=choose_action(s,rain)
            s,r=step(s,action,rain); total+=r; water+=10*action[0]
        elapsed=time.perf_counter()-start
        rows.append((name,total/8,water,s[0],s[1],elapsed))
    return rows

if __name__=="__main__":
    rows=evaluate_scenarios()
    out=Path(__file__).resolve().parents[1]/"outputs"/"q5_scenario_results.csv"
    with out.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["scenario","average_reward","water_units","final_moisture","final_crop_health","runtime_seconds"]); w.writerows(rows)
    for r in rows: print(r)
    print("Saved:",out)
