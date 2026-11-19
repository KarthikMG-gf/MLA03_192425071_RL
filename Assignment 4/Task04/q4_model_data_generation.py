"""Q4: Fit a simple least-squares transition model and generate synthetic tuples."""
import numpy as np, csv
from pathlib import Path

def build_demo_data(n=250, seed=11):
    rng=np.random.default_rng(seed)
    m=rng.uniform(.15,.8,n); temp=rng.uniform(18,38,n); humidity=rng.uniform(.25,.9,n)
    rain=rng.binomial(1,.25,n); action=rng.integers(0,2,n)
    nxt=np.clip(m+.18*action+.04*rain-.0015*np.maximum(temp-25,0)+rng.normal(0,.012,n),0,1)
    X=np.column_stack([np.ones(n),m,temp,humidity,rain,action])
    return X,nxt

def main():
    X,y=build_demo_data()
    rng=np.random.default_rng(3); idx=rng.permutation(len(y)); train, test=idx[:200],idx[200:]
    w=np.linalg.lstsq(X[train],y[train],rcond=None)[0]
    pred=X[test]@w; mae=float(np.mean(np.abs(pred-y[test])))
    print(f"Held-out transition MAE: {mae:.4f}")
    rows=[]
    for moisture, temp, humidity, rain, action in [(.35,30,.5,0,1),(.52,24,.7,1,0),(.25,36,.4,0,1)]:
        s=np.array([1,moisture,temp,humidity,rain,action])
        next_m=float(np.clip(s@w,0,1))
        reward=4-20*abs(next_m-.55)-1.2*action
        rows.append((moisture,temp,humidity,rain,action,reward,next_m))
    out=Path(__file__).resolve().parents[1]/"outputs"/"q4_synthetic_experiences.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["moisture","temperature","humidity","rain","action","reward","next_moisture"]); wr.writerows(rows)
    for r in rows: print("synthetic (s,a,r,s'):",r)
    print("Saved:",out)
if __name__=="__main__": main()
