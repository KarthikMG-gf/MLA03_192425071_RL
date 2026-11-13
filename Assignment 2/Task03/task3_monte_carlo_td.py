import numpy as np
gamma=.9
episodes=[[(0,-1),(1,3),(2,10)],[(0,-1),(1,3),(1,10)],[(0,-1),(2,10)]]
def returns(ep):
    g=0; out=[0]*len(ep)
    for i in range(len(ep)-1,-1,-1):g=ep[i][1]+gamma*g;out[i]=g
    return out
first={};every={}
for ep in episodes:
    seen=set()
    for (s,_),g in zip(ep,returns(ep)):
        every.setdefault(s,[]).append(g)
        if s not in seen:first.setdefault(s,[]).append(g);seen.add(s)
print("First-visit MC values:",{s:round(np.mean(v),3) for s,v in first.items()})
print("Every-visit MC values:",{s:round(np.mean(v),3) for s,v in every.items()})
v=5.; updated=v+.2*(3+.9*8-v)
print("TD(0) updated value:",round(updated,3))
rng=np.random.default_rng(7);q=np.zeros(4);counts=np.zeros(4);truth=[0,1,1.5,.5]
for _ in range(500):
    a=int(rng.integers(4)) if rng.random()<.1 else int(np.argmax(q))
    r=rng.normal(truth[a],.2);counts[a]+=1;q[a]+=(r-q[a])/counts[a]
print("MC control action-value estimates:",np.round(q,3))
print("Greedy action after learning:",int(np.argmax(q)))
