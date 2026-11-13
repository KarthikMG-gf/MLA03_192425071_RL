import numpy as np
gamma=.9; theta=.001
states=[0,1,2]; actions=[0,1]
P={(0,0):(1,-1),(0,1):(0,-2),(1,0):(2,10),(1,1):(1,-2)}
V=np.zeros(3)
while True:
    delta=0
    for s in [0,1]:
        old=V[s]; action_values=[]
        for a in actions:
            ns,reward=P[(s,a)]
            action_value=reward+gamma*V[ns]
            action_values.append(action_value)
        V[s]=max(action_values)
        delta=max(delta,abs(old-V[s]))
    if delta<theta:break
policy={s:int(np.argmax([P[(s,a)][1]+gamma*V[P[(s,a)][0]] for a in actions])) for s in [0,1]}
print("Optimal State Values:",np.round(V,3))
print("Optimal actions (0=Forward, 1=Stay):",policy)
