q=5.;r=4.;alpha=.2;gamma=.9
sarsa=q+alpha*(r+gamma*6-q)
ql=q+alpha*(r+gamma*8-q)
print(f"SARSA updated Q-value: {sarsa:.2f}")
print(f"Q-Learning updated Q-value: {ql:.2f}")
