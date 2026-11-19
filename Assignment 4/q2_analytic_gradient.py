"""Q2: Analytic gradient for a differentiable irrigation policy (NumPy)."""
import numpy as np

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))

def objective_and_gradient(theta, moisture=0.30, rain=0.0, temp=28.0):
    # u=sigmoid(z), z=theta0 + theta1*(target-moisture)
    deficit = 0.55-moisture
    z = theta[0] + theta[1]*deficit
    u = sigmoid(z)
    # m' = m + 0.18u + weather terms; reward= - (m'-target)^2 - 0.08*(10u)
    weather_loss = 0.015*max(temp-25,0)/10 - 0.04*rain
    m_next = np.clip(moisture + 0.18*u - weather_loss, 0, 1)
    error = m_next-0.55
    reward = -(error**2) - 0.8*u
    # In this example m' is not clipped at the selected operating point.
    dm_du = 0.18
    dr_dm = -2*error
    dr_du = dr_dm*dm_du - 0.8
    du_dz = u*(1-u)
    dz_dtheta = np.array([1.0, deficit])
    grad = dr_du*du_dz*dz_dtheta
    return float(reward), grad, float(u), float(m_next)

def main():
    theta=np.array([-0.4,1.2],dtype=float)
    lr=0.08
    for i in range(100):
        J,g,u,m=objective_and_gradient(theta)
        theta += lr*g  # gradient ascent because J is a reward
        if i in (0,9,49,99):
            print(f"iter={i+1:3d} reward={J:.4f} action={u:.3f} predicted_m={m:.3f} theta={theta.round(3)}")
    print("Final:", objective_and_gradient(theta))
if __name__=="__main__": main()
