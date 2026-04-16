import numpy as np
import matplotlib.pyplot as plt
from orbit_euler import update_physics
from orbit_euler_cromer import update_physics_cromer # type: ignore
from orbit_runge_kutta import calculate_energy, update_physics_rk4
import numpy as np
import matplotlib.pyplot as plt
import orbit_euler
import orbit_euler_cromer # type: ignore
import orbit_runge_kutta

dt = 86400 / 2  
num_steps = 365 * 2  

def run_analysis():
    
    initial_state = np.array([1.496e11, 0.0, 0.0, 29780.0])
    
    
    euler_err = []
    cromer_err = []
    rk4_err = []
    time_axis = np.arange(num_steps) * (dt / 86400) # Days

    # KEi
    ke0, pe0, te0 = orbit_runge_kutta.calculate_energy(initial_state)

    # Euler
    state = initial_state.copy()
    orbit_euler.pos = state[:2]
    orbit_euler.vel = state[2:]
    for _ in range(num_steps):
        orbit_euler.update_physics()
        _, _, te = orbit_euler.calculate_energy()
        euler_err.append(abs((te - te0) / te0))

    # Euler-Cromer
    state = initial_state.copy()
    orbit_euler_cromer.pos = state[:2]
    orbit_euler_cromer.vel = state[2:]
    for _ in range(num_steps):
        orbit_euler_cromer.update_physics_cromer()
        _, _, te = orbit_euler_cromer.calculate_energy()
        cromer_err.append(abs((te - te0) / te0))

    #RK4
    state = initial_state.copy()
    for _ in range(num_steps):
        state = orbit_runge_kutta.update_physics_rk4(state, dt)
        _, _, te = orbit_runge_kutta.calculate_energy(state)
        rk4_err.append(abs((te - te0) / te0))

   
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, euler_err, label='Euler', color='red')
    plt.plot(time_axis, cromer_err, label='Euler-Cromer', color='blue')
    plt.plot(time_axis, rk4_err, label='Runge_Kutta', color='green')

    plt.yscale('log')
    plt.xlabel('Time (Days)')
    plt.ylabel('Relative Energy Error $|(E - E_0) / E_0|$')
    plt.title('Integration Method Error Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    run_analysis()


import pandas as pd
#GTE SEC
G = 6.67430e-11
Mass_Sun = 1.9885e30
AU = 1.496e11
V_circ = np.sqrt(G * Mass_Sun / AU)

T_period = 2 * np.pi * np.sqrt(AU**3 / (G * Mass_Sun))

def get_accel(pos):
    r_mag = np.linalg.norm(pos)
    return -G * Mass_Sun * pos / r_mag**3

def run_sim(method, dt):
    pos = np.array([AU, 0.0])
    vel = np.array([0.0, V_circ])
    num_steps = int(T_period / dt)
    
    for _ in range(num_steps):
        if method == 'euler':
            a = get_accel(pos)
            pos, vel = pos + vel * dt, vel + a * dt
        elif method == 'euler_cromer':
            a = get_accel(pos)
            vel = vel + a * dt
            pos = pos + vel * dt
        elif method == 'rk4':
            def derivs(s):
                p, v = s[:2], s[2:]
                return np.concatenate([v, get_accel(p)])
            state = np.concatenate([pos, vel])
            k1 = derivs(state)
            k2 = derivs(state + k1 * dt / 2)
            k3 = derivs(state + k2 * dt / 2)
            k4 = derivs(state + k3 * dt)
            state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            pos, vel = state[:2], state[2:]
            
    
    return np.linalg.norm(pos - np.array([AU, 0.0]))


time_steps = [172800, 86400, 43200, 21600, 10800, 3600] # From 48h to 1h
results = []

for dt in time_steps:
    results.append({
        'dt': dt,
        'Euler': run_sim('euler', dt),
        'Euler-Cromer': run_sim('euler_cromer', dt),
        'RK4': run_sim('rk4', dt)
    })


df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.loglog(df['dt'], df['Euler'], 'r-o', label='Euler (1st Order)')
plt.loglog(df['dt'], df['Euler-Cromer'], 'g-o', label='Euler-Cromer')
plt.loglog(df['dt'], df['RK4'], 'b-o', label='RK4 (4th Order)')

plt.title('Global Truncation Error')
plt.xlabel('Time Step dt (seconds)')
plt.ylabel('Position Error after 1 Year (meters)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('/Users/bentleylin/Desktop/gte_convergence_plot.png')
plt.show()


df.to_csv('gte_data.csv', index=False)
