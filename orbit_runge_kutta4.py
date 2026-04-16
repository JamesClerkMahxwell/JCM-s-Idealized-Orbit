import numpy as np #
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Metric units
G = 6.67430e-11         
Mass_Sun = 1.9885e30        
AU = 1.496e11


state = np.array([AU, 0.0, 0.0, 29780.0])    

dt = 86400/2           
num_steps = 365

orbit_x, orbit_y = [], []
ke_list, pe_list, total_e_list = [], [], []
time_steps = []

def get_derivatives(current_state):
    """
    Given a state [x, y, vx, vy], return the derivatives [vx, vy, ax, ay].
    """
    pos = current_state[:2]
    r_mag = np.linalg.norm(pos)
    vel = current_state[2:]
    accel = -G * Mass_Sun * pos / r_mag**3
    return np.concatenate([vel, accel])

def update_physics_rk4(current_state, dt):
    k1 = get_derivatives(current_state)
    k2 = get_derivatives(current_state + k1 * dt / 2)
    k3 = get_derivatives(current_state + k2 * dt / 2)
    k4 = get_derivatives(current_state + k3 * dt)
    new_state = current_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

def calculate_energy(current_state):
    pos = current_state[:2]
    vel = current_state[2:]
    r_mag = np.linalg.norm(pos)
    v_mag = np.linalg.norm(vel)
    ke = 0.5 * v_mag**2 
    pe = -G * Mass_Sun / r_mag 
    return ke, pe, ke + pe

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.set_aspect('equal')
ax1.set_xlim(-1.5 * AU, 1.5 * AU)
ax1.set_ylim(-1.5 * AU, 1.5 * AU)
ax1.set_title("Earth Orbit (RK4 METHOD)")
sun, = ax1.plot(0, 0, 'yo', markersize=15, label="Sun")
earth, = ax1.plot([], [], 'bo', markersize=8)
trail, = ax1.plot([], [], 'b-', alpha=0.4)

ax2.set_xlim(0, num_steps)
ax2.set_ylim(-1e9, 1e9) 
ax2.set_title("Energy vs Time")
ax2.set_xlabel("Days")
ax2.set_ylabel("Specific Energy (J/kg)")
ke_line, = ax2.plot([], [], 'r-', label="Kinetic Energy")
pe_line, = ax2.plot([], [], 'g-', label="Potential Energy")
total_e_line, = ax2.plot([], [], 'k--', label="Total Energy")
ax2.legend(loc="upper right")

def init():
    earth.set_data([], [])
    trail.set_data([], [])
    ke_line.set_data([], [])
    pe_line.set_data([], [])
    total_e_line.set_data([], [])
    return earth, trail, ke_line, pe_line, total_e_line

def animate(i):
    global state
    # Runge Kutta 4
    state = update_physics_rk4(state, dt)
    
    x, y = state[0], state[1]
    ke, pe, te = calculate_energy(state)
    
    orbit_x.append(x)
    orbit_y.append(y)
    ke_list.append(ke)
    pe_list.append(pe)
    total_e_list.append(te)
    time_steps.append(i)
    
    earth.set_data([x], [y])
    trail.set_data(orbit_x, orbit_y)
    ke_line.set_data(time_steps, ke_list)
    pe_line.set_data(time_steps, pe_list)
    total_e_line.set_data(time_steps, total_e_list)
    
    return earth, trail, ke_line, pe_line, total_e_line

ani = FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=True, interval=20)
plt.tight_layout()
plt.show()
