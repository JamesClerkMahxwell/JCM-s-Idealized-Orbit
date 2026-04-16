import numpy as np #demonstration of energy conservation in an ideal system using Euler-Cromer integration technique
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Metric units
G = 6.67430e-11         
Mass_Sun = 1.9885e30        
AU = 1.496e11

pos = np.array([AU, 0.0])            
vel = np.array([0.0, 29780.0])    

dt = 86400/2             
num_steps = 365


orbit_x, orbit_y = [], []
ke_list, pe_list, total_e_list = [], [], []
time_steps = []

def update_physics_cromer():
    global pos, vel
    r_mag = np.linalg.norm(pos)
    accel = -G * Mass_Sun * pos / r_mag**3
    # Euler-Cromer Method
    vel = vel + accel * dt    
    pos = pos + vel * dt      
    return pos[0], pos[1]

def calculate_energy():
    r_mag = np.linalg.norm(pos)
    v_mag = np.linalg.norm(vel)
    ke = 0.5 * v_mag**2 #Kinetic Energy 
    pe = -G * Mass_Sun / r_mag #Potential Energy, derived via integration of GMm/r^2
    return ke, pe, ke + pe, #plotting the sum of the mechanical energy of the system


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


ax1.set_aspect('equal')
ax1.set_xlim(-1.5 * AU, 1.5 * AU)
ax1.set_ylim(-1.5 * AU, 1.5 * AU)
ax1.set_title("Earth Orbit(EULER-CROMER)")
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
  
    x, y = update_physics_cromer()
    ke, pe, te = calculate_energy()
    
    
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