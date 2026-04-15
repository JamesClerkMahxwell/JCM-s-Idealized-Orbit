import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#metric units, m, s
G = 6.67430e-11         
Mass_Sun = 1.9885e30        
AU = 1.496e11           
      


pos = np.array([AU, 0.0])            
vel = np.array([0.0, 29780.0])       


dt = 86400 / 2                       
num_steps = 730                     


orbit_x = []
orbit_y = []

def update_physics():
    global pos, vel
    r_mag = np.linalg.norm(pos)
    
    accel = -G * Mass_Sun * pos / r_mag**3
    
    
    vel += accel * dt    
    pos += vel * dt      
    
    return pos[0], pos[1]


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-1.5 * AU, 1.5 * AU)
ax.set_ylim(-1.5 * AU, 1.5 * AU)
ax.set_title("Earth Orbit")
ax.set_xlabel("Distance")
ax.set_ylabel("Distance")
sun, = ax.plot(0, 0, 'yo', markersize=15, label="Sun")
earth, = ax.plot([], [], 'bo', markersize=8, label="Earth")
trail, = ax.plot([], [], 'b-', alpha=0.3)
def init():
   earth.set_data([], [])
   trail.set_data([], [])
   return earth, trail


def animate(i):
   x, y = update_physics()
   orbit_x.append(x)
   orbit_y.append(y)
  
   earth.set_data([x], [y])
   trail.set_data(orbit_x, orbit_y)
   return earth, trail


ani = FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=True, interval=20)
plt.legend()
plt.show()
