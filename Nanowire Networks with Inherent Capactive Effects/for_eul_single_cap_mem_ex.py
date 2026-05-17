


import matplotlib.pyplot as plt
import numpy as np


# Define const.
μ = 1e-14
r_on = 100
r_off = 16000
D = 1e-8
τ = 10


ε0 = 8.85e-12
εr = 95     #McPherson, Joe W., et al. "Trends in the ultimate breakdown strength of high dielectric-constant materials."
ε = ε0*εr
A = 2e-12
c_on = ε*A/D


# Define time stuff
T = 10
dt = 1e-5
nt = int(T/dt)

t = np.linspace(0,T,nt)

# Input voltage
v = 1.5*np.sin(2*np.pi*t)

# Now, define initial state
x = 0.5

store_x = np.zeros_like(t)

# Current
mem_current = np.zeros_like(t)

# Initial values for numerical charge
c_prev = 0
v_prev = 0

for i in range(nt):

    # Simplify values
    g_on = 1/(r_on*x)
    g_off = 1/(r_off*(1-x))
    
    c = c_on*x
    
    i_c = c/dt
    den = g_on+g_off+i_c
    
    g_eff = g_on*(g_off+i_c)/den

    q_n = c_prev*v_prev
    i_prev = (g_on*q_n/dt)/den

    curr = g_eff*v[i]-i_prev

    k = μ*r_on/(D*D)
    l = k*curr
    f = 1-(x-1)**2 if curr<0 else 1-x**2
    #f = 1-(2*x-1)**2

    dx = l*f - x/τ
    #dx = l*f

    x = np.clip(x + dx*dt,0,1)

    store_x[i] = x
    mem_current[i] = curr

    c_prev = c
    v_int = (g_on*v[i] + q_n/dt)/den
    v_prev = v[i] - v_int


plt.figure(figsize=(8,8))
#plt.plot(v[:int(1/dt)], mem_current[:int(1/dt)], color='blue')
plt.plot(v, mem_current, color='blue')
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Memristor Hysteresis")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

plt.figure(figsize=(8,8))
#plt.plot(v[:int(1/dt)], mem_current[:int(1/dt)], color='blue', marker=",")
plt.plot(v, mem_current, color='blue', marker=",")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Memristor Hysteresis (Zoomed-In)")
plt.xlim(-1e-7,1e-7)
plt.ylim(-1e-11,1e-11)
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(t, c_on*store_x, color='blue')
plt.xlabel("Time (arb units until analyzed)")
plt.ylabel("Capacitance (F)")
plt.title("Memristor Capacitance")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(v, c_on*store_x, color='blue')
plt.xlabel("Voltage (V)")
plt.ylabel("Capacitance (F)")
plt.title("Memristor Capacitance-Voltage")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(t, store_x, color='blue')
plt.xlabel("Time (arb units)")
plt.ylabel("State x")
plt.title("Memristor State Trajectory")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()