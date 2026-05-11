# Define const.
μ = 1e-14
r_on = 100
r_off = 16000
D = 1e-8
τ = 1


ε0 = 8.85e-12
εr = 95     #McPherson, Joe W., et al. "Trends in the ultimate breakdown strength of high dielectric-constant materials."
ε = ε0*εr
#ε = 2.5e-9
A = 2e-12
c_on = ε*A/D


# Define time stuff
T = 10
dt = 1e-4
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
    #c = c_on*(1-x)     # My interpretation
    
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
