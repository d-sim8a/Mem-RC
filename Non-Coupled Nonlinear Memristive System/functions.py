import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

def param_network(N, params, pert_step, max_pert, idx_array):
    params = np.asarray(params)
    P = params.size
    param_array = np.zeros((N, P), dtype=float)
    param_array[:] = params  # start with baseline for every device

    # percentage perturbation per device in [-max_pert, max_pert] stepping pert_step
    choices = np.random.choice([-1, 1], size=N)
    steps = np.random.randint(0, int(max_pert / max(pert_step, 1)) + 1, size=N)
    pert_perc = choices * steps * pert_step / 100.0

    for i in range(N):
        for idx in idx_array:
            param_array[i, idx] = params[idx] * (1.0 + pert_perc[i])
    return param_array.T

def param_normal_network(N,params,var):
    '''
    Take the values of the params and make them the means and then make variance a percentage?
    '''
    params = np.asarray(params)
    P = params.size
    #param_array = np.zeros((N, P), dtype=float)
    #param_array[:] = params  # start with baseline for every device

    # percentage perturbation per device in [-max_pert, max_pert] stepping pert_step
    param_norm_array = np.zeros((N, P), dtype=float)
    for idx,param in enumerate(params):
        param_norm_array[:,idx] = np.random.normal(loc=param,scale=param*var,size=N)
    param_norm_array[0,:] = params
    return param_norm_array.T

def shift(t,v,nu,N):
    shifts = np.zeros(N)
    v_list = np.zeros((N,len(t)))
    for i in range(N):
        shift = nu*np.random.random_sample() # Random number [0,1]
        shift_idx = round(shift/(t[1] - t[0]))
        shift_list = np.zeros(shift_idx)
        v_shifted = np.concatenate((shift_list,v))[:len(t)]
        shifts[i] = shift_idx
        v_list[i] = v_shifted
    return v_list, shifts

def nonlinear_drift_vectorized(x, v, lam, eta, tau):
    f = 1 - (2 * x - 1) ** 2  # Joglekar window

    dx = lam * np.sinh(eta * v)*f - x / tau
    #dx = (lam * np.sinh(eta * v) - x / tau)*f

    return dx


def simulate_memristor_network_current(N, t, x, v_input, param_array):
    dt = t[1] - t[0]
    nt = len(t)

    output = np.zeros((nt, N))
    
    v_input = np.asarray(v_input).T
    alpha,beta,gamma,delta,lam,eta,tau = np.vsplit(param_array,7)
    store_x = np.zeros((nt, N))

    for i in range(nt):
        store_x[i] = x
        v_t = v_input[i]
        dx = nonlinear_drift_vectorized(x, v_t, lam, eta, tau)  # (N,)

        x = np.clip(x + dt * dx, 0, 1)
    
    schottky = alpha * (1-store_x)*(1-np.exp(-beta*v_input)) # Shape: (T,N)
    tunneling = gamma * store_x * np.sinh(delta*v_input) # Shape: (T,N)

    output = schottky + tunneling # Shape: (T,N)
        
    return output,store_x  # shape: (T, N)


def drift_diffusion_mm1(y,v,λ1,λ2,η1,η2,η3,η4,τ):
    dy = np.zeros_like(v)

    v_pos = v > 0
    v_neg = v < 0
    v_zero = v == 0

    f = (1 - (2*y - 1))**2

    if np.any(v_pos):
        f_pos = f[v_pos]
        dy[v_pos] = (λ1[v_pos]*(np.exp(η1[v_pos]*v[v_pos])-np.exp(η2[v_pos])) - y[v_pos]/τ[v_pos])*f_pos
    if np.any(v_neg):
        f_neg = f[v_neg]
        dy[v_neg] = (λ2[v_neg]*(np.exp(-η3[v_neg]*v[v_neg])-np.exp(η4[v_neg])) - y[v_neg]/τ[v_neg])*f_neg
    if np.any(v_zero):
        dy[v_zero] = 0.0

    return dy

def current_mm1tau(t, N, x, v, curr_params, state_params):
    dt = t[1] - t[0]
    nt = len(t)
    output = np.zeros((nt, N))
    store_x = np.zeros_like(output)

    α1, β1, α2, β2, γ, δ = np.vsplit(curr_params,6)
    λ1, λ2, η1, η2, η3, η4, τ =  np.vsplit(state_params,7)

    for i in range(nt):
        store_x[i] = x
        v_t = v[:,i]
        
        dx = drift_diffusion_mm1(x,v_t,λ1.flatten(),λ2.flatten(),η1.flatten(),η2.flatten(),η3.flatten(),η4.flatten(),τ.flatten())
        x = np.clip(x + dt * dx, 0, 1)

    schottky = α1 * (1 - store_x) * (1 - np.exp(-β1* v.T))
    tunneling = γ * store_x * np.sinh(δ * v.T)
    rectifier = α2*(1 - np.exp(-β2*v.T))
        
    output = (schottky + tunneling + rectifier)

    return output, store_x

def vecDynamicsMM2(t, y, v, params, const):

    λx, λy, η1, η2, η3, η4 = params[:,4], params[:,5], params[:,6], params[:,7], params[:,8], params[:,9]
    x, τ, ϵ = y    # Shape of each: (N) 
    ν, σ = const    # Shape of each (1)

    f1 = np.zeros_like(v)
    f2 = np.zeros_like(v)
    f3 = np.zeros_like(v)

    posV = v>0
    negV = v<0
    zeroV = v==0

    #print("Inside vecDynamicsMM2: \nShape params: ",np.shape(λx),'\nShape of states: ',np.shape(x),
         #'\nShape of const: ', np.shape(ν), '\nShape of voltage: ', np.shape(v))
    
    if np.any(posV):
        vPos = v[posV]
        xPos = x[posV]
        τPos = τ[posV]
        ϵPos = ϵ[posV]
        λxPos = λx[posV]
        η1Pos = η1[posV]
        η2Pos = η2[posV]

        exp = np.exp(η1Pos*vPos) - np.exp(η2Pos)
        win = 1.0 - (2.0*xPos -1.0)**2
        
        f1[posV] = (λxPos*exp - (xPos - ϵPos)/τPos)*win
        f2[posV] = ν*exp 
        f3[posV] = σ*exp*win
        
    if np.any(negV):
        vNeg = v[negV]
        xNeg = x[negV]
        τNeg = τ[negV]
        ϵNeg = ϵ[negV]
        λyNeg = λy[negV]
        η3Neg = η3[negV]
        η4Neg = η4[negV]

        exp = np.exp(-η3Neg*vNeg) - np.exp(η4Neg)
        win = 1.0 - (2.0*xNeg -1.0)**2

        
        f1[negV] = (λyNeg*exp - (xNeg - ϵNeg)/τNeg)*win
        f2[negV] = ν*exp 
        f3[negV] = σ*exp*win
    
    f1[zeroV] = 1e-8
    f2[zeroV] = 1e-8
    f3[zeroV] = 1e-8

def currentMM2(t,N,v,params,x0, const):
    dt = t[1] - t[0]
    T = len(t)
        
    output = np.zeros((T, N)) # Initialize current array

    xList = np.zeros((T,N))
    epsList = np.zeros_like(xList)
    tauList = np.zeros_like(xList)
    
    x, τ, ϵ = x0
    α, β, γ, δ = params[:,0], params[:,1], params[:,2], params[:,3]
    #λx, λy, η1, η2, η3, η4 = np.split(params[:,4:], 6)

    ν, σ = const # Note this is ν (greek letter) not v

    v = np.asarray(v)
    
    if v.ndim == 0:
        for i in range(T):
            xList[i], epsList[i], tauList[i] = x, ϵ, τ
            v_t = v[i]
            dx = vecDynamicsMM2(t,[x, τ, ϵ],v_t,params, const)
            x = x + dt*dx[0]
            τ = τ + dt*dx[1]
            ϵ = ϵ + dt*dx[2]
            
            x = np.clip(x, 0, 1)
            ϵ = np.clip(ϵ,0,1)
        
            schottky = α*(1 - x)*(1 - np.exp(-β*v_t))
            tunneling = γ*x*np.sinh(δ*v_t)
            output[i] = schottky + tunneling
    else:
        v = np.asarray(v).T
        for i in range(T):
            xList[i], epsList[i], tauList[i] = x, ϵ, τ
            v_t = v[i]
            dx = vecDynamicsMM2(t,[x, τ, ϵ],v_t,params, const)
            x = x + dt*dx[0]
            τ = τ + dt*dx[1]
            ϵ = ϵ + dt*dx[2]
            
            x = np.clip(x, 0, 1)
            ϵ = np.clip(ϵ, 0, 1)

            

        schottky = α*(1 - xList)*(1 - np.exp(-β*v)) # Shape: (T,N)
        tunneling = γ*xList*np.sinh(δ*v) # Shape: (T,N)
        output = schottky + tunneling # Shape: (T,N)
        
    return output, [xList, tauList, epsList]  # shape: (T, N)

def sim_train_plot_sst(t,x,v,nu,N,params,training_sig,train_times,train_ridge,param_var=True,input_delay=True):    # For Simple Schottky-Tunneling
    '''
    Description goes here
    '''
    dt = t[1] - t[0]

    if param_var:
        param_array = param_normal_network(N,params,1e-1)
    
    else:
        param_array = param_network(N,params,0,0,[i for i in range(len(params))])
    
    if input_delay:
        v_list_shift, shift_list = shift(t,v,nu,N)
    else:
        v_list_shift = np.tile(v,(N,1))

    output, states = simulate_memristor_network_current(N, t, x, v_list_shift, param_array)

    training_start, training_end = train_times

    sig = training_sig
    sig = np.expand_dims(sig,axis=1)
    sig = (sig-np.mean(sig))/np.std(sig)
    sup = sig[int(training_start/dt):int(training_end/dt),:]

    basis = output[int(training_start/dt):int(training_end/dt),:]

    phi = np.linalg.pinv(basis.T@basis+train_ridge*np.eye(len(output[0])))@(basis.T@sup)

    prediction = output@phi

    plt.figure(figsize=(10, 4))
    plt.plot(t, sig, label='Supervisor', color='blue')
    plt.plot(t, prediction, label='Prediction', color='red', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Memristor Network Output (State-Dependent Current)")
    plt.tight_layout()
    plt.xlim(training_end,training_end+10) if training_end+10 <= t[-1] else plt.xlim(training_end,t[-1])
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(1.1*np.min(sig),1.1*np.max(sig))
    plt.show()
    
    return {"Current Responses": output,
            "State Trajectories": states,
            "Parameter Set": param_array,
            "Inputs": v_list_shift,
            "Trained Weights": phi}

def sim_train_plot_mm1_rectifier(t,x,v,nu,N,curr_params,state_params,training_sig,train_times,train_ridge,param_var=True,input_delay=True):    # For Simple Schottky-Tunneling
    '''
    Description goes here
    '''
    dt = t[1] - t[0]

    if param_var:
        curr_params_network = param_normal_network(N,curr_params,1e-1)
        state_params_network = param_normal_network(N,state_params,1e-1)
    
    else:
        curr_params_network = param_network(N,curr_params,0,0,[i for i in range(len(curr_params))])
        state_params_network = param_network(N,state_params,0,0,[i for i in range(len(state_params))])   
    
    if input_delay:
        v_list_shift, shift_list = shift(t,v,nu,N)
    else:
        v_list_shift = np.tile(v,(N,1))

    # Generate input-output data
    output, states = current_mm1tau(t, N, x, v_list_shift, curr_params_network, state_params_network)

    training_start, training_end = train_times

    sig = training_sig
    sig = np.expand_dims(sig,axis=1)
    sig = (sig-np.mean(sig))/np.std(sig)
    sup = sig[int(training_start/dt):int(training_end/dt),:]

    basis = output[int(training_start/dt):int(training_end/dt),:]

    phi = np.linalg.pinv(basis.T@basis+train_ridge*np.eye(len(output[0])))@(basis.T@sup)

    prediction = output@phi

    plt.figure(figsize=(10, 4))
    plt.plot(t, sig, label='Supervisor', color='blue')
    plt.plot(t, prediction, label='Prediction', color='red', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Memristor Network Output (State-Dependent Current)")
    plt.tight_layout()
    plt.xlim(training_end,training_end+10) if training_end+10 <= t[-1] else plt.xlim(training_end,t[-1])
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(1.1*np.min(sig),1.1*np.max(sig))
    plt.show()
    
    return {"Current Responses": output,
            "State Trajectories": states,
            "Parameter Set (Current)": curr_params_network,
            "Parameter Set (States)": state_params_network,
            "Inputs": v_list_shift,
            "Trained Weights": phi}