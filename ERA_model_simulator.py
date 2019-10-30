import numpy as np # for basic vector computation in python
import scipy.integrate as integrate # needed to solve ODEs
import matplotlib.pyplot as plt # for plotting
from ipywidgets import interact, interact_manual # interactive widgets

# the function describing the right-hand side of the ODEs
# y[0] = E, y[1] = R, y[2] = A
def derivs(t, y, params):
    V = sum(y) # cell volume or mass
    E,R,A = y
    a = A/V # concentration of precursors
    total_prot_synth = R * params['sigma'] * a / (a + params['a_sat']) # rate of total protein synthesis
    dA_dt = params['k'] * E - total_prot_synth # metabolism produce A, protein synthesis consume A
    dE_dt = params['f_E'] * total_prot_synth # fraction of total synthesis allocated to E proteins (metabolism)
    dR_dt = (1-params['f_E']) * total_prot_synth # fraction allocated to R proteins (ribosomes)
    return np.array([dE_dt, dR_dt, dA_dt])

# method to simulate the model for a given duration and initial condition
def simulate(params, duration, E0, R0, A0):
    y0 = np.array([E0,R0,A0]) # initial condition
    ode_result = integrate.solve_ivp(lambda t,y: derivs(t,y,params), # the RHS for the ODEs
                                 (0,duration), # time interval
                                 y0, # initial condition
                                 dense_output=True,
                                 rtol=1e-6, atol=1e-9)
    E,R,A = ode_result.y
    V = E + R + A
    e, r, a = E/V, R/V, A/V
    t = ode_result.t
    return t, E, R, A, V, e, r, a

# now y[0] = e, y[1] = r, y[2] = a (concentrations)
def derivs_conc(t, y, params):
    e, r, a = y
    alpha = params['k'] * e
    total_prot_synth = r * params['sigma'] * a / (a + params['a_sat'])
    da_dt = params['k'] * e - total_prot_synth - alpha * a
    de_dt = params['f_E'] * total_prot_synth - alpha * e
    dr_dt = (1-params['f_E']) * total_prot_synth - alpha * r
    return np.array([de_dt, dr_dt, da_dt])

# method to simulate the (concentration) model for a given duration and initial condition
def simulate_conc(params, duration, e0, r0, a0):
    if e0 + r0 + a0 != 1:
        raise Exception('e0 + r0 + a0 should be 1')
    if e0 < 0 or r0 < 0 or a0 <0:
        raise Exception('initial concentrations should be positive')
    y0 = np.array([e0,r0,a0]) # initial condition
    ode_result = integrate.solve_ivp(lambda t,y: derivs_conc(t,y,params), # the RHS for the ODEs
                                 (0,duration), # time interval
                                 y0, # initial condition
                                 dense_output=True,
                                 rtol=1e-6, atol=1e-9)
    e,r,a = ode_result.y
    t = ode_result.t
    return t, e, r, a

def plot_conc_sim(t,e,r,a,gr):
    f, (ax1,ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
    ax1.plot(t, e, 'g', label='e')
    ax1.plot(t, r, 'b', label='r')
    ax1.plot(t, a, 'm', label='a')
    ax1.plot(t, e + r + a, '--k')
    ax1.legend()
    ax1.set_ylabel('concentration or mass fraction')
    ax1.set_ylim([0,1])
    ax2.plot(t, gr, 'r')
    ax2.set_ylim([0,1])
    ax2.set_ylabel('Instantaneous growth rate')
    ax2.set_title(f'Growth rate at the end: {gr[-1]:.3f}')
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time')
    plt.show()

def derivs_regulation(t, y, params, fE_reg_fun):
    e, r, a = y
    alpha = params['k'] * e
    total_prot_synth = r * params['sigma'] * a / (a + params['a_sat'])
    da_dt = params['k'] * e - total_prot_synth - alpha * a
    fE_from_reg = fE_reg_fun(e,r,a)
    if fE_from_reg < 0:
        fE_from_reg = 0
    if fE_from_reg > 1:
        fE_from_reg = 1
    de_dt = fE_from_reg * total_prot_synth - alpha * e
    dr_dt = (1-fE_from_reg) * total_prot_synth - alpha * r
    return np.array([de_dt, dr_dt, da_dt])

def simulate_regulation(params, fE_reg_fun, duration, e0, r0, a0):
    if e0 + r0 + a0 != 1:
        raise Exception('e0 + r0 + a0 should be 1')
    if e0 < 0 or r0 < 0 or a0 <0:
        raise Exception('initial concentrations should be positive')
    y0 = np.array([e0,r0,a0]) # initial condition
    ode_result = integrate.solve_ivp(lambda t,y: derivs_regulation(t,y,params,fE_reg_fun), # the RHS for the ODEs
                                 (0,duration), # time interval
                                 y0, # initial condition
                                 dense_output=True,
                                 rtol=1e-6, atol=1e-9)
    e,r,a = ode_result.y
    t = ode_result.t
    return t, e, r, a


def compare_regulation_functions(fE_reg_funs):
    fixed_params = {'sigma':1, 'a_sat':0.01}
    k_values = [0.1, 0.25, 1, 2.5]
    f, axs = plt.subplots(ncols=2,nrows=2,figsize=(15,10))
    for k,ax in zip(k_values, axs.ravel()):
        params = fixed_params
        params['k'] = k
        for fun_name,fE_fun in fE_reg_funs:
            t,e,r,a = simulate_regulation(params=params,
                                                              fE_reg_fun=fE_fun,
                                                              duration=40,
                                                              e0=0, r0=0.5, a0=0.5)
            ax.plot(t, params['k'] * e, label=fun_name)
        ax.set_title(f'k = {k}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Growth rate')
        ax.legend()
    plt.show()

print('Exploration of the model behavior (press Run Interact to update)')
@interact_manual
def display_model_behavior(fE=(0,1,0.01), k=(0.01,10,0.01), E_0=(0,10,1), R_0=(0,10,1), A_0=(0,10,1), duration=(5,60,5)):
    params = {'sigma':1, 'k':k, 'a_sat':0.01, 'f_E':fE}
    t, E, R, A, V, e, r, a = simulate(params=params, duration=duration, E0=E_0, R0=R_0, A0=A_0)
    growth_fit = np.polyfit(t[-5:], np.log(V[-5:]), 1)
    f, (ax1,ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
    ax1.plot(t, e, 'g', label='e')
    ax1.plot(t, r, 'b', label='r')
    ax1.plot(t, a, 'm', label='a')
    ax1.plot(t, e + r + a, '--k')
    ax1.legend()
    ax1.set_ylabel('concentration or mass fraction')
    ax2.semilogy(t, V, 'k')
    ax2.semilogy([t[0], t[-1]], np.exp(np.polyval(growth_fit, [t[0], t[-1]])), 'r')
    ax2.set_title(f'Growth rate at the end: {growth_fit[0]:.3f}')
    ax2.set_ylabel('Cell volume or mass')
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time')
    plt.show()

print('Exploration of model behavior using the concentration ODEs (press Run Interact to update)')
@interact_manual
def display_model_conc_behavior(fE=(0,1,0.01), k=(0.01,10,0.01), duration=(5,60,5)):
    params = {'sigma':1, 'k':k, 'a_sat':0.01, 'f_E':fE}
    e0, r0, a0 = 0, 0.5, 0.5
    t, e, r, a = simulate_conc(params=params, duration=duration, e0=e0, r0=r0, a0=a0)
    gr = params['k'] * e
    plot_conc_sim(t,e,r,a,gr)

print('Exploration of how the allocation strategy impacts the growth rate (press Run Interact to update)')
@interact_manual
def explore_allocation(k=(0.01,10,0.01), duration=(5,60,5)):
    params = {'sigma':1, 'k':k, 'a_sat':0.01}
    e0, r0, a0 = 0, 0.5, 0.5
    fE_vec = np.linspace(0,1,21)
    cmap = plt.cm.get_cmap()
    f, (ax1,ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
    end_growth_rates = []
    for fE in fE_vec:
        params['f_E'] = fE
        color = cmap.colors[np.int(fE*(cmap.N-1))]
        t, e, r, a = simulate_conc(params=params, duration=duration, e0=e0, r0=r0, a0=a0)
        end_growth_rates.append(params['k']*e[-1])
        ax1.plot(t, e, color=color)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('E concentration (e)')
    ax2.plot(fE_vec, end_growth_rates,'-ro')
    ax2.set_xlabel('fE')
    ax2.set_ylabel('End growth rate')
    plt.show()
