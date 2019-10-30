import numpy as np # for basic vector computation in python
import scipy.integrate as integrate # needed to solve ODEs
import matplotlib.pyplot as plt # for plotting

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
    
