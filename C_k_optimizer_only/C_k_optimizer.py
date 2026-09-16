# --- IMPORTS ---
import numpy as np
from scipy.signal import find_peaks
from params import DropletParams


# --- BACKTRACKING LINE SEARCH ALGORITHM ---
# INPUTS:
# C_k: current value of C_k
# p_k: search direction for C_k
# derivative: derivative of the error function at given C_k
# params: DropletParams dataclass w all parameters
#
# OUTPUTS:
# err: error value between current deformation and target deformation
def alpha_backtrack(C_k, p_k, derivative, params: 'DropletParams'):
    alpha = 1
    rho = 0.5
    mu = 1e-4

    phi_0 = err_from_C_k(C_k, params)

    while( err_from_C_k(C_k+(alpha*p_k), params) > phi_0 + mu*alpha*p_k*derivative ):
        alpha *= rho

    return (C_k + (alpha*p_k))


# --- FIND SEARCH DIRECTION ---
# INPUTS:
# C_k: current value of C_k
# params: DropletParams dataclass w all parameters
#
# OUTPUTS:
# err: error value between current deformation and target deformation
def find_search_dir(C_k, params: 'DropletParams'):
    h = 0.001 # Step size, change later?

    direction = -(err_from_C_k(C_k+h, params) - err_from_C_k(C_k, params) ) / h

    return (direction/np.abs(direction)), direction


# --- FUNCTION WE WISH TO MINIMIZE ---
# INPUTS:
# C_k: current value of C_k
# B_wo_C_k: B divided by C_k (B/C_k)
# params: DropletParams dataclass w all parameters
#
# OUTPUTS:
# err: error value between current deformation and target deformation
def err_from_C_k(C_k, params: 'DropletParams'):
    B = params.B_wo_C_k * C_k

    y_vals = (params.C/B) + np.exp(-params.A*params.timespace/2) * ( (params.y_0_init - (params.C/B))*np.cos(params.omega*params.timespace) + (1/params.omega)*(params.y_0_init + (params.A/2)*(params.y_0_init-(params.C/B))*np.sin(params.omega*params.timespace)) )

    peaks, _ = find_peaks(y_vals)
    updated_deformation = y_vals[peaks[0]]

    err = np.abs(updated_deformation - params.target_deformation)
    return err
    

# --- FIND C_D BEST FIT ---
# 
# INPUTS:
# y_vals: array of normalized displacements
# target_deform: target deformation value (peaks[0] should equal this eventually)
# C_k_init: initial guess for C_k
# 
# OUTPUTS:
# C_k_updated: updated value of C_k hits target deformation
def find_C_k(C_k_init, max_err, params: 'DropletParams'):

    C_k_updated = C_k_init

    while np.abs(err_from_C_k(C_k_updated, params)) >= max_err:
        p_k, derivative = find_search_dir(C_k_updated, params)
        C_k_updated = alpha_backtrack(C_k_updated, p_k, derivative, params)

    return C_k_updated, err_from_C_k(C_k_updated, params)
    