"""
hcl_pdra25: plasma.py

Contains functions for calculating plasma parameters, terms, etc.
"""

import numpy as np
from scipy import constants
from speasy.signal.resampling import interpolate


""" Routine to calculate the plasma beta from Speasy-like variables in default units """
def calculate_beta(n_inp, T_inp, B_mag_inp):

    # Interpolate magnetic field onto temperature cadence
    B_mag_ds_inp = interpolate(T_inp, B_mag_inp)

    # convert to SI units
    n = n_inp.values * 1e6
    T = T_inp.values * constants.elementary_charge
    B_mag = B_mag_ds_inp.values * 1e-9

    # calculate pressures
    P_th = np.multiply(n, T)
    P_mag = np.square(B_mag) / (2 * constants.mu_0)

    # Plasma beta is P_th/P_mag
    return np.squeeze(np.divide(P_th, P_mag))


""" Calculate the firehose parameter, alpha = 1 - mu0*(P_parr - P_perp)/B^2. If no T_perp2, use T_perp1 again. """
def calculate_alpha(n_inp, T_parr_inp, T_perp1_inp, T_perp2_inp, B_inp):

    # convert to SI units
    T_parr = T_parr_inp.values * constants.elementary_charge
    T_perp = (T_perp1_inp.values * constants.elementary_charge + T_perp2_inp.values * constants.elementary_charge)/2
    n = np.squeeze(n_inp.values) * 1e6
    B_mag = np.linalg.norm(B_inp.values, axis=1) * 1e-9

    # calculate the pressure components
    P_parr = np.einsum('t,t->t', T_parr, n)
    P_perp = np.einsum('t,t->t', T_perp, n)
    mu0_P_diff = constants.mu_0 * np.subtract(P_parr, P_perp)
    
    # calculate alpha
    alpha = 1 - np.einsum('t,t->t', mu0_P_diff, np.square(np.reciprocal(B_mag)))
    return alpha


""" Calculate the local Alfven speed from Speasy variables"""
def calculate_vA(n_inp, B_inp, species: str = 'p'):

    assert species in ['e','p'], "Available species options are 'e','i'."
    assert len(n_inp.values) == len(B_inp.values), "B must be interpolated onto n cadence prior to calling."

    # convert to SI units
    n = np.squeeze(n_inp.values) * 1e6
    B_mag = np.linalg.norm(B_inp.values, axis=1) * 1e-9

    # get correct mass for species selection
    m = constants.m_p if species == 'p' else constants.m_e
    rho = m * n
    v_A = np.einsum('t,t->t', B_mag, np.reciprocal(np.sqrt(constants.mu_0 * rho)))
    return v_A