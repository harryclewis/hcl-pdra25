"""
hcl_pdra25: plasma.py

Contains functions for calculating plasma parameters, terms, etc.
"""

import numpy as np
from scipy import constants
from speasy.signal.resampling import interpolate


""" Routine to calculate the plasma beta from Speasy-like variables in SI units """
def calculate_beta(n_SI, T_SI, B_mag_SI):

    # Interpolate magnetic field onto temperature cadence
    B_mag_ds_SI = interpolate(T_SI, B_mag_SI)

    # convert to SI units
    n = n_SI.values * 1e6
    T = T_SI.values * constants.elementary_charge
    B_mag = B_mag_ds_SI.values * 1e-9

    # calculate pressures
    P_th = np.multiply(n, T)
    P_mag = np.square(B_mag) / (2 * constants.mu_0)

    # Plasma beta is P_th/P_mag
    return np.squeeze(np.divide(P_th, P_mag))
