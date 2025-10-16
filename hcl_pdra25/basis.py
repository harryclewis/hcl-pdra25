"""
hcl_pdra25: basis.py

Routines to define and change bases and coordinate systems.
"""

import numpy as np


""" Define a local FAC system based on b and v (in the same coordinate system)"""
def local_FAC(b, v):

    b_hat = np.einsum('ti,t->ti', b.values, np.reciprocal(np.linalg.norm(b.values, axis=1)))
    v_hat = np.einsum('ti,t->ti', v.values, np.reciprocal(np.linalg.norm(v.values, axis=1)))

    # p1 represents b x v
    p1 = np.einsum('ti,t->ti', np.cross(b_hat, v_hat), np.reciprocal(np.linalg.norm(np.cross(b_hat, v_hat), axis=1)))

    # p2 represents b x b x v aka the b_perp component of v
    p2 = np.cross(b_hat, p1)

    # array of rotation matrices. To rotate do R^-1 T R, i.e. einsum('...ij,...jk,...kl->...il')
    rot = np.array([b_hat.T, p1.T, p2.T]).T
    return rot


""" Function to rotate tensor time series T by rotation matrix """
def rotate_tensor(R, T):

    # rotate time series by fixed R
    if len(R.shape) == 2:
        T_rot = np.einsum('ij,tjk,kl->til', np.linalg.inv(R), T, R)

    # rotate time series by time series of R
    elif len(R.shape) >= 3:
        T_rot = np.einsum('tij,tjk,tkl->til', np.linalg.inv(R), T, R)

    return T_rot


""" Function to rotate vector time series v by rotation matrix R """
def rotate_vector(R, v):

    # rotate time series by fixed R
    if len(R.shape) == 2:
        v_rot = np.einsum('ij,tj->ti', R, v)

    # rotate time series by time series of R
    elif len(R.shape) >= 3:
        v_rot = np.einsum('tij,tj->ti', R, v)

    return v_rot