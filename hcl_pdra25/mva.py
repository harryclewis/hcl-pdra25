"""
hcl_pdra25: mva.py

Minimum variance analysis functions
"""
# 3rd party imports
import numpy as np

# 1st party imports
from .data import get_parameter


""" Simple MVA script, inspired by irfu-python package"""
def mva(inp_data, flag='mvar'):

    n_t = inp_data.shape[0]
    idx_1, idx_2 = [[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]

    if flag in ["mvar", "<bn>=0"]:
        m_mu_nu_m = np.mean(inp_data[:, idx_1] * inp_data[:, idx_2], 0)
        m_mu_nu_m -= np.mean(inp_data, 0)[idx_1] * np.mean(inp_data, 0)[idx_2]
    else:
        m_mu_nu_m = np.mean(inp_data[:, idx_1] * inp_data[:, idx_2], 0)

    m_mu_nu = np.array(
        [m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]],
    )

    # Compute eigenvalues and eigenvectors
    [lamb, lmn] = np.linalg.eig(m_mu_nu)

    # Sort eigenvalues
    lamb, lmn = [lamb[lamb.argsort()[::-1]], lmn[:, lamb.argsort()[::-1]]]

    # ensure that the frame is right handed
    lmn[:, 2] = np.cross(lmn[:, 0], lmn[:, 1])

    out_data = (lmn.T @ inp_data.T).T

    return out_data, lamb, lmn


""" Routine to compute Hybrid-MVA """
def Hybrid_MVA(parameter, int1, int2, verbose=True):

    # Retrieve vector data
    win1 = get_parameter(parameter, int1[0], int1[1])
    win2 = get_parameter(parameter, int2[0], int2[1])
    win = get_parameter(parameter, int1[0], int2[1])

    # Average the intervals 
    B1 = np.nanmean(win1.values, axis=0)
    B2 = np.nanmean(win2.values, axis=0)

    # Determine the N vector
    N_vec = np.cross(B1, B2) / np.linalg.norm(np.cross(B1, B2))

    # Determine LMN
    L_vec_MVA, _, _, _, _, _ = MVA_Sonnerup(win.values, verbose=False)

    # Use max-var (L) to create coordinate system
    M_vec = np.cross(N_vec, L_vec_MVA)
    L_vec = np.cross(M_vec, N_vec)

    # Shorthand printing
    if verbose:
        print(f'L  = {L_vec}')
        print(f'M  = {M_vec}')
        print(f'N  = {N_vec}')

    return L_vec, M_vec, N_vec