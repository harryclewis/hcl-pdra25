"""
hcl_pdra25: mva.py

Minimum variance analysis functions
"""

import numpy as np


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