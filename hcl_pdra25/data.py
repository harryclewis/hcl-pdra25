"""
hcl_pdra25: data.py

Contains useful data functions.
"""

import cdflib
import numpy as np
import pandas as pd
from speasy import amda


""" Wrapper around speasy.amda.get_parameter to fill NaNs """
def get_parameter(name, ts, tf):
    param = amda.get_parameter(name, ts, tf)
    if not param is None:
        param.replace_fillval_by_nan(inplace=True)
    return param


""" Class to allow common syntax between custom variables and speasy variables """
class CustomVariable:
    def __init__(self, time, data):
        self.time = time
        self.values = data


""" Function to extract and trim data from a cdf file"""
def extract_from_CDF(filename, varname, t_s, t_f):

    # open the CDF file
    cdf_file = cdflib.CDF(filename)

    # extract the time and values
    t = pd.to_datetime(cdf_file.varget('TIME'), unit='s').to_numpy()
    v = cdf_file.varget(varname)

    # trim to desired range and return
    idx = np.argwhere(np.where((t >= pd.to_datetime(t_s).to_numpy()) & (t <= pd.to_datetime(t_f).to_numpy()), True, False))
    return CustomVariable(np.squeeze(t[idx]), np.squeeze(v[idx]))