"""
hcl_pdra25: data.py

Contains useful data functions.
"""


import numpy as np


""" Wrapper around speasy.amda.get_parameter to fill NaNs """
def get_parameter(name, ts, tf):
    from speasy import amda

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
    import cdflib
    import pandas as pd
    from speasy import amda

    # open the CDF file
    cdf_file = cdflib.CDF(filename)

    # extract the time and values
    t = pd.to_datetime(cdf_file.varget('TIME'), unit='s').to_numpy()
    v = cdf_file.varget(varname)

    # trim to desired range and return
    idx = np.argwhere(np.where((t >= pd.to_datetime(t_s).to_numpy()) & (t <= pd.to_datetime(t_f).to_numpy()), True, False))
    return CustomVariable(np.squeeze(t[idx]), np.squeeze(v[idx]))


"""Function to convert .tplot file into set of .npy files"""
def tplot_to_npy(filename: str = '', vars: list = [], save_dir: str = ''):
    from pyspedas import tplot_restore, get_data
    
    assert not filename == '', "'filename' argument must be a string e.g. 'data/var.tplot'"
    assert not vars == [], "Provide a list of tplot variable names to extract"
    assert not save_dir == '', "'save_dir' argument must be a directory path string e.g. 'data/npy_output/'"
    
    # load from the .tplot file into memory
    tplot_restore(filename)

    # loop through and save variables
    for var in vars:
        xr = get_data(var, xarray=True)
        t = xr.time.values
        d = xr.values
        np.save(save_dir+var+'_t', t, allow_pickle=False)
        np.save(save_dir+var, d, allow_pickle=False)