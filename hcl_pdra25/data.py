"""
hcl_pdra25: data.py

Contains useful data functions.
"""

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