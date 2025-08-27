"""
hcl_pdra25: remote.py

Functions specific to the remote sensing missions WISPR and EUI from PSP and SolO.
Note that routines in this section require SunPy and dependencies to be installed.
"""

from astropy.coordinates import SkyCoord
from astropy.timeseries import TimeSeries
import astropy.units as u
from astropy.visualization import ImageNormalize, PowerStretch
import astropy.wcs
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
import sunpy
from sunpy.coordinates.frames import Helioprojective


""" 
    Create a j-map from a sequence of map objects.
    Rotates the maps onto the preferred axis provided.
    Integrates (averages) the other axis to get j-map.

    Requires the spice kernel 'SPP_RTN' to be loaded from PSP.
"""
def produce_jmap(i_start, i_end, map_sequence, X_vec, Y_vec, Z_vec, rot_axis='X', mask_value=1e-14, bad_idxs=[]):

    length = (i_end-i_start)
    j_map_dates = []
    j_map = np.empty((length, 736))

    for t_ind, t_map in enumerate(map_sequence[i_start:i_end+1]):
        t_date = t_map.date

        """ 1. find the angle """
        # define coordinates at this time
        t_L_coord = SkyCoord(X_vec[0] * 1.5e6 * u.km, X_vec[1] * 1.5e6 * u.km, X_vec[2]* 1.5e6 * u.km, obstime=t_date, frame='spice_SPP_RTN', representation_type='cartesian')
        t_M_coord = SkyCoord(Y_vec[0] * 1.5e6 * u.km, Y_vec[1] * 1.5e6 * u.km, Y_vec[2] * 1.5e6 * u.km, obstime=t_date, frame='spice_SPP_RTN', representation_type='cartesian')
        t_N_coord = SkyCoord(Z_vec[0] * 1.5e6 * u.km, Z_vec[1] * 1.5e6 * u.km, Z_vec[2] * 1.5e6 * u.km, obstime=t_date, frame='spice_SPP_RTN', representation_type='cartesian')
        t_zero_coord = SkyCoord(0 * u.km, 0 * u.km, 0 * u.km, obstime=t_date, frame='spice_SPP_RTN', representation_type='cartesian')

        # transform the coordinates into the map frame
        t_L_hcs = t_L_coord.transform_to(t_map.coordinate_frame)
        t_M_hcs = t_M_coord.transform_to(t_map.coordinate_frame)
        t_N_hcs = t_N_coord.transform_to(t_map.coordinate_frame)
        t_zero_hcs = t_zero_coord.transform_to(t_map.coordinate_frame)

        # account for longitude wrapping around
        t_L_hcs_lon = t_L_hcs.data.lon.value*57.29583337126-360 if t_L_hcs.data.lon.value*57.29583337126 > 180 else t_L_hcs.data.lon.value*57.29583337126
        t_M_hcs_lon = t_M_hcs.data.lon.value*57.29583337126-360 if t_M_hcs.data.lon.value*57.29583337126 > 180 else t_M_hcs.data.lon.value*57.29583337126
        t_N_hcs_lon = t_N_hcs.data.lon.value*57.29583337126-360 if t_N_hcs.data.lon.value*57.29583337126 > 180 else t_N_hcs.data.lon.value*57.29583337126
        
        # calculate the counterclockwise angle in radians.
        if rot_axis == 'Y':
            theta = np.atan2((t_M_hcs.data.lat.value*57.29583337126 - t_zero_hcs.data.lat.value*57.29583337126), (t_M_hcs_lon - t_zero_hcs.data.lon.value*57.29583337126))
        else:
            raise Exception('Function produce_jmap is currently only set up to rotate onto the Y axis.')

        """ 2. Get the cut of the map"""
        # rotate the map
        t_map_rot = t_map.rotate(angle=(-90-theta*57.29583337126)*u.deg, missing=np.nan)

        # take a cut in the preferred area
        img_data = t_map_rot.data
        img_data = np.where(img_data<=mask_value,np.nan,img_data)
        img_data_masked = img_data[506:1241+1,377:411+1]
        img_data_integrated = np.nanmean(img_data_masked, axis=1)

        # add the slice to the j_map
        j_map_dates.append(t_date)
        j_map[t_ind,:] = img_data_integrated

    # convert j_map_dates into a more useful format
    j_map_dates = np.array([i.datetime64 for i in j_map_dates])

    # Return a plot to show that it worked OK
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot()

    for idx in bad_idxs:
        j_map = np.delete(j_map, (idx), axis=0)
        j_map_dates = np.delete(j_map_dates, (idx), axis=0)

    X = j_map_dates
    Y = np.arange(j_map.shape[-1]).T
    Z = j_map.T

    ax.imshow(Z, aspect='auto', interpolation='none', cmap='gray', norm=LogNorm(vmin=5e-14,vmax=4e-12))
    ax.set_ylim(575,325)
    ax.set_xticks(np.arange(0,96,5), minor=True)
    plt.show()

    return j_map, j_map_dates


"""
    Interpolate over 'bad' jmap indices (e.g. contaminated by dust)
    Uses a Savitzky-Golay filter to smooth, as a separate output

    Returns regular timeseries of dates, the interpolated data, and filtered data
"""
def interpolate_jmap(data, t_s="", remove_idxs: list=[], dt=300*u.s, n_samples=None, window_length: int=2, polyorder: int=0):

    assert t_s != "", "A start time is needed for the regular time series interpolation."

    if n_samples is None:
        n_samples = data.shape[0]

    # Define a regular time series
    ts_regular = np.array(TimeSeries(time_start=t_s, time_delta=dt, n_samples=n_samples).to_pandas().index.values)

    # Define a meshgrid for the interpolator object (currently works in data coords i.e. indices)
    _X = np.arange(0, data.shape[0])
    _Y = np.arange(0, data.shape[-1])
    _Xg, _Yg = np.meshgrid(_X, _Y, indexing='ij', sparse=True)

    # Remove the 'bad' indices from data (axis 0) and _X
    for idx in remove_idxs:
        data = np.delete(data, (idx), axis=0)
        _X = np.delete(_X, (idx), axis=0)

    # Define interpolator object
    interp = RegularGridInterpolator((_X, _Y), data)

    # Interpolate data to regular grid
    data_interp = interp((_Xg, _Yg))

    # Apply filter to interpolated data
    data_filter = savgol_filter(data_interp, window_length=window_length, polyorder=polyorder, axis=0)

    return ts_regular, data_interp, data_filter


""" Combine two WISPR maps, a la heliophysicsPy Summer School 2024 """
def combine_wispr_maps(inner_map, outer_map):

    # define a normal map
    wispr_norm = ImageNormalize(vmin=0, vmax=0.5e-11, stretch=PowerStretch(1/2.2))

    # get a reference coord at the center of the inner_map
    ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=Helioprojective(observer=inner_map.observer_coordinate, obstime=inner_map.date))
    
    # create wider map
    outshape = (360*2, int(360*3.5))
    new_header = sunpy.map.make_fitswcs_header(
        outshape, 
        ref_coord,
        reference_pixel=u.Quantity([60*u.pixel, 430*u.pixel]), 
        scale=u.Quantity([0.1*u.deg/u.pixel, 0.1*u.deg/u.pixel]), 
        projection_code="CAR",
    )

    # reproject both maps onto new wider map
    out_wcs = astropy.wcs.WCS(new_header)
    with Helioprojective.assume_spherical_screen(inner_map.observer_coordinate):
        array, footprint = reproject_and_coadd((inner_map, outer_map), out_wcs, outshape,
                                               reproject_function=reproject_interp, match_background=True)

    combined_map = sunpy.map.Map((array, new_header))
    combined_map.plot_settings["norm"] = wispr_norm
    combined_map.plot_settings["cmap"] = "viridis"
    return combined_map