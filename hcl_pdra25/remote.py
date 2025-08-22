"""
hcl_pdra25: remote.py

Functions specific to the remote sensing missions WISPR and EUI from PSP and SolO.
Note that routines in this section require SunPy and dependencies to be installed.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ImageNormalize, PowerStretch
import astropy.wcs
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
import sunpy
from sunpy.coordinates.frames import Helioprojective


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